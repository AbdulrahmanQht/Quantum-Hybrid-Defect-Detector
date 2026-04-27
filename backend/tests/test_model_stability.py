"""
tests/test_model_stability.py
==============================
Merged from test_determinism.py + test_soak.py.

Covers two complementary dimensions of model stability:

PART A — Determinism and reproducibility
    Verifies that the same weights + input always produce identical outputs.
    Required for:
        - Academic publication (reported numbers must be reproducible)
        - Debugging (non-determinism masks real bugs)
        - CI (flaky inference means flaky tests)
    Tests:
        1. Inference determinism     — model(x) == model(x) every run
        2. Checkpoint round-trip     — save → reload → same outputs
        3. Batch-size invariance     — model(x[0:1]) == model(x)[0] (BatchNorm check)
        4. EMA apply/restore         — quantum weight save/restore is lossless
        5. Preprocessing determinism — inference transform is deterministic

PART B — Soak and memory leak detection
    Runs 500 sequential inferences and checks:
        - CPU/GPU memory growth < 50 MB  (tensor accumulation / CUDA cache leak)
        - Prediction class does not drift (BatchNorm in wrong mode)
        - Model parameters unchanged     (no silent weight mutation)
        - File descriptors do not leak   (logger / preprocessing handles)
        - EMA shadow not mutated during eval mode

Both parts share the same model and device fixtures (loaded once per module).

Results saved to:
    data/results_tests/determinism_cnn.json
    data/results_tests/determinism_reproducibility.json
    data/results_tests/soak_cnn.json
    data/results_tests/soak_qnn_cpu.json
    data/results_tests/soak_combined.json
    data/results_tests/model_stability_combined.json

Run:
    pytest tests/test_model_stability.py -v
    pytest tests/test_model_stability.py -v -k "Determinism"
    pytest tests/test_model_stability.py -v -k "Soak"
    pytest tests/test_model_stability.py -v -s          # prints per-100 memory readings
"""

from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

BACKEND_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BACKEND_DIR))

RESULTS_DIR = BACKEND_DIR / "data" / "results_tests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Deformation", "Deposition", "Disconnect",
    "Misalignment", "Obstacle", "Rupture",
]

# ── Soak config ─────────────────────────────────────────────────────────────
N_SOAK_IMAGES         = 500
MEMORY_GROWTH_LIMIT_MB = 50.0
SAMPLE_EVERY          = 100

# ── Batch-size invariance tolerance ─────────────────────────────────────────
BATCH_TOL = 2e-4

_CNN_CKPT = next(
    (str(BACKEND_DIR / p) for p in [
        "models/cnn.pth",
    ] if (BACKEND_DIR / p).exists()), None
)
_QNN_CPU_CKPT = next(
    (str(BACKEND_DIR / p) for p in [
        "models/qnn_cpu.pth",
    ] if (BACKEND_DIR / p).exists()), None
)
_QNN_GPU_CKPT = next(
    (str(BACKEND_DIR / p) for p in [
        "models/qnn_gpu.pth",
    ] if (BACKEND_DIR / p).exists()), None
)


# ═══════════════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════════════

def _save(data: dict, filename: str) -> None:
    path = RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"\nSaved → {path}")


def _cpu_memory_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / 1e6
        except ImportError:
            return 0.0


def _gpu_memory_mb() -> float:
    return torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


def _param_snapshot(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: param.data.clone() for name, param in model.named_parameters()}


def _params_unchanged(
    model: torch.nn.Module, snapshot: Dict[str, torch.Tensor]
) -> Tuple[bool, Optional[str]]:
    for name, param in model.named_parameters():
        if name in snapshot and not torch.equal(param.data, snapshot[name]):
            diff = (param.data - snapshot[name]).abs().max().item()
            return False, f"'{name}' changed (max diff {diff:.2e})"
    return True, None


def _count_fds() -> int:
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except Exception:
        return -1


# ═══════════════════════════════════════════════════════════════════════════
# Shared fixtures  (scope="module" — each model loaded once for all tests)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def fixed_images():
    """50 deterministic images — identical across machines."""
    torch.manual_seed(2024)
    return torch.rand(50, 3, 384, 384)


@pytest.fixture(scope="module")
def single_image():
    torch.manual_seed(99)
    return torch.rand(1, 3, 384, 384)


@pytest.fixture(scope="module")
def soak_images():
    """500 fixed images for soak run."""
    torch.manual_seed(42)
    return torch.rand(N_SOAK_IMAGES, 3, 384, 384)


@pytest.fixture(scope="module")
def cnn_model(device):
    if _CNN_CKPT is None:
        pytest.skip("CNN checkpoint not found")
    from backend.models.cnn import CNN
    m = CNN(num_classes=6)
    m.load_model(_CNN_CKPT, device)
    m.eval()
    return m


@pytest.fixture(scope="module")
def qnn_cpu_model(device):
    if _QNN_CPU_CKPT is None:
        pytest.skip("QNN-CPU checkpoint not found")
    from backend.models.qnn_cpu import HybridQnnCPU
    m = HybridQnnCPU(num_classes=6, n_qubits=6, q_depth=2)
    m.load_model(_QNN_CPU_CKPT, device)
    m.eval()
    return m


@pytest.fixture(scope="module")
def qnn_gpu_model():
    if not torch.cuda.is_available():
        pytest.skip("QNN_GPU tests require a CUDA device")
    if _QNN_GPU_CKPT is None:
        pytest.skip("QNN-GPU checkpoint not found")
    gpu_device = torch.device("cuda")
    from backend.models.qnn_gpu import HybridQnnGPU
    m = HybridQnnGPU(num_classes=6, n_qubits=6, q_depth=2)
    m.load_model(_QNN_GPU_CKPT, gpu_device)
    m.eval()
    return m


# ── Soak helper (used by both CNN and QNN_CPU soak fixtures) ─────────────────

def _run_soak(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    model_name: str,
) -> dict:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    snap     = _param_snapshot(model)
    fd_start = _count_fds()
    cpu_start, gpu_start = _cpu_memory_mb(), _gpu_memory_mb()
    cpu_readings, gpu_readings = [cpu_start], [gpu_start]

    first_img = images[0:1].to(device)
    with torch.no_grad():
        ref_pred = int(model(first_img).argmax(1).item())

    t_start = time.perf_counter()
    drifts  = 0
    for i in range(N_SOAK_IMAGES):
        with torch.no_grad():
            model(images[i:i+1].to(device))
        if i % SAMPLE_EVERY == 0:
            with torch.no_grad():
                if int(model(first_img).argmax(1).item()) != ref_pred:
                    drifts += 1
            cpu_now, gpu_now = _cpu_memory_mb(), _gpu_memory_mb()
            cpu_readings.append(cpu_now)
            gpu_readings.append(gpu_now)
            print(
                f"[{model_name}] {i+1}/{N_SOAK_IMAGES} "
                f"CPU {cpu_now:.1f} MB  GPU {gpu_now:.1f} MB"
            )

    elapsed  = time.perf_counter() - t_start
    fd_end   = _count_fds()
    ok, err  = _params_unchanged(model, snap)
    cpu_grow = max(cpu_readings) - cpu_start
    gpu_grow = max(gpu_readings) - gpu_start

    return {
        "model": model_name,
        "n_images": N_SOAK_IMAGES,
        "elapsed_s": round(elapsed, 2),
        "throughput_img_per_s": round(N_SOAK_IMAGES / elapsed, 2),
        "cpu_memory_start_mb": round(cpu_start, 2),
        "cpu_memory_peak_mb": round(max(cpu_readings), 2),
        "cpu_memory_growth_mb": round(cpu_grow, 2),
        "gpu_memory_start_mb": round(gpu_start, 2),
        "gpu_memory_peak_mb": round(max(gpu_readings), 2),
        "gpu_memory_growth_mb": round(gpu_grow, 2),
        "memory_growth_limit_mb": MEMORY_GROWTH_LIMIT_MB,
        "cpu_leak_detected": cpu_grow > MEMORY_GROWTH_LIMIT_MB,
        "gpu_leak_detected": gpu_grow > MEMORY_GROWTH_LIMIT_MB,
        "prediction_drifts": drifts,
        "prediction_drift_detected": drifts > 0,
        "model_params_unchanged": ok,
        "param_error": err,
        "fd_start": fd_start,
        "fd_end": fd_end,
        "fd_leak_detected": (fd_end - fd_start) > 20 if fd_start > 0 else None,
    }


@pytest.fixture(scope="module")
def cnn_soak_report(cnn_model, soak_images, device):
    r = _run_soak(cnn_model, soak_images, device, "CNN")
    r["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save(r, "soak_cnn.json")
    return r


@pytest.fixture(scope="module")
def qnn_cpu_soak_report(qnn_cpu_model, soak_images, device):
    r = _run_soak(qnn_cpu_model, soak_images, device, "QNN_CPU")
    r["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save(r, "soak_qnn_cpu.json")
    return r

@pytest.fixture(scope="module")
def qnn_gpu_soak_report(qnn_gpu_model, soak_images):
    gpu_device = torch.device("cuda")
    r = _run_soak(qnn_gpu_model, soak_images, gpu_device, "QNN_GPU")
    r["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save(r, "soak_qnn_gpu.json")
    return r
# ═══════════════════════════════════════════════════════════════════════════
# PART A — DETERMINISM TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestInferenceDeterminism:
    """Same weights + same input must produce bit-identical output every call."""

    def test_cnn_deterministic_over_50_images(self, cnn_model, fixed_images, device):
        imgs = fixed_images.to(device)
        with torch.no_grad():
            out1, out2 = cnn_model(imgs), cnn_model(imgs)
        assert torch.equal(out1, out2), (
            "CNN inference is not deterministic. Verify model.eval() is called."
        )

    def test_cnn_deterministic_across_five_runs(self, cnn_model, single_image, device):
        img = single_image.to(device)
        with torch.no_grad():
            outputs = [cnn_model(img) for _ in range(5)]
        for i, out in enumerate(outputs[1:], 1):
            assert torch.equal(outputs[0], out), (
                f"CNN run {i+1} output differs from run 1."
            )

    def test_qnn_cpu_deterministic_over_10_images(self, qnn_cpu_model, fixed_images, device):
        imgs = fixed_images[:10].to(device)
        with torch.no_grad():
            out1, out2 = qnn_cpu_model(imgs), qnn_cpu_model(imgs)
        assert torch.equal(out1, out2), (
            "QNN_CPU inference is not deterministic. "
            "Check _apply_ema() / _restore_live() are balanced."
        )
        
    @pytest.mark.requires_gpu
    def test_qnn_gpu_deterministic_over_10_images(self, qnn_gpu_model, fixed_images):
        imgs = fixed_images[:10].to(torch.device("cuda"))
        with torch.no_grad():
            out1, out2 = qnn_gpu_model(imgs), qnn_gpu_model(imgs)
        assert torch.equal(out1, out2), "QNN_GPU inference is not deterministic."

    def test_predict_method_deterministic(self, cnn_model, single_image, device):
        r1 = cnn_model.predict(single_image, device, CLASS_NAMES)
        r2 = cnn_model.predict(single_image, device, CLASS_NAMES)
        assert r1["predicted_index"] == r2["predicted_index"]
        assert abs(r1["confidence"] - r2["confidence"]) < 1e-5

    def test_save_determinism_report(self, cnn_model, fixed_images, device):
        imgs = fixed_images.to(device)
        with torch.no_grad():
            out1, out2 = cnn_model(imgs), cnn_model(imgs)
        max_diff = (out1 - out2).abs().max().item()
        _save({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": "CNN", "n_images": 50, "n_runs": 2,
            "max_logit_diff_between_runs": max_diff,
            "deterministic": max_diff == 0.0,
        }, "determinism_cnn.json")
        assert max_diff == 0.0


class TestCheckpointRoundTrip:
    """Save → reload weights → outputs must match exactly."""

    def test_cnn_checkpoint_round_trip(self, cnn_model, fixed_images, device):
        imgs = fixed_images[:8].to(device)
        with torch.no_grad():
            out_before = cnn_model(imgs).clone()
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            path = tmp.name
        try:
            cnn_model.save_model(path)
            from backend.models.cnn import CNN
            m2 = CNN(num_classes=6)
            m2.load_model(path, device)
            m2.eval()
            with torch.no_grad():
                out_after = m2(imgs)
            assert torch.equal(out_before, out_after), (
                "CNN checkpoint round-trip failed — weight serialization is lossy."
            )
        finally:
            Path(path).unlink(missing_ok=True)

    def test_qnn_cpu_checkpoint_round_trip_with_ema(
        self, qnn_cpu_model, fixed_images, device
    ):
        imgs = fixed_images[:4].to(device)
        with torch.no_grad():
            out_before = qnn_cpu_model(imgs).clone()
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            path = tmp.name
        try:
            qnn_cpu_model.save_model(path)
            from backend.models.qnn_cpu import HybridQnnCPU
            m2 = HybridQnnCPU(num_classes=6, n_qubits=6, q_depth=2)
            m2.load_model(path, device)
            m2.eval()
            with torch.no_grad():
                out_after = m2(imgs)
            max_diff = (out_before - out_after).abs().max().item()
            assert max_diff < 1e-5, (
                f"QNN_CPU checkpoint round-trip max diff {max_diff:.2e} — "
                "EMA weights may not be saved/restored correctly."
            )
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.requires_gpu
    def test_qnn_gpu_checkpoint_round_trip_with_ema(self, qnn_gpu_model, fixed_images):
        imgs = fixed_images[:4].to(torch.device("cuda"))
        with torch.no_grad():
            out_before = qnn_gpu_model(imgs).clone()
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            path = tmp.name
        try:
            qnn_gpu_model.save_model(path)
            from backend.models.qnn_gpu import HybridQnnGPU
            m2 = HybridQnnGPU(num_classes=6, n_qubits=6, q_depth=2)
            m2.load_model(path, torch.device("cuda"))
            m2.eval()
            with torch.no_grad():
                out_after = m2(imgs)
            max_diff = (out_before - out_after).abs().max().item()
            assert max_diff < 1e-5, f"QNN_GPU checkpoint round-trip max diff {max_diff:.2e}"
        finally:
            Path(path).unlink(missing_ok=True)
class TestBatchSizeInvariance:
    """
    model(x[0:1]) must equal model(x)[0] within float32 tolerance.
    Failure indicates BatchNorm is in training mode — a common eval bug.
    """

    def test_cnn_single_vs_batch(self, cnn_model, fixed_images, device):
        imgs = fixed_images[:8].to(device)
        with torch.no_grad():
            batch = cnn_model(imgs)
            single = cnn_model(imgs[0:1])
        diff = (batch[0] - single[0]).abs().max().item()
        assert diff < BATCH_TOL, (
            f"CNN batch-size invariance FAIL: diff {diff:.2e}. "
            "BatchNorm may be in train() mode."
        )

    def test_cnn_all_samples_in_batch_consistent(self, cnn_model, fixed_images, device):
        imgs = fixed_images[:8].to(device)
        with torch.no_grad():
            batch = cnn_model(imgs)
        for i in range(8):
            with torch.no_grad():
                s = cnn_model(imgs[i:i+1])
            assert (batch[i] - s[0]).abs().max().item() < BATCH_TOL, (
                f"CNN sample {i}: batch vs single logit diff exceeds tolerance"
            )

    def test_qnn_cpu_single_vs_batch(self, qnn_cpu_model, fixed_images, device):
        imgs = fixed_images[:4].to(device)
        with torch.no_grad():
            batch  = qnn_cpu_model(imgs)
            single = qnn_cpu_model(imgs[0:1])
        assert (batch[0] - single[0]).abs().max().item() < BATCH_TOL


    @pytest.mark.requires_gpu
    def test_qnn_gpu_single_vs_batch(self, qnn_gpu_model, fixed_images):
        imgs = fixed_images[:4].to(torch.device("cuda"))
        with torch.no_grad():
            batch = qnn_gpu_model(imgs)
            single = qnn_gpu_model(imgs[0:1])
        assert (batch[0] - single[0]).abs().max().item() < BATCH_TOL

class TestEMALosslessness:
    """_apply_ema() → _restore_live() must leave quantum weights byte-identical."""

    def test_ema_restore_is_lossless(self, qnn_cpu_model):
        before = {
            name: param.data.clone()
            for name, param in qnn_cpu_model.q_layer.named_parameters()
        }
        backup = qnn_cpu_model._apply_ema()
        qnn_cpu_model._restore_live(backup)
        for name, param in qnn_cpu_model.q_layer.named_parameters():
            assert torch.equal(param.data, before[name]), (
                f"EMA restore introduced drift in '{name}'."
            )

    @pytest.mark.requires_gpu
    def test_qnn_gpu_ema_restore_is_lossless(self, qnn_gpu_model):
        before = {
            name: param.data.clone()
            for name, param in qnn_gpu_model.q_layer.named_parameters()
        }
        backup = qnn_gpu_model._apply_ema()
        qnn_gpu_model._restore_live(backup)
        for name, param in qnn_gpu_model.q_layer.named_parameters():
            assert torch.equal(param.data, before[name]), f"EMA restore drift in '{name}'."


    @pytest.mark.requires_gpu
    def test_qnn_gpu_ema_apply_changes_weights_when_shadow_differs(self, qnn_gpu_model):
        shadow = qnn_gpu_model._quantum_shadow
        if not shadow:
            pytest.skip("EMA shadow is empty")

        # Freshly loaded models may have EMA shadow identical to live weights.
        live_matches_shadow = all(
            torch.equal(param.data, shadow[name])
            for name, param in qnn_gpu_model.q_layer.named_parameters()
            if name in shadow
        )
        if live_matches_shadow:
            pytest.skip("EMA shadow matches live weights on loaded model")

        backup = qnn_gpu_model._apply_ema()
        changed = any(
            not torch.equal(param.data, backup[name])
            for name, param in qnn_gpu_model.q_layer.named_parameters()
            if name in backup
        )
        qnn_gpu_model._restore_live(backup)
        assert changed, "EMA apply did not swap in shadow weights"

        
    def test_ema_apply_changes_weights_when_shadow_differs(self, qnn_cpu_model):
        shadow = qnn_cpu_model._quantum_shadow
        if not shadow:
            pytest.skip("EMA shadow is empty — model loaded without training history")

        live_matches_shadow = all(
            torch.equal(param.data, shadow[name])
            for name, param in qnn_cpu_model.q_layer.named_parameters()
            if name in shadow
        )
        if live_matches_shadow:
            pytest.skip("EMA shadow matches live weights on loaded model")

        backup = qnn_cpu_model._apply_ema()
        changed = any(
            not torch.equal(param.data, backup[name])
            for name, param in qnn_cpu_model.q_layer.named_parameters()
            if name in backup
        )
        qnn_cpu_model._restore_live(backup)
        assert changed, (
            "EMA apply did not swap in shadow weights"
        )



class TestPreprocessingDeterminism:
    """Inference transform must be fully deterministic (no augmentation in eval mode)."""

    def test_inference_transform_is_deterministic(self):
        try:
            from backend.data.preprocessing import PreProcessing
            from PIL import Image
        except ImportError:
            pytest.skip("PreProcessing or Pillow not available")

        t = PreProcessing.get_transforms(img_width=384, img_height=384, is_training=False)
        img = Image.new("RGB", (256, 256), color=(100, 150, 200))
        assert torch.equal(t(img), t(img)), (
            "Inference transform is not deterministic. "
            "RandomFlip/ColorJitter must not be active in eval mode."
        )

    def test_save_reproducibility_report(self, cnn_model, fixed_images, device):
        imgs = fixed_images[:5].to(device)
        runs = []
        for _ in range(3):
            with torch.no_grad():
                runs.append(cnn_model(imgs).tolist())
        all_match = all(
            abs(runs[0][i][j] - runs[r][i][j]) < 1e-6
            for r in range(1, 3) for i in range(5) for j in range(6)
        )
        _save({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "test": "reproducibility_across_3_runs",
            "n_images": 5, "n_runs": 3,
            "all_outputs_identical": all_match,
        }, "determinism_reproducibility.json")
        assert all_match


# ═══════════════════════════════════════════════════════════════════════════
# PART B — SOAK TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestCNNSoak:

    def test_no_cpu_memory_leak(self, cnn_soak_report):
        grow = cnn_soak_report["cpu_memory_growth_mb"]
        assert grow <= MEMORY_GROWTH_LIMIT_MB, (
            f"CNN CPU memory grew {grow:.1f} MB over {N_SOAK_IMAGES} inferences "
            f"(limit {MEMORY_GROWTH_LIMIT_MB} MB). "
            "Likely cause: tensors accumulating without torch.no_grad()."
        )

    def test_no_gpu_memory_leak(self, cnn_soak_report):
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        grow = cnn_soak_report["gpu_memory_growth_mb"]
        assert grow <= MEMORY_GROWTH_LIMIT_MB, (
            f"CNN GPU memory grew {grow:.1f} MB over {N_SOAK_IMAGES} inferences."
        )

    def test_no_prediction_drift(self, cnn_soak_report):
        drifts = cnn_soak_report["prediction_drifts"]
        assert drifts == 0, (
            f"CNN predictions drifted {drifts} times over {N_SOAK_IMAGES} inferences. "
            "Repeated inference is altering model state — check BatchNorm mode."
        )

    def test_model_params_unchanged(self, cnn_soak_report):
        assert cnn_soak_report["model_params_unchanged"], (
            f"CNN weights changed during inference soak: {cnn_soak_report['param_error']}"
        )

    def test_no_file_descriptor_leak(self, cnn_soak_report):
        fd_leak = cnn_soak_report.get("fd_leak_detected")
        if fd_leak is None:
            pytest.skip("FD counting not supported on this platform")
        grow = cnn_soak_report["fd_end"] - cnn_soak_report["fd_start"]
        assert not fd_leak, (
            f"CNN inference leaked {grow} file descriptors over {N_SOAK_IMAGES} runs."
        )


class TestQNNCPUSoak:

    def test_no_cpu_memory_leak(self, qnn_cpu_soak_report):
        grow = qnn_cpu_soak_report["cpu_memory_growth_mb"]
        assert grow <= MEMORY_GROWTH_LIMIT_MB, (
            f"QNN_CPU CPU memory grew {grow:.1f} MB over {N_SOAK_IMAGES} inferences."
        )

    def test_no_prediction_drift(self, qnn_cpu_soak_report):
        drifts = qnn_cpu_soak_report["prediction_drifts"]
        assert drifts == 0, (
            f"QNN_CPU predictions drifted {drifts} times. "
            "EMA apply/restore may be corrupting model state."
        )

    def test_model_params_unchanged(self, qnn_cpu_soak_report):
        assert qnn_cpu_soak_report["model_params_unchanged"], (
            f"QNN_CPU weights changed during inference soak: "
            f"{qnn_cpu_soak_report['param_error']}"
        )

    def test_ema_shadow_unchanged_during_eval(self, qnn_cpu_model, soak_images, device):
        """_update_ema() must not be called in eval mode."""
        shadow_before = {k: v.clone() for k, v in qnn_cpu_model._quantum_shadow.items()}
        img = soak_images[0:1].to(device)
        for _ in range(20):
            with torch.no_grad():
                qnn_cpu_model(img)
        for name, val in shadow_before.items():
            current = qnn_cpu_model._quantum_shadow.get(name)
            if current is not None:
                assert torch.equal(val, current), (
                    f"EMA shadow for '{name}' changed during eval inference. "
                    "_update_ema() must only be called during training."
                )

@pytest.mark.requires_gpu
class TestQNNGPUSoak:

    def test_no_cpu_memory_leak(self, qnn_gpu_soak_report):
        grow = qnn_gpu_soak_report["cpu_memory_growth_mb"]
        assert grow <= MEMORY_GROWTH_LIMIT_MB

    def test_no_gpu_memory_leak(self, qnn_gpu_soak_report):
        grow = qnn_gpu_soak_report["gpu_memory_growth_mb"]
        assert grow <= MEMORY_GROWTH_LIMIT_MB

    def test_no_prediction_drift(self, qnn_gpu_soak_report):
        drifts = qnn_gpu_soak_report["prediction_drifts"]
        assert drifts == 0

    def test_model_params_unchanged(self, qnn_gpu_soak_report):
        assert qnn_gpu_soak_report["model_params_unchanged"], (
            f"QNN_GPU weights changed during inference soak: "
            f"{qnn_gpu_soak_report['param_error']}"
        )

    def test_ema_shadow_unchanged_during_eval(self, qnn_gpu_model, soak_images):
        shadow_before = {k: v.clone() for k, v in qnn_gpu_model._quantum_shadow.items()}
        img = soak_images[0:1].to(torch.device("cuda"))
        for _ in range(20):
            with torch.no_grad():
                qnn_gpu_model(img)
        for name, val in shadow_before.items():
            current = qnn_gpu_model._quantum_shadow.get(name)
            if current is not None:
                assert torch.equal(val, current), (
                    f"EMA shadow for '{name}' changed during eval inference."
                )

# ═══════════════════════════════════════════════════════════════════════════
# Combined stability report
# ═══════════════════════════════════════════════════════════════════════════

class TestStabilityCombinedReport:

    def test_save_combined_stability_report(
        self, cnn_soak_report, qnn_cpu_soak_report, qnn_gpu_soak_report
    ):
        def _verdict(r: dict) -> str:
            if (
                not r.get("cpu_leak_detected")
                and not r.get("gpu_leak_detected")
                and not r.get("prediction_drift_detected")
                and r.get("model_params_unchanged")
            ):
                return "PASS"
            return "FAIL"

        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_soak_images": N_SOAK_IMAGES,
            "memory_growth_limit_mb": MEMORY_GROWTH_LIMIT_MB,
            "models": {
                "CNN": {
                    "cpu_memory_growth_mb": cnn_soak_report["cpu_memory_growth_mb"],
                    "gpu_memory_growth_mb": cnn_soak_report["gpu_memory_growth_mb"],
                    "prediction_drifts": cnn_soak_report["prediction_drifts"],
                    "model_params_unchanged": cnn_soak_report["model_params_unchanged"],
                    "throughput_img_per_s": cnn_soak_report["throughput_img_per_s"],
                    "verdict": _verdict(cnn_soak_report),
                },
                "QNN_CPU": {
                    "cpu_memory_growth_mb": qnn_cpu_soak_report["cpu_memory_growth_mb"],
                    "gpu_memory_growth_mb": qnn_cpu_soak_report["gpu_memory_growth_mb"],
                    "prediction_drifts": qnn_cpu_soak_report["prediction_drifts"],
                    "model_params_unchanged": qnn_cpu_soak_report["model_params_unchanged"],
                    "throughput_img_per_s": qnn_cpu_soak_report["throughput_img_per_s"],
                    "verdict": _verdict(qnn_cpu_soak_report),
                },
                "QNN_GPU": {
                    "cpu_memory_growth_mb": qnn_gpu_soak_report["cpu_memory_growth_mb"],
                    "gpu_memory_growth_mb": qnn_gpu_soak_report["gpu_memory_growth_mb"],
                    "prediction_drifts": qnn_gpu_soak_report["prediction_drifts"],
                    "model_params_unchanged": qnn_gpu_soak_report["model_params_unchanged"],
                    "throughput_img_per_s": qnn_gpu_soak_report["throughput_img_per_s"],
                    "verdict": _verdict(qnn_gpu_soak_report),
                },
            },
        }
        _save(report, "soak_combined.json")

        # Also save a top-level combined report covering both determinism and soak
        det_cnn = RESULTS_DIR / "determinism_cnn.json"
        combined = {
            "generated_at": report["generated_at"],
            "determinism": json.loads(det_cnn.read_text()) if det_cnn.exists() else {},
            "soak": report,
        }
        _save(combined, "model_stability_combined.json")
        assert len(report["models"]) == 3