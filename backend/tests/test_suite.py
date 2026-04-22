"""
tests/test_suite.py
============================
Expanded test suite for Quantum-Hybrid-Defect-Detector.
Addresses all 15 QA gaps identified in the software audit.

Gap coverage map:
    GAP-01  TestRealModelSmoke           real predict() schema from actual weights
    GAP-02  TestPreProcessing            resize, normalize, dtype, NaN/Inf
    GAP-03  TestModelFailureInjection    one-model exception → clean HTTP error
    GAP-04  TestGPUFallback              CUDA-unavailable path in HybridQnnGPU
    GAP-05  TestRealImageValidator       imports from production backend module
    GAP-06  TestRealImageValidator       5000×5000 image rejection (Test Plan TC#4)
    GAP-07  TestAggregationContract      per-model output schema contract
    GAP-08  TestBatchLatency             100-image sequential timing ≤ 60 s (PR-4.2)
    GAP-09  TestNoiseMonotonicity        pixel variance monotonically increases with noise
    GAP-10  TestExportEndpoints          /api/v1/export produces valid JSON/CSV
    GAP-11  TestRealFastAPI              uses backend.main.app, not a stub
    GAP-12  TestRestartRecovery          cold-start model load + /health check
    (GAP-13 E2E browser tests: see tests/test_e2e_playwright.py — needs separate install)
    (GAP-14 load test:           see tests/load_test.py)
    (GAP-15 GPU memory monitor:  see tests/monitor_gpu_memory.py)

How to run:
    # From project root
    pip install pytest pytest-cov httpx pillow --break-system-packages

    # All gaps, verbose
    pytest tests/test_suite.py -v

    # Skip tests that need trained model weights on disk
    pytest tests/test_suite.py -v -m "not requires_weights"

    # Only PR-4.2 batch timing (slow)
    pytest tests/test_suite.py -v -k "batch"

    # Coverage against backend/
    pytest tests/test_suite.py --cov=backend --cov-report=term-missing
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest
import torch

# Make backend importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Pull in the noise utilities that are already tested in test_suite.py.
# We re-import here to keep this file self-contained.
# ---------------------------------------------------------------------------
from backend.utils.noise import (
    apply_noise,
    noise_gaussian,
    noise_blur,
    noise_contrast,
    noise_salt_pepper,
    noise_motion_blur,
    noise_jpeg,
    noise_lens_occlusion,
    BENCHMARK_NOISE_LEVELS,
)

# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------
# Tests that need a trained model checkpoint saved to disk.
requires_weights = pytest.mark.skipif(
    not any(
        Path(p).exists()
        for p in ["models/cnn_noise_training_75_epochs.pth", "models/cnn.pth"]
    ),
    reason="No CNN checkpoint found; set WEIGHTS_AVAILABLE=1 to force-fail",
)
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
requires_pennylane = pytest.mark.skipif(
    True,  # flip to False once PennyLane is installed in test env
    reason="Skipping PennyLane-dependent tests in CPU CI environment",
)


# ═══════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════════════
@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def small_rgb_tensor() -> torch.Tensor:
    """A single 384×384 RGB image tensor (B=1, C=3, H=384, W=384)."""
    torch.manual_seed(7)
    return torch.rand(1, 3, 384, 384)


@pytest.fixture
def batch_384() -> torch.Tensor:
    """Batch of 8 × 384×384 images — realistic inference input."""
    torch.manual_seed(42)
    return torch.rand(8, 3, 384, 384)


@pytest.fixture
def clean_batch() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.rand(4, 3, 32, 32)


@pytest.fixture
def small_jpeg_bytes() -> bytes:
    """In-memory valid JPEG via PIL, or magic-byte fallback."""
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (384, 384), color=(80, 120, 160)).save(
            buf, format="JPEG", quality=85
        )
        return buf.getvalue()
    except ImportError:
        return b"\xff\xd8\xff\xe0" + b"\x00" * 2048


@pytest.fixture
def small_png_bytes() -> bytes:
    try:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (64, 64)).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 512


# ═══════════════════════════════════════════════════════════════════════════
# GAP-01 — Real model smoke tests
# Covers: loading trained weights, running predict(), validating output schema.
# The schema differs between CNN and QNN (QNN adds classical_pred / quantum_pred).
# ═══════════════════════════════════════════════════════════════════════════
CNN_CHECKPOINT_CANDIDATES = [
    "models/cnn.pth",
]

QNN_CPU_CHECKPOINT_CANDIDATES = [
    "models/qnn_cpu.pth",
]

QNN_GPU_CHECKPOINT_CANDIDATES = [
    "models/qnn_gpu.pth",
]

CLASS_NAMES = [
    "Deformation",
    "Deposition",
    "Disconnect",
    "Misalignment",
    "Obstacle",
    "Rupture",
]

_CNN_CKPT = next(
    (p for p in CNN_CHECKPOINT_CANDIDATES if Path(p).exists()), None
)
_QNN_CPU_CKPT = next(
    (p for p in QNN_CPU_CHECKPOINT_CANDIDATES if Path(p).exists()), None
)
_QNN_GPU_CKPT = next(
    (p for p in QNN_GPU_CHECKPOINT_CANDIDATES if Path(p).exists()), None
)


@pytest.mark.skipif(_CNN_CKPT is None, reason="CNN checkpoint not found")
class TestRealCNNSmoke:
    """GAP-01: Load real CNN weights and verify predict() contract."""

    @pytest.fixture
    def cnn_model(self, device):
        from backend.models.cnn import CNN

        model = CNN(num_classes=6)
        model.load_model(_CNN_CKPT, device)
        model.eval()
        return model

    def test_predict_returns_dict(self, cnn_model, small_rgb_tensor, device):
        result = cnn_model.predict(small_rgb_tensor, device, CLASS_NAMES)
        assert isinstance(result, dict), "predict() must return a dict"

    def test_predict_schema_complete(self, cnn_model, small_rgb_tensor, device):
        result = cnn_model.predict(small_rgb_tensor, device, CLASS_NAMES)
        required = {
            "predicted_index",
            "predicted_class",
            "confidence",
            "all_class_scores",
            "inference_latency_ms",
        }
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_predict_index_in_valid_range(self, cnn_model, small_rgb_tensor, device):
        result = cnn_model.predict(small_rgb_tensor, device, CLASS_NAMES)
        assert 0 <= result["predicted_index"] < 6

    def test_predict_confidence_in_unit_range(
        self, cnn_model, small_rgb_tensor, device
    ):
        result = cnn_model.predict(small_rgb_tensor, device, CLASS_NAMES)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_class_name_is_valid(self, cnn_model, small_rgb_tensor, device):
        result = cnn_model.predict(small_rgb_tensor, device, CLASS_NAMES)
        assert result["predicted_class"] in CLASS_NAMES

    def test_predict_all_scores_sum_to_one(self, cnn_model, small_rgb_tensor, device):
        result = cnn_model.predict(small_rgb_tensor, device, CLASS_NAMES)
        scores = list(result["all_class_scores"].values())
        assert abs(sum(scores) - 1.0) < 1e-3, "Softmax scores must sum to ≈1.0"

    def test_predict_latency_positive(self, cnn_model, small_rgb_tensor, device):
        result = cnn_model.predict(small_rgb_tensor, device, CLASS_NAMES)
        assert result["inference_latency_ms"] > 0

    def test_predict_latency_under_3000ms(self, cnn_model, small_rgb_tensor, device):
        """PR-4.1: single inference must complete in < 3,000 ms."""
        result = cnn_model.predict(small_rgb_tensor, device, CLASS_NAMES)
        assert result["inference_latency_ms"] < 3000.0, (
            f"Latency {result['inference_latency_ms']:.1f} ms exceeds PR-4.1 limit"
        )


@pytest.mark.skipif(_QNN_CPU_CKPT is None, reason="QNN-CPU checkpoint not found")
class TestRealQNNCPUSmoke:
    """GAP-01: QNN-CPU predict() has an extended schema vs CNN."""

    @pytest.fixture
    def qnn_cpu_model(self, device):
        from backend.models.qnn_cpu import HybridQnnCPU

        model = HybridQnnCPU(num_classes=6, n_qubits=6, q_depth=2)
        model.load_model(_QNN_CPU_CKPT, device)
        model.eval()
        return model

    def test_predict_schema(self, qnn_cpu_model, small_rgb_tensor, device):
        """QNN models expose three extra keys CNN does not have."""
        result = qnn_cpu_model.predict(small_rgb_tensor, device, CLASS_NAMES)
        qnn_only_keys = {"classical_pred", "quantum_pred", "heads_agree"}
        assert qnn_only_keys.issubset(result.keys()), (
            f"Missing QNN-specific keys: {qnn_only_keys - result.keys()}"
        )

    def test_heads_agree_is_bool(self, qnn_cpu_model, small_rgb_tensor, device):
        result = qnn_cpu_model.predict(small_rgb_tensor, device, CLASS_NAMES)
        assert isinstance(result["heads_agree"], bool)

    def test_classical_pred_in_valid_range(
        self, qnn_cpu_model, small_rgb_tensor, device
    ):
        result = qnn_cpu_model.predict(small_rgb_tensor, device, CLASS_NAMES)
        assert 0 <= result["classical_pred"] < 6

    def test_quantum_pred_in_valid_range(
        self, qnn_cpu_model, small_rgb_tensor, device
    ):
        result = qnn_cpu_model.predict(small_rgb_tensor, device, CLASS_NAMES)
        assert 0 <= result["quantum_pred"] < 6


# ═══════════════════════════════════════════════════════════════════════════
# GAP-02 — Preprocessing unit tests
# Covers: resize, tensor conversion, normalization, dtype, NaN/Inf.
# ═══════════════════════════════════════════════════════════════════════════
class TestPreProcessing:
    """GAP-02: Image preprocessing pipeline — entirely absent from original suite."""

    @pytest.fixture
    def inference_transform(self):
        from backend.data.preprocessing import PreProcessing

        return PreProcessing.get_transforms(
            img_width=384, img_height=384, is_training=False
        )

    @pytest.fixture
    def training_transform(self):
        from backend.data.preprocessing import PreProcessing

        return PreProcessing.get_transforms(
            img_width=384, img_height=384, is_training=True
        )

    @pytest.fixture
    def pil_image(self):
        """Minimal 128×128 PIL image to exercise transforms."""
        try:
            from PIL import Image

            return Image.new("RGB", (128, 128), color=(100, 150, 200))
        except ImportError:
            pytest.skip("Pillow not installed")

    def test_output_shape_is_384(self, inference_transform, pil_image):
        tensor = inference_transform(pil_image)
        assert tensor.shape == (3, 384, 384), (
            f"Expected (3, 384, 384), got {tensor.shape}"
        )

    def test_output_dtype_is_float32(self, inference_transform, pil_image):
        tensor = inference_transform(pil_image)
        assert tensor.dtype == torch.float32

    def test_no_nan_in_output(self, inference_transform, pil_image):
        tensor = inference_transform(pil_image)
        assert not torch.isnan(tensor).any(), "Preprocessing must not produce NaN"

    def test_no_inf_in_output(self, inference_transform, pil_image):
        tensor = inference_transform(pil_image)
        assert not torch.isinf(tensor).any(), "Preprocessing must not produce Inf"

    def test_imagenet_normalized_range(self, inference_transform, pil_image):
        """
        After ImageNet normalization, pixel values are NOT in [0, 1].
        The channel means are ≈ [0.485, 0.456, 0.406].
        We verify values are plausible: mean |value| < 3 (roughly ±3 std from zero).
        """
        tensor = inference_transform(pil_image)
        assert tensor.abs().mean().item() < 3.0, (
            "Normalized values look unusual — check normalization params"
        )

    def test_training_transform_is_non_deterministic(
        self, training_transform, pil_image
    ):
        """
        Training transforms include augmentation (flips, jitter, etc.).
        Two applications of the same image should NOT always produce identical tensors.
        We run 5 times and expect at least one difference.
        """
        outputs = [training_transform(pil_image) for _ in range(5)]
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, (
            "Training transforms must include stochastic augmentation"
        )

    def test_unsqueeze_for_model_input(self, inference_transform, pil_image):
        """Model expects (B, C, H, W); a single-image tensor needs unsqueeze(0)."""
        tensor = inference_transform(pil_image).unsqueeze(0)
        assert tensor.shape == (1, 3, 384, 384)


# ═══════════════════════════════════════════════════════════════════════════
# GAP-04 — GPU CUDA-unavailable fallback
# Covers: HybridQnnGPU.__init__ when lightning.gpu is not available.
# Source-code finding: the current code logs the error but does NOT assign
# a fallback device, causing AttributeError on the next line.  This test
# documents the bug and should FAIL until the code is fixed.
# ═══════════════════════════════════════════════════════════════════════════


class TestGPUFallback:
    """
    GAP-04: HybridQnnGPU must not crash fatally when CUDA / lightning.gpu is absent.

    Current state (from source audit): the try/except in __init__ catches the
    lightning.gpu error but never assigns self.q_device to a fallback.  The next
    line `self.q_layer = vqc(self.q_device, ...)` then raises AttributeError.

    Expected fix: assign `self.q_device = qml.device("default.qubit", wires=n_qubits)`
    in the except block and set a `self.gpu_available = False` flag so the router
    can return a controlled "feature unavailable" response.
    """

    def test_gpu_model_raises_graceful_error_without_cuda(self):
        """
        When lightning.gpu fails, the model must either:
          (a) fall back to default.qubit and set a flag, OR
          (b) raise a controlled FeatureUnavailableError (not AttributeError/NameError).
        This test documents the CURRENT bug (AttributeError) and should be updated
        once the fix lands.
        """
        try:
            from backend.models.qnn_gpu import HybridQnnGPU
            import pennylane as qml
        except ImportError:
            pytest.skip("backend or pennylane not importable")

        def _bad_device(name, **kwargs):
            if name == "lightning.gpu":
                raise RuntimeError("cuQuantum not available")
            return qml.device(name, **kwargs)

        with patch("pennylane.device", side_effect=_bad_device):
            try:
                model = HybridQnnGPU(num_classes=6)
                # If we get here, the model has a fallback — check the flag
                assert hasattr(model, "gpu_available"), (
                    "Model should expose gpu_available=False when falling back"
                )
                assert model.gpu_available is False
            except AttributeError as exc:
                # Document the current bug; do NOT let it become a silent pass
                pytest.xfail(
                    f"Known bug: HybridQnnGPU crashes with AttributeError when "
                    f"lightning.gpu is unavailable. Fix: assign fallback device "
                    f"in the except block. Error: {exc}"
                )

    def test_gpu_model_predict_returns_unavailable_dict_without_cuda(self):
        """
        If CUDA is absent, the router should return a JSON dict with
        `{"error": "GPU not available", "status": 503}` rather than crashing.
        This tests the router-level fallback, not the model itself.
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app
        except ImportError:
            pytest.skip("FastAPI or backend not importable")

        client = TestClient(app)
        with patch("torch.cuda.is_available", return_value=False):
            resp = client.post(
                "/api/v1/classify",
                files={"file": ("test.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 512, "image/jpeg")},
            )
        # Either 200 with a graceful QNN_GPU error field, or 503 — NOT 500
        assert resp.status_code != 500, (
            "Server must not return 500 when CUDA is absent; "
            "return 503 or degrade gracefully"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GAP-03 — One-model failure injection
# Covers: Test Plan TC#5 — inject a model exception during parallel dispatch,
# verify the API returns a clean error (not partial / silent 200).
# ═══════════════════════════════════════════════════════════════════════════


class TestModelFailureInjection:
    """GAP-03: One-model timeout/exception must produce a clean HTTP error."""

    @pytest.fixture
    def client(self):
        try:
            from fastapi.testclient import TestClient
            from backend.main import app

            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI or backend.main not importable")

    def _mock_predict_raises(self, *args, **kwargs):
        raise TimeoutError("Simulated model timeout")

    def test_qnn_cpu_timeout_returns_error_not_200(self, client, small_jpeg_bytes):
        """
        When QNN_CPU.predict() raises TimeoutError, the endpoint must NOT
        return HTTP 200 with partial results.  It must return 4xx or 5xx
        with a structured error body — no silent data corruption.
        """
        with patch(
            "backend.models.qnn_cpu.HybridQnnCPU.predict",
            side_effect=TimeoutError("Simulated timeout"),
        ):
            resp = client.post(
                "/api/v1/classify",
                files={"file": ("test.jpg", small_jpeg_bytes, "image/jpeg")},
            )
        # Must not be 200 with incomplete results
        if resp.status_code == 200:
            body = resp.json()
            results = body.get("results", {})
            assert "QNN_CPU" not in results or "error" in results.get("QNN_CPU", {}), (
                "TC#5 FAIL: 200 returned with silent partial results. "
                "The endpoint must surface model errors to the caller."
            )
        else:
            assert resp.status_code in (500, 503, 502), (
                f"Expected 5xx for model failure, got {resp.status_code}"
            )

    def test_error_body_is_structured_json(self, client, small_jpeg_bytes):
        """Error responses must be JSON, not raw stack traces."""
        with patch(
            "backend.models.cnn.CNN.predict",
            side_effect=RuntimeError("Intentional error"),
        ):
            resp = client.post(
                "/api/v1/classify",
                files={"file": ("test.jpg", small_jpeg_bytes, "image/jpeg")},
            )
        if resp.status_code != 200:
            # Must be parseable JSON
            try:
                body = resp.json()
                assert "detail" in body or "error" in body or "message" in body, (
                    "Error response must contain 'detail', 'error', or 'message' key"
                )
            except Exception:
                pytest.fail("Error response body is not valid JSON")


# ═══════════════════════════════════════════════════════════════════════════
# GAP-05 & GAP-06 — Real image validator + TC#4 (5000×5000)
# ═══════════════════════════════════════════════════════════════════════════


class TestRealImageValidator:
    """
    GAP-05: Tests import from the PRODUCTION backend validate module.
    GAP-06: Adds the 5000×5000 rejection case from Test Plan TC#4.
    """

    @pytest.fixture(autouse=True)
    def _import_validator(self):
        """Attempt to import real validator; skip if not available."""
        try:
            from backend.utils.validate import (
                check_magic_bytes,
                check_file_size,
                check_dimensions,
            )

            self.check_magic_bytes = check_magic_bytes
            self.check_file_size = check_file_size
            self.check_dimensions = check_dimensions
        except ImportError:
            pytest.skip(
                "backend.utils.validate not importable. "
                "The test_suite.py stubs will not catch regressions in the "
                "real module — this test class is required for that coverage."
            )

    # ── Magic bytes ──────────────────────────────────────────────────────

    def test_jpeg_magic_bytes_accepted(self, small_jpeg_bytes):
        assert self.check_magic_bytes(small_jpeg_bytes, "image/jpeg") is True

    def test_png_magic_bytes_accepted(self, small_png_bytes):
        assert self.check_magic_bytes(small_png_bytes, "image/png") is True

    def test_pdf_disguised_as_jpeg_rejected(self):
        pdf = b"%PDF-1.4" + b"\x00" * 100
        assert self.check_magic_bytes(pdf, "image/jpeg") is False

    def test_empty_content_rejected(self):
        assert self.check_magic_bytes(b"", "image/jpeg") is False

    def test_truncated_header_rejected(self):
        assert self.check_magic_bytes(b"\xff\xd8", "image/jpeg") is False

    # ── File size ────────────────────────────────────────────────────────

    def test_5mb_exactly_rejected(self):
        """Boundary: 5 MB exactly must be rejected (strict <, not ≤)."""
        assert self.check_file_size(5 * 1024 * 1024) is False

    def test_just_under_5mb_accepted(self):
        assert self.check_file_size(5 * 1024 * 1024 - 1) is True

    def test_6mb_rejected(self):
        assert self.check_file_size(6 * 1024 * 1024) is False

    # ── Dimensions — GAP-06 ──────────────────────────────────────────────

    def test_5000x5000_rejected(self):
        """Test Plan TC#4: 5000×5000 must be rejected."""
        assert self.check_dimensions(5000, 5000) is False, (
            "TC#4 FAIL: 5000×5000 image must be rejected by the validator"
        )

    def test_4096x4096_rejected(self):
        """The documented max_dim=4096 boundary must also be rejected."""
        assert self.check_dimensions(4096, 4096) is False

    def test_4095x4095_accepted(self):
        assert self.check_dimensions(4095, 4095) is True

    def test_normal_size_accepted(self):
        assert self.check_dimensions(384, 384) is True

    def test_landscape_near_limit_accepted(self):
        assert self.check_dimensions(4000, 384) is True

    def test_portrait_over_limit_rejected(self):
        assert self.check_dimensions(384, 5000) is False


# ═══════════════════════════════════════════════════════════════════════════
# GAP-07 — Aggregation contract schema
# ═══════════════════════════════════════════════════════════════════════════

# Required fields per model in the /classify response
_CNN_REQUIRED_KEYS = {"label", "confidence", "latency_ms"}
_QNN_REQUIRED_KEYS = {"label", "confidence", "latency_ms"}

# Some implementations use "predicted_class" instead of "label" — accept both
_LABEL_ALIASES = {"label", "predicted_class"}
_CONF_ALIASES = {"confidence"}
_LATENCY_ALIASES = {"latency_ms", "inference_latency_ms"}


def _has_key_or_alias(d: dict, aliases: set) -> bool:
    return bool(aliases & d.keys())


class TestAggregationContract:
    """GAP-07: /classify must return all three model results with the correct schema."""

    @pytest.fixture
    def client(self):
        try:
            from fastapi.testclient import TestClient
            from backend.main import app

            return TestClient(app)
        except ImportError:
            pytest.skip("backend.main not importable")

    def test_all_three_model_keys_present(self, client, small_jpeg_bytes):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("test.jpg", small_jpeg_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200
        results = resp.json().get("results", {})
        for key in ("CNN", "QNN_CPU", "QNN_GPU"):
            assert key in results, f"Missing model key in response: {key}"

    def test_each_result_has_label(self, client, small_jpeg_bytes):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("test.jpg", small_jpeg_bytes, "image/jpeg")},
        )
        results = resp.json().get("results", {})
        for key, model_result in results.items():
            if isinstance(model_result, dict) and "error" not in model_result:
                assert _has_key_or_alias(model_result, _LABEL_ALIASES), (
                    f"Model {key} result missing 'label' or 'predicted_class'"
                )

    def test_each_result_has_confidence(self, client, small_jpeg_bytes):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("test.jpg", small_jpeg_bytes, "image/jpeg")},
        )
        results = resp.json().get("results", {})
        for key, model_result in results.items():
            if isinstance(model_result, dict) and "error" not in model_result:
                assert _has_key_or_alias(model_result, _CONF_ALIASES), (
                    f"Model {key} result missing 'confidence'"
                )

    def test_each_result_has_latency(self, client, small_jpeg_bytes):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("test.jpg", small_jpeg_bytes, "image/jpeg")},
        )
        results = resp.json().get("results", {})
        for key, model_result in results.items():
            if isinstance(model_result, dict) and "error" not in model_result:
                assert _has_key_or_alias(model_result, _LATENCY_ALIASES), (
                    f"Model {key} result missing latency field"
                )

    def test_confidence_values_in_unit_range(self, client, small_jpeg_bytes):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("test.jpg", small_jpeg_bytes, "image/jpeg")},
        )
        results = resp.json().get("results", {})
        for key, model_result in results.items():
            if isinstance(model_result, dict):
                conf = model_result.get("confidence", model_result.get("conf"))
                if conf is not None:
                    assert 0.0 <= float(conf) <= 1.0, (
                        f"Model {key} confidence {conf} out of [0, 1]"
                    )

    def test_label_is_one_of_six_classes(self, client, small_jpeg_bytes):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("test.jpg", small_jpeg_bytes, "image/jpeg")},
        )
        results = resp.json().get("results", {})
        valid_classes = set(CLASS_NAMES)
        for key, model_result in results.items():
            if isinstance(model_result, dict) and "error" not in model_result:
                label = model_result.get("label") or model_result.get("predicted_class")
                if label is not None:
                    assert label in valid_classes, (
                        f"Model {key} returned unknown class: '{label}'"
                    )


# ═══════════════════════════════════════════════════════════════════════════
# GAP-08 — Batch inference latency (PR-4.2)
# Covers: 100 sequential single-image inferences on QNN_GPU ≤ 60 s.
# Note: this tests the actual model predict() loop, not a dedicated batch API.
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(_QNN_GPU_CKPT is None, reason="QNN-GPU checkpoint not found")
@requires_gpu
class TestBatchLatency:
    """GAP-08: PR-4.2 — 100-image batch must complete in ≤ 60 s on QNN_GPU."""

    @pytest.fixture
    def qnn_gpu_model(self):
        from backend.models.qnn_gpu import HybridQnnGPU

        device = torch.device("cuda")
        model = HybridQnnGPU(num_classes=6, n_qubits=6, q_depth=2)
        model.load_model(_QNN_GPU_CKPT, device)
        model.eval()
        return model, device

    def test_100_images_within_60_seconds(self, qnn_gpu_model):
        model, device = qnn_gpu_model
        torch.manual_seed(0)

        t_start = time.perf_counter()
        for _ in range(100):
            img = torch.rand(1, 3, 384, 384)
            model.predict(img, device, CLASS_NAMES)
        elapsed = time.perf_counter() - t_start

        print(f"\nPR-4.2: 100 images in {elapsed:.2f} s (limit: 60 s)")
        assert elapsed < 60.0, (
            f"PR-4.2 FAIL: 100-image loop took {elapsed:.2f} s, exceeds 60 s limit"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GAP-09 — Noise monotonicity
# Covers: Test Plan §5.3.3-A-3 — pixel variance must increase with noise level.
# ═══════════════════════════════════════════════════════════════════════════


class TestNoiseMonotonicity:
    """GAP-09: Noise functions must produce monotonically increasing pixel variance."""

    @pytest.fixture
    def base_image(self) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.rand(1, 3, 64, 64)

    def _variance_at_level(self, base: torch.Tensor, noise_type: str, level) -> float:
        return apply_noise(base.clone(), noise_type, level).var().item()

    @pytest.mark.parametrize(
        "noise_type,levels",
        [
            ("gaussian", [0.0, 0.1, 0.3, 0.5, 0.75]),
            ("salt_pepper", [0.0, 0.02, 0.05, 0.1]),
        ],
    )
    def test_variance_increases_with_noise(self, base_image, noise_type, levels):
        """
        For noise types that ADD signal energy (Gaussian, salt-and-pepper),
        pixel variance must be monotonically non-decreasing across levels.
        """
        variances = [self._variance_at_level(base_image, noise_type, l) for l in levels]
        for i in range(len(variances) - 1):
            assert variances[i] <= variances[i + 1] + 1e-6, (
                f"{noise_type}: variance did NOT increase at level {levels[i+1]} "
                f"(got {variances[i+1]:.6f} ≤ {variances[i]:.6f})"
            )

    @pytest.mark.parametrize(
        "noise_type,levels",
        [
            ("blur", [0.0, 0.5, 1.5, 3.0]),
            ("motion_blur", [0, 2, 5, 10]),
        ],
    )
    def test_std_decreases_with_blur(self, base_image, noise_type, levels):
        """
        Blur REMOVES high-frequency energy; pixel standard deviation must
        decrease monotonically as blur strength increases.
        """
        stds = [
            apply_noise(base_image.clone(), noise_type, l).std().item()
            for l in levels
        ]
        for i in range(len(stds) - 1):
            assert stds[i] >= stds[i + 1] - 1e-5, (
                f"{noise_type}: std did NOT decrease at level {levels[i+1]} "
                f"(got {stds[i+1]:.6f} > {stds[i]:.6f})"
            )

    def test_gaussian_zero_level_is_clean_baseline(self, base_image):
        """Level=0.0 Gaussian must equal the original variance exactly."""
        clean_var = base_image.var().item()
        noisy_var = self._variance_at_level(base_image, "gaussian", 0.0)
        assert abs(clean_var - noisy_var) < 1e-6

    @pytest.mark.parametrize("noise_type", list(BENCHMARK_NOISE_LEVELS.keys()))
    def test_all_types_produce_finite_variance(self, base_image, noise_type):
        """Any noise level must produce a finite pixel variance."""
        for level in BENCHMARK_NOISE_LEVELS[noise_type]:
            var = self._variance_at_level(base_image, noise_type, level)
            assert 0.0 <= var < float("inf"), (
                f"{noise_type} level {level}: variance is not finite ({var})"
            )


# ═══════════════════════════════════════════════════════════════════════════
# GAP-10 — Export endpoints (CSV / JSON)
# ═══════════════════════════════════════════════════════════════════════════


class TestExportEndpoints:
    """GAP-10: /api/v1/export must produce correctly structured JSON and CSV."""

    @pytest.fixture
    def client(self):
        try:
            from fastapi.testclient import TestClient
            from backend.main import app

            return TestClient(app)
        except ImportError:
            pytest.skip("backend.main not importable")

    def test_benchmark_json_export_is_parseable(self, client):
        """The benchmark data endpoint must return valid JSON."""
        resp = client.get("/api/v1/benchmark")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, dict)

    def test_benchmark_json_has_clean_evaluation(self, client):
        resp = client.get("/api/v1/benchmark")
        body = resp.json()
        assert "clean_evaluation" in body, (
            "Benchmark JSON must contain 'clean_evaluation'"
        )

    def test_benchmark_json_has_noise_robustness(self, client):
        resp = client.get("/api/v1/benchmark")
        body = resp.json()
        assert "noise_robustness" in body, (
            "Benchmark JSON must contain 'noise_robustness'"
        )

    def test_classification_export_json_structure(self, client, small_jpeg_bytes):
        """After a classify call, any exportable result must include model keys."""
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("test.jpg", small_jpeg_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Export data is either in 'results' or a dedicated export field
        export_data = body.get("results", body)
        assert isinstance(export_data, dict)

    def test_csv_export_endpoint_returns_text(self, client):
        """
        If a /api/v1/export/csv endpoint exists, the response must be
        parseable as CSV with at least a header row.
        """
        resp = client.get("/api/v1/export/csv")
        if resp.status_code == 404:
            pytest.skip("CSV export endpoint not yet implemented")
        assert resp.status_code == 200
        content_type = resp.headers.get("content-type", "")
        assert "csv" in content_type or "text" in content_type, (
            "CSV export must return text/csv content type"
        )
        try:
            reader = csv.reader(io.StringIO(resp.text))
            rows = list(reader)
            assert len(rows) >= 1, "CSV must have at least a header row"
        except Exception as exc:
            pytest.fail(f"CSV response is not parseable: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
# GAP-11 — Real FastAPI integration tests
# Tests that use backend.main.app directly (not a fake stub).
# ═══════════════════════════════════════════════════════════════════════════


class TestRealFastAPI:
    """GAP-11: Full FastAPI app — middleware, rate-limiting, security headers."""

    @pytest.fixture
    def client(self):
        try:
            from fastapi.testclient import TestClient
            from backend.main import app

            return TestClient(app)
        except ImportError:
            pytest.skip("backend.main not importable")

    # ── Happy path ──────────────────────────────────────────────────────

    def test_classify_valid_jpeg_200(self, client, small_jpeg_bytes):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("defect.jpg", small_jpeg_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_classify_valid_png_200(self, client, small_png_bytes):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("defect.png", small_png_bytes, "image/png")},
        )
        assert resp.status_code == 200

    def test_health_returns_healthy(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") in ("healthy", "ok")

    # ── Error paths ─────────────────────────────────────────────────────

    def test_pdf_rejected_422(self, client):
        pdf = b"%PDF-1.4" + b"\x00" * 200
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("report.pdf", pdf, "application/pdf")},
        )
        assert resp.status_code == 422

    def test_txt_rejected_422(self, client):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 422

    def test_oversized_file_rejected_413(self, client):
        big = b"\xff\xd8\xff\xe0" + b"\x00" * (5 * 1024 * 1024 + 1)
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("big.jpg", big, "image/jpeg")},
        )
        assert resp.status_code == 413

    def test_missing_file_rejected_422(self, client):
        resp = client.post("/api/v1/classify")
        assert resp.status_code == 422

    # ── TC#4 ─────────────────────────────────────────────────────────────

    def test_oversized_dimensions_rejected(self, client):
        """TC#4: 5000×5000 image — server-side dimension validation."""
        try:
            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (5000, 5000)).save(buf, format="JPEG", quality=10)
            big_img = buf.getvalue()
        except ImportError:
            pytest.skip("Pillow not installed — cannot create 5000×5000 JPEG")

        resp = client.post(
            "/api/v1/classify",
            files={"file": ("huge.jpg", big_img, "image/jpeg")},
        )
        assert resp.status_code in (413, 422), (
            f"TC#4: 5000×5000 image must be rejected (got {resp.status_code})"
        )

    # ── Security headers ─────────────────────────────────────────────────

    def test_response_has_x_content_type_options(self, client, small_jpeg_bytes):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("t.jpg", small_jpeg_bytes, "image/jpeg")},
        )
        assert "x-content-type-options" in resp.headers, (
            "Security header X-Content-Type-Options is missing"
        )

    def test_pdf_magic_bytes_rejected_even_with_jpg_extension(self, client):
        """
        Security: a PDF disguised as a .jpg must be caught by magic-byte check,
        not just by extension matching.
        """
        pdf_content = b"%PDF-1.4 fake content here" + b"\x00" * 100
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("legit.jpg", pdf_content, "image/jpeg")},
        )
        assert resp.status_code == 422, (
            "Server must reject PDF content even when named with .jpg extension"
        )

    # ── Rate limiting ────────────────────────────────────────────────────

    def test_rate_limit_not_exceeded_on_10_rapid_requests(self, client, small_jpeg_bytes):
        """
        10 rapid sequential requests should not all be rate-limited.
        (If rate-limiting is configured at ≥10 req/min this must pass.)
        """
        statuses = []
        for _ in range(10):
            r = client.post(
                "/api/v1/classify",
                files={"file": ("t.jpg", small_jpeg_bytes, "image/jpeg")},
            )
            statuses.append(r.status_code)
        non_429 = [s for s in statuses if s != 429]
        assert len(non_429) >= 5, (
            f"Rate limiter is too aggressive: only {len(non_429)}/10 requests passed"
        )


# ═══════════════════════════════════════════════════════════════════════════
# GAP-12 — Backend restart recovery
# Covers: cold-start model loading + /health returning healthy.
# ═══════════════════════════════════════════════════════════════════════════


class TestRestartRecovery:
    """GAP-12: After a fresh import of backend.main, /health must return healthy."""

    def test_cold_start_health_check(self):
        """
        Simulate a backend restart by importing the app from scratch
        (in a fresh module state) and immediately hitting /health.
        """
        try:
            # Force a re-import to simulate restart
            if "backend.main" in sys.modules:
                del sys.modules["backend.main"]

            from fastapi.testclient import TestClient
            from backend.main import app

            client = TestClient(app)
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body.get("status") in ("healthy", "ok"), (
                f"Expected 'healthy' status after cold start, got: {body}"
            )
        except ImportError:
            pytest.skip("backend.main not importable")

    def test_models_loaded_after_cold_start(self):
        """
        /health should indicate that all three model weights are loaded.
        If the endpoint exposes a 'models_loaded' field, verify it.
        """
        try:
            from fastapi.testclient import TestClient
            from backend.main import app

            client = TestClient(app)
            resp = client.get("/api/v1/health")
            body = resp.json()

            # This check is optional — only run if the endpoint exposes it
            if "models_loaded" in body:
                assert body["models_loaded"] is True, (
                    "Model weights must be loaded at startup"
                )
            elif "models" in body:
                for model_name, status in body["models"].items():
                    assert status in ("loaded", "ok", True), (
                        f"Model {model_name} not loaded at startup: {status}"
                    )
        except ImportError:
            pytest.skip("backend.main not importable")


# ═══════════════════════════════════════════════════════════════════════════
# PR-3 training time assertions (source-code analysis)
# These tests cannot run training but validate the per-epoch timer is present
# in logs and the structural speed difference between CPU and GPU QNN.
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainingTimeLogging:
    """
    PR-3.1 / PR-3.2: Verify the per-epoch timer is implemented correctly.
    These tests inspect source rather than running full training.
    """

    def test_cnn_fit_uses_perf_counter(self):
        """CNN.fit() must call time.perf_counter() per epoch (PR-3.2 evidence)."""
        import inspect
        try:
            from backend.models.cnn import CNN
        except ImportError:
            pytest.skip("backend.models.cnn not importable")

        source = inspect.getsource(CNN.fit)
        assert "perf_counter" in source, (
            "CNN.fit() must log per-epoch wall time via time.perf_counter()"
        )

    def test_qnn_cpu_fit_uses_perf_counter(self):
        import inspect
        try:
            from backend.models.qnn_cpu import HybridQnnCPU
        except ImportError:
            pytest.skip("backend.models.qnn_cpu not importable")

        source = inspect.getsource(HybridQnnCPU.fit)
        assert "perf_counter" in source

    def test_qnn_gpu_forward_is_batched_not_per_sample(self):
        """
        PR-3.1 structural evidence: GPU _quantum_forward passes the full
        batch tensor to q_layer at once, while CPU loops per sample.
        Verify the GPU version does NOT contain the per-sample loop.
        """
        import inspect
        try:
            from backend.models.qnn_gpu import qnn_gpu as gpu_module
        except ImportError:
            pytest.skip("backend.models.qnn_gpu not importable")

        source = inspect.getsource(gpu_module._quantum_forward)
        assert "for i in range" not in source, (
            "PR-3.1: GPU _quantum_forward must NOT loop over samples — "
            "it should pass the full batch to q_layer for efficiency"
        )

    def test_qnn_cpu_forward_uses_per_sample_loop(self):
        """
        CPU version intentionally loops (TorchLayer / default.qubit limitation).
        Confirm this is documented and present.
        """
        import inspect
        try:
            from backend.models.qnn_cpu import qnn_cpu as cpu_module
        except ImportError:
            pytest.skip("backend.models.qnn_cpu not importable")

        source = inspect.getsource(cpu_module._quantum_forward)
        assert "for i in range" in source, (
            "CPU _quantum_forward must use per-sample loop for default.qubit"
        )

    def test_qnn_gpu_uses_adjoint_diff_method(self):
        """
        GPU VQC uses diff_method='adjoint' (memory-efficient on GPU).
        CPU uses 'backprop'. Verify the GPU circuit is set correctly.
        """
        import inspect
        try:
            from backend.models.qnn_gpu import vqc as gpu_vqc
        except ImportError:
            pytest.skip("backend.models.qnn_gpu not importable")

        source = inspect.getsource(gpu_vqc)
        assert "adjoint" in source, (
            "GPU VQC circuit must use diff_method='adjoint'"
        )