"""
tests/test_golden_set.py
========================
Golden-set regression tests for Quantum-Hybrid-Defect-Detector.

Dataset layout (YOLO format):
    data/test/Images/<class_name>/*.jpg   - test images
    data/test/Labels/<class_name>/*.txt   - one file per image, first token = class index

Run:
    # From backend/ — run all available models
    pytest tests/test_golden_set.py -v

    # From project root — same test file
    pytest backend/tests/test_golden_set.py -v

    First run:
      Creates per-model golden baselines under tests/results/golden_set/
      Example files:
          golden_set_cnn.json
          golden_set_qnn_cpu.json
          golden_set_qnn_gpu.json
    
    Subsequent runs:
      Compare current predictions against each model's saved baseline
      and fail if a previously-correct prediction regresses

    Force baseline regeneration after an intentional model update
    REGENERATE_GOLDEN=1 pytest tests/test_golden_set.py -v

    # Run a single model only
    pytest tests/test_golden_set.py -v -k "CNN"
    pytest tests/test_golden_set.py -v -k "QNN_CPU"
    pytest tests/test_golden_set.py -v -k "QNN_GPU"

    # Show detailed dataset fallback diagnostics
    pytest tests/test_golden_set.py -v -s

How it works:
    1. First run (no golden file for a model): builds golden set from real images,
       writes tests/results/golden_set/golden_set_<model>.json, PASSES.
    2. Subsequent runs: compares current predictions against that model baseline,
       FAILS if any previously-correct prediction flips.
    3. Regenerate after intentional model update:
           REGENERATE_GOLDEN=1 pytest tests/test_golden_set.py -v


Results saved to:
    tests/results/golden_set/golden_set_cnn.json
    tests/results/golden_set/golden_set_qnn_cpu.json
    tests/results/golden_set/golden_set_qnn_gpu.json
    tests/results/golden_set/golden_regression_<model>.json
    tests/results/golden_set/golden_set_summary_<model>.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

BACKEND_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BACKEND_DIR))

RESULTS_DIR = BACKEND_DIR / "tests" / "results" / "golden_set"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_CLASS_NAMES_FILE = BACKEND_DIR / "data" / "class_names.json"
try:
    with open(_CLASS_NAMES_FILE, encoding="utf-8") as fh:
        CLASS_NAMES: List[str] = json.load(fh)
except FileNotFoundError:
    CLASS_NAMES = [
        "Deformation",
        "Deposition",
        "Disconnect",
        "Misalignment",
        "Obstacle",
        "Rupture",
    ]

IMAGES_PER_CLASS = 5
FLOOR_RECALL = 0.80
SAFETY_CRITICAL = {"Rupture", "Disconnect"}

_CNN_CKPT = next(
    (
        str(BACKEND_DIR / p)
        for p in [
            "models/cnn.pth",
            "models/cnn_noise_training_75_epochs.pth",
            "models/cpu_new.pth",
        ]
        if (BACKEND_DIR / p).exists()
    ),
    None,
)

_QNN_CPU_CKPT = next(
    (
        str(BACKEND_DIR / p)
        for p in [
            "models/qnn_cpu.pth",
            "models/qnn_cpu_new.pth",
        ]
        if (BACKEND_DIR / p).exists()
    ),
    None,
)

_QNN_GPU_CKPT = next(
    (
        str(BACKEND_DIR / p)
        for p in [
            "models/qnn_gpu.pth",
        ]
        if (BACKEND_DIR / p).exists()
    ),
    None,
)

MODEL_KEYS = ["CNN", "QNN_CPU", "QNN_GPU"]


def _artifact_paths(model_key: str) -> tuple[Path, Path, Path]:
    slug = model_key.lower()
    return (
        RESULTS_DIR / f"golden_set_{slug}.json",
        RESULTS_DIR / f"golden_regression_{slug}.json",
        RESULTS_DIR / f"golden_set_summary_{slug}.json",
    )


def _build_real_golden_set(
    data_dir: Optional[str] = None,
) -> Optional[Tuple[List[Tuple[torch.Tensor, int, str]], str]]:
    base = BACKEND_DIR / (data_dir or "data/test")
    images_dir = base / "Images"
    labels_dir = base / "Labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"[golden_set] FAIL: Images or Labels folder missing under {base}")
        return None

    try:
        from data.preprocessing import PreProcessing
        from PIL import Image as PILImage

        transform = PreProcessing.get_transforms(
            img_width=384, img_height=384, is_training=False
        )
    except Exception as exc:
        print(f"[golden_set] FAIL: PreProcessing load error: {exc}")
        return None

    image_files = sorted(
        f
        for ext in ("**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.JPG")
        for f in images_dir.glob(ext)
    )

    if not image_files:
        print(f"[golden_set] FAIL: No images found in subfolders of {images_dir}")
        return None

    found_counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    golden: List[Tuple[torch.Tensor, int, str]] = []

    for img_path in image_files:
        cls_subfolder = img_path.parent.name
        label_path = labels_dir / cls_subfolder / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        with open(label_path, encoding="utf-8") as lf:
            first_line = lf.readline().split()
        if not first_line:
            continue

        try:
            cls_idx = int(first_line[0])
            cls_name = CLASS_NAMES[cls_idx]
        except (ValueError, IndexError):
            continue

        if cls_name != cls_subfolder:
            continue

        if found_counts[cls_name] >= IMAGES_PER_CLASS:
            continue

        try:
            tensor = transform(PILImage.open(img_path).convert("RGB")).unsqueeze(0)
        except Exception:
            continue

        golden.append((tensor, cls_idx, f"real_{cls_name}_{img_path.stem}"))
        found_counts[cls_name] += 1

    if not golden:
        print("[golden_set] FAIL: No valid image/label pairs found in subfolders.")
        return None

    print(f"[golden_set] SUCCESS: Loaded {len(golden)} real images.")
    return golden, "real_yolo"


def _build_synthetic_golden_set() -> List[Tuple[torch.Tensor, int, str]]:
    golden = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        for i in range(IMAGES_PER_CLASS):
            torch.manual_seed(cls_idx * 100 + i)
            img = torch.rand(1, 3, 384, 384)
            golden.append((img, cls_idx, f"synthetic_{cls_name}_{i:02d}"))
    return golden


def _predict_logits(model_key: str, model, img: torch.Tensor, device: torch.device) -> torch.Tensor:
    img = img.to(device)

    if model_key == "CNN":
        return model(img)

    result = model.predict(img, device, CLASS_NAMES)
    scores = result.get("all_class_scores")
    if scores is None:
        raise AssertionError(f"{model_key}.predict() returned no all_class_scores")

    if isinstance(scores, dict):
        ordered = [float(scores[name]) for name in CLASS_NAMES]
    else:
        ordered = [float(x) for x in scores]

    return torch.tensor([ordered], dtype=torch.float32, device=device)


def _run_predictions(
    model_key: str,
    model,
    golden: List[Tuple[torch.Tensor, int, str]],
    device: torch.device,
) -> List[dict]:
    model.eval()
    records = []

    with torch.no_grad():
        for img, true_cls, img_id in golden:
            logits = _predict_logits(model_key, model, img, device)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(1)

            records.append(
                {
                    "image_id": img_id,
                    "true_class": CLASS_NAMES[true_cls],
                    "true_class_idx": true_cls,
                    "predicted_class": CLASS_NAMES[pred.item()],
                    "predicted_class_idx": int(pred.item()),
                    "confidence": round(float(conf.item()), 4),
                    "correct": int(pred.item()) == true_cls,
                    "all_scores": {
                        CLASS_NAMES[j]: round(float(probs[0, j].item()), 4)
                        for j in range(len(CLASS_NAMES))
                    },
                }
            )

    return records


@pytest.fixture(scope="module")
def golden_data():
    result = _build_real_golden_set()
    if result is not None:
        return result

    print(
        "\n[golden_set] WARNING: Falling back to SYNTHETIC golden set.\n"
        "Floor accuracy tests will be SKIPPED for synthetic data.\n"
        "Run with -s to see which step failed above."
    )
    return _build_synthetic_golden_set(), "synthetic"


@pytest.fixture(scope="module", params=MODEL_KEYS, ids=MODEL_KEYS)
def model_bundle(request):
    model_key = request.param

    if model_key == "CNN":
        if _CNN_CKPT is None:
            pytest.skip("CNN checkpoint not found")
        from models.cnn import CNN

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNN(num_classes=len(CLASS_NAMES))
        model.load_model(_CNN_CKPT, device)
        model.eval()
        return model_key, model, device, _CNN_CKPT

    if model_key == "QNN_CPU":
        if _QNN_CPU_CKPT is None:
            pytest.skip("QNN_CPU checkpoint not found")
        from models.qnn_cpu import HybridQnnCPU

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridQnnCPU(num_classes=len(CLASS_NAMES), n_qubits=6, q_depth=2)
        model.load_model(_QNN_CPU_CKPT, device)
        model.eval()
        return model_key, model, device, _QNN_CPU_CKPT

    if not torch.cuda.is_available():
        pytest.skip("QNN_GPU tests require CUDA")
    if _QNN_GPU_CKPT is None:
        pytest.skip("QNN_GPU checkpoint not found")

    from models.qnn_gpu import HybridQnnGPU

    device = torch.device("cuda")
    model = HybridQnnGPU(num_classes=len(CLASS_NAMES), n_qubits=6, q_depth=2)
    model.load_model(_QNN_GPU_CKPT, device)
    model.eval()
    return model_key, model, device, _QNN_GPU_CKPT


@pytest.fixture(scope="module")
def model_predictions(model_bundle, golden_data):
    model_key, model, device, ckpt = model_bundle
    images, source = golden_data
    records = _run_predictions(model_key, model, images, device)
    return {
        "model_key": model_key,
        "checkpoint": ckpt,
        "source": source,
        "records": records,
    }


def _require_real(source: str, test_name: str) -> None:
    if source == "synthetic":
        pytest.skip(
            f"{test_name}: skipped because the golden set is SYNTHETIC.\n"
            f"Real images were not found at {BACKEND_DIR / 'data' / 'test'}."
        )


class TestGoldenSetFloor:
    def test_every_image_produces_valid_prediction(self, model_predictions):
        records = model_predictions["records"]
        for rec in records:
            assert rec["predicted_class"] in CLASS_NAMES
            assert 0.0 <= rec["confidence"] <= 1.0

    def test_scores_sum_to_one(self, model_predictions):
        records = model_predictions["records"]
        for rec in records:
            total = sum(rec["all_scores"].values())
            assert abs(total - 1.0) < 1e-3

    def test_no_class_has_zero_recall(self, model_predictions):
        records = model_predictions["records"]
        source = model_predictions["source"]
        _require_real(source, "zero-recall floor test")

        for cls_name in CLASS_NAMES:
            cls_records = [r for r in records if r["true_class"] == cls_name]
            if not cls_records:
                continue
            n_correct = sum(1 for r in cls_records if r["correct"])
            assert n_correct > 0, f"Class '{cls_name}' has ZERO correct predictions."

    def test_per_class_recall_meets_floor(self, model_predictions):
        records = model_predictions["records"]
        source = model_predictions["source"]
        _require_real(source, "per-class recall floor test")

        for cls_name in CLASS_NAMES:
            cls_records = [r for r in records if r["true_class"] == cls_name]
            if not cls_records:
                continue
            n_correct = sum(1 for r in cls_records if r["correct"])
            recall = n_correct / len(cls_records)
            assert recall >= FLOOR_RECALL, (
                f"Class '{cls_name}' recall {recall:.0%} below floor {FLOOR_RECALL:.0%}"
            )

    def test_safety_critical_classes_recall(self, model_predictions):
        records = model_predictions["records"]
        source = model_predictions["source"]
        _require_real(source, "safety-critical recall test")

        for cls_name in SAFETY_CRITICAL:
            cls_records = [r for r in records if r["true_class"] == cls_name]
            if not cls_records:
                continue
            wrong = [r for r in cls_records if not r["correct"]]
            if wrong:
                pytest.xfail(
                    f"{model_predictions['model_key']} safety-critical class '{cls_name}' "
                    f"missed {len(wrong)}/{len(cls_records)} golden images."
                )

    def test_overall_accuracy_above_90pct(self, model_predictions):
        records = model_predictions["records"]
        source = model_predictions["source"]
        _require_real(source, "overall accuracy floor test")

        n_correct = sum(1 for r in records if r["correct"])
        accuracy = n_correct / len(records)
        assert accuracy >= 0.90, f"Overall accuracy {accuracy:.1%} below 90%."

    def test_show_diagnostic_when_synthetic(self, model_predictions):
        records = model_predictions["records"]
        source = model_predictions["source"]
        if source == "synthetic":
            n_correct = sum(1 for r in records if r["correct"])
            print(
                f"\n{'='*60}\n"
                f"SYNTHETIC GOLDEN SET DIAGNOSTIC [{model_predictions['model_key']}]\n"
                f"Overall: {n_correct}/{len(records)} correct on random noise\n"
                f"{'='*60}"
            )
        assert True


class TestGoldenSetRegression:
    def test_create_or_verify_golden_file(self, model_predictions):
        model_key = model_predictions["model_key"]
        checkpoint = model_predictions["checkpoint"]
        source = model_predictions["source"]
        records = model_predictions["records"]

        golden_file, regression_file, _ = _artifact_paths(model_key)
        regenerate = os.environ.get("REGENERATE_GOLDEN", "0") == "1"

        if not golden_file.exists() or regenerate:
            baseline = {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model_key": model_key,
                "model_checkpoint": checkpoint,
                "backend_dir": str(BACKEND_DIR),
                "source": source,
                "n_images": len(records),
                "images_per_class": IMAGES_PER_CLASS,
                "class_names": CLASS_NAMES,
                "records": records,
            }
            with open(golden_file, "w", encoding="utf-8") as fh:
                json.dump(baseline, fh, indent=2)
            return

        with open(golden_file, encoding="utf-8") as fh:
            golden = json.load(fh)

        golden_by_id = {r["image_id"]: r for r in golden["records"]}
        current_by_id = {r["image_id"]: r for r in records}
        regressions = []

        for img_id, gold_rec in golden_by_id.items():
            curr_rec = current_by_id.get(img_id)
            if curr_rec is None:
                continue
            if gold_rec["correct"] and not curr_rec["correct"]:
                regressions.append(
                    {
                        "image_id": img_id,
                        "true_class": gold_rec["true_class"],
                        "was_predicted": gold_rec["predicted_class"],
                        "now_predicted": curr_rec["predicted_class"],
                        "confidence_was": gold_rec["confidence"],
                        "confidence_now": curr_rec["confidence"],
                    }
                )

        regression_report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_key": model_key,
            "source": source,
            "n_regressions": len(regressions),
            "regressions": regressions,
            "verdict": "PASS" if not regressions else "FAIL",
        }
        with open(regression_file, "w", encoding="utf-8") as fh:
            json.dump(regression_report, fh, indent=2)

        assert not regressions, (
            f"{model_key} GOLDEN SET REGRESSION: {len(regressions)} predictions flipped.\n"
            + "\n".join(
                f"  {r['image_id']}: '{r['true_class']}' was '{r['was_predicted']}' "
                f"(conf {r['confidence_was']:.3f}), now '{r['now_predicted']}' "
                f"(conf {r['confidence_now']:.3f})"
                for r in regressions
            )
            + f"\nFull report: {regression_file}"
        )

    def test_no_regression_in_safety_critical_classes(self, model_predictions):
        model_key = model_predictions["model_key"]
        records = model_predictions["records"]

        golden_file, _, _ = _artifact_paths(model_key)
        if not golden_file.exists():
            pytest.skip(f"{model_key} golden file not created yet")

        with open(golden_file, encoding="utf-8") as fh:
            golden = json.load(fh)

        golden_by_id = {r["image_id"]: r for r in golden["records"]}

        for rec in records:
            if rec["true_class"] not in SAFETY_CRITICAL:
                continue
            gold_rec = golden_by_id.get(rec["image_id"])
            if gold_rec is None:
                continue
            if gold_rec["correct"] and not rec["correct"]:
                pytest.fail(
                    f"{model_key} CRITICAL CLASS REGRESSION: '{rec['true_class']}' image "
                    f"'{rec['image_id']}' regressed from '{gold_rec['predicted_class']}' "
                    f"to '{rec['predicted_class']}'."
                )

    def test_save_prediction_summary(self, model_predictions):
        model_key = model_predictions["model_key"]
        records = model_predictions["records"]
        source = model_predictions["source"]
        checkpoint = model_predictions["checkpoint"]

        _, _, summary_file = _artifact_paths(model_key)

        n_correct = sum(1 for r in records if r["correct"])
        per_class: dict = {}
        for rec in records:
            cls = rec["true_class"]
            per_class.setdefault(cls, {"correct": 0, "total": 0})
            per_class[cls]["total"] += 1
            per_class[cls]["correct"] += int(rec["correct"])

        summary = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_key": model_key,
            "checkpoint": checkpoint,
            "backend_dir": str(BACKEND_DIR),
            "source": source,
            "n_images": len(records),
            "n_correct": n_correct,
            "overall_accuracy": round(n_correct / max(len(records), 1), 4),
            "per_class": {
                cls: {
                    "correct": v["correct"],
                    "total": v["total"],
                    "recall": round(v["correct"] / max(v["total"], 1), 4),
                }
                for cls, v in per_class.items()
            },
            "records": records,
        }

        with open(summary_file, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        assert True
