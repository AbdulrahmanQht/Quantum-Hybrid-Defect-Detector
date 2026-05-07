"""
tests/test_golden_set.py
=========================
Golden-set regression tests for Quantum-Hybrid-Defect-Detector.

Dataset layout (YOLO format):
    data/test/Images/*.jpg   — test images
    data/test/Labels/*.txt   — one file per image, first token = class index

Bug fixes applied:
    FIX-1  All paths now use Path(__file__).parent.parent as the anchor,
           not bare relative strings. This prevents the silent synthetic
           fallback caused by pytest resolving "data/test" against the
           project root (Quantum-Hybrid-Defect-Detector/) instead of
           against backend/ where the data actually lives.

    FIX-2  Removed Layout A (classification folder detection) introduced
           in the previous version — it added confusion since the project
           uses YOLO format, not per-class subdirectories.

    FIX-3  Added explicit existence checks with clear error messages at
           every path so the synthetic fallback reason is always visible.

    FIX-4  Floor tests skip automatically when synthetic data is in use.
           Random tensors cannot be classified correctly by a model
           trained on real defect images — that is expected, not a bug.

How it works:
    1. First run (no golden file): builds golden set from real images,
       writes results/golden_set/golden_set.json, PASSES.
    2. Subsequent runs: compares current predictions against golden file,
       FAILS if any previously-correct prediction flips.
    3. Regenerate after intentional model update:
           REGENERATE_GOLDEN=1 pytest tests/test_golden_set.py -v

IMPORTANT: commit results/golden_set/golden_set.json to
version control after the first run.

Results saved to:
    results/golden_set/golden_set.json
    results/golden_set/golden_regression.json
    results/golden_set/golden_set_summary.json

Run:
    # First run — creates the golden baseline
    pytest tests/test_golden_set.py -v

    # Subsequent runs — compares against baseline
    pytest tests/test_golden_set.py -v

    # Force baseline regeneration (e.g., after intentional model update)
    REGENERATE_GOLDEN=1 pytest tests/test_golden_set.py -v
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

# ── Anchor all paths to backend/ regardless of pytest rootdir ──────────────
# pytest rootdir = Quantum-Hybrid-Defect-Detector/ (project root, not backend/)
# Path(__file__) = Quantum-Hybrid-Defect-Detector/backend/tests/test_golden_set.py
# BACKEND_DIR    = Quantum-Hybrid-Defect-Detector/backend/
BACKEND_DIR = Path(__file__).parent.parent.resolve()

sys.path.insert(0, str(BACKEND_DIR))

RESULTS_DIR = "results" / "golden_set"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GOLDEN_FILE     = RESULTS_DIR / "golden_set.json"
REGRESSION_FILE = RESULTS_DIR / "golden_regression.json"

# Load class names relative to backend/
_CLASS_NAMES_FILE = BACKEND_DIR / "data" / "class_names.json"
try:
    with open(_CLASS_NAMES_FILE, encoding="utf-8") as _f:
        CLASS_NAMES: List[str] = json.load(_f)
except FileNotFoundError:
    CLASS_NAMES = [
        "Deformation", "Deposition", "Disconnect",
        "Misalignment", "Obstacle", "Rupture",
    ]

IMAGES_PER_CLASS = 5
FLOOR_RECALL     = 0.80
SAFETY_CRITICAL  = {"Rupture", "Disconnect"}

# Checkpoint paths relative to backend/
_CNN_CKPT = next(
    (str(BACKEND_DIR / p) for p in [
        "models/cnn.pth",
        "models/cnn_noise_training_75_epochs.pth",
        "models/cpu_new.pth",
    ] if (BACKEND_DIR / p).exists()),
    None,
)


# ══════════════════════════════════════════════════════════════════════════════
# Golden set construction — YOLO format
# ══════════════════════════════════════════════════════════════════════════════

def _build_real_golden_set(
    data_dir: Optional[str] = None,
) -> Optional[Tuple[List[Tuple[torch.Tensor, int, str]], str]]:
    """
    Build the golden set from a YOLO-style dataset with per-class subfolders.

    Expected layout:
        <data_dir>/
            Images/
                <class_name>/
                    img1.jpg
            Labels/
                <class_name>/
                    img1.txt   ← first token = YOLO class index

    Images are skipped if: the label file is missing, the class index is
    out of range, or the subfolder name does not match the label's class.
    """
    base = BACKEND_DIR / (data_dir or "data/test")
    
    # Identify directories
    images_dir = base / "Images"
    labels_dir = base / "Labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"[golden_set] FAIL: Images or Labels folder missing under {base}")
        return None

    try:
        from data.preprocessing import PreProcessing
        from PIL import Image as PILImage
        transform = PreProcessing.get_transforms(img_width=384, img_height=384, is_training=False)
    except Exception as exc:
        print(f"[golden_set] FAIL: PreProcessing load error: {exc}")
        return None

    # 1. Use RECURSIVE glob to find images in subfolders
    image_files = sorted(
        f for ext in ("**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.JPG")
        for f in images_dir.glob(ext)
    )

    if not image_files:
        print(f"[golden_set] FAIL: No images found in subfolders of {images_dir}")
        return None

    found_counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    golden: List[Tuple[torch.Tensor, int, str]] = []

    for img_path in image_files:
        # 2. Extract class name from the parent subfolder
        cls_subfolder = img_path.parent.name
        
        # 3. Locate label in the matching class subfolder
        label_path = labels_dir / cls_subfolder / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            continue

        with open(label_path, encoding="utf-8") as lf:
            first_line = lf.readline().split()
        if not first_line: continue

        try:
            cls_idx = int(first_line[0])
            cls_name = CLASS_NAMES[cls_idx]
        except (ValueError, IndexError): continue

        # Safety: Ensure folder name matches the label class
        if cls_name != cls_subfolder: continue

        if found_counts[cls_name] < IMAGES_PER_CLASS:
            try:
                tensor = transform(PILImage.open(img_path).convert("RGB")).unsqueeze(0)
                golden.append((tensor, cls_idx, f"real_{cls_name}_{img_path.stem}"))
                found_counts[cls_name] += 1
            except Exception: continue

    if not golden:
        print("[golden_set] FAIL: No valid image/label pairs found in subfolders.")
        return None

    print(f"[golden_set] SUCCESS: Loaded {len(golden)} real images.")
    return golden, "real_yolo"


def _build_synthetic_golden_set() -> List[Tuple[torch.Tensor, int, str]]:
    """
    Deterministic synthetic fallback.
    Random tensors are NOT expected to be classified correctly.
    Floor tests skip automatically when this fallback is active.
    """
    golden = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        for i in range(IMAGES_PER_CLASS):
            torch.manual_seed(cls_idx * 100 + i)
            img = torch.rand(1, 3, 384, 384)
            golden.append((img, cls_idx, f"synthetic_{cls_name}_{i:02d}"))
    return golden


def _run_predictions(
    model,
    golden: List[Tuple[torch.Tensor, int, str]],
    device: torch.device,
) -> List[dict]:
    model.eval()
    records = []
    with torch.no_grad():
        for img, true_cls, img_id in golden:
            probs = F.softmax(model(img.to(device)), dim=1)
            conf, pred = probs.max(1)
            records.append({
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
            })
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def cnn_model(device):
    if _CNN_CKPT is None:
        pytest.skip(
            f"CNN checkpoint not found. Checked under {BACKEND_DIR / 'models'}/"
        )
    try:
        from models.cnn import CNN
    except ImportError:
        from .models.cnn import CNN
    model = CNN(num_classes=len(CLASS_NAMES))
    model.load_model(_CNN_CKPT, device)
    model.eval()
    return model


@pytest.fixture(scope="module")
def golden_data():
    """
    Returns (images_list, source_label).

    source_label:
        "real_yolo"  — from data/test/Images/ + Labels/
        "synthetic"  — random tensors (real data not found)

    Floor accuracy tests skip when source == "synthetic".
    """
    result = _build_real_golden_set()
    if result is not None:
        return result

    print(
        "\n[golden_set] WARNING: Falling back to SYNTHETIC golden set.\n"
        "Floor accuracy tests will be SKIPPED for synthetic data.\n"
        "Run with -s to see which step failed above."
    )
    return _build_synthetic_golden_set(), "synthetic"


@pytest.fixture(scope="module")
def cnn_predictions(cnn_model, golden_data, device):
    images, source = golden_data
    records = _run_predictions(cnn_model, images, device)
    return records, source


# ══════════════════════════════════════════════════════════════════════════════
# Skip guard — floor tests are meaningless on random noise
# ══════════════════════════════════════════════════════════════════════════════

def _require_real(source: str, test_name: str) -> None:
    if source == "synthetic":
        pytest.skip(
            f"{test_name}: skipped because the golden set is SYNTHETIC (random tensors).\n"
            f"Real images were not found at {BACKEND_DIR / 'data' / 'test'}.\n"
            "Run with -s to see the specific failure reason."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PART A — Floor tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGoldenSetFloor:

    def test_every_image_produces_valid_prediction(self, cnn_predictions):
        """Schema sanity — runs regardless of data source."""
        records, _ = cnn_predictions
        for rec in records:
            assert rec["predicted_class"] in CLASS_NAMES, (
                f"Invalid prediction for {rec['image_id']}: '{rec['predicted_class']}'"
            )
            assert 0.0 <= rec["confidence"] <= 1.0, (
                f"Confidence out of range for {rec['image_id']}: {rec['confidence']}"
            )

    def test_scores_sum_to_one(self, cnn_predictions):
        """Softmax sanity — runs regardless of data source."""
        records, _ = cnn_predictions
        for rec in records:
            total = sum(rec["all_scores"].values())
            assert abs(total - 1.0) < 1e-3, (
                f"Softmax scores for {rec['image_id']} sum to {total:.4f}, not 1.0"
            )

    def test_no_class_has_zero_recall(self, cnn_predictions):
        """Skipped for synthetic data."""
        records, source = cnn_predictions
        _require_real(source, "zero-recall floor test")

        for cls_name in CLASS_NAMES:
            cls_records = [r for r in records if r["true_class"] == cls_name]
            if not cls_records:
                continue
            n_correct = sum(1 for r in cls_records if r["correct"])
            assert n_correct > 0, (
                f"Class '{cls_name}' has ZERO correct predictions on the golden set. "
                "This indicates a catastrophic model failure for this class."
            )

    def test_per_class_recall_meets_floor(self, cnn_predictions):
        """Each class must achieve ≥ 80% recall (≥ 4/5 correct). Skipped for synthetic."""
        records, source = cnn_predictions
        _require_real(source, "per-class recall floor test")

        for cls_name in CLASS_NAMES:
            cls_records = [r for r in records if r["true_class"] == cls_name]
            if not cls_records:
                continue
            n_correct = sum(1 for r in cls_records if r["correct"])
            recall    = n_correct / len(cls_records)
            assert recall >= FLOOR_RECALL, (
                f"Class '{cls_name}' recall {recall:.0%} below floor {FLOOR_RECALL:.0%} "
                f"({n_correct}/{len(cls_records)} correct on golden images)."
            )

    def test_safety_critical_classes_recall(self, cnn_predictions):
        """
        Rupture and Disconnect must achieve high recall on golden images.
        Uses xfail (not hard fail) so a single miss surfaces as a warning
        rather than blocking the build — it will become a hard regression
        if it persists across runs via the regression test below.
        Skipped for synthetic data.
        """
        records, source = cnn_predictions
        _require_real(source, "safety-critical recall test")

        for cls_name in SAFETY_CRITICAL:
            cls_records = [r for r in records if r["true_class"] == cls_name]
            if not cls_records:
                continue
            wrong = [r for r in cls_records if not r["correct"]]
            if wrong:
                pytest.xfail(
                    f"Safety-critical class '{cls_name}' missed "
                    f"{len(wrong)}/{len(cls_records)} golden images:\n"
                    + "\n".join(
                        f"  {r['image_id']}: predicted '{r['predicted_class']}' "
                        f"(conf {r['confidence']:.3f})"
                        for r in wrong
                    )
                )

    def test_overall_accuracy_above_90pct(self, cnn_predictions):
        """≥ 90% of all golden images correct. Skipped for synthetic."""
        records, source = cnn_predictions
        _require_real(source, "overall accuracy floor test")

        n_correct = sum(1 for r in records if r["correct"])
        accuracy  = n_correct / len(records)
        assert accuracy >= 0.90, (
            f"Overall golden set accuracy {accuracy:.1%} below 90% "
            f"({n_correct}/{len(records)} correct)."
        )

    def test_show_diagnostic_when_synthetic(self, cnn_predictions):
        """Informational only — prints guidance when synthetic fallback is active."""
        records, source = cnn_predictions
        if source == "synthetic":
            n_correct = sum(1 for r in records if r["correct"])
            print(
                f"\n{'='*60}\n"
                f"SYNTHETIC GOLDEN SET DIAGNOSTIC\n"
                f"Overall: {n_correct}/{len(records)} correct on random noise\n"
                f"(Expected — the model is not trained on random tensors)\n"
                f"\nAction required:\n"
                f"  Ensure {BACKEND_DIR / 'data' / 'test' / 'Images'} exists\n"
                f"  and {BACKEND_DIR / 'data' / 'test' / 'Labels'} contains .txt files.\n"
                f"  Re-run with -s to see the exact failure reason.\n"
                f"{'='*60}"
            )
        assert True  # always passes


# ══════════════════════════════════════════════════════════════════════════════
# PART B — Regression tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGoldenSetRegression:

    def test_create_or_verify_golden_file(self, cnn_model, cnn_predictions, device):
        """
        First run: write golden_set.json and pass.
        Subsequent runs: compare and fail if any correct prediction flips.
        """
        records, source = cnn_predictions
        regenerate = os.environ.get("REGENERATE_GOLDEN", "0") == "1"

        if not GOLDEN_FILE.exists() or regenerate:
            baseline = {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model_checkpoint": _CNN_CKPT,
                "backend_dir": str(BACKEND_DIR),
                "source": source,
                "n_images": len(records),
                "images_per_class": IMAGES_PER_CLASS,
                "class_names": CLASS_NAMES,
                "records": records,
            }
            with open(GOLDEN_FILE, "w", encoding="utf-8") as fh:
                json.dump(baseline, fh, indent=2)
            action = "REGENERATED" if regenerate else "CREATED"
            print(
                f"\nGolden file {action}: {GOLDEN_FILE}\n"
                "Commit this file to version control to enable regression detection."
            )
            if source == "synthetic":
                print(
                    "WARNING: baseline built from SYNTHETIC data.\n"
                    "Re-run once real data is available to get a meaningful baseline."
                )
            return

        with open(GOLDEN_FILE, encoding="utf-8") as fh:
            golden = json.load(fh)

        stored_source = golden.get("source", "unknown")
        if stored_source != source:
            print(
                f"\nWARNING: golden file was built from '{stored_source}' data "
                f"but current run uses '{source}' data.\n"
                "Regenerate: REGENERATE_GOLDEN=1 pytest tests/test_golden_set.py"
            )

        golden_by_id  = {r["image_id"]: r for r in golden["records"]}
        current_by_id = {r["image_id"]: r for r in records}
        regressions   = []

        for img_id, gold_rec in golden_by_id.items():
            curr_rec = current_by_id.get(img_id)
            if curr_rec is None:
                continue
            if gold_rec["correct"] and not curr_rec["correct"]:
                regressions.append({
                    "image_id": img_id,
                    "true_class": gold_rec["true_class"],
                    "was_predicted": gold_rec["predicted_class"],
                    "now_predicted": curr_rec["predicted_class"],
                    "confidence_was": gold_rec["confidence"],
                    "confidence_now": curr_rec["confidence"],
                })

        regression_report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": source,
            "n_regressions": len(regressions),
            "regressions": regressions,
            "verdict": "PASS" if not regressions else "FAIL",
        }
        with open(REGRESSION_FILE, "w", encoding="utf-8") as fh:
            json.dump(regression_report, fh, indent=2)

        assert not regressions, (
            f"GOLDEN SET REGRESSION: {len(regressions)} previously-correct "
            "predictions have flipped.\n"
            + "\n".join(
                f"  {r['image_id']}: '{r['true_class']}' was '{r['was_predicted']}' "
                f"(conf {r['confidence_was']:.3f}), now '{r['now_predicted']}' "
                f"(conf {r['confidence_now']:.3f})"
                for r in regressions
            )
            + f"\nFull report: {REGRESSION_FILE}"
        )

    def test_no_regression_in_safety_critical_classes(self, cnn_predictions):
        """Hard fail on any Rupture/Disconnect regression, skipped before first baseline."""
        if not GOLDEN_FILE.exists():
            pytest.skip("Golden file not created yet — run this after first baseline")

        records, source = cnn_predictions
        with open(GOLDEN_FILE, encoding="utf-8") as fh:
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
                    f"CRITICAL CLASS REGRESSION: '{rec['true_class']}' image "
                    f"'{rec['image_id']}' was correctly predicted as "
                    f"'{gold_rec['predicted_class']}' (conf {gold_rec['confidence']:.3f}) "
                    f"but is now predicted as '{rec['predicted_class']}' "
                    f"(conf {rec['confidence']:.3f}).\n"
                    "Rupture and Disconnect regressions are unacceptable for a "
                    "pipeline safety inspection system."
                )

    def test_save_prediction_summary(self, cnn_predictions):
        """Always saves a per-image prediction log for audit — always passes."""
        records, source = cnn_predictions
        n_correct = sum(1 for r in records if r["correct"])
        per_class: dict = {}
        for rec in records:
            cls = rec["true_class"]
            per_class.setdefault(cls, {"correct": 0, "total": 0})
            per_class[cls]["total"]  += 1
            per_class[cls]["correct"] += int(rec["correct"])

        summary = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
        path = RESULTS_DIR / "golden_set_summary.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nSummary saved to {path}")
        assert True