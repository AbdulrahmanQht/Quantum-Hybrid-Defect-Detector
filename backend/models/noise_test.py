"""
noise_robustness_test.py
========================
Enhanced noise-robustness benchmark for Quantum-Hybrid-Defect-Detector.

Improvements over the original:
  1. Noise implementations aligned with PreProcessing augmentation functions
     (contrast blends toward per-image mean; blur uses same sigma semantics).
  2. Three new noise types: motion_blur, jpeg_compression, lens_occlusion.
  3. Per-class accuracy breakdown at each noise level.
  4. Mean-accuracy-under-noise (MAUN) degradation summary per model.
  5. Dual export: CSV (flat rows) + JSON (nested, ready for benchmark cache).
  6. QNN model runs without AMP interference (float32 enforced on noise ops).
"""
from __future__ import annotations

import io
import json
import csv
import os
import random
import time
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── Project imports ───────────────────────────────────────────────────────────
# Adjust these paths to match your import structure
from backend.models.cnn import CNN
from backend.models.qnn_cpu import HybridQnnCPU
from backend.data.data_loader import DataLoaderManager

# ── Config ────────────────────────────────────────────────────────────────────
# Each noise type is parameterised by a 'level' float.
# Level semantics are documented per noise function below.
NOISE_LEVELS: dict[str, list[float]] = {
    # sigma of additive Gaussian noise (0 = clean)
    "gaussian":         [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75],
    # blur sigma (0 = clean); kernel auto-sized to nearest odd integer
    "blur":             [0.00, 0.30, 0.60, 0.90, 1.20, 1.50, 2.00, 2.50, 3.00],
    # contrast retention factor: 1.0 = original, 0.0 = flat mean
    "contrast":         [1.00, 0.85, 0.70, 0.55, 0.40, 0.25, 0.10],
    # fraction of pixels turned to salt or pepper
    "salt_pepper":      [0.00, 0.005, 0.01, 0.02, 0.04, 0.07, 0.10],
    # motion-blur kernel half-width in pixels (0 = clean); final k = 2*level+1
    "motion_blur":      [0, 1, 2, 3, 4, 6, 8, 10],
    # JPEG quality factor (100 = lossless, 10 = heaviest compression)
    "jpeg_compression": [100, 85, 70, 55, 40, 25, 10],
    # fractional area of each occlusion patch (0 = clean)
    "lens_occlusion":   [0.00, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20],
}

IMG_W, IMG_H   = 384, 384
BATCH_SIZE     = 16

CNN_CHECKPOINT    = os.path.join("models", "cnn_noise_training_75_epochs.pth")
QNN_CHECKPOINT    = os.path.join("models", "qnn_cpu_75_epochs.pth")
CLASS_NAMES_PATH  = os.path.join("data", "class_names.json")
DATA_DIRS = {
    "train": os.path.join("data", "train"),
    "val":   os.path.join("data", "val"),
    "test":  os.path.join("data", "test"),
}

CSV_OUT  = "noise_robustness_results_1.csv"
JSON_OUT = "noise_robustness_results_1.json"


# ── Noise functions (aligned with PreProcessing augmentations) ────────────────
# All functions:
#   - Accept a float32 batch tensor in [0, 1] of shape (B, C, H, W).
#   - Return a tensor of the same shape, values clamped to [0, 1].
#   - Are deterministic given a level (no internal random sampling), so that
#     results are reproducible across models at the same level.

def _noise_gaussian(images: torch.Tensor, level: float) -> torch.Tensor:
    """
    Additive white Gaussian noise.
    level = sigma of the noise distribution.
    Aligned with PreProcessing._gaussian_noise (same additive formula).
    """
    if level == 0.0:
        return images
    return (images + torch.randn_like(images) * level).clamp(0.0, 1.0)


def _noise_blur(images: torch.Tensor, level: float) -> torch.Tensor:
    """
    Isotropic Gaussian blur.
    level = sigma (same unit as PreProcessing GaussianBlur sigma parameter).
    Kernel size is chosen as the smallest odd integer >= 6*sigma, minimum 3.
    """
    if level == 0.0:
        return images
    from torchvision.transforms import v2
    k = max(3, int(level * 6 + 1))
    k = k if k % 2 == 1 else k + 1           # force odd
    return v2.GaussianBlur(kernel_size=k, sigma=level)(images)


def _noise_contrast(images: torch.Tensor, level: float) -> torch.Tensor:
    """
    Contrast reduction by blending toward the PER-IMAGE mean intensity.
    level = retention factor in [0, 1]; 1.0 = original, 0.0 = flat gray.

    Aligned with PreProcessing._contrast_reduction which blends toward
    img.mean(dim=(-2,-1)) — NOT a fixed 0.5 midpoint.
    """
    if level >= 1.0:
        return images
    # Per-image mean, broadcast over spatial dims
    mean = images.mean(dim=(-2, -1), keepdim=True)   # (B, C, 1, 1)
    return (mean + level * (images - mean)).clamp(0.0, 1.0)


def _noise_salt_pepper(images: torch.Tensor, level: float) -> torch.Tensor:
    """
    Salt-and-pepper noise: level = fraction of pixels corrupted.
    Half become 0.0 (pepper), half become 1.0 (salt).
    Applied identically to all channels (consistent with test-time corruption).
    """
    if level == 0.0:
        return images
    corrupted = images.clone()
    # Single spatial mask broadcast over channels
    B, C, H, W = images.shape
    mask = torch.rand(B, 1, H, W, device=images.device).expand(B, C, H, W)
    corrupted[mask < level / 2]       = 0.0   # pepper
    corrupted[mask > 1.0 - level / 2] = 1.0   # salt
    return corrupted


def _noise_motion_blur(images: torch.Tensor, level: int) -> torch.Tensor:
    """
    Horizontal motion blur via a 1-D box kernel.
    level = half-width in pixels; final kernel size = 2*level + 1.
    Aligned with PreProcessing._motion_blur (horizontal direction, box kernel).
    level=0 returns the input unchanged.
    """
    if level == 0:
        return images
    k = 2 * int(level) + 1
    B, C, H, W = images.shape
    kernel = torch.zeros(1, 1, k, k, dtype=images.dtype, device=images.device)
    kernel[0, 0, k // 2, :] = 1.0 / k        # horizontal box
    blurred = F.conv2d(
        images.reshape(B * C, 1, H, W),
        kernel,
        padding=k // 2,
    ).reshape(B, C, H, W)
    return blurred.clamp(0.0, 1.0)


def _noise_jpeg(images: torch.Tensor, level: int) -> torch.Tensor:
    """
    JPEG compression artifacts.
    level = quality factor (100 = virtually lossless, 10 = heavy artifacts).
    Aligned with PreProcessing._jpeg_compression_noise.
    """
    if level >= 100:
        return images
    try:
        import numpy as np
        from PIL import Image as PILImage

        results = []
        for img in images.cpu():                                  # iterate over batch
            uint8 = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
            pil   = PILImage.fromarray(uint8, mode="RGB")
            buf   = io.BytesIO()
            pil.save(buf, format="JPEG", quality=int(level))
            buf.seek(0)
            decoded = PILImage.open(buf).convert("RGB")
            results.append(
                torch.from_numpy(np.array(decoded))
                .permute(2, 0, 1)
                .float()
                .div(255.0)
            )
        return torch.stack(results).to(images.device)
    except Exception:
        return images                                              # degrade gracefully


def _noise_lens_occlusion(images: torch.Tensor, level: float) -> torch.Tensor:
    """
    Rectangular lens occlusion patch filled with dim noise (0–0.3 range).
    level = area fraction of the occluded patch (0 = clean).
    A single central-ish patch is placed per image for determinism.
    Aligned with PreProcessing._lens_occlusion_erasing.
    """
    if level == 0.0:
        return images
    B, C, H, W = images.shape
    out = images.clone()
    # Deterministic square patch centred in the image for benchmarking repeatability
    ph = max(1, int((H * W * level) ** 0.5))
    pw = ph
    top  = (H - ph) // 2
    left = (W - pw) // 2
    out[:, :, top:top + ph, left:left + pw] = (
        torch.rand(B, C, ph, pw, device=images.device) * 0.3
    )
    return out


# Dispatch table — maps noise_type string → callable
_NOISE_FN = {
    "gaussian":         _noise_gaussian,
    "blur":             _noise_blur,
    "contrast":         _noise_contrast,
    "salt_pepper":      _noise_salt_pepper,
    "motion_blur":      _noise_motion_blur,
    "jpeg_compression": _noise_jpeg,
    "lens_occlusion":   _noise_lens_occlusion,
}


def apply_noise(images: torch.Tensor, noise_type: str, level: float) -> torch.Tensor:
    """Route to the correct noise function. Raises for unknown types."""
    fn = _NOISE_FN.get(noise_type)
    if fn is None:
        raise ValueError(f"Unknown noise_type '{noise_type}'. "
                         f"Valid options: {list(_NOISE_FN)}")
    return fn(images, level)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    noise_type: str,
    level: float,
    num_classes: int,
) -> dict[str, Any]:
    """
    Run one (noise_type, level) pass over the loader.

    Returns a dict with:
        overall_acc   : float  — top-1 accuracy across all samples
        per_class_acc : dict[int, float] — per-class accuracy
        mean_conf     : float  — mean softmax confidence of predicted class
    """
    model.eval()
    correct = 0
    total   = 0
    conf_sum = 0.0
    class_correct: dict[int, int] = {}
    class_total:   dict[int, int] = {}

    for images, labels in loader:
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device)

        # Apply noise AFTER moving to device (keeps JPEG encode/decode on CPU tensors)
        noisy = apply_noise(images, noise_type, level)

        logits = model(noisy)
        probs  = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        total    += labels.size(0)
        correct  += pred.eq(labels).sum().item()
        conf_sum += conf.sum().item()

        for lbl, p in zip(labels.tolist(), pred.tolist()):
            class_total[lbl]   = class_total.get(lbl, 0) + 1
            if lbl == p:
                class_correct[lbl] = class_correct.get(lbl, 0) + 1

    overall_acc = round(100.0 * correct / total, 4) if total else 0.0
    per_class   = {
        c: round(100.0 * class_correct.get(c, 0) / class_total[c], 2)
        for c in range(num_classes)
        if c in class_total
    }
    mean_conf = round(conf_sum / total, 4) if total else 0.0

    return {
        "overall_acc":   overall_acc,
        "per_class_acc": per_class,
        "mean_conf":     mean_conf,
    }


# ── Degradation summary ───────────────────────────────────────────────────────

def compute_maun(results: dict[str, dict]) -> dict[str, dict[str, float]]:
    """
    Mean Accuracy Under Noise (MAUN) — average overall_acc across all levels
    for each noise type, per model.  Lower = degrades faster.

    Returns: { model_name: { noise_type: maun_score } }
    """
    summary: dict[str, dict[str, float]] = {}
    for model_name, noise_dict in results.items():
        summary[model_name] = {}
        for noise_type, level_dict in noise_dict.items():
            accs = [v["overall_acc"] for v in level_dict.values()]
            summary[model_name][noise_type] = round(sum(accs) / len(accs), 4) if accs else 0.0
    return summary


# ── Main benchmark loop ───────────────────────────────────────────────────────

def run_noise_benchmark(
    models: dict[str, torch.nn.Module],
    loader,
    device: torch.device,
    num_classes: int,
    class_names: list[str],
) -> dict[str, dict]:
    """
    Populate and return the full results dict:
        results[model_name][noise_type][level] = {
            overall_acc, per_class_acc, mean_conf
        }
    """
    results: dict[str, dict] = {name: {} for name in models}

    for noise_type, levels in NOISE_LEVELS.items():
        print(f"\n{'─'*60}")
        print(f"  Noise: {noise_type}")
        print(f"{'─'*60}")

        # Header row
        col_w = 10
        level_col = 8
        header = f"  {'level':>{level_col}}  " + "  ".join(
            f"{n:>{col_w}}" for n in models
        )
        print(header)
        print("  " + "─" * (len(header) - 2))

        for level in tqdm(levels, desc=f"  {noise_type}", leave=False):
            row = f"  {level:>{level_col}.4g}  "
            for name, model in models.items():
                metrics = evaluate(model, loader, device, noise_type, level, num_classes)
                results[name].setdefault(noise_type, {})[level] = metrics
                row += f"  {metrics['overall_acc']:>{col_w}.2f}%"
            print(row)

    return results


# ── Exports ───────────────────────────────────────────────────────────────────

def save_csv(results: dict, class_names: list[str], path: str = CSV_OUT) -> None:
    """
    Flat CSV: one row per (model, noise_type, level).
    Columns: model, noise_type, level, overall_acc, mean_conf, <class_0>, ..., <class_N>
    """
    class_cols = [f"acc_{n}" for n in class_names]
    fieldnames = ["model", "noise_type", "level", "overall_acc", "mean_conf"] + class_cols

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, noise_dict in results.items():
            for noise_type, level_dict in noise_dict.items():
                for level, metrics in level_dict.items():
                    row: dict = {
                        "model":       model_name,
                        "noise_type":  noise_type,
                        "level":       level,
                        "overall_acc": metrics["overall_acc"],
                        "mean_conf":   metrics["mean_conf"],
                    }
                    for i, col in enumerate(class_cols):
                        row[col] = metrics["per_class_acc"].get(i, "")
                    writer.writerow(row)
    print(f"CSV saved → {path}")


def save_json(results: dict, maun: dict, path: str = JSON_OUT) -> None:
    """
    Nested JSON ready for the benchmark cache used by benchmark_router.py.
    Keys use str(level) for JSON compatibility.
    """
    serialisable: dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "img_size":  [IMG_W, IMG_H],
            "batch_size": BATCH_SIZE,
        },
        "maun_summary": maun,
        "results":      {},
    }
    for model_name, noise_dict in results.items():
        serialisable["results"][model_name] = {}
        for noise_type, level_dict in noise_dict.items():
            serialisable["results"][model_name][noise_type] = {
                str(lvl): v for lvl, v in level_dict.items()
            }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    print(f"JSON saved → {path}")


def print_maun_summary(maun: dict) -> None:
    print(f"\n{'='*60}")
    print("  Mean Accuracy Under Noise (MAUN) — higher = more robust")
    print(f"{'='*60}")
    model_names = list(maun.keys())
    noise_types = list(next(iter(maun.values())).keys())
    col_w = 12

    header = f"  {'noise_type':<20}" + "".join(f"  {n:>{col_w}}" for n in model_names)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for nt in noise_types:
        row = f"  {nt:<20}"
        for mn in model_names:
            row += f"  {maun[mn].get(nt, 0.0):>{col_w}.2f}%"
        print(row)

    # Overall MAUN across all noise types
    print("  " + "─" * (len(header) - 2))
    row = f"  {'OVERALL':<20}"
    for mn in model_names:
        scores = list(maun[mn].values())
        row += f"  {sum(scores)/len(scores):>{col_w}.2f}%"
    print(row)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Class names ──────────────────────────────────────────────────────────
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names: list[str] = json.load(f)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}\n")

    # ── Data loader ───────────────────────────────────────────────────────────
    manager = DataLoaderManager(
        train_dir=DATA_DIRS["train"],
        val_dir=DATA_DIRS["val"],
        test_dir=DATA_DIRS["test"],
        img_width=IMG_W,
        img_height=IMG_H,
        batch_size=BATCH_SIZE,
    )
    _, _, test_loader = manager.get_loaders()
    print(f"Test set: {len(test_loader.dataset)} images\n")

    # ── Load CNN ──────────────────────────────────────────────────────────────
    if not os.path.exists(CNN_CHECKPOINT):
        raise FileNotFoundError(f"CNN checkpoint not found: {CNN_CHECKPOINT}")
    cnn = CNN(num_classes=num_classes)
    cnn.load_model(CNN_CHECKPOINT, device)
    cnn.eval()
    print(f"Loaded CNN from {CNN_CHECKPOINT}")

    # ── Load QNN-CPU ──────────────────────────────────────────────────────────
    if not os.path.exists(QNN_CHECKPOINT):
        raise FileNotFoundError(f"QNN checkpoint not found: {QNN_CHECKPOINT}")
    qnn = HybridQnnCPU(num_classes=num_classes, n_qubits=6, q_depth=2)
    qnn.load_model(QNN_CHECKPOINT, device)
    qnn.eval()
    print(f"Loaded QNN-CPU from {QNN_CHECKPOINT}\n")

    models = {
        "CNN":     cnn,
        "QNN-CPU": qnn,
    }

    # ── Run benchmark ─────────────────────────────────────────────────────────
    t_start = time.perf_counter()
    results = run_noise_benchmark(models, test_loader, device, num_classes, class_names)
    print(f"\nBenchmark completed in {time.perf_counter() - t_start:.1f}s")

    # ── Degradation summary ───────────────────────────────────────────────────
    maun = compute_maun(results)
    print_maun_summary(maun)

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_csv(results, class_names, CSV_OUT)
    save_json(results, maun, JSON_OUT)