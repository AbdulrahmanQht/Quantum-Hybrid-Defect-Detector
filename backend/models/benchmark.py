"""
Benchmark Runner — Quantum-Hybrid-Defect-Detector
==================================================
Run this script ONCE after training to generate benchmark_results.json
and a set of companion CSV files.

The FastAPI /api/v1/benchmark endpoint reads benchmark_results.json
and serves it instantly.

What this computes:
    1. Clean accuracy on the full test set (all 3 models)
    2. Per-class Accuracy / Precision / Recall / F1  (all 3 models)
    3. Macro & Weighted averages per model
    4. Confusion matrix per model
    5. Noise robustness sweep — accuracy at N levels across 4 noise types (all 3 models)
    6. Average single-image inference latency (all 3 models)
    7. Config metadata — test set size, resolution, epochs, qubits, batch size

CSV exports:
    - per_class_metrics.csv      — per-class P / R / F1 / Acc for all models
    - summary_metrics.csv        — macro & weighted averages + overall accuracy
    - noise_robustness.csv       — accuracy per noise_type / level per model
    - inference_latency.csv      — avg latency (ms) per model
    - confusion_matrix_<model>.csv — one file per model
"""

from __future__ import annotations

import os
import csv
import json
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.models.cnn import CNN
from backend.data.preprocessing import PreProcessing
from backend.models.qnn_cpu import HybridQnnCPU
from backend.models.qnn_gpu import HybridQnnGPU
from backend.data.data_loader import DataLoaderManager
from backend.utils.logger import Logger

logger = Logger()

# Configuration — mirrors your training setup
CONFIG = {
    "img_width":        384,
    "img_height":       384,
    "batch_size":       16,
    "training_epochs":  50,
    "n_qubits":         6,
    "q_depth":          2,
    "noise_levels": {
        "gaussian":    [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75],
        "blur":        [0.00, 0.30, 0.60, 0.90, 1.20, 1.50, 2.00, 2.50, 3.00],
        "contrast":    [1.00, 0.85, 0.70, 0.55, 0.40, 0.25, 0.10],
        "salt_pepper": [0.00, 0.005, 0.01, 0.02, 0.04, 0.07, 0.10],
    },
    # Number of single images used to measure average inference latency
    "latency_samples":  50,
}

# 1. Get the directory where benchmark.py lives (.../backend/models/)
_current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the 'backend' root (.../backend/)
BACKEND_ROOT = os.path.dirname(_current_dir)

# 3. Update paths using BACKEND_ROOT
CHECKPOINT_PATHS = {
    "CNN": os.path.join(BACKEND_ROOT, "models", "cnn.pth"),
    "QNN_CPU": os.path.join(BACKEND_ROOT, "models", "qnn_cpu.pth"),
    "QNN_GPU": os.path.join(BACKEND_ROOT, "models", "qnn_gpu_6_qubits.pth"),
}

CLASS_NAMES_PATH = os.path.join(BACKEND_ROOT, "data", "class_names.json")
OUTPUT_PATH = os.path.join(BACKEND_ROOT, "data", "benchmark", "benchmark_results.json")

# Ensure this directory exists relative to backend
OUTPUT_DIR = os.path.join(BACKEND_ROOT, "data" , "benchmark")

DATA_DIRS = {
    "train": os.path.join(BACKEND_ROOT, "data", "train"),
    "val": os.path.join(BACKEND_ROOT, "data", "val"),
    "test": os.path.join(BACKEND_ROOT, "data", "test"),
}

ALL_MODEL_KEYS = ["CNN", "QNN_CPU", "QNN_GPU"]

# Model loading
def load_models(num_classes: int, device: torch.device, checkpoint_dir: str = "models",) -> dict[str, torch.nn.Module]:
    """
    Load all three models from their checkpoints.
    QNN_GPU is skipped gracefully if CUDA is unavailable.
    """
    models: dict[str, torch.nn.Module] = {}

    def _ckpt(key: str) -> str:
        return CHECKPOINT_PATHS[key]

    # CNN
    path = _ckpt("CNN")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CNN checkpoint not found: {path}")
    cnn = CNN(num_classes=num_classes)
    cnn.load_model(path, device)
    cnn.eval()
    models["CNN"] = cnn
    logger.info(f"CNN loaded from {path}")

    # QNN_CPU
    path = _ckpt("QNN_CPU")
    if not os.path.exists(path):
        raise FileNotFoundError(f"QNN_CPU checkpoint not found: {path}")
    qnn_cpu = HybridQnnCPU(num_classes=num_classes)
    qnn_cpu.load_model(path, device)
    qnn_cpu.eval()
    models["QNN_CPU"] = qnn_cpu
    logger.info(f"QNN_CPU loaded from {path}")

    # QNN_GPU
    if torch.cuda.is_available():
        path = _ckpt("QNN_GPU")
        if not os.path.exists(path):
            logger.warning(f"QNN_GPU checkpoint not found: {path}. Skipping.")
        else:
            qnn_gpu = HybridQnnGPU(num_classes=num_classes)
            qnn_gpu.load_model(path, device)
            qnn_gpu.eval()
            models["QNN_GPU"] = qnn_gpu
            logger.info(f"QNN_GPU loaded from {path}")
    else:
        logger.warning("CUDA unavailable — QNN_GPU will be excluded from benchmark.")

    return models

# Experiment 1 — Clean accuracy + per-class metrics + confusion matrix
@torch.no_grad()
def run_clean_evaluation(model: torch.nn.Module, test_loader: DataLoader, device: torch.device, class_names: list[str], model_label: str,) -> dict:
    """
    Full-pass evaluation on the clean test set.
    Returns:
        - overall accuracy
        - per-class precision / recall / F1 / accuracy
        - macro and weighted averages
        - confusion matrix (2-D list, rows = true, cols = predicted)
    """
    model.eval()
    all_preds:  list[int] = []
    all_labels: list[int] = []

    for images, labels in tqdm(test_loader, desc=f"[Clean Eval] {model_label}", colour="cyan"):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    n_classes = len(class_names)
    labels_range = list(range(n_classes))

    # Overall accuracy
    correct  = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = round(100.0 * correct / len(all_labels), 4)

    # Per-class metrics via sklearn
    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        all_labels, all_preds, labels=labels_range, zero_division=0
    )

    # Confusion matrix (needed for per-class accuracy)
    cm = confusion_matrix(all_labels, all_preds, labels=labels_range)
    # Per-class accuracy = TP / (all samples truly belonging to that class)
    class_totals  = cm.sum(axis=1)                           # shape: (n_classes,)
    class_correct = cm.diagonal()                            # TP per class
    per_class_acc = class_correct / class_totals.clip(min=1) # avoid div-by-zero

    per_class: dict[str, dict] = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "accuracy":  round(float(per_class_acc[i]) * 100, 2),
            "precision": round(float(precision_arr[i]) * 100, 2),
            "recall":    round(float(recall_arr[i])    * 100, 2),
            "f1":        round(float(f1_arr[i])        * 100, 2),
            "support":   int(support_arr[i]),
        }

    # Macro & Weighted averages
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro",    zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    averages = {
        "macro": {
            "precision": round(float(p_macro)    * 100, 2),
            "recall":    round(float(r_macro)    * 100, 2),
            "f1":        round(float(f1_macro)   * 100, 2),
        },
        "weighted": {
            "precision": round(float(p_weighted)    * 100, 2),
            "recall":    round(float(r_weighted)    * 100, 2),
            "f1":        round(float(f1_weighted)   * 100, 2),
        },
    }

    # Confusion matrix as 2-D list (JSON-serialisable) 
    cm_list = cm.tolist()   # list[list[int]]

    # Print full sklearn report
    report_str = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )
    logger.info(f"\n[{model_label}] Clean Evaluation Report:\n{report_str}")

    return {
        "accuracy":         accuracy,
        "per_class":        per_class,
        "averages":         averages,
        "confusion_matrix": cm_list,
        "n_samples":        len(all_labels),
    }

# Experiment 2 — Noise robustness sweep
def _apply_noise(images: torch.Tensor, noise_type: str, level: float) -> torch.Tensor:
    """
    Applies the requested noise type at the given intensity level.
    p=1.0 is always used here (benchmarking requires deterministic application).
 
    noise_type  level meaning
    ----------  ---------------------------------------------------------
    gaussian    std dev added to pixel values (0.0 = clean)
    blur        GaussianBlur sigma; kernel size is derived from level
    contrast    blend factor: 1.0 = full contrast, 0.0 = flat grey
    salt_pepper fraction of pixels corrupted (0.0 = clean)
    """
    if level == 0.0 or (noise_type == "contrast" and level == 1.0):
        return images
 
    if noise_type == "gaussian":
        return (images + torch.randn_like(images) * level).clamp(0.0, 1.0)
 
    if noise_type == "blur":
        from torchvision.transforms import v2
        k = max(3, int(level * 10) | 1)   # odd kernel, grows with level
        return v2.GaussianBlur(kernel_size=k, sigma=level)(images)
 
    if noise_type == "contrast":
        # Delegate to PreProcessing static method; force p=1.0 so it always fires.
        return PreProcessing._contrast_reduction(images, factor_range=(level, level), p=1.0)
 
    if noise_type == "salt_pepper":
        # Delegate to PreProcessing static method; force p=1.0 so it always fires.
        return PreProcessing._salt_and_pepper_noise(images, amount_range=(level, level), p=1.0)
 
    raise ValueError(f"Unknown noise_type '{noise_type}'. Choose: gaussian, blur, contrast, salt_pepper")

@torch.no_grad()
def _accuracy_with_noise(model: torch.nn.Module, data_loader: DataLoader, device: torch.device, sigma: float,) -> float:
    """Single-pass accuracy with Gaussian noise applied at the given sigma."""
    model.eval()
    correct = 0
    total   = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        if sigma > 0.0:
            images = (images + torch.randn_like(images) * sigma).clamp(0.0, 1.0)
        preds   = model(images).argmax(dim=1)
        total   += labels.size(0)
        correct += preds.eq(labels).sum().item()

    return round(100.0 * correct / total, 4)

def run_noise_sweep(models: dict[str, torch.nn.Module], test_loader: DataLoader, device: torch.device, noise_levels: dict[str, list[float]],) -> dict[str, list[dict]]:
    """
    Runs the robustness sweep across all four noise types and their levels.
 
    Returns a dict keyed by noise_type. Each value is a list of dicts:
        [{"level": float, "CNN": acc, "QNN_CPU": acc, "QNN_GPU": acc | None}, ...]
 
    contrast levels are descending (1.0 = clean, lower = more degraded),
    which is the opposite axis direction to the other three noise types.
    This is preserved as-is so the frontend can render it correctly.
    """
    results: dict[str, list[dict]] = {}
 
    for noise_type, levels in noise_levels.items():
        logger.info(f"\n  -- Noise type: {noise_type} --")
        rows: list[dict] = []
 
        for level in levels:
            row: dict = {"level": level}
 
            for label, model in models.items():
                acc = _accuracy_with_noise(model, test_loader, device, noise_type, level)
                row[label] = acc
                logger.info(f"    {noise_type} level={level:.3f} | {label} → {acc:.2f}%")
 
            # Ensure all model keys are present (QNN_GPU may be absent on CPU machine)
            for key in ALL_MODEL_KEYS:
                row.setdefault(key, None)
 
            rows.append(row)
            print(f"  {noise_type} level={level:.3f} → {row}")
 
        results[noise_type] = rows
 
    return results

# Experiment 3 — Inference latency
def run_latency_benchmark(models: dict[str, torch.nn.Module], test_loader: DataLoader, device: torch.device, n_samples: int,) -> dict[str, Optional[float]]:
    """
    Measures average single-image inference latency in milliseconds.
    Runs n_samples images one at a time (batch size = 1) and averages.
    The first few samples are used as warm-up and discarded.
    """
    WARMUP = 5
    results: dict[str, Optional[float]] = {}

    # Collect n_samples + WARMUP individual images from the test set
    single_images: list[torch.Tensor] = []
    for images, _ in test_loader:
        for i in range(images.size(0)):
            single_images.append(images[i].unsqueeze(0))
            if len(single_images) >= n_samples + WARMUP:
                break
        if len(single_images) >= n_samples + WARMUP:
            break

    for label, model in models.items():
        model.eval()
        latencies: list[float] = []

        with torch.no_grad():
            for idx, img in enumerate(single_images):
                img = img.to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(img)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000

                if idx >= WARMUP:
                    latencies.append(elapsed_ms)

        avg_ms = round(sum(latencies) / len(latencies), 3)
        results[label] = avg_ms
        logger.info(f"  Latency | {label} → avg {avg_ms:.2f} ms over {n_samples} samples")

    for key in ALL_MODEL_KEYS:
        results.setdefault(key, None)

    return results


def export_confusion_matrix_plots(clean_results: dict[str, dict], class_names: list[str], output_dir: str,) -> None:
    """
    Generates and saves a confusion matrix heatmap (PNG) for each model.
    """
    for model_key, result in clean_results.items():
        cm = result.get("confusion_matrix")
        if cm is None:
            continue

        cm_array = np.array(cm)

        # Set up the matplotlib figure
        plt.figure(figsize=(10, 8))

        # Draw the heatmap using seaborn
        sns.heatmap(
            cm_array,
            annot=True,  # Display the numbers in the cells
            fmt="d",  # Format as integers
            cmap="Blues",  # Use a blue color scale
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,  # Hide the colorbar for cleaner look
        )

        # Formatting labels and titles
        plt.title(f"Confusion Matrix — {model_key}", fontsize=16, pad=15)
        plt.ylabel("True Class", fontsize=12)
        plt.xlabel("Predicted Class", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save the plot
        path = os.path.join(output_dir, f"confusion_matrix_{model_key}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure to free up memory

        logger.info(f"Plot saved: {path}")


# CSV exports
def _na(value) -> str:
    """Format a numeric value or return 'N/A' for None."""
    return str(value) if value is not None else "N/A"

def export_per_class_metrics_csv(clean_results: dict[str, dict], class_names: list[str], output_dir: str,) -> None:
    """
    One row per (model, class).
    Columns: model, class, accuracy, precision, recall, f1, support
    """
    path = os.path.join(output_dir, "per_class_metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "class", "accuracy_%", "precision_%", "recall_%", "f1_%", "support"])
        for model_key, result in clean_results.items():
            for cls in class_names:
                m = result["per_class"].get(cls, {})
                writer.writerow([
                    model_key,
                    cls,
                    _na(m.get("accuracy")),
                    _na(m.get("precision")),
                    _na(m.get("recall")),
                    _na(m.get("f1")),
                    _na(m.get("support")),
                ])
    logger.info(f"CSV saved: {path}")

def export_summary_metrics_csv(clean_results: dict[str, dict], output_dir: str,) -> None:
    """
    One row per model.
    Columns: model, overall_accuracy, macro_p, macro_r, macro_f1,
             weighted_p, weighted_r, weighted_f1, n_samples
    """
    path = os.path.join(output_dir, "summary_metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "overall_accuracy_%",
            "macro_precision_%",  "macro_recall_%",    "macro_f1_%",
            "weighted_precision_%","weighted_recall_%", "weighted_f1_%",
            "n_samples",
        ])
        for model_key, result in clean_results.items():
            avg = result.get("averages", {})
            macro    = avg.get("macro",    {})
            weighted = avg.get("weighted", {})
            writer.writerow([
                model_key,
                _na(result.get("accuracy")),
                _na(macro.get("precision")),    _na(macro.get("recall")),    _na(macro.get("f1")),
                _na(weighted.get("precision")), _na(weighted.get("recall")), _na(weighted.get("f1")),
                _na(result.get("n_samples")),
            ])
    logger.info(f"CSV saved: {path}")

def export_noise_robustness_csv(noise_results: dict[str, list[dict]], output_dir: str,) -> None:
    """
    Single CSV with all noise types.
    Columns: noise_type, level, CNN_accuracy_%, QNN_CPU_accuracy_%, QNN_GPU_accuracy_%
 
    contrast levels decrease from 1.0 (clean) — opposite axis to other types.
    The noise_type column lets the frontend split and render each curve separately.
    """
    path = os.path.join(output_dir, "noise_robustness.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["noise_type", "level", "CNN_accuracy_%", "QNN_CPU_accuracy_%", "QNN_GPU_accuracy_%"])
        for noise_type, rows in noise_results.items():
            for row in rows:
                writer.writerow([
                    noise_type,
                    row["level"],
                    _na(row.get("CNN")),
                    _na(row.get("QNN_CPU")),
                    _na(row.get("QNN_GPU")),
                ])
    logger.info(f"CSV saved: {path}")

def export_latency_csv(latency_results: dict[str, Optional[float]], output_dir: str,) -> None:
    """
    One row per model.
    Columns: model, avg_latency_ms
    """
    path = os.path.join(output_dir, "inference_latency.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "avg_latency_ms"])
        for model_key in ALL_MODEL_KEYS:
            writer.writerow([model_key, _na(latency_results.get(model_key))])
    logger.info(f"CSV saved: {path}")


def export_confusion_matrix_csvs(clean_results: dict[str, dict], class_names: list[str], output_dir: str,) -> None:
    """
    One CSV per model.
    Row index = true class, column index = predicted class.
    First row and column are labelled with class names.
    """
    for model_key, result in clean_results.items():
        cm = result.get("confusion_matrix")
        if cm is None:
            continue
        path = os.path.join(output_dir, f"confusion_matrix_{model_key}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Header row: empty top-left cell, then predicted class names
            writer.writerow(["true \\ predicted"] + class_names)
            for true_idx, row in enumerate(cm):
                writer.writerow([class_names[true_idx]] + row)
        logger.info(f"CSV saved: {path}")

# Main
def run_benchmark() -> dict:
    cfg    = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running benchmark on device: {device}")

    # Load class names
    class_names_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", CLASS_NAMES_PATH
    )
    if not os.path.exists(class_names_path):
        raise FileNotFoundError(f"class_names.json not found at {class_names_path}")
    with open(class_names_path, encoding="utf-8") as f:
        class_names: list[str] = json.load(f)
    num_classes = len(class_names)
    logger.info(f"Classes ({num_classes}): {class_names}")

    # Data loaders ─
    manager = DataLoaderManager(
        train_dir=DATA_DIRS["train"],
        val_dir=DATA_DIRS["val"],
        test_dir=DATA_DIRS["test"],
        img_width=cfg["img_width"],
        img_height=cfg["img_height"],
        batch_size=cfg["batch_size"],
    )
    _, _, test_loader = manager.get_loaders()

    test_set_size = len(test_loader.dataset)
    logger.info(f"Test set size: {test_set_size} images")

    # Load models
    models = load_models(num_classes, device)

    # Experiment 1: Clean evaluation
    logger.info("\n=== Experiment 1: Clean Evaluation ===")
    clean_results: dict[str, dict] = {}
    for label, model in models.items():
        clean_results[label] = run_clean_evaluation(
            model, test_loader, device, class_names, label
        )

    # Experiment 2: Noise robustness sweep
    logger.info("\n=== Experiment 2: Noise Robustness Sweep ===")
    noise_results = run_noise_sweep(models, test_loader, device, cfg["noise_levels"])

    # Experiment 3: Inference latency
    logger.info("\n=== Experiment 3: Inference Latency ===")
    latency_results = run_latency_benchmark(
        models, test_loader, device, cfg["latency_samples"]
    )

    # Assemble final JSON
    benchmark_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device":       str(device),
        "config": {
            "test_set_size":    test_set_size,
            "image_resolution": f"{cfg['img_width']}x{cfg['img_height']}",
            "batch_size":       cfg["batch_size"],
            "training_epochs":  cfg["training_epochs"],
            "n_qubits":         cfg["n_qubits"],
            "q_depth":          cfg["q_depth"],
            "class_names":      class_names,
            "noise_sigmas":     cfg["noise_sigmas"],
        },
        # Per-model: overall accuracy, per-class breakdown, averages, confusion matrix
        "clean_evaluation":     clean_results,
        # List of {sigma, CNN, QNN_CPU, QNN_GPU}
        "noise_robustness":     noise_results,
        # Average single-image inference latency in ms
        "inference_latency_ms": latency_results,
    }

    export_confusion_matrix_plots(clean_results, class_names, OUTPUT_DIR)
    # Save JSON
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    logger.info(f"JSON saved: {OUTPUT_PATH}")

    # Export CSVs
    logger.info("\n=== Exporting CSVs ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    export_per_class_metrics_csv(clean_results, class_names, OUTPUT_DIR)
    export_summary_metrics_csv(clean_results, OUTPUT_DIR)
    export_noise_robustness_csv(noise_results, OUTPUT_DIR)
    export_latency_csv(latency_results, OUTPUT_DIR)
    export_confusion_matrix_csvs(clean_results, class_names, OUTPUT_DIR)

    print(f"\nJSON: {OUTPUT_PATH}")
    print(f"CSVs: {OUTPUT_DIR}/")
    print("per_class_metrics.csv")
    print("summary_metrics.csv")
    print("noise_robustness.csv")
    print("inference_latency.csv")
    print("confusion_matrix_CNN.csv")
    print("confusion_matrix_QNN_CPU.csv")
    print("confusion_matrix_QNN_GPU.csv  (if CUDA available)")

    return benchmark_data

if __name__ == "__main__":
    # PyTorch device
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] PyTorch device: {torch_device}")
    if torch_device.type == "cuda":
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

    run_benchmark()