"""
Benchmark — Quantum-Hybrid-Defect-Detector
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
    5. Noise robustness sweep — accuracy at N levels across 7 noise types:
           gaussian, blur, contrast, salt_pepper,
           motion_blur, jpeg_compression, lens_occlusion
       Includes per-class accuracy and Mean Accuracy Under Noise (MAUN)
    6. Average single-image inference latency (all 3 models)
    7. Config metadata — test set size, resolution, epochs, qubits, batch size

CSV exports:
    per_class_metrics.csv
    summary_metrics.csv
    noise_robustness.csv             — overall accuracy per (noise_type, level, model)
    noise_robustness_per_class.csv   — per-class accuracy under each noise condition
    noise_maun_summary.csv           — Mean Accuracy Under Noise per (model, noise_type)
    inference_latency.csv
    confusion_matrix_<model>.csv     — one file per model

Changes vs. original:
    - Replaced 4-type Gaussian-only noise sweep with the full 7-type suite from
      noise_robustness_test.py (shared via backend.utils.noise).
    - Fixed silent bug: _accuracy_with_noise previously ignored noise_type and
      only applied Gaussian; run_noise_sweep was passing noise_type but the
      function signature did not accept it.
    - Added per-class accuracy collection inside the noise sweep.
    - Added compute_maun() — Mean Accuracy Under Noise degradation summary.
    - Added export_noise_per_class_csv() and export_maun_csv().
    - Fixed JSON key: was cfg["noise_sigmas"] (KeyError), now cfg["noise_levels"].
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
from sklearn.metrics import (classification_report, confusion_matrix, precision_recall_fscore_support,)
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.data.preprocessing import PreProcessing
from backend.data.data_loader import DataLoaderManager
from backend.models.cnn import CNN
from backend.models.qnn_cpu import HybridQnnCPU
from backend.models.qnn_gpu import HybridQnnGPU
from backend.models.noise import apply_noise, BENCHMARK_NOISE_LEVELS
from backend.utils.logger import Logger

logger = Logger()

# Configuration
CONFIG = {
    "img_width":       384,
    "img_height":      384,
    "batch_size":      16,
    "training_epochs": 50,
    "n_qubits":        6,
    "q_depth":         2,
    "noise_levels":    BENCHMARK_NOISE_LEVELS,
    "latency_samples": 50, # Number of single images used to measure average inference latency
}

_current_dir  = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT  = os.path.dirname(_current_dir)

CHECKPOINT_PATHS = {
    "CNN":     os.path.join(BACKEND_ROOT, "models", "cnn.pth"),
    "QNN_CPU": os.path.join(BACKEND_ROOT, "models", "qnn_cpu.pth"),
    "QNN_GPU": os.path.join(BACKEND_ROOT, "models", "qnn_gpu.pth"),
}

CLASS_NAMES_PATH = os.path.join(BACKEND_ROOT, "data", "class_names.json")
OUTPUT_PATH      = os.path.join(BACKEND_ROOT, "data", "benchmark", "benchmark_results.json")
OUTPUT_DIR       = os.path.join(BACKEND_ROOT, "data", "benchmark")

DATA_DIRS = {
    "train": os.path.join(BACKEND_ROOT, "data", "train"),
    "val":   os.path.join(BACKEND_ROOT, "data", "val"),
    "test":  os.path.join(BACKEND_ROOT, "data", "test"),
}

ALL_MODEL_KEYS = ["CNN", "QNN_CPU", "QNN_GPU"]


#  Model loading 
def load_models(
    num_classes: int,
    device: torch.device,
) -> dict[str, torch.nn.Module]:
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

    # QNN_GPU — optional
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


#  Experiment 1 — Clean accuracy + per-class metrics + confusion matrix ─
@torch.no_grad()
def run_clean_evaluation(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    model_label: str,
) -> dict:
    """
    Full-pass evaluation on the clean test set.
    Returns overall accuracy, per-class P/R/F1/acc, macro & weighted
    averages, and the confusion matrix.
    """
    model.eval()
    all_preds:  list[int] = []
    all_labels: list[int] = []

    for images, labels in tqdm(test_loader, desc=f"[Clean Eval] {model_label}", colour="cyan"):
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    n_classes    = len(class_names)
    labels_range = list(range(n_classes))

    correct  = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = round(100.0 * correct / len(all_labels), 4)

    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        all_labels, all_preds, labels=labels_range, zero_division=0
    )

    cm            = confusion_matrix(all_labels, all_preds, labels=labels_range)
    class_totals  = cm.sum(axis=1)
    class_correct = cm.diagonal()
    per_class_acc = class_correct / class_totals.clip(min=1)

    per_class: dict[str, dict] = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "accuracy":  round(float(per_class_acc[i]) * 100, 2),
            "precision": round(float(precision_arr[i]) * 100, 2),
            "recall":    round(float(recall_arr[i])    * 100, 2),
            "f1":        round(float(f1_arr[i])        * 100, 2),
            "support":   int(support_arr[i]),
        }

    p_macro, r_macro, f1_macro, _       = precision_recall_fscore_support(
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

    report_str = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )
    logger.info(f"\n[{model_label}] Clean Evaluation Report:\n{report_str}")

    return {
        "accuracy":         accuracy,
        "per_class":        per_class,
        "averages":         averages,
        "confusion_matrix": cm.tolist(),
        "n_samples":        len(all_labels),
    }


#  Experiment 2 — Noise robustness sweep (7 types) ─
@torch.no_grad()
def _evaluate_with_noise(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    noise_type: str,
    level: float,
    num_classes: int,
) -> dict:
    """
    Single-pass accuracy under a specific (noise_type, level) combination.

    Returns:
        overall_acc   : float — top-1 accuracy across all samples (%)
        per_class_acc : dict[int, float] — per-class accuracy (%)
    """
    model.eval()
    correct = 0
    total = 0
    conf_sum = 0.0
    class_correct: dict[int, int] = {}
    class_total:   dict[int, int] = {}

    for images, labels in loader:
        # Cast to float32 BEFORE noise injection — QNN layers dislike fp16
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device)

        noisy = apply_noise(images, noise_type, level)
        probs = torch.softmax(model(noisy), dim=1)
        conf, preds = probs.max(dim=1)

        total   += labels.size(0)
        correct += preds.eq(labels).sum().item()
        conf_sum += conf.sum().item()

        for lbl, p in zip(labels.tolist(), preds.tolist()):
            class_total[lbl]   = class_total.get(lbl, 0) + 1
            if lbl == p:
                class_correct[lbl] = class_correct.get(lbl, 0) + 1

    overall_acc = round(100.0 * correct / total, 4) if total else 0.0
    mean_conf   = round(100.0 * conf_sum / total, 4) if total else 0.0
    per_class   = {
        c: round(100.0 * class_correct.get(c, 0) / class_total[c], 2)
        for c in range(num_classes)
        if c in class_total
    }
    return {"overall_acc": overall_acc, "per_class_acc": per_class, "mean_conf": mean_conf}


def run_noise_sweep(
    models: dict[str, torch.nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    noise_levels: dict[str, list],
    num_classes: int,
) -> dict[str, list[dict]]:
    """
    Runs the robustness sweep across all seven noise types and their levels.

    Returns a dict keyed by noise_type. Each value is a list of row dicts:
        [
          {
            "level":        float,
            "CNN":          float | None,   # overall accuracy (%)
            "QNN_CPU":      float | None,
            "QNN_GPU":      float | None,
            "per_class": {
              "CNN":     {class_idx: acc, ...},
              "QNN_CPU": {class_idx: acc, ...},
              "QNN_GPU": {class_idx: acc, ...} | None,
            }
          },
          ...
        ]

    contrast levels descend from 1.0 (clean) — the opposite axis direction to
    the other noise types — which is preserved for frontend rendering.
    """
    results: dict[str, list[dict]] = {}

    for noise_type, levels in noise_levels.items():
        logger.info(f"\n  -- Noise type: {noise_type} ({len(levels)} levels) --")
        rows: list[dict] = []

        for level in tqdm(levels, desc=f"  {noise_type}", leave=False):
            row: dict             = {"level": level}
            per_class_data: dict  = {}

            for label, model in models.items():
                metrics = _evaluate_with_noise(model, test_loader, device, noise_type, level, num_classes)
                
                row[label] = metrics["overall_acc"]
                row[f"{label}_mean_conf"] = metrics["mean_conf"]
                per_class_data[label] = metrics["per_class_acc"]
                logger.info(f" {noise_type} level={level:.4g} | {label} → " f"{metrics['overall_acc']:.2f}%")

            # Ensure every model key is present (QNN_GPU may be absent on CPU)
            for key in ALL_MODEL_KEYS:
                row.setdefault(key, None)
                row.setdefault(f"{key}_mean_conf", None)
                per_class_data.setdefault(key, None)

            row["per_class"] = per_class_data
            rows.append(row)

        results[noise_type] = rows

    return results


def compute_maun(
    noise_results: dict[str, list[dict]],
) -> dict[str, dict[str, Optional[float]]]:
    """
    Mean Accuracy Under Noise (MAUN).

    Averages each model's overall accuracy across ALL levels for each noise
    type. Lower MAUN = model degrades faster under that noise type.

    Returns: {model_key: {noise_type: maun_score}}
    """
    maun: dict[str, dict[str, Optional[float]]] = {k: {} for k in ALL_MODEL_KEYS}

    for noise_type, rows in noise_results.items():
        for key in ALL_MODEL_KEYS:
            accs = [r[key] for r in rows if r.get(key) is not None]
            maun[key][noise_type] = (
                round(sum(accs) / len(accs), 4) if accs else None
            )

    return maun


#  Experiment 3 — Inference latency 
def run_latency_benchmark(
    models: dict[str, torch.nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    n_samples: int,
) -> dict[str, Optional[float]]:
    """
    Measures average single-image inference latency in milliseconds.
    Runs n_samples images one at a time (batch=1). First WARMUP samples
    are discarded.
    """
    WARMUP = 5
    results: dict[str, Optional[float]] = {}

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


#  Confusion matrix plots
def export_confusion_matrix_plots(
    clean_results: dict[str, dict],
    class_names: list[str],
    output_dir: str,
) -> None:
    for model_key, result in clean_results.items():
        cm = result.get("confusion_matrix")
        if cm is None:
            continue
        cm_array = np.array(cm)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_array,
            annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
        )
        plt.title(f"Confusion Matrix — {model_key}", fontsize=16, pad=15)
        plt.ylabel("True Class", fontsize=12)
        plt.xlabel("Predicted Class", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        path = os.path.join(output_dir, f"confusion_matrix_{model_key}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved: {path}")


#  CSV exports 
def _na(value) -> str:
    return str(value) if value is not None else "N/A"

def export_per_class_metrics_csv(
    clean_results: dict[str, dict],
    class_names: list[str],
    output_dir: str,
) -> None:
    """One row per (model, class) — clean-test metrics."""
    path = os.path.join(output_dir, "per_class_metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "class", "accuracy_%", "precision_%",
                         "recall_%", "f1_%", "support"])
        for model_key, result in clean_results.items():
            for cls in class_names:
                m = result["per_class"].get(cls, {})
                writer.writerow([
                    model_key, cls,
                    _na(m.get("accuracy")), _na(m.get("precision")),
                    _na(m.get("recall")),   _na(m.get("f1")),
                    _na(m.get("support")),
                ])
    logger.info(f"CSV saved: {path}")


def export_summary_metrics_csv(
    clean_results: dict[str, dict],
    output_dir: str,
) -> None:
    """One row per model — overall accuracy and macro/weighted averages."""
    path = os.path.join(output_dir, "summary_metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "overall_accuracy_%",
            "macro_precision_%", "macro_recall_%",    "macro_f1_%",
            "weighted_precision_%", "weighted_recall_%", "weighted_f1_%",
            "n_samples",
        ])
        for model_key, result in clean_results.items():
            avg      = result.get("averages", {})
            macro    = avg.get("macro",    {})
            weighted = avg.get("weighted", {})
            writer.writerow([
                model_key,
                _na(result.get("accuracy")),
                _na(macro.get("precision")),    _na(macro.get("recall")),
                _na(macro.get("f1")),
                _na(weighted.get("precision")), _na(weighted.get("recall")),
                _na(weighted.get("f1")),
                _na(result.get("n_samples")),
            ])
    logger.info(f"CSV saved: {path}")


def export_noise_robustness_csv(
    noise_results: dict[str, list[dict]],
    output_dir: str,
) -> None:
    """
    Overall accuracy per (noise_type, level, model).
    One row per (noise_type, level).
    """
    path = os.path.join(output_dir, "noise_robustness.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "noise_type", "level",
            "CNN_accuracy_%", "QNN_CPU_accuracy_%", "QNN_GPU_accuracy_%",
            "CNN_mean_conf_%",    "QNN_CPU_mean_conf_%",    "QNN_GPU_mean_conf_%",
        ])
        for noise_type, rows in noise_results.items():
            for row in rows:
                writer.writerow([
                    noise_type, row["level"],
                    _na(row.get("CNN")),
                    _na(row.get("QNN_CPU")),
                    _na(row.get("QNN_GPU")),
                    _na(row.get("CNN_mean_conf")),      
                    _na(row.get("QNN_CPU_mean_conf")),  
                    _na(row.get("QNN_GPU_mean_conf")),  
                ])
    logger.info(f"CSV saved: {path}")


def export_noise_per_class_csv(
    noise_results: dict[str, list[dict]],
    class_names: list[str],
    output_dir: str,
) -> None:
    """
    Per-class accuracy under each noise condition.
    Columns: noise_type, level, model, <class_0>, ..., <class_N>
    """
    path = os.path.join(output_dir, "noise_robustness_per_class.csv")
    class_headers = [f"acc_{name}" for name in class_names]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["noise_type", "level", "model"] + class_headers)

        for noise_type, rows in noise_results.items():
            for row in rows:
                per_class_data = row.get("per_class", {})
                for model_key in ALL_MODEL_KEYS:
                    class_accs = per_class_data.get(model_key)
                    if class_accs is None:
                        col_vals = ["N/A"] * len(class_names)
                    else:
                        col_vals = [
                            _na(class_accs.get(i)) for i in range(len(class_names))
                        ]
                    writer.writerow(
                        [noise_type, row["level"], model_key] + col_vals
                    )
    logger.info(f"CSV saved: {path}")


def export_maun_csv(
    maun: dict[str, dict[str, Optional[float]]],
    output_dir: str,
) -> None:
    """
    Mean Accuracy Under Noise summary.
    Columns: model, <noise_type_0>, ..., <noise_type_N>, overall_maun
    """
    path = os.path.join(output_dir, "noise_maun_summary.csv")
    # Collect all noise type names from any model entry that has data
    noise_types: list[str] = []
    for model_data in maun.values():
        for nt in model_data:
            if nt not in noise_types:
                noise_types.append(nt)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + noise_types + ["overall_maun"])

        for model_key in ALL_MODEL_KEYS:
            row_vals = [_na(maun.get(model_key, {}).get(nt)) for nt in noise_types]
            scores   = [v for v in maun.get(model_key, {}).values() if v is not None]
            overall  = round(sum(scores) / len(scores), 4) if scores else None
            writer.writerow([model_key] + row_vals + [_na(overall)])

    logger.info(f"CSV saved: {path}")


def export_latency_csv(
    latency_results: dict[str, Optional[float]],
    output_dir: str,
) -> None:
    path = os.path.join(output_dir, "inference_latency.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "avg_latency_ms"])
        for model_key in ALL_MODEL_KEYS:
            writer.writerow([model_key, _na(latency_results.get(model_key))])
    logger.info(f"CSV saved: {path}")


def export_confusion_matrix_csvs(
    clean_results: dict[str, dict],
    class_names: list[str],
    output_dir: str,
) -> None:
    for model_key, result in clean_results.items():
        cm = result.get("confusion_matrix")
        if cm is None:
            continue
        path = os.path.join(output_dir, f"confusion_matrix_{model_key}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true \\ predicted"] + class_names)
            for true_idx, row in enumerate(cm):
                writer.writerow([class_names[true_idx]] + row)
        logger.info(f"CSV saved: {path}")


#  Main
def run_benchmark() -> dict:
    cfg    = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running benchmark on device: {device}")

    # Load class names
    if not os.path.exists(CLASS_NAMES_PATH):
        raise FileNotFoundError(f"class_names.json not found: {CLASS_NAMES_PATH}")
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names: list[str] = json.load(f)
    num_classes = len(class_names)
    logger.info(f"Classes ({num_classes}): {class_names}")

    # Data loaders
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

    # Experiment 2: Noise robustness sweep (7 types)
    logger.info("\n=== Experiment 2: Noise Robustness Sweep (7 types) ===")
    noise_results = run_noise_sweep(
        models, test_loader, device,
        cfg["noise_levels"],
        num_classes,
    )

    # MAUN degradation summary
    maun = compute_maun(noise_results)
    logger.info("\n--- MAUN Summary ---")
    for model_key, nt_scores in maun.items():
        scores = [v for v in nt_scores.values() if v is not None]
        overall = round(sum(scores) / len(scores), 2) if scores else None
        logger.info(f"  {model_key}: overall MAUN = {overall}%  |  {nt_scores}")

    # Experiment 3: Inference latency
    logger.info("\n=== Experiment 3: Inference Latency ===")
    latency_results = run_latency_benchmark(
        models, test_loader, device, cfg["latency_samples"]
    )

    # Strip non-JSON-serialisable per_class nested dict for top-level noise entry.
    # We store it separately — keep the JSON lean for the FastAPI cache.
    noise_results_slim = {
        noise_type: [
            {k: v for k, v in row.items() if k != "per_class"}
            for row in rows
        ]
        for noise_type, rows in noise_results.items()
    }

    # Assemble final JSON
    # FIX: was cfg["noise_sigmas"] (KeyError) — corrected to cfg["noise_levels"]
    noise_level_config = {nt: lvls for nt, lvls in cfg["noise_levels"].items()}

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
            "noise_levels":     noise_level_config,  # ← was noise_sigmas (bug)
        },
        "clean_evaluation":     clean_results,
        "noise_robustness":     noise_results_slim,  # overall accuracy only
        "noise_maun_summary":   maun,
        "inference_latency_ms": latency_results,
    }

    # Save plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    export_confusion_matrix_plots(clean_results, class_names, OUTPUT_DIR)

    # Save JSON
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    logger.info(f"JSON saved: {OUTPUT_PATH}")

    # Export CSVs
    logger.info("\n=== Exporting CSVs ===")
    export_per_class_metrics_csv(clean_results, class_names, OUTPUT_DIR)
    export_summary_metrics_csv(clean_results, OUTPUT_DIR)
    export_noise_robustness_csv(noise_results, OUTPUT_DIR)
    export_noise_per_class_csv(noise_results, class_names, OUTPUT_DIR)  
    export_maun_csv(maun, OUTPUT_DIR)                                    
    export_latency_csv(latency_results, OUTPUT_DIR)
    export_confusion_matrix_csvs(clean_results, class_names, OUTPUT_DIR)

    print(f"\nJSON:  {OUTPUT_PATH}")
    print(f"CSVs:  {OUTPUT_DIR}/")
    for name in [
        "per_class_metrics.csv", "summary_metrics.csv",
        "noise_robustness.csv", "noise_robustness_per_class.csv",
        "noise_maun_summary.csv", "inference_latency.csv",
        "confusion_matrix_CNN.csv", "confusion_matrix_QNN_CPU.csv",
        "confusion_matrix_QNN_GPU.csv  (if CUDA available)",
    ]:
        print(f"       {name}")

    return benchmark_data

if __name__ == "__main__":
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] PyTorch device: {torch_device}")
    if torch_device.type == "cuda":
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

    run_benchmark()