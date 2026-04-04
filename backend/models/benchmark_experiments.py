"""
benchmark_experiments.py
═══════════════════════════════════════════════════════════════════════════════
Full Comparative Experiment Suite
Hybrid Quantum-Classical Neural Network vs Classical CNN
Industrial Defect Detection — Graduation Project

Experiments run:
    1. Clean accuracy comparison        (Accuracy / Precision / Recall / F1)
    2. Confusion matrices               (per model)
    3. Noise robustness sweep           (Gaussian σ = 0.0 → 0.20)
    4. Inference latency benchmark      (single image + batch)
    5. Per-class accuracy comparison    (bar chart)
    6. Results export                   (CSV + JSON)

Usage:
    Run from backend/:
        python -m benchmark_experiments

    Flags (edit CONFIG below):
        CNN_PATH        — path to saved CNN weights
        QNN_CPU_PATH    — path to saved QNN CPU weights
        QNN_GPU_PATH    — path to saved QNN GPU (lightning.gpu) weights
                          set to None if not yet trained
        NOISE_SIGMAS    — list of Gaussian noise levels to sweep
        BATCH_SIZE      — dataloader batch size
        N_LATENCY_RUNS  — number of forward passes for latency averaging
"""

import os
import csv
import json
import time
import copy
import warnings
import argparse
from pathlib import Path
from typing import Optional

import torch
import matplotlib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")  # headless — no display needed
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from models.cnn_new import CnnNew
from data.data_loader import DataLoaderManager
from backend.models.qnn_cpu import HybridQnnCPU

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these paths before running
# ════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # Model checkpoint paths
    "CNN_PATH": "models/cnn_new.pth",
    "QNN_CPU_PATH": "models/qnn_cpu.pth",
    "QNN_GPU_PATH": None,  # set to path when lightning.gpu model is ready
    # Data
    "TEST_DIR": "data/test",
    "VAL_DIR": "data/val",
    "TRAIN_DIR": "data/train",
    "CLASS_NAMES_JSON": "data/class_names.json",
    "IMG_SIZE": 384,
    "BATCH_SIZE": 16,
    # Noise sweep
    "NOISE_SIGMAS": [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20],
    # Latency
    "N_LATENCY_RUNS": 50,  # forward passes to average over
    # Output
    "OUTPUT_DIR": "models/results",
    "PLOTS_DIR": "models/plots",
    # Quantum
    "N_QUBITS": 6,
    "Q_DEPTH": 2,
}

# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════


def setup_dirs():
    Path(CONFIG["OUTPUT_DIR"])
    Path(CONFIG["PLOTS_DIR"])


def load_class_names():
    with open(CONFIG["CLASS_NAMES_JSON"], "r", encoding="utf-8") as f:
        return json.load(f)


def add_gaussian_noise(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian noise and clamp to [0, 1]."""
    if sigma <= 0:
        return images
    return (images + torch.randn_like(images) * sigma).clamp(0.0, 1.0)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════════════════
# Model loading
# ════════════════════════════════════════════════════════════════════════════


def load_cnn(num_classes: int, device: torch.device) -> Optional[nn.Module]:
    path = CONFIG["CNN_PATH"]
    if not Path(path).exists():
        print(f"[WARN] CNN checkpoint not found at {path}. Skipping CNN.")
        return None
    model = CnnModule(num_classes=num_classes)
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[OK]   CNN loaded from {path}")
    return model


def load_qnn_cpu(num_classes: int, device: torch.device) -> Optional[nn.Module]:
    path = CONFIG["QNN_CPU_PATH"]
    if not Path(path).exists():
        print(f"[WARN] QNN-CPU checkpoint not found at {path}. Skipping QNN-CPU.")
        return None
    model = StrongHybridQNN(
        num_classes=num_classes,
        n_qubits=CONFIG["N_QUBITS"],
        q_depth=CONFIG["Q_DEPTH"],
        q_device_name="default.qubit",
        use_train_noise=False,
    )
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[OK]   QNN-CPU loaded from {path}")
    return model


def load_qnn_gpu(num_classes: int, device: torch.device) -> Optional[nn.Module]:
    path = CONFIG["QNN_GPU_PATH"]
    if path is None:
        print("[INFO] QNN-GPU path not set in CONFIG. Skipping QNN-GPU.")
        return None
    if not Path(path).exists():
        print(f"[WARN] QNN-GPU checkpoint not found at {path}. Skipping QNN-GPU.")
        return None
    model = StrongHybridQNN(
        num_classes=num_classes,
        n_qubits=CONFIG["N_QUBITS"],
        q_depth=CONFIG["Q_DEPTH"],
        q_device_name="lightning.gpu",
        use_train_noise=False,
    )
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[OK]   QNN-GPU loaded from {path}")
    return model


# ════════════════════════════════════════════════════════════════════════════
# Experiment 1 — Clean Accuracy + Full Classification Report
# ════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def run_clean_evaluation(
    model: nn.Module, loader, device: torch.device, class_names: list, model_name: str
) -> dict:
    """
    Full evaluation on clean test set.
    Returns accuracy, per-class precision/recall/F1, and raw predictions.
    """
    print(f"\n{'─'*60}")
    print(f"  Clean Evaluation: {model_name}")
    print(f"{'─'*60}")

    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in tqdm(
        loader, desc=f"[{model_name}] Clean eval", colour="cyan"
    ):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
    avg_loss = total_loss / len(loader)

    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    print(f"  Accuracy : {accuracy:.2f}%")
    print(f"  Avg Loss : {avg_loss:.4f}")
    print(f"\n{report_str}")

    return {
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "avg_loss": round(avg_loss, 4),
        "report_dict": report_dict,
        "report_str": report_str,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }


# ════════════════════════════════════════════════════════════════════════════
# Experiment 2 — Confusion Matrix
# ════════════════════════════════════════════════════════════════════════════


def plot_confusion_matrix(result: dict, class_names: list):
    model_name = result["model"]
    cm = confusion_matrix(result["all_labels"], result["all_preds"])
    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=30)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, pad=12)
    plt.tight_layout()
    path = os.path.join(
        CONFIG["PLOTS_DIR"], f"confusion_{model_name.replace(' ', '_')}.png"
    )
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] Confusion matrix → {path}")


# ════════════════════════════════════════════════════════════════════════════
# Experiment 3 — Noise Robustness Sweep
# ════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def run_noise_sweep(
    model: nn.Module, loader, device: torch.device, model_name: str
) -> list:
    """
    Evaluates model accuracy across Gaussian noise levels.
    Returns list of (sigma, accuracy) tuples.
    """
    print(f"\n  Noise sweep: {model_name}")
    rows = []
    for sigma in CONFIG["NOISE_SIGMAS"]:
        correct = 0
        total = 0
        for images, labels in loader:
            images = add_gaussian_noise(images, sigma).to(device)
            labels = labels.to(device)
            preds = model(images).argmax(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        acc = 100.0 * correct / total
        rows.append((sigma, round(acc, 4)))
        print(f"    σ={sigma:.2f}  →  Accuracy: {acc:.2f}%")
    return rows


def plot_noise_sweep(sweep_results: dict):
    """
    sweep_results: {model_name: [(sigma, acc), ...]}
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    markers = ["o", "s", "^"]
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    for (name, rows), marker, color in zip(sweep_results.items(), markers, colors):
        sigmas = [r[0] for r in rows]
        accs = [r[1] for r in rows]
        ax.plot(
            sigmas,
            accs,
            marker=marker,
            label=name,
            color=color,
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel("Gaussian Noise σ", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Noise Robustness — All Models", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    plt.tight_layout()
    path = os.path.join(CONFIG["PLOTS_DIR"], "noise_robustness.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  [Saved] Noise robustness plot → {path}")


# ════════════════════════════════════════════════════════════════════════════
# Experiment 4 — Inference Latency
# ════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def run_latency_benchmark(
    model: nn.Module, device: torch.device, model_name: str
) -> dict:
    """
    Measures single-image and batch (32) inference latency.
    Runs N_LATENCY_RUNS warm-up + timed passes.
    """
    print(f"\n  Latency benchmark: {model_name}")
    img_size = CONFIG["IMG_SIZE"]

    # ── Single image ─────────────────────────────────────────────────────
    single = torch.randn(1, 3, img_size, img_size).to(device)

    # Warm-up
    for _ in range(5):
        model(single)

    # Timed runs
    times = []
    for _ in range(CONFIG["N_LATENCY_RUNS"]):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(single)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    single_mean = round(float(np.mean(times)), 3)
    single_std = round(float(np.std(times)), 3)

    # ── Batch of 32 ──────────────────────────────────────────────────────
    batch = torch.randn(32, 3, img_size, img_size).to(device)

    for _ in range(3):
        model(batch)

    batch_times = []
    for _ in range(CONFIG["N_LATENCY_RUNS"] // 5):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_times.append((time.perf_counter() - t0) * 1000)

    batch_mean = round(float(np.mean(batch_times)), 3)
    batch_std = round(float(np.std(batch_times)), 3)

    print(f"    Single image : {single_mean:.1f} ± {single_std:.1f} ms")
    print(f"    Batch (32)   : {batch_mean:.1f}  ± {batch_std:.1f}  ms")

    return {
        "model": model_name,
        "single_mean_ms": single_mean,
        "single_std_ms": single_std,
        "batch32_mean_ms": batch_mean,
        "batch32_std_ms": batch_std,
    }


def plot_latency(latency_results: list):
    names = [r["model"] for r in latency_results]
    single_means = [r["single_mean_ms"] for r in latency_results]
    batch_means = [r["batch32_mean_ms"] for r in latency_results]
    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(
        x - w / 2,
        single_means,
        w,
        label="Single image (ms)",
        color="#2196F3",
        alpha=0.85,
    )
    bars2 = ax.bar(
        x + w / 2, batch_means, w, label="Batch 32 (ms)", color="#FF5722", alpha=0.85
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Inference Latency Comparison", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    path = os.path.join(CONFIG["PLOTS_DIR"], "inference_latency.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] Latency plot → {path}")


# ════════════════════════════════════════════════════════════════════════════
# Experiment 5 — Per-Class Accuracy Bar Chart
# ════════════════════════════════════════════════════════════════════════════


def plot_per_class_accuracy(clean_results: list, class_names: list):
    n_classes = len(class_names)
    n_models = len(clean_results)
    x = np.arange(n_classes)
    w = 0.8 / n_models
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, result in enumerate(clean_results):
        rd = result["report_dict"]
        accs = []
        for cls in class_names:
            recall = rd.get(cls, {}).get("recall", 0.0)
            accs.append(recall * 100)
        offset = (i - n_models / 2 + 0.5) * w
        bars = ax.bar(
            x + offset,
            accs,
            w,
            label=result["model"],
            color=colors[i % len(colors)],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Recall / Per-Class Accuracy (%)", fontsize=12)
    ax.set_title("Per-Class Accuracy Comparison — All Models", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(CONFIG["PLOTS_DIR"], "per_class_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] Per-class accuracy plot → {path}")


# ════════════════════════════════════════════════════════════════════════════
# Experiment 6 — Overall Accuracy Bar Chart
# ════════════════════════════════════════════════════════════════════════════


def plot_overall_accuracy(clean_results: list):
    names = [r["model"] for r in clean_results]
    accs = [r["accuracy"] for r in clean_results]
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, accs, color=colors[: len(names)], alpha=0.85, width=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Overall Accuracy Comparison", fontsize=13)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(CONFIG["PLOTS_DIR"], "overall_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] Overall accuracy plot → {path}")


# ════════════════════════════════════════════════════════════════════════════
# Export — CSV + JSON
# ════════════════════════════════════════════════════════════════════════════


def export_results(
    clean_results: list, noise_results: dict, latency_results: list, class_names: list
):

    # ── 1. Overall accuracy CSV ───────────────────────────────────────────
    acc_path = os.path.join(CONFIG["OUTPUT_DIR"], "accuracy_comparison.csv")
    with open(acc_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Model", "Accuracy (%)", "Avg Loss", "Macro F1", "Weighted F1"]
        )
        for r in clean_results:
            rd = r["report_dict"]
            writer.writerow(
                [
                    r["model"],
                    r["accuracy"],
                    r["avg_loss"],
                    round(rd["macro avg"]["f1-score"] * 100, 2),
                    round(rd["weighted avg"]["f1-score"] * 100, 2),
                ]
            )
    print(f"  [Saved] Accuracy CSV → {acc_path}")

    # ── 2. Per-class metrics CSV ──────────────────────────────────────────
    cls_path = os.path.join(CONFIG["OUTPUT_DIR"], "per_class_metrics.csv")
    with open(cls_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Class", "Precision (%)", "Recall (%)", "F1 (%)"])
        for r in clean_results:
            rd = r["report_dict"]
            for cls in class_names:
                metrics = rd.get(cls, {})
                writer.writerow(
                    [
                        r["model"],
                        cls,
                        round(metrics.get("precision", 0) * 100, 2),
                        round(metrics.get("recall", 0) * 100, 2),
                        round(metrics.get("f1-score", 0) * 100, 2),
                    ]
                )
    print(f"  [Saved] Per-class CSV → {cls_path}")

    # ── 3. Noise robustness CSV ───────────────────────────────────────────
    noise_path = os.path.join(CONFIG["OUTPUT_DIR"], "noise_robustness.csv")
    with open(noise_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Noise Sigma"] + list(noise_results.keys())
        writer.writerow(header)
        # Assumes all models have same sigma list
        first = next(iter(noise_results.values()))
        for idx, (sigma, _) in enumerate(first):
            row = [sigma] + [noise_results[m][idx][1] for m in noise_results]
            writer.writerow(row)
    print(f"  [Saved] Noise robustness CSV → {noise_path}")

    # ── 4. Latency CSV ────────────────────────────────────────────────────
    lat_path = os.path.join(CONFIG["OUTPUT_DIR"], "latency_benchmark.csv")
    with open(lat_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Single Image Mean (ms)",
                "Single Image Std (ms)",
                "Batch-32 Mean (ms)",
                "Batch-32 Std (ms)",
            ]
        )
        for r in latency_results:
            writer.writerow(
                [
                    r["model"],
                    r["single_mean_ms"],
                    r["single_std_ms"],
                    r["batch32_mean_ms"],
                    r["batch32_std_ms"],
                ]
            )
    print(f"  [Saved] Latency CSV → {lat_path}")

    # ── 5. Full JSON dump ─────────────────────────────────────────────────
    json_payload = {
        "clean_accuracy": [
            {
                "model": r["model"],
                "accuracy": r["accuracy"],
                "avg_loss": r["avg_loss"],
                "report": {
                    k: v for k, v in r["report_dict"].items() if k not in ("accuracy",)
                },
            }
            for r in clean_results
        ],
        "noise_robustness": {
            model: [{"sigma": s, "accuracy": a} for s, a in rows]
            for model, rows in noise_results.items()
        },
        "latency": latency_results,
    }
    json_path = os.path.join(CONFIG["OUTPUT_DIR"], "full_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    print(f"  [Saved] Full JSON → {json_path}")


# ════════════════════════════════════════════════════════════════════════════
# Summary Table (printed to console)
# ════════════════════════════════════════════════════════════════════════════


def print_summary(clean_results: list, noise_results: dict, latency_results: list):
    sep = "═" * 80
    print(f"\n{sep}")
    print("  EXPERIMENT SUMMARY")
    print(sep)

    # Accuracy
    print("\n  ┌─ Clean Test Accuracy ─────────────────────────────────────┐")
    for r in clean_results:
        rd = r["report_dict"]
        macro_f1 = round(rd["macro avg"]["f1-score"] * 100, 2)
        print(
            f"  │  {r['model']:<20} Acc: {r['accuracy']:>6.2f}%   Macro-F1: {macro_f1:>6.2f}%  │"
        )
    print("  └───────────────────────────────────────────────────────────┘")

    # Noise drop at highest sigma
    if noise_results:
        max_sigma = CONFIG["NOISE_SIGMAS"][-1]
        print(f"\n  ┌─ Accuracy Drop at σ={max_sigma} ─────────────────────────────┐")
        for model, rows in noise_results.items():
            clean_acc = rows[0][1]  # sigma=0.0
            noisy_acc = rows[-1][1]  # sigma=max
            drop = clean_acc - noisy_acc
            print(
                f"  │  {model:<20}  Clean: {clean_acc:>6.2f}%   Noisy: {noisy_acc:>6.2f}%   Drop: {drop:>5.2f}%  │"
            )
        print("  └───────────────────────────────────────────────────────────┘")

    # Latency
    if latency_results:
        print("\n  ┌─ Inference Latency ────────────────────────────────────────┐")
        for r in latency_results:
            print(
                f"  │  {r['model']:<20}  Single: {r['single_mean_ms']:>7.1f} ms   "
                f"Batch-32: {r['batch32_mean_ms']:>7.1f} ms  │"
            )
        print("  └───────────────────────────────────────────────────────────┘")

    print(f"\n{sep}\n")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═" * 80)
    print("  HYBRID QNN vs CNN — FULL BENCHMARK EXPERIMENT SUITE")
    print("═" * 80 + "\n")

    setup_dirs()
    device = get_device()
    class_names = load_class_names()
    num_classes = len(class_names)
    print(f"  Device      : {device}")
    print(f"  Classes     : {num_classes} → {class_names}")
    print(f"  Image size  : {CONFIG['IMG_SIZE']}×{CONFIG['IMG_SIZE']}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────
    manager = DataLoaderManager(
        train_dir=CONFIG["TRAIN_DIR"],
        val_dir=CONFIG["VAL_DIR"],
        test_dir=CONFIG["TEST_DIR"],
        img_width=CONFIG["IMG_SIZE"],
        img_height=CONFIG["IMG_SIZE"],
        batch_size=CONFIG["BATCH_SIZE"],
    )
    _, _, test_loader = manager.get_loaders()

    # ── Load models ───────────────────────────────────────────────────────
    print("Loading models...")
    models = {}

    cnn = load_cnn(num_classes, device)
    if cnn:
        models["Classical CNN"] = cnn

    qnn_cpu = load_qnn_cpu(num_classes, device)
    if qnn_cpu:
        models["Hybrid QNN (CPU)"] = qnn_cpu

    qnn_gpu = load_qnn_gpu(num_classes, device)
    if qnn_gpu:
        models["Hybrid QNN (GPU)"] = qnn_gpu

    if not models:
        print("[ERROR] No models loaded. Check checkpoint paths in CONFIG.")

    print(f"\n  Models ready: {list(models.keys())}\n")

    # ════════════════════════════════════════════════════════════════════
    # Experiment 1 + 2 — Clean accuracy + confusion matrices
    # ════════════════════════════════════════════════════════════════════
    print("═" * 60)
    print("  EXPERIMENT 1 — Clean Accuracy & Classification Report")
    print("═" * 60)

    clean_results = []
    for name, model in models.items():
        result = run_clean_evaluation(model, test_loader, device, class_names, name)
        clean_results.append(result)
        plot_confusion_matrix(result, class_names)

    # ════════════════════════════════════════════════════════════════════
    # Experiment 3 — Noise robustness sweep
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  EXPERIMENT 2 — Noise Robustness Sweep")
    print(f"  Sigma levels: {CONFIG['NOISE_SIGMAS']}")
    print("═" * 60)

    noise_results = {}
    for name, model in models.items():
        rows = run_noise_sweep(model, test_loader, device, name)
        noise_results[name] = rows

    plot_noise_sweep(noise_results)

    # ════════════════════════════════════════════════════════════════════
    # Experiment 4 — Inference latency
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  EXPERIMENT 3 — Inference Latency Benchmark")
    print(
        f"  Runs per model: {CONFIG['N_LATENCY_RUNS']} (single) / "
        f"{CONFIG['N_LATENCY_RUNS']//5} (batch)"
    )
    print("═" * 60)

    latency_results = []
    for name, model in models.items():
        lat = run_latency_benchmark(model, device, name)
        latency_results.append(lat)

    plot_latency(latency_results)

    # ════════════════════════════════════════════════════════════════════
    # Experiment 5 — Per-class accuracy chart
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  EXPERIMENT 4 — Per-Class Accuracy Comparison")
    print("═" * 60)
    plot_per_class_accuracy(clean_results, class_names)
    plot_overall_accuracy(clean_results)

    # ════════════════════════════════════════════════════════════════════
    # Export
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  EXPORTING RESULTS")
    print("═" * 60)
    export_results(clean_results, noise_results, latency_results, class_names)

    # ════════════════════════════════════════════════════════════════════
    # Final summary
    # ════════════════════════════════════════════════════════════════════
    print_summary(clean_results, noise_results, latency_results)

    print(f"  All plots  → {CONFIG['PLOTS_DIR']}/")
    print(f"  All data   → {CONFIG['OUTPUT_DIR']}/")
    print("\n  Done.\n")
