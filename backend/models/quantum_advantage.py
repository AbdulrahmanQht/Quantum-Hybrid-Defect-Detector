"""
Quantum Advantage Runner — Hybrid QNN Defect Detector
------------------------------------------------------------------------
Run AFTER benchmark_runner.py. This script proves the quantum mechanisms
in HybridQnnCPU and HybridQnnGPU are real, active, and contributing.

Answers the question: "Is this actually quantum?"

What this computes:
    1. Feature Orthogonality      — cosine similarity between classical (z) and
                                    quantum (q_emb) branch features. Near 0 = the
                                    quantum branch learns genuinely different things.

    2. Branch Ablation            — zeroes out each branch independently to isolate
                                    the quantum branch's direct accuracy contribution.

    3. Re-upload Ablation         — disables data re-uploading at inference time.
                                    The accuracy drop quantifies the expressive power
                                    added by re-uploading (Pérez-Salinas et al.).

    4. Entanglement Entropy       — extracts the full statevector (via a mirrored
                                    default.qubit circuit) and computes von Neumann
                                    entropy for each qubit bipartition.
                                    NOTE: lightning.gpu does not expose statevectors,
                                    so GPU model weights are loaded into a temporary
                                    default.qubit device for this experiment only.

    5. Quantum Gradient Variance  — measures gradient variance of the trained quantum
                                    weights across test batches. Near-zero = barren
                                    plateau. High = healthy quantum signal.

    6. Noise Robustness Ablation  — NEW: runs full / classical_only / quantum_only
                                    forward passes under 5 noise types at multiple
                                    intensity levels.
                                    Key metric:
                                        quantum_noise_gain[level] =
                                            full_acc[level] − classical_only_acc[level]
                                    An increasing gain as noise intensifies proves the
                                    quantum branch specifically improves noise tolerance.

Outputs:
    data/QA/quantum_advantage_results.json
    data/QA/qa_feature_orthogonality.csv
    data/QA/qa_branch_ablation.csv
    data/QA/qa_reupload_ablation.csv
    data/QA/qa_entanglement_entropy.csv
    data/QA/qa_gradient_variance.csv
    data/QA/qa_noise_ablation.csv              ← new (Experiment 6)
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from torch.utils.data import DataLoader
from tqdm import tqdm

from backend.models.cnn import CNN
from backend.models.qnn_cpu import HybridQnnCPU
from backend.models.qnn_gpu import HybridQnnGPU
from backend.data.data_loader import DataLoaderManager
from backend.utils.logger import Logger
from backend.utils.noise import apply_noise, QA_NOISE_LEVELS    # ← shared util

logger = Logger()

# ── Config ─────────────────────────────────────────────────────────────────────

CONFIG = {
    "img_width":        384,
    "img_height":       384,
    "batch_size":       16,
    "n_qubits":         6,
    "q_depth":          2,
    "grad_var_batches": 20,
    "entropy_samples":  128,
}

_current_dir  = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT  = os.path.dirname(_current_dir)

CHECKPOINT_PATHS = {
    "CNN":     os.path.join(BACKEND_ROOT, "models", "cnn.pth"),
    "QNN_CPU": os.path.join(BACKEND_ROOT, "models", "qnn_cpu.pth"),
    "QNN_GPU": os.path.join(BACKEND_ROOT, "models", "qnn_gpu_6_qubits.pth"),
}
CLASS_NAMES_PATH = os.path.join(BACKEND_ROOT, "data", "class_names.json")
OUTPUT_PATH      = os.path.join(BACKEND_ROOT, "data", "QA", "quantum_advantage_results.json")
OUTPUT_DIR       = os.path.join(BACKEND_ROOT, "data", "QA")

DATA_DIRS = {
    "train": os.path.join(BACKEND_ROOT, "data", "train"),
    "val":   os.path.join(BACKEND_ROOT, "data", "val"),
    "test":  os.path.join(BACKEND_ROOT, "data", "test"),
}

QNN_MODEL_KEYS = ["QNN_CPU", "QNN_GPU"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _na(value) -> str:
    return str(value) if value is not None else "N/A"


@torch.no_grad()
def _evaluate(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Generic accuracy evaluation given an arbitrary forward function."""
    correct = 0
    total   = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds   = forward_fn(images).argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
    return round(100.0 * correct / total, 4)


@torch.no_grad()
def _evaluate_noisy(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    noise_type: str,
    level: float,
) -> float:
    """
    Accuracy evaluation with noise applied to every batch before forwarding.
    forward_fn may be any of: full model, classical_only, quantum_only.
    """
    correct = 0
    total   = 0
    for images, labels in loader:
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device)
        noisy  = apply_noise(images, noise_type, level)
        preds  = forward_fn(noisy).argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
    return round(100.0 * correct / total, 4)


def _extract_classical_and_quantum(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run backbone + quantum branch and return:
        z      — 512-D classical pooled features
        q_emb  — 128-D post-quantum projected features
    Works for both HybridQnnCPU and HybridQnnGPU.
    """
    x = images.to(device)
    dtype = next(model.backbone.parameters()).dtype
    x = x.to(dtype=dtype)
    h = model.backbone(x)
    z = F.adaptive_avg_pool2d(h, (1, 1)).flatten(1)

    q_in = model.pre_quantum(z)

    if isinstance(model, HybridQnnCPU):
        with torch.amp.autocast(device_type=device.type, enabled=False):
            q_raw = model.q_layer(q_in.to("cpu").float()).to(device)
    else:
        with torch.amp.autocast(device_type=device.type, enabled=False):
            q_raw = model.q_layer(q_in.float())

    q_emb = model.post_quantum(q_raw)
    return z, q_emb


# ── Experiment 1 — Feature Orthogonality ──────────────────────────────────────

@torch.no_grad()
def run_feature_orthogonality(
    models: dict[str, nn.Module],
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, Optional[float]]:
    """
    Cosine similarity between classical (z[:128]) and quantum (q_emb) features.
    ~0 = orthogonal (ideal);  ~1 = quantum mirrors classical (no advantage).
    """
    results: dict[str, Optional[float]] = {}

    for label, model in models.items():
        model.eval()
        sims: list[float] = []

        for images, _ in tqdm(test_loader, desc=f"[Orthogonality] {label}", colour="cyan"):
            z, q_emb = _extract_classical_and_quantum(model, images, device)
            z_proj   = z[:, : q_emb.size(1)]
            z_norm   = F.normalize(z_proj.float(), dim=1)
            q_norm   = F.normalize(q_emb.float(),  dim=1)
            sims.append((z_norm * q_norm).sum(dim=1).mean().item())

        avg_sim       = round(float(np.mean(sims)), 6)
        results[label] = avg_sim
        logger.info(f"  [{label}] Classical-Quantum cosine similarity: {avg_sim:.4f}")

    return results


# ── Experiment 2 — Branch Ablation ────────────────────────────────────────────

@torch.no_grad()
def run_branch_ablation(
    models: dict[str, nn.Module],
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, Optional[float]]]:
    """
    Three conditions: full / classical_only (q=0) / quantum_only (z=0).
    quantum_gain = full_acc − classical_only_acc.
    """
    results: dict[str, dict] = {}

    for label, model in models.items():
        model.eval()

        full_acc     = _evaluate(model, test_loader, device)
        q_dim        = model.post_quantum[-2].normalized_shape[0]

        def classical_only(x: torch.Tensor) -> torch.Tensor:
            z, _ = _extract_classical_and_quantum(model, x, device)
            q_blank = torch.zeros(x.size(0), q_dim, device=device, dtype=z.dtype)
            return model.classifier(torch.cat([z, q_blank], dim=1))

        classical_acc = _evaluate(classical_only, test_loader, device)
        z_dim         = model.backbone_dim

        def quantum_only(x: torch.Tensor) -> torch.Tensor:
            z, q_emb = _extract_classical_and_quantum(model, x, device)
            z_blank  = torch.zeros(x.size(0), z_dim, device=device, dtype=z.dtype)
            return model.classifier(torch.cat([z_blank, q_emb], dim=1))

        quantum_acc   = _evaluate(quantum_only, test_loader, device)
        quantum_gain  = round(full_acc - classical_acc, 4)

        results[label] = {
            "full_accuracy":           full_acc,
            "classical_only_accuracy": classical_acc,
            "quantum_only_accuracy":   quantum_acc,
            "quantum_gain_%":          quantum_gain,
        }
        logger.info(
            f"  [{label}] Full: {full_acc:.2f}% | Classical-only: {classical_acc:.2f}% | "
            f"Quantum-only: {quantum_acc:.2f}% | Gain: {quantum_gain:+.2f}%"
        )

    return results


# ── Experiment 3 — Re-upload Ablation ─────────────────────────────────────────

def _make_no_reupload_layer(
    original_layer: nn.Module,
    n_qubits: int,
    q_depth: int,
    device_name: str = "default.qubit",
) -> nn.Module:
    """
    Builds an identical VQC with reupload disabled, copies trained weights.
    Always uses default.qubit (CPU sim) — GPU adjoint diff not needed here.
    """
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit_no_reupload(inputs, weights):
        qml.AngleEmbedding(features=inputs * math.pi, wires=range(n_qubits), rotation="Y")
        for layer_idx in range(q_depth):
            for i in range(n_qubits):
                qml.RY(weights[layer_idx, i, 0], wires=i)
                qml.RZ(weights[layer_idx, i, 1], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    no_reupload_layer = qml.qnn.TorchLayer(
        circuit_no_reupload, {"weights": (q_depth, n_qubits, 2)}
    )
    with torch.no_grad():
        no_reupload_layer.weights.copy_(original_layer.weights.cpu())
    return no_reupload_layer


@torch.no_grad()
def run_reupload_ablation(
    models: dict[str, nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    n_qubits: int,
    q_depth: int,
) -> dict[str, dict[str, Optional[float]]]:
    results: dict[str, dict] = {}

    for label, model in models.items():
        model.eval()
        with_reupload_acc  = _evaluate(model, test_loader, device)
        no_reupload_layer  = _make_no_reupload_layer(model.q_layer, n_qubits, q_depth)
        no_reupload_layer.eval()

        def forward_no_reupload(x: torch.Tensor) -> torch.Tensor:
            z, _ = _extract_classical_and_quantum(model, x, device)
            q_in = model.pre_quantum(z)
            with torch.amp.autocast(device_type="cpu", enabled=False):
                q_raw = no_reupload_layer(q_in.cpu().float()).to(device)
            q_emb = model.post_quantum(q_raw)
            return model.classifier(torch.cat([z, q_emb], dim=1))

        without_reupload_acc   = _evaluate(forward_no_reupload, test_loader, device)
        reupload_contribution  = round(with_reupload_acc - without_reupload_acc, 4)

        results[label] = {
            "with_reupload_accuracy":    with_reupload_acc,
            "without_reupload_accuracy": without_reupload_acc,
            "reupload_contribution_%":   reupload_contribution,
        }
        logger.info(
            f"  [{label}] With: {with_reupload_acc:.2f}% | "
            f"Without: {without_reupload_acc:.2f}% | "
            f"Gain: {reupload_contribution:+.2f}%"
        )

    return results


# ── Experiment 4 — Entanglement Entropy ───────────────────────────────────────

def _build_statevector_circuit(
    n_qubits: int,
    q_depth: int,
    trained_weights: torch.Tensor,
) -> Callable:
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def statevector_circuit(inputs, weights):
        qml.AngleEmbedding(features=inputs * math.pi, wires=range(n_qubits), rotation="Y")
        for layer_idx in range(q_depth):
            for i in range(n_qubits):
                qml.RY(weights[layer_idx, i, 0], wires=i)
                qml.RZ(weights[layer_idx, i, 1], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            if layer_idx < q_depth - 1:
                qml.AngleEmbedding(
                    features=inputs * math.pi, wires=range(n_qubits), rotation="Y"
                )
        return qml.state()

    return statevector_circuit


def _von_neumann_entropy(
    state_np: np.ndarray, qubit_idx: int, n_qubits: int
) -> float:
    state_np = state_np.reshape([2] * n_qubits)
    axes     = [qubit_idx] + [i for i in range(n_qubits) if i != qubit_idx]
    state_np = state_np.transpose(axes).reshape(2, -1)
    rho      = state_np @ state_np.conj().T
    eigvals  = np.linalg.eigvalsh(rho.real)
    eigvals  = eigvals[eigvals > 1e-12]
    return round(float(-np.sum(eigvals * np.log2(eigvals))), 6)


@torch.no_grad()
def run_entanglement_entropy(
    models: dict[str, nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    n_qubits: int,
    q_depth: int,
    n_samples: int,
) -> dict[str, dict]:
    results: dict[str, dict] = {}

    sample_q_inputs: list[torch.Tensor] = []
    model_ref = next(iter(models.values()))
    model_ref.eval()
    for images, _ in test_loader:
        for i in range(images.size(0)):
            img   = images[i].unsqueeze(0).to(device)
            dtype = next(model_ref.backbone.parameters()).dtype
            h     = model_ref.backbone(img.to(dtype=dtype))
            z     = F.adaptive_avg_pool2d(h, (1, 1)).flatten(1)
            q_in  = model_ref.pre_quantum(z).squeeze(0)
            sample_q_inputs.append(q_in.cpu().float())
            if len(sample_q_inputs) >= n_samples:
                break
        if len(sample_q_inputs) >= n_samples:
            break

    logger.info(f"  Collected {len(sample_q_inputs)} pre-quantum inputs for entropy")

    for label, model in models.items():
        model.eval()
        trained_weights = model.q_layer.weights.detach().cpu().float()
        sv_circuit      = _build_statevector_circuit(n_qubits, q_depth, trained_weights)
        per_qubit_entropies: dict[int, list[float]] = {i: [] for i in range(n_qubits)}

        for q_in in tqdm(sample_q_inputs, desc=f"[Entropy] {label}", colour="magenta"):
            state_tensor = sv_circuit(q_in, trained_weights)
            state_np     = state_tensor.detach().numpy().astype(complex)
            for qubit_idx in range(n_qubits):
                per_qubit_entropies[qubit_idx].append(
                    _von_neumann_entropy(state_np, qubit_idx, n_qubits)
                )

        mean_per_qubit = {
            f"qubit_{i}": round(float(np.mean(per_qubit_entropies[i])), 6)
            for i in range(n_qubits)
        }
        overall_mean = round(float(np.mean([
            v for vals in per_qubit_entropies.values() for v in vals
        ])), 6)

        results[label] = {
            "mean_entropy_per_qubit": mean_per_qubit,
            "overall_mean_entropy":   overall_mean,
            "interpretation": (
                "0.0 = no entanglement; 1.0 = maximally entangled. "
                "Values above 0.3 confirm genuine quantum correlations."
            ),
        }
        logger.info(f"  [{label}] Mean entropy: {overall_mean:.4f}")

    return results


# ── Experiment 5 — Quantum Gradient Variance ──────────────────────────────────

def run_gradient_variance(
    models_all: dict[str, nn.Module],
    cnn_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    n_batches: int,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    criterion = nn.CrossEntropyLoss()

    for label, model in models_all.items():
        model.eval()
        q_weights = model.q_layer.weights
        q_weights.requires_grad_(True)
        grad_vars: list[float] = []
        grad_means: list[float] = []

        for i, (images, labels) in enumerate(test_loader):
            if i >= n_batches:
                break
            images, labels = images.to(device), labels.to(device)
            if q_weights.grad is not None:
                q_weights.grad.zero_()
            loss = criterion(model(images), labels)
            loss.backward()
            if q_weights.grad is not None:
                g = q_weights.grad.detach().cpu().float()
                grad_vars.append(g.var().item())
                grad_means.append(g.abs().mean().item())

        q_weights.requires_grad_(False)
        results[label] = {
            "target":             "quantum_layer_weights",
            "mean_grad_variance": round(float(np.mean(grad_vars)),  8),
            "mean_grad_abs_mean": round(float(np.mean(grad_means)), 8),
            "n_batches":          n_batches,
            "interpretation": (
                "Near-zero variance = barren plateau risk. "
                "Compare to CNN baseline — higher QNN variance = stronger quantum signal."
            ),
        }

    # CNN baseline
    cnn_model.eval()
    final_linear: Optional[nn.Linear] = None
    for m in reversed(list(cnn_model.modules())):
        if isinstance(m, nn.Linear):
            final_linear = m
            break

    if final_linear is not None:
        final_linear.weight.requires_grad_(True)
        cnn_grad_vars:  list[float] = []
        cnn_grad_means: list[float] = []

        for i, (images, labels) in enumerate(test_loader):
            if i >= n_batches:
                break
            images, labels = images.to(device), labels.to(device)
            if final_linear.weight.grad is not None:
                final_linear.weight.grad.zero_()
            loss = criterion(cnn_model(images), labels)
            loss.backward()
            if final_linear.weight.grad is not None:
                g = final_linear.weight.grad.detach().cpu().float()
                cnn_grad_vars.append(g.var().item())
                cnn_grad_means.append(g.abs().mean().item())

        final_linear.weight.requires_grad_(False)
        results["CNN_baseline"] = {
            "target":             "final_linear_layer_weights",
            "mean_grad_variance": round(float(np.mean(cnn_grad_vars)),  8),
            "mean_grad_abs_mean": round(float(np.mean(cnn_grad_means)), 8),
            "n_batches":          n_batches,
            "interpretation":     "CNN final-layer gradient variance for comparison.",
        }

    return results


# ── Experiment 6 — Noise Robustness Ablation (NEW) ────────────────────────────

@torch.no_grad()
def run_noise_ablation(
    models: dict[str, nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    noise_levels: dict[str, list] = QA_NOISE_LEVELS,
) -> dict[str, dict]:
    """
    Runs the branch ablation (full / classical_only / quantum_only) under
    noise to quantify the quantum branch's specific contribution to noise
    robustness.

    Key metric:
        quantum_noise_gain = full_acc − classical_only_acc

    If quantum_noise_gain *increases* as the noise level rises, the quantum
    branch is specifically helping the model cope with degraded inputs — a
    direct empirical answer to the research question.

    Uses QA_NOISE_LEVELS (a reduced subset of the full benchmark grid) to
    keep runtime reasonable for the QNN forward passes.

    Returns:
        {
          model_key: {
            noise_type: [
              {
                "level":                float,
                "full_accuracy_%":      float,
                "classical_only_%":     float,
                "quantum_only_%":       float,
                "quantum_noise_gain_%": float,   # full − classical_only
              },
              ...
            ]
          }
        }
    """
    results: dict[str, dict] = {}

    for label, model in models.items():
        model.eval()
        results[label] = {}

        # Cache branch dimensions once per model
        q_dim = model.post_quantum[-2].normalized_shape[0]
        z_dim = model.backbone_dim

        # Define the three forward functions for this model
        def full_forward(x: torch.Tensor) -> torch.Tensor:
            return model(x)

        def classical_only_forward(x: torch.Tensor) -> torch.Tensor:
            z, _ = _extract_classical_and_quantum(model, x, device)
            q_blank = torch.zeros(x.size(0), q_dim, device=device, dtype=z.dtype)
            return model.classifier(torch.cat([z, q_blank], dim=1))

        def quantum_only_forward(x: torch.Tensor) -> torch.Tensor:
            z, q_emb = _extract_classical_and_quantum(model, x, device)
            z_blank  = torch.zeros(x.size(0), z_dim, device=device, dtype=z.dtype)
            return model.classifier(torch.cat([z_blank, q_emb], dim=1))

        for noise_type, levels in noise_levels.items():
            logger.info(f"\n  [{label}] Noise ablation — {noise_type}")
            rows: list[dict] = []

            for level in tqdm(levels, desc=f"  {label}/{noise_type}", leave=False):
                full_acc      = _evaluate_noisy(full_forward,       test_loader, device, noise_type, level)
                classical_acc = _evaluate_noisy(classical_only_forward, test_loader, device, noise_type, level)
                quantum_acc   = _evaluate_noisy(quantum_only_forward,   test_loader, device, noise_type, level)
                gain          = round(full_acc - classical_acc, 4)

                row = {
                    "level":                level,
                    "full_accuracy_%":      full_acc,
                    "classical_only_%":     classical_acc,
                    "quantum_only_%":       quantum_acc,
                    "quantum_noise_gain_%": gain,
                }
                rows.append(row)

                logger.info(
                    f"    level={level:.4g} | full={full_acc:.2f}%  "
                    f"classical={classical_acc:.2f}%  quantum={quantum_acc:.2f}%  "
                    f"gain={gain:+.2f}%"
                )

            results[label][noise_type] = rows

    return results


# ── CSV Exports ────────────────────────────────────────────────────────────────

def export_orthogonality_csv(results: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "qa_feature_orthogonality.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "cosine_similarity", "interpretation"])
        for model_key, val in results.items():
            if val is not None:
                interp = (
                    "orthogonal (good)"           if abs(val) < 0.3  else
                    "partially correlated"         if abs(val) < 0.7  else
                    "highly correlated (no advantage)"
                )
            else:
                interp = "N/A"
            writer.writerow([model_key, _na(val), interp])
    logger.info(f"CSV saved: {path}")


def export_ablation_csv(results: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "qa_branch_ablation.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "full_accuracy_%", "classical_only_accuracy_%",
            "quantum_only_accuracy_%", "quantum_gain_%",
        ])
        for model_key, r in results.items():
            writer.writerow([
                model_key,
                _na(r.get("full_accuracy")),
                _na(r.get("classical_only_accuracy")),
                _na(r.get("quantum_only_accuracy")),
                _na(r.get("quantum_gain_%")),
            ])
    logger.info(f"CSV saved: {path}")


def export_reupload_csv(results: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "qa_reupload_ablation.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "with_reupload_accuracy_%",
            "without_reupload_accuracy_%", "reupload_contribution_%",
        ])
        for model_key, r in results.items():
            writer.writerow([
                model_key,
                _na(r.get("with_reupload_accuracy")),
                _na(r.get("without_reupload_accuracy")),
                _na(r.get("reupload_contribution_%")),
            ])
    logger.info(f"CSV saved: {path}")


def export_entropy_csv(results: dict, n_qubits: int, output_dir: str) -> None:
    path = os.path.join(output_dir, "qa_entanglement_entropy.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        qubit_headers = [f"qubit_{i}_entropy" for i in range(n_qubits)]
        writer.writerow(["model", "overall_mean_entropy"] + qubit_headers)
        for model_key, r in results.items():
            per_qubit = r.get("mean_entropy_per_qubit", {})
            writer.writerow(
                [model_key, _na(r.get("overall_mean_entropy"))]
                + [_na(per_qubit.get(f"qubit_{i}")) for i in range(n_qubits)]
            )
    logger.info(f"CSV saved: {path}")


def export_grad_variance_csv(results: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "qa_gradient_variance.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "target_layer", "mean_grad_variance", "mean_grad_abs_mean"])
        for model_key, r in results.items():
            writer.writerow([
                model_key, r.get("target", "N/A"),
                _na(r.get("mean_grad_variance")),
                _na(r.get("mean_grad_abs_mean")),
            ])
    logger.info(f"CSV saved: {path}")


def export_noise_ablation_csv(results: dict, output_dir: str) -> None:
    """
    Flat CSV for Experiment 6 (noise robustness ablation).
    Columns: model, noise_type, level, full_accuracy_%, classical_only_%,
             quantum_only_%, quantum_noise_gain_%
    """
    path = os.path.join(output_dir, "qa_noise_ablation.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "noise_type", "level",
            "full_accuracy_%", "classical_only_%",
            "quantum_only_%",  "quantum_noise_gain_%",
        ])
        for model_key, noise_dict in results.items():
            for noise_type, rows in noise_dict.items():
                for row in rows:
                    writer.writerow([
                        model_key, noise_type, row["level"],
                        _na(row.get("full_accuracy_%")),
                        _na(row.get("classical_only_%")),
                        _na(row.get("quantum_only_%")),
                        _na(row.get("quantum_noise_gain_%")),
                    ])
    logger.info(f"CSV saved: {path}")


# ── Model Loading ──────────────────────────────────────────────────────────────

def load_models(
    num_classes: int,
    device: torch.device,
) -> tuple[dict[str, nn.Module], nn.Module]:
    qnn_models: dict[str, nn.Module] = {}

    cnn = CNN(num_classes=num_classes)
    cnn.load_model(CHECKPOINT_PATHS["CNN"], device)
    cnn.eval()
    logger.info("CNN loaded (gradient variance baseline)")

    qnn_cpu = HybridQnnCPU(num_classes=num_classes)
    qnn_cpu.load_model(CHECKPOINT_PATHS["QNN_CPU"], device)
    qnn_cpu.eval()
    qnn_models["QNN_CPU"] = qnn_cpu
    logger.info("QNN_CPU loaded")

    if torch.cuda.is_available():
        qnn_gpu = HybridQnnGPU(num_classes=num_classes)
        qnn_gpu.load_model(CHECKPOINT_PATHS["QNN_GPU"], device)
        qnn_gpu.eval()
        qnn_models["QNN_GPU"] = qnn_gpu
        logger.info("QNN_GPU loaded")
    else:
        logger.warning("CUDA unavailable — QNN_GPU excluded.")

    return qnn_models, cnn


# ── Main ───────────────────────────────────────────────────────────────────────

def run_quantum_advantage() -> dict:
    cfg    = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Quantum Advantage Runner on device: {device}")

    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names: list[str] = json.load(f)
    num_classes = len(class_names)
    logger.info(f"Classes ({num_classes}): {class_names}")

    manager = DataLoaderManager(
        train_dir=DATA_DIRS["train"],
        val_dir=DATA_DIRS["val"],
        test_dir=DATA_DIRS["test"],
        img_width=cfg["img_width"],
        img_height=cfg["img_height"],
        batch_size=cfg["batch_size"],
    )
    _, _, test_loader = manager.get_loaders()
    logger.info(f"Test set: {len(test_loader.dataset)} images")

    qnn_models, cnn_model = load_models(num_classes, device)

    logger.info("\n=== Experiment 1: Feature Orthogonality ===")
    orthogonality_results = run_feature_orthogonality(qnn_models, test_loader, device)

    logger.info("\n=== Experiment 2: Branch Ablation ===")
    ablation_results = run_branch_ablation(qnn_models, test_loader, device)

    logger.info("\n=== Experiment 3: Re-upload Ablation ===")
    reupload_results = run_reupload_ablation(
        qnn_models, test_loader, device,
        n_qubits=cfg["n_qubits"], q_depth=cfg["q_depth"],
    )

    logger.info("\n=== Experiment 4: Entanglement Entropy ===")
    entropy_results = run_entanglement_entropy(
        qnn_models, test_loader, device,
        n_qubits=cfg["n_qubits"], q_depth=cfg["q_depth"],
        n_samples=cfg["entropy_samples"],
    )

    logger.info("\n=== Experiment 5: Quantum Gradient Variance ===")
    grad_variance_results = run_gradient_variance(
        qnn_models, cnn_model, test_loader, device,
        n_batches=cfg["grad_var_batches"],
    )

    logger.info("\n=== Experiment 6: Noise Robustness Ablation ===")
    logger.info(
        "  Running full / classical-only / quantum-only under 5 noise types.\n"
        "  quantum_noise_gain = full_acc - classical_only_acc at each level."
    )
    noise_ablation_results = run_noise_ablation(
        qnn_models, test_loader, device,
        noise_levels=QA_NOISE_LEVELS,
    )

    # Assemble JSON
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device":       str(device),
        "config": {
            "n_qubits":        cfg["n_qubits"],
            "q_depth":         cfg["q_depth"],
            "entropy_samples": cfg["entropy_samples"],
            "grad_var_batches":cfg["grad_var_batches"],
            "class_names":     class_names,
            "qa_noise_levels": QA_NOISE_LEVELS,
            "notes": {
                "entanglement_entropy": (
                    "QNN_GPU uses lightning.gpu with adjoint differentiation which "
                    "does not expose intermediate statevectors. Weights are mirrored "
                    "onto a default.qubit circuit for this experiment."
                ),
                "noise_ablation": (
                    "full: normal forward on noisy input. "
                    "classical_only: quantum branch zeroed. "
                    "quantum_only: classical branch zeroed. "
                    "quantum_noise_gain = full − classical_only. "
                    "Increasing gain as noise rises = quantum branch improves robustness."
                ),
            },
        },
        "experiment_1_feature_orthogonality": orthogonality_results,
        "experiment_2_branch_ablation":        ablation_results,
        "experiment_3_reupload_ablation":      reupload_results,
        "experiment_4_entanglement_entropy":   entropy_results,
        "experiment_5_gradient_variance":      grad_variance_results,
        "experiment_6_noise_ablation":         noise_ablation_results,   # ← new
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"JSON saved: {OUTPUT_PATH}")

    logger.info("\n=== Exporting CSVs ===")
    export_orthogonality_csv(orthogonality_results, OUTPUT_DIR)
    export_ablation_csv(ablation_results, OUTPUT_DIR)
    export_reupload_csv(reupload_results, OUTPUT_DIR)
    export_entropy_csv(entropy_results, cfg["n_qubits"], OUTPUT_DIR)
    export_grad_variance_csv(grad_variance_results, OUTPUT_DIR)
    export_noise_ablation_csv(noise_ablation_results, OUTPUT_DIR)         # ← new

    # Summary
    print("\n" + "=" * 70)
    print("QUANTUM ADVANTAGE SUMMARY")
    print("=" * 70)
    for model_key in qnn_models:
        print(f"\n  {model_key}")
        orth = orthogonality_results.get(model_key)
        if orth is not None:
            tag = "orthogonal ✓" if abs(orth) < 0.3 else "~ partial"
            print(f"    Feature orthogonality:     {orth:.4f}  ({tag})")
        abl = ablation_results.get(model_key, {})
        print(f"    Quantum branch gain:       {abl.get('quantum_gain_%', 'N/A'):+.2f}%")
        reu = reupload_results.get(model_key, {})
        print(f"    Re-upload contribution:    {reu.get('reupload_contribution_%', 'N/A'):+.2f}%")
        ent = entropy_results.get(model_key, {})
        overall_ent = ent.get("overall_mean_entropy", 0)
        tag_e = "entangled ✓" if overall_ent > 0.3 else "~ weak"
        print(f"    Mean entanglement entropy: {overall_ent:.4f}  ({tag_e})")
        # Noise ablation summary — Gaussian gain at lowest vs highest level
        if model_key in noise_ablation_results:
            gaussian_rows = noise_ablation_results[model_key].get("gaussian", [])
            if len(gaussian_rows) >= 2:
                clean_gain = gaussian_rows[0].get("quantum_noise_gain_%", 0)
                noisy_gain = gaussian_rows[-1].get("quantum_noise_gain_%", 0)
                print(
                    f"    Gaussian noise gain:       "
                    f"{clean_gain:+.2f}% (clean) → {noisy_gain:+.2f}% (σ=0.50)"
                )
    print("=" * 70)

    return output


if __name__ == "__main__":
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] PyTorch device: {torch_device}")
    if torch_device.type == "cuda":
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

    try:
        qml_dev = qml.device("default.qubit", wires=1)
        print(f"[INFO] PennyLane device: {qml_dev}")
    except Exception as e:
        print(f"[WARN] PennyLane check failed: {e}")

    run_quantum_advantage()