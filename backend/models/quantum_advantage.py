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
                                    entropy for each qubit bipartition. Proves the
                                    CNOT ring is genuinely entangling features.
                                    NOTE: lightning.gpu does not expose statevectors,
                                    so GPU model weights are loaded into a temporary
                                    default.qubit device for this experiment only.

    5. Quantum Gradient Variance  — measures gradient variance of the trained quantum
                                    weights across test batches. Near-zero = barren
                                    plateau. High = healthy quantum signal. Compared
                                    against the CNN's final-layer gradient variance.

Outputs:
    data/QA/quantum_advantage_results.json
    data/QA/qa_feature_orthogonality.csv
    data/QA/qa_branch_ablation.csv
    data/QA/qa_reupload_ablation.csv
    data/QA/qa_entanglement_entropy.csv
    data/QA/qa_gradient_variance.csv
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

logger = Logger()

# Config
CONFIG = {
    "img_width":       384,
    "img_height":      384,
    "batch_size":      16,
    "n_qubits":        6,
    "q_depth":         2,
    # Batches used for gradient variance (full test set is slow with grad enabled)
    "grad_var_batches": 20,
    # Samples used for entanglement entropy (statevector is expensive)
    "entropy_samples": 128,
}

# 1. Get the directory where benchmark.py lives (.../backend/models/)
_current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Get the 'backend' root (.../backend/)
BACKEND_ROOT = os.path.dirname(_current_dir)

CHECKPOINT_PATHS = {
    "CNN": os.path.join(BACKEND_ROOT, "models", "cnn.pth"),
    "QNN_CPU": os.path.join(BACKEND_ROOT, "models", "qnn_cpu.pth"),
    "QNN_GPU": os.path.join(BACKEND_ROOT, "models", "qnn_gpu_6_qubits.pth"),
}
CLASS_NAMES_PATH = os.path.join(BACKEND_ROOT, "data", "class_names.json")
OUTPUT_PATH = os.path.join(BACKEND_ROOT, "data", "QA", "quantum_advantage_results.json")

# Ensure this directory exists relative to backend
OUTPUT_DIR = os.path.join(BACKEND_ROOT, "data", "QA")

DATA_DIRS = {
    "train": os.path.join(BACKEND_ROOT, "data", "train"),
    "val": os.path.join(BACKEND_ROOT, "data", "val"),
    "test": os.path.join(BACKEND_ROOT, "data", "test"),
}

QNN_MODEL_KEYS = ["QNN_CPU", "QNN_GPU"]

# Helpers
def _na(value) -> str:
    return str(value) if value is not None else "N/A"

@torch.no_grad()
def _evaluate(forward_fn: Callable[[torch.Tensor], torch.Tensor], loader: DataLoader, device: torch.device,) -> float:
    """Generic accuracy evaluation given an arbitrary forward function."""
    correct = 0
    total   = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = forward_fn(images).argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
    return round(100.0 * correct / total, 4)


def _extract_classical_and_quantum(model: nn.Module, images: torch.Tensor, device: torch.device,) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the backbone + quantum branch and return:
        z      — 512-D classical pooled features
        q_emb  — 128-D post-quantum projected features
    Works for both HybridQnnCPU and HybridQnnGPU.
    """
    x = images.to(device)
    dtype = next(model.backbone.parameters()).dtype
    x = x.to(dtype=dtype)
    h = model.backbone(x)
    z = F.adaptive_avg_pool2d(h, (1, 1)).flatten(1)

    q_in  = model.pre_quantum(z)

    # CPU model sends q_input to CPU; GPU model keeps it on device
    if isinstance(model, HybridQnnCPU):
        with torch.amp.autocast(device_type=device.type, enabled=False):
            q_raw = model.q_layer(q_in.to("cpu").float()).to(device)
    else:
        with torch.amp.autocast(device_type=device.type, enabled=False):
            q_raw = model.q_layer(q_in.float())

    q_emb = model.post_quantum(q_raw)
    return z, q_emb

# Experiment 1 — Feature Orthogonality
@torch.no_grad()
def run_feature_orthogonality(models: dict[str, nn.Module], test_loader: DataLoader, device: torch.device,) -> dict[str, Optional[float]]:
    """
    Measures cosine similarity between the classical branch features
    (z, first 128 dims) and the quantum branch features (q_emb, 128-D).

    Interpretation:
        ~0.0   quantum learns orthogonal / complementary features (ideal)
        ~1.0   quantum just mirrors classical branch (no advantage)
        ~-1.0  anti-correlated (also non-trivial)
    """
    results: dict[str, Optional[float]] = {}

    for label, model in models.items():
        model.eval()
        sims: list[float] = []

        for images, _ in tqdm(test_loader, desc=f"[Orthogonality] {label}", colour="cyan"):
            z, q_emb = _extract_classical_and_quantum(model, images, device)

            # Project z to same dimensionality as q_emb for fair comparison
            z_proj = z[:, :q_emb.size(1)]

            z_norm = F.normalize(z_proj.float(), dim=1)
            q_norm = F.normalize(q_emb.float(), dim=1)

            batch_sim = (z_norm * q_norm).sum(dim=1).mean().item()
            sims.append(batch_sim)

        avg_sim = round(float(np.mean(sims)), 6)
        results[label] = avg_sim
        logger.info(f"  [{label}] Classical-Quantum cosine similarity: {avg_sim:.4f}")

    return results

# Experiment 2 — Branch Ablation
@torch.no_grad()
def run_branch_ablation(models: dict[str, nn.Module], test_loader: DataLoader, device: torch.device,) -> dict[str, dict[str, Optional[float]]]:
    """
    Three conditions per model:
        full            — normal forward pass
        classical_only  — quantum branch replaced with zeros
        quantum_only    — classical branch replaced with zeros

    quantum_gain = full_acc - classical_only_acc
    This is the headline number proving the quantum branch adds real accuracy.
    """
    results: dict[str, dict] = {}

    for label, model in models.items():
        model.eval()

        # Full forward
        full_acc = _evaluate(model, test_loader, device)

        # Classical only (zero q_emb)
        q_dim = model.post_quantum[-2].normalized_shape[0]  # LayerNorm input dim = 128

        def classical_only(x: torch.Tensor) -> torch.Tensor:
            z, _ = _extract_classical_and_quantum(model, x, device)
            q_blank = torch.zeros(x.size(0), q_dim, device=device, dtype=z.dtype)
            return model.classifier(torch.cat([z, q_blank], dim=1))

        classical_acc = _evaluate(classical_only, test_loader, device)

        # Quantum only (zero z)
        z_dim = model.backbone_dim  # 512

        def quantum_only(x: torch.Tensor) -> torch.Tensor:
            z, q_emb = _extract_classical_and_quantum(model, x, device)
            z_blank = torch.zeros(x.size(0), z_dim, device=device, dtype=z.dtype)
            return model.classifier(torch.cat([z_blank, q_emb], dim=1))

        quantum_acc = _evaluate(quantum_only, test_loader, device)

        quantum_gain = round(full_acc - classical_acc, 4)

        results[label] = {
            "full_accuracy":          full_acc,
            "classical_only_accuracy": classical_acc,
            "quantum_only_accuracy":  quantum_acc,
            "quantum_gain_%":         quantum_gain,
        }

        logger.info(
            f"  [{label}] Full: {full_acc:.2f}% | "
            f"Classical-only: {classical_acc:.2f}% | "
            f"Quantum-only: {quantum_acc:.2f}% | "
            f"Quantum gain: {quantum_gain:+.2f}%"
        )

    return results

# Experiment 3 — Re-upload Ablation
def _make_no_reupload_layer(original_layer: nn.Module, n_qubits: int, q_depth: int, device_name: str = "default.qubit",) -> nn.Module:
    """
    Builds an identical VQC but with reupload=False, then copies
    the trained weights from the original layer into it.
    Uses default.qubit regardless of original device — we only need
    inference here, not GPU acceleration.
    """
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit_no_reupload(inputs, weights):
        qml.AngleEmbedding(
            features=inputs * math.pi, wires=range(n_qubits), rotation="Y"
        )
        for layer_idx in range(q_depth):
            for i in range(n_qubits):
                qml.RY(weights[layer_idx, i, 0], wires=i)
                qml.RZ(weights[layer_idx, i, 1], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            # NOTE: re-uploading intentionally omitted
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    no_reupload_layer = qml.qnn.TorchLayer(
        circuit_no_reupload, {"weights": (q_depth, n_qubits, 2)}
    )

    # Copy trained weights exactly
    with torch.no_grad():
        no_reupload_layer.weights.copy_(original_layer.weights.cpu())

    return no_reupload_layer


@torch.no_grad()
def run_reupload_ablation(models: dict[str, nn.Module], test_loader: DataLoader, device: torch.device, n_qubits: int, q_depth: int,) -> dict[str, dict[str, Optional[float]]]:
    """
    Compares accuracy with and without data re-uploading.
    The drop quantifies the expressive power added by re-uploading.
    Uses the same trained weights — only the circuit topology changes.
    """
    results: dict[str, dict] = {}

    for label, model in models.items():
        model.eval()

        # Baseline: normal forward
        with_reupload_acc = _evaluate(model, test_loader, device)

        # Build patched layer (no re-upload) with identical weights
        no_reupload_layer = _make_no_reupload_layer(
            model.q_layer, n_qubits, q_depth
        )
        no_reupload_layer.eval()

        def forward_no_reupload(x: torch.Tensor) -> torch.Tensor:
            z, _ = _extract_classical_and_quantum(model, x, device)
            q_in = model.pre_quantum(z)
            # Always run on CPU (default.qubit)
            with torch.amp.autocast(device_type="cpu", enabled=False):
                q_raw = no_reupload_layer(q_in.cpu().float()).to(device)
            q_emb = model.post_quantum(q_raw)
            return model.classifier(torch.cat([z, q_emb], dim=1))

        without_reupload_acc = _evaluate(forward_no_reupload, test_loader, device)

        reupload_contribution = round(with_reupload_acc - without_reupload_acc, 4)

        results[label] = {
            "with_reupload_accuracy":    with_reupload_acc,
            "without_reupload_accuracy": without_reupload_acc,
            "reupload_contribution_%":   reupload_contribution,
        }

        logger.info(
            f"  [{label}] With re-upload: {with_reupload_acc:.2f}% | "
            f"Without: {without_reupload_acc:.2f}% | "
            f"Re-upload gain: {reupload_contribution:+.2f}%"
        )

    return results

# Experiment 4 — Entanglement Entropy
def _build_statevector_circuit(n_qubits: int, q_depth: int, trained_weights: torch.Tensor,) -> tuple[Callable, torch.Tensor]:
    """
    Mirrors the exact VQC topology on default.qubit (statevector sim).
    Returns the qnode and a reference to the weight tensor so we can
    pass trained weights at call time.

    This is the ONLY way to extract the full quantum state from a
    lightning.gpu model — we rebuild the circuit on a CPU simulator
    with identical weights.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def statevector_circuit(inputs, weights):
        qml.AngleEmbedding(
            features=inputs * math.pi, wires=range(n_qubits), rotation="Y"
        )
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
        return qml.state()  # full 2^n_qubits complex statevector

    return statevector_circuit

def _von_neumann_entropy(state_np: np.ndarray, qubit_idx: int, n_qubits: int) -> float:
    """
    Computes the von Neumann entropy of qubit `qubit_idx` by partial-tracing
    over all other qubits. Entropy = 0  product state, = 1  maximally entangled.
    """
    dim = 2 ** n_qubits
    state_np = state_np.reshape([2] * n_qubits)

    # Keep `qubit_idx`, trace over everything else
    # Reshape: (2, 2^(n-1)) by moving target qubit to front
    axes = list(range(n_qubits))
    axes.pop(qubit_idx)
    axes = [qubit_idx] + axes
    state_np = state_np.transpose(axes).reshape(2, -1)

    # Reduced density matrix: rho_A = psi_A psi_A†
    rho = state_np @ state_np.conj().T  # 2×2

    eigvals = np.linalg.eigvalsh(rho.real)
    eigvals = eigvals[eigvals > 1e-12]
    entropy = float(-np.sum(eigvals * np.log2(eigvals)))
    return round(entropy, 6)


@torch.no_grad()
def run_entanglement_entropy(models: dict[str, nn.Module], test_loader: DataLoader, device: torch.device, n_qubits: int, q_depth: int,  n_samples: int,) -> dict[str, dict]:
    """
    Extracts the quantum statevector for n_samples inputs per model,
    computes von Neumann entropy for each qubit bipartition,
    and reports mean entropy per qubit and overall mean.

    For QNN_GPU: weights are copied from lightning.gpu model into a
    temporary default.qubit circuit (same topology, same weights).
    This is necessary because lightning.gpu uses adjoint differentiation
    which does not expose intermediate statevectors.
    """
    results: dict[str, dict] = {}

    # Collect n_samples pre-quantum inputs from test set
    sample_q_inputs: list[torch.Tensor] = []
    for images, _ in test_loader:
        for i in range(images.size(0)):
            img = images[i].unsqueeze(0).to(device)
            model_ref = next(iter(models.values()))
            model_ref.eval()
            dtype = next(model_ref.backbone.parameters()).dtype
            h = model_ref.backbone(img.to(dtype=dtype))
            z = F.adaptive_avg_pool2d(h, (1, 1)).flatten(1)
            q_in = model_ref.pre_quantum(z).squeeze(0)  # (n_qubits,)
            sample_q_inputs.append(q_in.cpu().float())
            if len(sample_q_inputs) >= n_samples:
                break
        if len(sample_q_inputs) >= n_samples:
            break

    logger.info(f"  Collected {len(sample_q_inputs)} pre-quantum inputs for entropy")

    for label, model in models.items():
        model.eval()

        # Extract trained weights  CPU float
        trained_weights = model.q_layer.weights.detach().cpu().float()

        # Build statevector circuit (always default.qubit)
        sv_circuit = _build_statevector_circuit(n_qubits, q_depth, trained_weights)

        per_qubit_entropies: dict[int, list[float]] = {i: [] for i in range(n_qubits)}

        for q_in in tqdm(sample_q_inputs, desc=f"[Entropy] {label}", colour="magenta"):
            state_tensor = sv_circuit(q_in, trained_weights)
            state_np = state_tensor.detach().numpy().astype(complex)

            for qubit_idx in range(n_qubits):
                ent = _von_neumann_entropy(state_np, qubit_idx, n_qubits)
                per_qubit_entropies[qubit_idx].append(ent)

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
            # Interpretation guide saved alongside results
            "interpretation": (
                "0.0 = no entanglement (classical separable state); "
                "1.0 = maximally entangled qubit. "
                "Values above 0.3 confirm genuine quantum correlations from the CNOT ring."
            ),
        }

        logger.info(
            f"  [{label}] Mean entanglement entropy: {overall_mean:.4f} "
            f"| Per qubit: {mean_per_qubit}"
        )

    return results

# Experiment 5 — Quantum Gradient Variance
def run_gradient_variance(
    models_all: dict[str, nn.Module], cnn_model: nn.Module, test_loader: DataLoader, device: torch.device, n_batches: int,) -> dict[str, dict]:
    """
    Measures gradient variance of:
        - quantum layer weights (QNN models)
        - final linear layer weights (CNN baseline for comparison)

    Near-zero variance = barren plateau (quantum signal is vanishing).
    High variance = healthy, expressive gradient signal.

    Importantly, we do NOT retrain — we just do a forward+backward pass
    through the frozen model to observe the gradient landscape.
    """
    results: dict[str, dict] = {}
    criterion = nn.CrossEntropyLoss()

    # QNN models: quantum layer gradient variance
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

            # Zero existing grads
            if q_weights.grad is not None:
                q_weights.grad.zero_()

            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()

            if q_weights.grad is not None:
                g = q_weights.grad.detach().cpu().float()
                grad_vars.append(g.var().item())
                grad_means.append(g.abs().mean().item())

        # Restore no-grad state
        q_weights.requires_grad_(False)

        results[label] = {
            "target":             "quantum_layer_weights",
            "mean_grad_variance": round(float(np.mean(grad_vars)),  8),
            "mean_grad_abs_mean": round(float(np.mean(grad_means)), 8),
            "n_batches":          n_batches,
            "interpretation": (
                "Quantum layer gradient variance across test batches. "
                "Near-zero = barren plateau risk. "
                "Compare to CNN baseline — higher QNN variance = stronger quantum signal."
            ),
        }
        logger.info(
            f"  [{label}] Q-layer grad variance: {results[label]['mean_grad_variance']:.8f} "
            f"| abs mean: {results[label]['mean_grad_abs_mean']:.8f}"
        )

    # CNN baseline: final linear layer gradient variance
    cnn_model.eval()
    # Find the last Linear layer in the CNN
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

            logits = cnn_model(images)
            loss   = criterion(logits, labels)
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
        logger.info(
            f"  [CNN_baseline] Final linear grad variance: "
            f"{results['CNN_baseline']['mean_grad_variance']:.8f}"
        )

    return results

# CSV Exports
def export_orthogonality_csv(results: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "qa_feature_orthogonality.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "cosine_similarity", "interpretation"])
        for model_key, val in results.items():
            interp = (
                "orthogonal (good)" if val is not None and abs(val) < 0.3
                else "partially correlated" if val is not None and abs(val) < 0.7
                else "highly correlated (no advantage)"
            )
            writer.writerow([model_key, _na(val), interp])
    logger.info(f"CSV saved: {path}")

def export_ablation_csv(results: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "qa_branch_ablation.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "full_accuracy_%",
            "classical_only_accuracy_%",
            "quantum_only_accuracy_%",
            "quantum_gain_%",
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
            "model",
            "with_reupload_accuracy_%",
            "without_reupload_accuracy_%",
            "reupload_contribution_%",
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
            row = [
                model_key,
                _na(r.get("overall_mean_entropy")),
            ] + [_na(per_qubit.get(f"qubit_{i}")) for i in range(n_qubits)]
            writer.writerow(row)
    logger.info(f"CSV saved: {path}")

def export_grad_variance_csv(results: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "qa_gradient_variance.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "target_layer", "mean_grad_variance", "mean_grad_abs_mean"])
        for model_key, r in results.items():
            writer.writerow([
                model_key,
                r.get("target", "N/A"),
                _na(r.get("mean_grad_variance")),
                _na(r.get("mean_grad_abs_mean")),
            ])
    logger.info(f"CSV saved: {path}")


# Model Loading
def load_models(num_classes: int, device: torch.device,) -> tuple[dict[str, nn.Module], nn.Module]:
    """
    Returns (qnn_models, cnn_model).
    QNN_GPU is skipped gracefully if CUDA is unavailable.
    CNN is loaded separately for gradient variance baseline.
    """
    qnn_models: dict[str, nn.Module] = {}

    # CNN (gradient variance baseline only)
    cnn = CNN(num_classes=num_classes)
    cnn.load_model(CHECKPOINT_PATHS["CNN"], device)
    cnn.eval()
    logger.info("CNN loaded (gradient variance baseline)")

    # QNN_CPU
    qnn_cpu = HybridQnnCPU(num_classes=num_classes)
    qnn_cpu.load_model(CHECKPOINT_PATHS["QNN_CPU"], device)
    qnn_cpu.eval()
    qnn_models["QNN_CPU"] = qnn_cpu
    logger.info("QNN_CPU loaded")

    # QNN_GPU
    if torch.cuda.is_available():
        qnn_gpu = HybridQnnGPU(num_classes=num_classes)
        qnn_gpu.load_model(CHECKPOINT_PATHS["QNN_GPU"], device)
        qnn_gpu.eval()
        qnn_models["QNN_GPU"] = qnn_gpu
        logger.info("QNN_GPU loaded")
    else:
        logger.warning(
            "CUDA unavailable — QNN_GPU excluded. "
            "Entanglement entropy will still run for QNN_CPU."
        )

    return qnn_models, cnn

# Main
def run_quantum_advantage() -> dict:
    cfg    = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Quantum Advantage Runner on device: {device}")

    # Load class names
    class_names_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", CLASS_NAMES_PATH
    )
    with open(class_names_path, encoding="utf-8") as f:
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
    logger.info(f"Test set: {len(test_loader.dataset)} images")

    # Load models
    qnn_models, cnn_model = load_models(num_classes, device)

    # Experiment 1: Feature Orthogonality 
    logger.info("\n=== Experiment 1: Feature Orthogonality ===")
    orthogonality_results = run_feature_orthogonality(qnn_models, test_loader, device)

    # Experiment 2: Branch Ablation
    logger.info("\n=== Experiment 2: Branch Ablation ===")
    ablation_results = run_branch_ablation(qnn_models, test_loader, device)

    # Experiment 3: Re-upload Ablation
    logger.info("\n=== Experiment 3: Re-upload Ablation ===")
    reupload_results = run_reupload_ablation(
        qnn_models, test_loader, device,
        n_qubits=cfg["n_qubits"],
        q_depth=cfg["q_depth"],
    )

    # Experiment 4: Entanglement Entropy
    logger.info("\n=== Experiment 4: Entanglement Entropy ===")
    logger.info(
        "NOTE: QNN_GPU weights are mirrored onto default.qubit for this "
        "experiment — lightning.gpu (adjoint diff) does not expose statevectors."
    )
    entropy_results = run_entanglement_entropy(
        qnn_models, test_loader, device,
        n_qubits=cfg["n_qubits"],
        q_depth=cfg["q_depth"],
        n_samples=cfg["entropy_samples"],
    )

    # Experiment 5: Gradient Variance
    logger.info("\n=== Experiment 5: Quantum Gradient Variance ===")
    grad_variance_results = run_gradient_variance(
        qnn_models, cnn_model, test_loader, device,
        n_batches=cfg["grad_var_batches"],
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
            "notes": {
                "entanglement_entropy": (
                    "QNN_GPU uses lightning.gpu with adjoint differentiation which "
                    "does not expose intermediate statevectors. For this experiment, "
                    "trained weights are loaded into an equivalent default.qubit "
                    "circuit (same topology, same weights) to enable statevector extraction."
                ),
                "branch_ablation": (
                    "classical_only: quantum branch replaced with zero tensor. "
                    "quantum_only: classical backbone branch replaced with zero tensor. "
                    "quantum_gain = full_accuracy - classical_only_accuracy."
                ),
                "gradient_variance": (
                    "Gradient variance measured over frozen trained weights via "
                    "forward+backward pass on test batches. No retraining occurs."
                ),
            },
        },
        "experiment_1_feature_orthogonality": orthogonality_results,
        "experiment_2_branch_ablation":       ablation_results,
        "experiment_3_reupload_ablation":     reupload_results,
        "experiment_4_entanglement_entropy":  entropy_results,
        "experiment_5_gradient_variance":     grad_variance_results,
    }

    # Save JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"JSON saved: {OUTPUT_PATH}")

    # Export CSVs
    logger.info("\n=== Exporting CSVs ===")
    export_orthogonality_csv(orthogonality_results, OUTPUT_DIR)
    export_ablation_csv(ablation_results, OUTPUT_DIR)
    export_reupload_csv(reupload_results, OUTPUT_DIR)
    export_entropy_csv(entropy_results, cfg["n_qubits"], OUTPUT_DIR)
    export_grad_variance_csv(grad_variance_results, OUTPUT_DIR)

    print(f"\nJSON   {OUTPUT_PATH}")
    print(f"CSVs   {OUTPUT_DIR}/")
    print("  qa_feature_orthogonality.csv")
    print("  qa_branch_ablation.csv")
    print("  qa_reupload_ablation.csv")
    print("  qa_entanglement_entropy.csv")
    print("  qa_gradient_variance.csv")

    # Print Summary 
    print("\n" + "=" * 60)
    print("QUANTUM ADVANTAGE SUMMARY")
    print("=" * 60)
    for model_key in qnn_models:
        print(f"\n  {model_key}")
        orth = orthogonality_results.get(model_key)
        print(f"    Feature orthogonality (cosine sim): {orth:.4f}  "
              f"{' orthogonal' if orth is not None and abs(orth) < 0.3 else '~ partial'}")
        abl = ablation_results.get(model_key, {})
        print(f"    Quantum branch gain:  {abl.get('quantum_gain_%', 'N/A'):+.2f}%")
        reu = reupload_results.get(model_key, {})
        print(f"    Re-upload contribution: {reu.get('reupload_contribution_%', 'N/A'):+.2f}%")
        ent = entropy_results.get(model_key, {})
        print(f"    Mean entanglement entropy: {ent.get('overall_mean_entropy', 'N/A'):.4f}  "
              f"{' entangled' if ent.get('overall_mean_entropy', 0) > 0.3 else '~ weak'}")
    print("=" * 60)

    return output


if __name__ == "__main__":
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] PyTorch device: {torch_device}")
    if torch_device.type == "cuda":
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

    try:
        import pennylane as qml

        qml_dev = qml.device("default.qubit", wires=1)
        print(f"[INFO] PennyLane CPU device: {qml_dev}")
    except Exception as e:
        print(f"[WARN] PennyLane CPU device failed: {e}")
    run_quantum_advantage()
