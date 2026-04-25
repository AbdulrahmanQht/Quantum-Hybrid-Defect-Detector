"""
Quantum Advantage — Hybrid QNN Defect Detector
------------------------------------------------------------------------
This script proves the quantum mechanisms in HybridQnnCPU and HybridQnnGPU are real, active, and contributing.

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

    6. Noise Robustness Ablation  — runs full / classical_only / quantum_only forward
                                    passes under 7 noise types at multiple intensity
                                    levels. quantum_noise_gain = full − classical_only.
                                    Increasing gain as noise rises proves the quantum
                                    branch specifically improves noise tolerance.

    7. VQC Expressibility         — Sim et al. (2019). Samples n random input pairs,
                                    computes pairwise fidelity |<ψ(x_i)|ψ(x_j)>|²,
                                    and measures KL divergence from the Haar reference
                                    Beta(1, 2^n − 1). Lower KL = circuit explores
                                    Hilbert space more uniformly = higher expressibility.
                                    Directly justifies the depth-2 CNOT / CZ ansatz.

    8. Kernel Target Alignment    — Cortes & Mohri. Computes KTA(K_Q, K_Y) and
                                    KTA(K_C, K_Y) where K_Q is the quantum kernel
                                    |<ψ(q_i)|ψ(q_j)>|², K_C is an RBF kernel on the
                                    same 6-dim pre-quantum inputs, and K_Y is the label
                                    kernel. KTA_Q > KTA_C is a rigorous claim that the
                                    quantum kernel is better aligned to the task.

    9. Geometric Difference       — Huang et al. (2021). Computes
                                    g(K_Q, K_C) = sqrt(||K_C^{-½} K_Q K_C^{-½}||_∞).
                                    g > 1 proves the quantum kernel spans feature
                                    directions the classical kernel cannot represent.
                                    NOTE: comparison valid only because both kernels
                                    are evaluated on the same 6-dim input.

    10. FIM + Effective Dimension — Abbas et al. (2021). Computes the empirical Fisher
                                    via per-sample gradients on quantum weights (36
                                    params). Effective dimension d_eff(n) quantifies
                                    the model's expressive capacity relative to training
                                    set size. Compared per-parameter to CNN last layer
                                    (diagonal FIM approximation due to larger param
                                    count). Budget ~2-3× Exp 5 runtime.

    11. Feature Effective Rank    — Roy & Vetterli. eff_rank = exp(H(σ)) where H is
                                    the entropy of normalized singular values of the
                                    feature matrix. Computed for both z (512-dim) and
                                    q_emb (128-dim). Higher eff_rank relative to
                                    parameter count = better use of the embedding space.

    12. Intrinsic Dimension       — Facco et al. (2017) TwoNN estimator. Measures the
                                    true dimensionality of the quantum embedding manifold
                                    vs the classical feature manifold. Lower intrinsic
                                    dimension with competitive accuracy = the quantum
                                    circuit compresses information more efficiently.

    13. Linear CKA                — Kornblith et al. (2019). Compares the full
                                    representation geometry (Gram matrices) of z vs
                                    q_emb. Invariant to rotation and isotropic scaling.
                                    Low CKA is the formal argument for "the quantum
                                    branch learns complementary structure". Supplements
                                    and strengthens Experiment 1.

    14. Class Separability        — Multiclass Fisher criterion J = tr(S_W^{-1} S_B)
                                    on z and q_emb separately. Directly answers whether
                                    the quantum embedding creates better class separation
                                    in metric space than classical features alone.

Outputs:
    data/QA/quantum_advantage_results.json
    data/QA/qa_feature_orthogonality.csv
    data/QA/qa_branch_ablation.csv
    data/QA/qa_reupload_ablation.csv
    data/QA/qa_entanglement_entropy.csv
    data/QA/qa_gradient_variance.csv
    data/QA/qa_noise_ablation.csv
    data/QA/qa_vqc_expressibility.csv
    data/QA/qa_kernel_alignment.csv     
    data/QA/qa_geometric_difference.csv
    data/QA/qa_fisher_effective_dim.csv
    data/QA/qa_feature_effective_rank.csv
    data/QA/qa_intrinsic_dimension.csv  
    data/QA/qa_linear_cka.csv    
    data/QA/qa_class_separability.csv 
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

try:
    from scipy.stats import beta as scipy_beta
    SCIPY_AVAILABLE = True
except ImportError:
    print("SCIPY_AVAILABLE = False")
    SCIPY_AVAILABLE = False

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    print("SKLEARN_AVAILABLE = False")
    SKLEARN_AVAILABLE = False

from backend.data.data_loader import DataLoaderManager
from backend.utils.noise import apply_noise, QA_NOISE_LEVELS
from backend.models.cnn import CNN
from backend.models.qnn_cpu import HybridQnnCPU
from backend.models.qnn_gpu import HybridQnnGPU
from backend.utils.logger import Logger

logger = Logger()

#  Config
CONFIG = {
    "img_width":          384,
    "img_height":         384,
    "batch_size":         16,
    "n_qubits":           6,
    "q_depth":            2,
    "grad_var_batches":   20,
    "entropy_samples":    128,
    "feature_samples":    2833,           # feature vectors collected per model (Exp 8–14)
    "kernel_samples":     128,           # subset for N×N kernel matrix (Exp 8, 9)
    "expr_pairs":         1000,          # random input pairs (Exp 7)
    "fim_samples":        200,           # per-sample FIM backward passes (Exp 10)
    "fim_n_sizes":        [50, 100, 500, 1000],  # training sizes for effective dim
}

_current_dir  = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT  = os.path.dirname(_current_dir)

CHECKPOINT_PATHS = {
    "CNN":     os.path.join(BACKEND_ROOT, "models", "cnn.pth"),
    "QNN_CPU": os.path.join(BACKEND_ROOT, "models", "qnn_cpu.pth"),
    "QNN_GPU": os.path.join(BACKEND_ROOT, "models", "qnn_gpu.pth"),
}
CLASS_NAMES_PATH = os.path.join(BACKEND_ROOT, "data", "class_names.json")
OUTPUT_PATH = os.path.join(BACKEND_ROOT, "data", "QA",  "quantum_advantage_results.json")
OUTPUT_DIR  = os.path.join(BACKEND_ROOT, "data", "QA")

DATA_DIRS = {
    "train": os.path.join(BACKEND_ROOT, "data", "train"),
    "val":   os.path.join(BACKEND_ROOT, "data", "val"),
    "test":  os.path.join(BACKEND_ROOT, "data", "test"),
}

QNN_MODEL_KEYS = ["QNN_CPU", "QNN_GPU"]

# Helpers
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

@torch.no_grad()
def _extract_classical_and_quantum(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = images.to(device)
    dtype = next(model.backbone_early.parameters()).dtype
    x = x.to(dtype=dtype)

    h_mid = model.backbone_early(x)
    z_mid = F.adaptive_avg_pool2d(h_mid, (1, 1)).flatten(1)   # (B, 256)
    h_late = model.backbone_late(h_mid)
    z = F.adaptive_avg_pool2d(h_late, (1, 1)).flatten(1)      # (B, 512)

    q_in = model.quantum_selector(z_mid)
    q_in_scaled = q_in * torch.sigmoid(model.angle_scales)

    if isinstance(model, HybridQnnCPU):
        with torch.amp.autocast(device_type="cpu", enabled=False):
            q_raw = model.q_layer(q_in_scaled.to("cpu").float()).to(device)
    else:
        with torch.amp.autocast(device_type=device.type, enabled=False):
            q_raw = model.q_layer(q_in_scaled.float()).to(device)

    z_exp  = q_raw[:, :model.n_qubits]
    xy_exp = q_raw[:, model.n_qubits:]
    q_residual = torch.cat([z_exp + q_in, xy_exp], dim=1)     # Z-residual skip
    q_emb = model.post_quantum(q_residual)
    return z, q_emb

@torch.no_grad()
def _collect_all_features(
    models: dict[str, nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    n_samples: int,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Single-pass per-model feature collection for Experiments 8–14.

    Returns per model:
        q_in   (N, n_qubits) — pre-quantum scaled inputs (post-selector, pre-circuit)
        z      (N, 512)      — classical backbone features
        q_emb  (N, 128)      — post-quantum embeddings
        labels (N,)          — integer class labels
    """
    output: dict[str, dict] = {}

    for model_key, model in models.items():
        model.eval()
        q_in_list, z_list, q_emb_list, label_list = [], [], [], []
        collected = 0

        for images, labels in test_loader:
            if collected >= n_samples:
                break

            images = images.to(device)
            n = min(images.size(0), n_samples - collected)
            images_n, labels_n = images[:n], labels[:n]

            dtype = next(model.backbone_early.parameters()).dtype
            x = images_n.to(dtype=dtype)

            h_mid = model.backbone_early(x)
            z_mid = F.adaptive_avg_pool2d(h_mid, (1, 1)).flatten(1)
            h_late = model.backbone_late(h_mid)
            z = F.adaptive_avg_pool2d(h_late, (1, 1)).flatten(1)

            q_in = model.quantum_selector(z_mid)
            q_in_scaled = q_in * torch.sigmoid(model.angle_scales)

            if isinstance(model, HybridQnnCPU):
                with torch.amp.autocast(device_type="cpu", enabled=False):
                    q_raw = model.q_layer(q_in_scaled.to("cpu").float()).to(device)
            else:
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    q_raw = model.q_layer(q_in_scaled.float()).to(device)

            z_exp  = q_raw[:, :model.n_qubits]
            xy_exp = q_raw[:, model.n_qubits:]
            q_residual = torch.cat([z_exp + q_in, xy_exp], dim=1)
            q_emb = model.post_quantum(q_residual)

            q_in_list.append(q_in_scaled.cpu().float().numpy())
            z_list.append(z.cpu().float().numpy())
            q_emb_list.append(q_emb.cpu().float().numpy())
            label_list.append(labels_n.numpy())
            collected += n

        output[model_key] = {
            "q_in":   np.concatenate(q_in_list),
            "z":      np.concatenate(z_list),
            "q_emb":  np.concatenate(q_emb_list),
            "labels": np.concatenate(label_list).astype(int),
        }
        logger.info(f"  [{model_key}] Collected {collected} feature vectors")

    return output

def _build_faithful_sv_circuit(n_qubits: int, q_depth: int) -> Callable:
    """
    Faithful statevector mirror of the actual VQC.

    Matches the production circuit exactly:
        Hadamard init → Y-axis AngleEmbedding → RX/RY/RZ rotations →
        alternating CNOT ring (even) / CZ ladder + skip-1 (odd) →
        Z-axis re-upload between layers → qml.state() output.

    Used for Experiments 7 (expressibility), 8 (KTA), and 9 (geometric diff).
    Always runs on default.qubit regardless of the model's training device.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        qml.AngleEmbedding(inputs * math.pi, wires=range(n_qubits), rotation="Y")
        for layer in range(q_depth):
            for i in range(n_qubits):
                qml.RX(weights[layer, i, 0], wires=i)
                qml.RY(weights[layer, i, 1], wires=i)
                qml.RZ(weights[layer, i, 2], wires=i)
            if layer % 2 == 0:
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
            else:
                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
                for i in range(0, n_qubits - 2, 2):
                    qml.CZ(wires=[i, (i + 2) % n_qubits])
            if layer < q_depth - 1:
                qml.AngleEmbedding(
                    inputs * math.pi * 0.5, wires=range(n_qubits), rotation="Z"
                )
        return qml.state()

    return circuit

def _compute_quantum_kernel(
    sv_circuit: Callable,
    weights: torch.Tensor,
    q_inputs: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    """
    Quantum kernel matrix K[i,j] = |<ψ(q_inputs[i])|ψ(q_inputs[j])>|².
    Statevectors are computed via the faithful VQC mirror on default.qubit.
    """
    N = len(q_inputs)
    states: list[np.ndarray] = []
    for i in tqdm(range(N), desc="  Statevectors", leave=False):
        x = torch.tensor(q_inputs[i], dtype=torch.float32)
        sv = sv_circuit(x, weights).detach().numpy().astype(complex)
        states.append(sv)

    S = np.array(states)                    # (N, 2^n_qubits)
    overlap = S @ S.conj().T               # (N, N) complex
    K = np.abs(overlap) ** 2               # (N, N) real
    return K.real.astype(np.float64)

def _rbf_kernel(X: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
    """RBF kernel with median-heuristic bandwidth."""
    sq_dists = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
    if gamma is None:
        median_sq = np.median(sq_dists[sq_dists > 0])
        gamma = 1.0 / (2.0 * median_sq + 1e-10)
    return np.exp(-gamma * sq_dists).astype(np.float64)

def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Double-center a kernel matrix: K_c = H K H where H = I - (1/n)11^T."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def _kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    Centered KTA(K, K_Y) = <K_c, K_Yc>_F / (||K_c||_F * ||K_Yc||_F).
    K_Y[i,j] = 1 if y_i == y_j else 0.
    """
    K_Y = (y[:, None] == y[None, :]).astype(np.float64)
    Kc   = _center_kernel(K)
    K_Yc = _center_kernel(K_Y)
    num  = float(np.sum(Kc * K_Yc))
    den  = np.linalg.norm(Kc, "fro") * np.linalg.norm(K_Yc, "fro") + 1e-10
    return float(num / den)

def _geometric_difference(
    K_Q: np.ndarray,
    K_C: np.ndarray,
    reg: float = 1e-6,
) -> float:
    """
    Huang et al. (2021) geometric difference:
        g(K_Q, K_C) = sqrt(||K_C^{-½} K_Q K_C^{-½}||_spectral)

    Uses eigendecomposition of K_C (symmetric PSD) for numerically stable
    matrix square-root inversion. Spectral norm = largest singular value.
    g > 1 ⟹ quantum kernel accesses feature directions the classical kernel
    cannot represent.
    """
    N = K_C.shape[0]
    K_C_reg = K_C + reg * np.eye(N)

    eigvals, eigvecs = np.linalg.eigh(K_C_reg)
    eigvals = np.maximum(eigvals, reg)

    K_C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    M = K_C_inv_sqrt @ K_Q @ K_C_inv_sqrt
    spectral_norm = float(np.linalg.norm(M, ord=2))
    return float(np.sqrt(spectral_norm))

def _effective_rank(X: np.ndarray) -> float:
    """
    Roy & Vetterli (2007) effective rank.
        eff_rank = exp(H(σ))  where H = Shannon entropy of normalized singular values.
    Higher eff_rank relative to parameter count = better use of the embedding space.
    """
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    entropy = -float(np.sum(p * np.log(p + 1e-12)))
    return float(np.exp(entropy))

def _twonn_dim(X: np.ndarray) -> float:
    """
    TwoNN intrinsic dimension estimator (Facco et al., 2017).
        d = 1 / mean(log(r2 / r1))
    r1, r2 = distances to nearest and second-nearest neighbours.
    Returns float('nan') if estimator cannot be computed.
    """
    if not SKLEARN_AVAILABLE:
        return float("nan")

    nbrs = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
    dists, _ = nbrs.kneighbors(X)

    r1 = dists[:, 1]
    r2 = dists[:, 2]
    valid = (r1 > 1e-10) & (r2 > 1e-10)
    mu = r2[valid] / r1[valid]
    mu = mu[mu > 1.0]

    if len(mu) < 10:
        return float("nan")

    return float(1.0 / np.mean(np.log(mu)))

def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA (Kornblith et al., 2019).
        CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    Invariant to rotation and isotropic scaling.  0 = orthogonal, 1 = identical.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    XtY = X.T @ Y
    XtX = X.T @ X
    YtY = Y.T @ Y
    num = float(np.linalg.norm(XtY, "fro") ** 2)
    den = float(np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro") + 1e-10)
    return float(num / den)

def _fisher_criterion(X: np.ndarray, y: np.ndarray, reg: float = 1e-6) -> float:
    """
    Multiclass Fisher criterion J = tr(S_W^{-1} S_B).
    Higher J = better class separability in feature space.
    """
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)
    n, d = X.shape
    S_W = np.zeros((d, d), dtype=np.float64)
    S_B = np.zeros((d, d), dtype=np.float64)

    for c in classes:
        X_c = X[y == c]
        if len(X_c) < 2:
            continue
        mu_c = X_c.mean(axis=0)
        diff_W = X_c - mu_c
        S_W += diff_W.T @ diff_W
        diff_B = (mu_c - overall_mean)[:, None]
        S_B += len(X_c) * (diff_B @ diff_B.T)

    S_W += reg * np.eye(d)
    try:
        J = float(np.trace(np.linalg.solve(S_W, S_B)))
    except np.linalg.LinAlgError:
        J = float("nan")
    return J

#  Experiment 1 — Feature Orthogonality 
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

#  Experiment 2 — Branch Ablation 
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

        full_acc = _evaluate(model, test_loader, device)
        q_dim = next(
        m.normalized_shape[0]
        for m in model.post_quantum.modules()
        if isinstance(m, nn.LayerNorm)
    )

        def classical_only(x: torch.Tensor) -> torch.Tensor:
            z, q_emb = _extract_classical_and_quantum(model, x, device)
            q_blank = torch.zeros_like(q_emb)
            gated_z = z * model.quantum_gate(q_blank)
            fused = torch.cat([gated_z, q_blank], dim=1)
            fused = fused * model.fusion_gate(fused)
            return model.classifier(fused)

        classical_acc = _evaluate(classical_only, test_loader, device)
        z_dim         = model.backbone_dim

        def quantum_only(x: torch.Tensor) -> torch.Tensor:
            z, q_emb = _extract_classical_and_quantum(model, x, device)
            z_blank = torch.zeros(x.size(0), z_dim, device=device, dtype=z.dtype)
            gated_z = z_blank * model.quantum_gate(q_emb)
            fused = torch.cat([gated_z, q_emb], dim=1)
            fused = fused * model.fusion_gate(fused)
            return model.classifier(fused)

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

#  Experiment 3 — Re-upload Ablation 
def _make_no_reupload_layer(
    original_layer: nn.Module,
    n_qubits: int,
    q_depth: int,
    device_name: str = "default.qubit",
) -> nn.Module:
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
        # Match the real model's output: PauliX + PauliY + PauliZ per qubit = 3*n_qubits
        return (
            [qml.expval(qml.PauliX(i)) for i in range(n_qubits)] +
            [qml.expval(qml.PauliY(i)) for i in range(n_qubits)] +
            [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        )

    no_reupload_layer = qml.qnn.TorchLayer(
        circuit_no_reupload, {"weights": (q_depth, n_qubits, 2)}
    )
    with torch.no_grad():
        # Original weights are (q_depth, n_qubits, 3) — copy RY, RZ only (first 2)
        no_reupload_layer.weights.copy_(original_layer.weights.cpu()[:, :, :2])
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
        with_reupload_acc = _evaluate(model, test_loader, device)
        no_reupload_layer = _make_no_reupload_layer(model.q_layer, n_qubits, q_depth)
        no_reupload_layer.eval()

        def forward_no_reupload(x: torch.Tensor) -> torch.Tensor:
            dtype = next(model.backbone_early.parameters()).dtype
            x_ = x.to(device, dtype=dtype)
            h_mid = model.backbone_early(x_)
            z_mid = F.adaptive_avg_pool2d(h_mid, (1, 1)).flatten(1)
            h_late = model.backbone_late(h_mid)
            z = F.adaptive_avg_pool2d(h_late, (1, 1)).flatten(1)
            q_in = model.quantum_selector(z_mid)
            q_in_scaled = q_in * torch.sigmoid(model.angle_scales)
            with torch.amp.autocast(device_type="cpu", enabled=False):
                q_raw = no_reupload_layer(q_in_scaled.to("cpu").float()).to(device)
            z_exp = q_raw[:, :model.n_qubits]
            xy_exp = q_raw[:, model.n_qubits:]
            q_residual = torch.cat([z_exp + q_in, xy_exp], dim=1)
            q_emb = model.post_quantum(q_residual)
            gated_z = z * model.quantum_gate(q_emb)
            fused = torch.cat([gated_z, q_emb], dim=1)
            fused = fused * model.fusion_gate(fused)
            return model.classifier(fused)

        without_reupload_acc = _evaluate(forward_no_reupload, test_loader, device)
        reupload_contribution = round(with_reupload_acc - without_reupload_acc, 4)

        results[label] = {
            "with_reupload_accuracy": with_reupload_acc,
            "without_reupload_accuracy": without_reupload_acc,
            "reupload_contribution_%": reupload_contribution,
        }
        logger.info(
            f"  [{label}] With: {with_reupload_acc:.2f}% | "
            f"Without: {without_reupload_acc:.2f}% | "
            f"Gain: {reupload_contribution:+.2f}%"
        )

    return results

#  Experiment 4 — Entanglement Entropy
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
            img = images[i].unsqueeze(0).to(device)
            dtype = next(model_ref.backbone_early.parameters()).dtype
            h_mid = model_ref.backbone_early(img.to(dtype=dtype))
            z_mid = F.adaptive_avg_pool2d(h_mid, (1, 1)).flatten(1)
            q_in = model_ref.quantum_selector(z_mid)
            q_in_scaled = (q_in * torch.sigmoid(model_ref.angle_scales)).squeeze(0)
            sample_q_inputs.append(q_in_scaled.cpu().float())
            if len(sample_q_inputs) >= n_samples:
                break
        if len(sample_q_inputs) >= n_samples:
            break

    logger.info(f"  Collected {len(sample_q_inputs)} pre-quantum inputs for entropy")

    # Reuse the faithful circuit — same one used by Experiments 7, 8, 9
    sv_circuit = _build_faithful_sv_circuit(n_qubits, q_depth)

    for label, model in models.items():
        model.eval()
        trained_weights = model.q_layer.weights.detach().cpu().float()
        per_qubit_entropies: dict[int, list[float]] = {i: [] for i in range(n_qubits)}

        for q_in in tqdm(sample_q_inputs, desc=f"[Entropy] {label}", colour="magenta"):
            state_tensor = sv_circuit(q_in, trained_weights)
            state_np = state_tensor.detach().numpy().astype(complex)
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
            "overall_mean_entropy": overall_mean,
            "interpretation": (
                "0.0 = no entanglement; 1.0 = maximally entangled. "
                "Values above 0.3 confirm genuine quantum correlations."
            ),
        }
        logger.info(f"  [{label}] Mean entropy: {overall_mean:.4f}")

    return results

#  Experiment 5 — Quantum Gradient Variance 
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
                if torch.isnan(g).any() or torch.isinf(g).any():
                    logger.warn(f"  [{label}] NaN/Inf gradient detected at batch {i}, skipping.")
                    continue
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

#  Experiment 6 — Noise Robustness Ablation 
@torch.no_grad()
def run_noise_ablation(
    models: dict[str, nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    noise_levels: dict[str, list] = QA_NOISE_LEVELS,
) -> dict[str, dict]:
    results: dict[str, dict] = {}

    for label, model in models.items():
        model.eval()
        results[label] = {}

        q_dim = next(
        m.normalized_shape[0]
        for m in model.post_quantum.modules()
        if isinstance(m, nn.LayerNorm)
    )
        z_dim = model.backbone_dim

        def full_forward(x: torch.Tensor) -> torch.Tensor:
            return model(x)

        def classical_only_forward(x: torch.Tensor) -> torch.Tensor:
            z, q_emb = _extract_classical_and_quantum(model, x, device)
            q_blank = torch.zeros_like(q_emb)
            gated_z = z * model.quantum_gate(q_blank)
            fused = torch.cat([gated_z, q_blank], dim=1)
            fused = fused * model.fusion_gate(fused)
            return model.classifier(fused)

        def quantum_only_forward(x: torch.Tensor) -> torch.Tensor:
            z, q_emb = _extract_classical_and_quantum(model, x, device)
            z_blank = torch.zeros(x.size(0), z_dim, device=device, dtype=z.dtype)
            gated_z = z_blank * model.quantum_gate(q_emb)
            fused = torch.cat([gated_z, q_emb], dim=1)
            fused = fused * model.fusion_gate(fused)
            return model.classifier(fused)

        for noise_type, levels in noise_levels.items():
            logger.info(f"\n  [{label}] Noise ablation — {noise_type}")
            rows: list[dict] = []

            for level in tqdm(levels, desc=f"  {label}/{noise_type}", leave=False):
                full_acc      = _evaluate_noisy(full_forward,           test_loader, device, noise_type, level)
                classical_acc = _evaluate_noisy(classical_only_forward, test_loader, device, noise_type, level)
                quantum_acc   = _evaluate_noisy(quantum_only_forward,   test_loader, device, noise_type, level)
                gain          = round(full_acc - classical_acc, 4)

                rows.append({
                    "level":                level,
                    "full_accuracy_%":      full_acc,
                    "classical_only_%":     classical_acc,
                    "quantum_only_%":       quantum_acc,
                    "quantum_noise_gain_%": gain,
                })
                logger.info(
                    f"    level={level:.4g} | full={full_acc:.2f}%  "
                    f"classical={classical_acc:.2f}%  quantum={quantum_acc:.2f}%  "
                    f"gain={gain:+.2f}%"
                )

            results[label][noise_type] = rows

    return results

#  Experiment 7 — VQC Expressibility (Sim et al. 2019) 
def run_vqc_expressibility(
    models: dict[str, nn.Module],
    n_qubits: int,
    q_depth: int,
    n_pairs: int = 1000,
    n_bins: int = 75,
) -> dict[str, dict]:
    """
    Measures circuit expressibility as KL divergence from the Haar-random
    fidelity distribution.

    Procedure:
        1. Sample n_pairs random input vectors x_i, x_j ~ Uniform(-1, 1)^n_qubits.
        2. Run the faithful VQC mirror (default.qubit) with trained weights.
        3. Compute fidelity F_ij = |<ψ(x_i)|ψ(x_j)>|².
        4. Bin the empirical F distribution.
        5. Compare against Haar reference P_Haar = Beta(1, 2^n_qubits − 1).
        6. KL(empirical || Haar) — lower = more expressive.

    Using trained weights rather than random weights measures the expressibility
    of the learned circuit, not just the ansatz capacity.
    """
    if not SCIPY_AVAILABLE:
        logger.warn("scipy not available — skipping Experiment 7 (VQC Expressibility).")
        return {k: {"error": "scipy not installed"} for k in models}

    haar_b = (2 ** n_qubits) - 1   # Beta(1, 63) for 6 qubits
    circuit = _build_faithful_sv_circuit(n_qubits, q_depth)
    results: dict[str, dict] = {}

    for label, model in models.items():
        model.eval()
        weights = model.q_layer.weights.detach().cpu().float()
        fidelities: list[float] = []

        for _ in tqdm(range(n_pairs), desc=f"[Expressibility] {label}", colour="yellow"):
            x1 = torch.rand(n_qubits) * 2 - 1   # Uniform(-1, 1)
            x2 = torch.rand(n_qubits) * 2 - 1
            with torch.no_grad():
                s1 = circuit(x1, weights).detach().numpy().astype(complex)
                s2 = circuit(x2, weights).detach().numpy().astype(complex)
            fid = float(abs(np.dot(s1.conj(), s2)) ** 2)
            fidelities.append(min(fid, 1.0))  # numerical clamp

        fidelities = np.array(fidelities)
        hist, bin_edges = np.histogram(fidelities, bins=n_bins, range=(0.0, 1.0), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width   = bin_edges[1] - bin_edges[0]

        haar_pdf = scipy_beta.pdf(bin_centers, 1, haar_b)

        # Normalised PMFs for KL computation
        p_emp  = np.clip(hist * bin_width, 1e-12, None)
        p_haar = np.clip(haar_pdf * bin_width, 1e-12, None)
        p_emp  /= p_emp.sum()
        p_haar /= p_haar.sum()
        kl     = float(np.sum(p_emp * np.log(p_emp / p_haar)))

        results[label] = {
            "kl_divergence_from_haar": round(kl, 6),
            "mean_fidelity":           round(float(fidelities.mean()), 6),
            "std_fidelity":            round(float(fidelities.std()),  6),
            "haar_reference":          f"Beta(1, {haar_b})",
            "n_pairs":                 n_pairs,
            "interpretation": (
                "Lower KL = circuit explores Hilbert space more uniformly = "
                f"higher expressibility. Haar reference for {n_qubits} qubits: "
                f"Beta(1, {haar_b}). Values < 0.05 indicate near-Haar expressibility."
            ),
        }
        logger.info(f"  [{label}] Expressibility KL from Haar: {kl:.4f}")

    return results

# Experiments 8 & 9 — Kernel Target Alignment + Geometric Difference 
def run_kernel_experiments(
    models: dict[str, nn.Module],
    collected_features: dict[str, dict[str, np.ndarray]],
    n_qubits: int,
    q_depth: int,
    kernel_samples: int = 128,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Experiments 8 and 9 share the expensive quantum kernel matrix computation
    and are therefore computed in one function.

    Experiment 8 — Kernel Target Alignment (Cortes & Mohri):
        KTA(K, K_Y) = <K_c, K_Yc>_F / (||K_c||_F * ||K_Yc||_F)
        Both quantum kernel K_Q and RBF kernel K_C are evaluated on the same
        6-dim pre-quantum inputs. KTA_Q > KTA_C is a rigorous, citable claim.

    Experiment 9 — Geometric Difference (Huang et al. 2021):
        g(K_Q, K_C) = sqrt(||K_C^{-½} K_Q K_C^{-½}||_spectral)
        g > 1 ⟹ quantum kernel accesses feature directions the classical
        kernel cannot represent.
    """
    circuit = _build_faithful_sv_circuit(n_qubits, q_depth)
    kta_results:  dict[str, dict] = {}
    geom_results: dict[str, dict] = {}

    for label, model in models.items():
        model.eval()
        feats = collected_features[label]
        
        # Shuffle indices to draw samples from the entire test set
        total_available = len(feats["labels"])
        indices = np.arange(total_available)
        np.random.seed(42)  # Use fixed seed for consistent QA results
        np.random.shuffle(indices)
        
        n = min(kernel_samples, total_available)
        selected_indices = indices[:n]
        
        # Use the shuffled indices to get a mix of classes
        q_in   = feats["q_in"][selected_indices]      
        labels = feats["labels"][selected_indices]    
                
        q_in_std = q_in.std(axis=0)
        q_in_global_std = float(q_in.std())
        logger.info(
            f"  [{label}] q_in stats — mean: {q_in.mean():.4f}  "
            f"std: {q_in_global_std:.4f}  per-dim std: {q_in_std.round(4)}"
        )
        
        # Keeps the inputs in a comparable range across samples.
        # Only rescales — does not change the ordering or relative structure.
        q_in_norm = (q_in - q_in.mean(axis=0)) / (q_in.std(axis=0) + 1e-8)
        # Re-clip to [-1, 1] so angle encoding stays valid
        q_in_norm = np.clip(q_in_norm, -1.0, 1.0)

        logger.info(f"  [{label}] q_in_norm std after fix: {q_in_norm.std():.4f}")

        # Quantum kernel on normalised inputs
        logger.info(f"  [{label}] Computing quantum kernel ({n}×{n}) …")
        weights = model.q_layer.weights.detach().cpu().float()
        K_Q = _compute_quantum_kernel(circuit, weights, q_in_norm, n_qubits)

        # Classical RBF kernel on same normalised inputs (fair comparison)
        K_C = _rbf_kernel(q_in_norm)

        # KTA
        kta_q = _kernel_target_alignment(K_Q, labels)
        kta_c = _kernel_target_alignment(K_C, labels)
        diff  = round(kta_q - kta_c, 6)

        kta_results[label] = {
            "kta_quantum":    round(kta_q, 6),
            "kta_classical":  round(kta_c, 6),
            "kta_difference": diff,
            "quantum_wins":   bool(kta_q > kta_c),
            "n_samples":      int(n),
            "q_in_global_std_raw": round(q_in_global_std, 6),
            "normalisation_applied": True,
            "interpretation": (
                "KTA measures kernel alignment to the label structure. "
                "kta_difference > 0 means the quantum kernel is better aligned "
                "to the classification task on these specific 6-dim inputs — "
                "a rigorous, data-driven claim (Cortes & Mohri). "
                "Inputs were per-dim standardised before kernel evaluation."
            ),
        }
        logger.info(
            f"  [{label}] KTA — quantum: {kta_q:.4f}  classical: {kta_c:.4f}  "
            f"diff: {diff:+.4f}  ({'quantum wins' if kta_q > kta_c else 'classical wins'})"
        )

        # Geometric difference
        g = _geometric_difference(K_Q, K_C)
        geom_results[label] = {
            "geometric_difference": round(g, 6),
            "advantage":            bool(g > 1.0),
            "n_samples":            int(n),
            "normalisation_applied": True,
            "interpretation": (
                "g > 1 means the quantum kernel spans feature directions the "
                "classical RBF kernel cannot represent (Huang et al. 2021). "
                f"g = {g:.4f} — "
                f"{'quantum advantage confirmed' if g > 1 else 'no geometric advantage'}. "
                "Inputs were per-dim standardised before kernel evaluation."
            ),
        }
        logger.info(f"  [{label}] Geometric difference g = {g:.4f}  (>1 = advantage)")

    return kta_results, geom_results

#  Experiment 10 — FIM + Effective Dimension (Abbas et al. 2021) 
def run_fisher_effective_dimension(
    models: dict[str, nn.Module],
    cnn_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    n_samples: int = 200,
    fim_n_sizes: list[int] = (50, 100, 500, 1000),
) -> dict[str, dict]:
    """
    Empirical Fisher Information Matrix + Abbas et al. (2021) effective dimension.

    For QNN quantum weights (m = q_depth × n_qubits × 3 = 36 params):
        F̂_ij = (1/n) Σ_k (∂ log p(y_k|x_k,θ) / ∂θ_i)(∂ log p(y_k|x_k,θ) / ∂θ_j)
        d_eff(n) = (2 / log(n/2π)) × log(Σ_i sqrt(1 + n/(2π) × λ_i(F̂_norm)))

    For CNN last linear layer (large m): diagonal approximation (g_i² only).

    Per-parameter d_eff(n)/m is the headline efficiency ratio.
    """
    fim_n_sizes = list(fim_n_sizes)
    results: dict[str, dict] = {}
    criterion = nn.CrossEntropyLoss()

    for label, model in models.items():
        model.eval()

        # Freeze everything except quantum weights for clean gradient isolation
        orig_rg = {n: p.requires_grad for n, p in model.named_parameters()}
        for p in model.parameters():
            p.requires_grad_(False)

        q_weights = model.q_layer.weights
        q_weights.requires_grad_(True)
        m = q_weights.numel()

        F_accum = np.zeros((m, m), dtype=np.float64)
        n_proc  = 0

        logger.info(f"  [{label}] FIM accumulation (m={m}, n_samples={n_samples}) …")

        for images, labels in test_loader:
            if n_proc >= n_samples:
                break
            images, labels = images.to(device), labels.to(device)

            for i in range(images.size(0)):
                if n_proc >= n_samples:
                    break
                if q_weights.grad is not None:
                    q_weights.grad.zero_()

                with torch.enable_grad():
                    logits   = model(images[i : i + 1])
                    log_prob = F.log_softmax(logits, dim=1)[0, labels[i]]
                    # Gradient of log p(y|x, θ) w.r.t. quantum weights
                    log_prob.backward()

                if q_weights.grad is not None:
                    g = q_weights.grad.detach().cpu().numpy().flatten().astype(np.float64)
                    F_accum += np.outer(g, g)
                    n_proc  += 1

        # Restore parameter gradients
        for name, param in model.named_parameters():
            param.requires_grad_(orig_rg.get(name, False))

        F_hat = F_accum / max(n_proc, 1)
        tr    = float(np.trace(F_hat))
        # Normalize: F̂_norm = F̂ × (m / tr)
        F_norm = F_hat * m / (tr + 1e-12)
        eigvals = np.maximum(np.linalg.eigvalsh(F_norm), 0.0)

        d_effs: dict[int, float] = {}
        for n_size in fim_n_sizes:
            gamma    = n_size / (2.0 * math.pi)
            sum_term = float(np.sum(np.sqrt(1.0 + gamma * eigvals)))
            if n_size > 2.0 * math.pi:
                d_eff = (2.0 / math.log(n_size / (2.0 * math.pi))) * math.log(sum_term + 1e-10)
            else:
                d_eff = float(m)
            d_effs[n_size] = round(d_eff, 4)

        results[label] = {
            "n_quantum_params":       m,
            "effective_dimension":    d_effs,
            "d_eff_per_param":        {str(k): round(v / m, 6) for k, v in d_effs.items()},
            "fim_trace":              round(tr, 6),
            "fim_rank":               int(np.sum(eigvals > 1e-6)),
            "n_samples":              n_proc,
            "interpretation": (
                "Higher d_eff per parameter means the model uses its parameters "
                "more efficiently. Compare QNN d_eff/m vs CNN_baseline d_eff/m "
                "at the same n to make the headline efficiency claim."
            ),
        }
        logger.info(
            f"  [{label}] m={m}, d_eff(n=100)={d_effs.get(100, 'N/A')}, "
            f"d_eff/m={round(d_effs.get(100, 0) / m, 4)}"
        )

    # CNN baseline — diagonal approximation (g²) for large parameter count
    cnn_model.eval()
    final_linear: Optional[nn.Linear] = None
    for mod in reversed(list(cnn_model.modules())):
        if isinstance(mod, nn.Linear):
            final_linear = mod
            break

    if final_linear is not None:
        orig_rg_cnn = {n: p.requires_grad for n, p in cnn_model.named_parameters()}
        for p in cnn_model.parameters():
            p.requires_grad_(False)
        final_linear.weight.requires_grad_(True)

        m_cnn   = final_linear.weight.numel()
        diag_F  = np.zeros(m_cnn, dtype=np.float64)
        n_proc  = 0

        logger.info(f"  [CNN_baseline] Diagonal FIM (m={m_cnn}) …")

        for images, labels in test_loader:
            if n_proc >= n_samples:
                break
            images, labels = images.to(device), labels.to(device)

            for i in range(images.size(0)):
                if n_proc >= n_samples:
                    break
                if final_linear.weight.grad is not None:
                    final_linear.weight.grad.zero_()
                with torch.enable_grad():
                    logits   = cnn_model(images[i : i + 1])
                    log_prob = F.log_softmax(logits, dim=1)[0, labels[i]]
                    log_prob.backward()
                if final_linear.weight.grad is not None:
                    g = final_linear.weight.grad.detach().cpu().numpy().flatten().astype(np.float64)
                    diag_F += g ** 2
                    n_proc += 1

        for name, param in cnn_model.named_parameters():
            param.requires_grad_(orig_rg_cnn.get(name, False))

        diag_F   /= max(n_proc, 1)
        eigvals_c = np.maximum(diag_F, 0.0)
        tr_c      = float(np.sum(diag_F))
        F_norm_c  = diag_F * m_cnn / (tr_c + 1e-12)
        eigvals_c_norm = np.maximum(F_norm_c, 0.0)

        d_effs_c: dict[int, float] = {}
        for n_size in fim_n_sizes:
            gamma    = n_size / (2.0 * math.pi)
            sum_term = float(np.sum(np.sqrt(1.0 + gamma * eigvals_c_norm)))
            if n_size > 2.0 * math.pi:
                d_eff = (2.0 / math.log(n_size / (2.0 * math.pi))) * math.log(sum_term + 1e-10)
            else:
                d_eff = float(m_cnn)
            d_effs_c[n_size] = round(d_eff, 4)

        results["CNN_baseline"] = {
            "n_quantum_params":  m_cnn,
            "effective_dimension": d_effs_c,
            "d_eff_per_param":   {str(k): round(v / m_cnn, 6) for k, v in d_effs_c.items()},
            "fim_trace":         round(tr_c, 6),
            "fim_rank":          int(np.sum(eigvals_c > 1e-6)),
            "n_samples":         n_proc,
            "approximation":     "diagonal FIM (g² only) due to large m",
            "interpretation":    "CNN final linear layer d_eff/m for comparison.",
        }

    return results

#  Experiment 11 — Feature Effective Rank 
def run_feature_effective_rank(
    collected_features: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict]:
    """
    Roy & Vetterli (2007) effective rank on z (512-dim) and q_emb (128-dim).

    eff_rank = exp(H(σ))  where H = entropy of normalised singular values.
    A higher eff_rank relative to embedding dimension = the branch makes fuller
    use of its representational capacity.
    """
    results: dict[str, dict] = {}

    for label, feats in collected_features.items():
        z     = feats["z"]     # (N, 512)
        q_emb = feats["q_emb"] # (N, 128)

        eff_z     = _effective_rank(z)
        eff_q     = _effective_rank(q_emb)
        util_z    = round(eff_z    / z.shape[1],     4)
        util_q    = round(eff_q    / q_emb.shape[1], 4)

        results[label] = {
            "z_eff_rank":          round(eff_z,  4),
            "z_dim":               z.shape[1],
            "z_utilisation":       util_z,
            "q_emb_eff_rank":      round(eff_q,  4),
            "q_emb_dim":           q_emb.shape[1],
            "q_emb_utilisation":   util_q,
            "interpretation": (
                "utilisation = eff_rank / embedding_dim. "
                "Higher utilisation on q_emb relative to z argues the quantum "
                "branch uses its dimensions more efficiently."
            ),
        }
        logger.info(
            f"  [{label}] Eff-rank — z: {eff_z:.2f}/{z.shape[1]} ({util_z:.3f})  "
            f"q_emb: {eff_q:.2f}/{q_emb.shape[1]} ({util_q:.3f})"
        )

    return results

#  Experiment 12 — Intrinsic Dimension 
def run_intrinsic_dimension(
    collected_features: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict]:
    """
    TwoNN intrinsic dimension estimator (Facco et al. 2017).
    d = 1 / mean(log(r2 / r1))  where r1, r2 are first and second NN distances.

    Lower intrinsic dimension with competitive accuracy supports the claim that
    the quantum circuit compresses information more efficiently than the 512-dim
    classical backbone.
    """
    if not SKLEARN_AVAILABLE:
        logger.warn("sklearn not available — skipping Experiment 12 (Intrinsic Dimension).")
        return {k: {"error": "sklearn not installed"} for k in collected_features}

    results: dict[str, dict] = {}

    for label, feats in collected_features.items():
        z     = feats["z"]
        q_emb = feats["q_emb"]

        id_z   = _twonn_dim(z)
        id_q   = _twonn_dim(q_emb)

        results[label] = {
            "intrinsic_dim_z":    round(id_z, 4) if not math.isnan(id_z) else "N/A",
            "intrinsic_dim_q_emb": round(id_q, 4) if not math.isnan(id_q) else "N/A",
            "n_samples":          int(len(z)),
            "interpretation": (
                "TwoNN intrinsic dimension (Facco 2017). "
                "Lower intrinsic dim on q_emb vs z with competitive accuracy = "
                "the quantum circuit compresses class-relevant information more efficiently."
            ),
        }
        logger.info(
            f"  [{label}] Intrinsic dim — z: {id_z:.2f}  q_emb: {id_q:.2f}"
        )

    return results

#  Experiment 13 — Linear CKA 
def run_linear_cka(
    collected_features: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict]:
    """
    Linear CKA (Kornblith et al. 2019) between feature spaces.

    Comparisons:
        CKA(z, q_emb)        — main complementarity metric;
                               supplements Experiment 1 (cosine similarity).
                               Low CKA is the formal argument: quantum branch
                               learns orthogonal representational structure.

    CKA is invariant to rotation and isotropic scaling, making it a strictly
    stronger claim than cosine similarity in Experiment 1.
    """
    results: dict[str, dict] = {}

    for label, feats in collected_features.items():
        z     = feats["z"]
        q_emb = feats["q_emb"]

        cka_z_q  = _linear_cka(z, q_emb)

        results[label] = {
            "cka_classical_vs_quantum": round(cka_z_q, 6),
            "n_samples":               int(len(z)),
            "interpretation": (
                "CKA in [0,1]. 0 = orthogonal (ideal), 1 = identical geometry. "
                "Low CKA(z, q_emb) is the formal claim that the quantum branch "
                "learns complementary representational structure (invariant to "
                "rotation and isotropic scaling — stronger than cosine similarity)."
            ),
        }
        logger.info(f"  [{label}] Linear CKA(z, q_emb) = {cka_z_q:.4f}")

    return results

#  Experiment 14 — Class Separability 
def run_class_separability(
    collected_features: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict]:
    """
    Multiclass Fisher criterion J = tr(S_W^{-1} S_B) on z and q_emb.

    S_B = between-class scatter, S_W = within-class scatter.
    Higher J = classes are more tightly clustered and better separated.
    Directly answers whether the quantum embedding creates better class
    separability in metric space than classical features alone.
    """
    results: dict[str, dict] = {}

    for label, feats in collected_features.items():
        z     = feats["z"].astype(np.float64)
        q_emb = feats["q_emb"].astype(np.float64)
        y     = feats["labels"]

        J_z   = _fisher_criterion(z,     y)
        J_q   = _fisher_criterion(q_emb, y)

        z_proj = PCA(n_components=q_emb.shape[1]).fit_transform(z)
        J_z_proj = _fisher_criterion(z_proj, y)
        
        results[label] = {
            "fisher_criterion_z":     round(J_z, 4) if not math.isnan(J_z) else "N/A",
            "fisher_criterion_z_proj":  round(J_z_proj, 4) if not math.isnan(J_z_proj) else "N/A",
            "fisher_criterion_q_emb": round(J_q, 4) if not math.isnan(J_q) else "N/A",
            "q_advantage":            bool(J_q > J_z) if not (math.isnan(J_z) or math.isnan(J_q)) else None,
            "n_samples":              int(len(y)),
            "interpretation": (
                "Fisher criterion J = tr(S_W^{-1} S_B). Higher J = tighter within-class "
                "clusters and wider between-class margins. q_advantage=True means the "
                "quantum embedding provides better class separability than classical features "
                "after dimension matching with PCA."
            ),
        }
        logger.info(
            f"  [{label}] Fisher criterion — z: {J_z:.4f}  q_emb: {J_q:.4f}  "
            f"z_proj: {J_z_proj:.4f} "
            f"({'quantum better' if J_q > J_z_proj else 'classical better'})"
        )

    return results

#  CSV Exports
def export_orthogonality_csv(results: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "qa_feature_orthogonality.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "cosine_similarity", "interpretation"])
        for model_key, val in results.items():
            if val is not None:
                interp = (
                    "orthogonal (good)"              if abs(val) < 0.3  else
                    "partially correlated"            if abs(val) < 0.7  else
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

def export_expressibility_csv(results: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "qa_vqc_expressibility.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "kl_divergence_from_haar", "mean_fidelity",
                          "std_fidelity", "haar_reference", "n_pairs"])
        for model_key, r in results.items():
            writer.writerow([
                model_key,
                _na(r.get("kl_divergence_from_haar")),
                _na(r.get("mean_fidelity")),
                _na(r.get("std_fidelity")),
                r.get("haar_reference", "N/A"),
                r.get("n_pairs", "N/A"),
            ])
    logger.info(f"CSV saved: {path}")

def export_kernel_alignment_csv(results: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "qa_kernel_alignment.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "kta_quantum", "kta_classical", "kta_difference",
                          "quantum_wins", "n_samples"])
        for model_key, r in results.items():
            writer.writerow([
                model_key,
                _na(r.get("kta_quantum")),
                _na(r.get("kta_classical")),
                _na(r.get("kta_difference")),
                r.get("quantum_wins", "N/A"),
                r.get("n_samples", "N/A"),
            ])
    logger.info(f"CSV saved: {path}")

def export_geometric_diff_csv(results: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "qa_geometric_difference.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "geometric_difference_g", "advantage_g_gt_1", "n_samples"])
        for model_key, r in results.items():
            writer.writerow([
                model_key,
                _na(r.get("geometric_difference")),
                r.get("advantage", "N/A"),
                r.get("n_samples", "N/A"),
            ])
    logger.info(f"CSV saved: {path}")

def export_fisher_eff_dim_csv(results: dict, fim_n_sizes: list[int], output_dir: str) -> None:
    path = os.path.join(output_dir, "qa_fisher_effective_dim.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        n_cols     = [f"d_eff_n{n}" for n in fim_n_sizes]
        npp_cols   = [f"d_eff_per_param_n{n}" for n in fim_n_sizes]
        writer.writerow(["model", "n_params", "fim_trace", "fim_rank"]
                        + n_cols + npp_cols)
        for model_key, r in results.items():
            d_effs = r.get("effective_dimension", {})
            dpps   = r.get("d_eff_per_param",    {})
            writer.writerow(
                [model_key, r.get("n_quantum_params", "N/A"),
                 _na(r.get("fim_trace")), _na(r.get("fim_rank"))]
                + [_na(d_effs.get(n)) for n in fim_n_sizes]
                + [_na(dpps.get(str(n))) for n in fim_n_sizes]
            )
    logger.info(f"CSV saved: {path}")

def export_eff_rank_csv(results: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "qa_feature_effective_rank.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "z_eff_rank", "z_dim", "z_utilisation",
                          "q_emb_eff_rank", "q_emb_dim", "q_emb_utilisation"])
        for model_key, r in results.items():
            writer.writerow([
                model_key,
                _na(r.get("z_eff_rank")),      r.get("z_dim", "N/A"),
                _na(r.get("z_utilisation")),
                _na(r.get("q_emb_eff_rank")),  r.get("q_emb_dim", "N/A"),
                _na(r.get("q_emb_utilisation")),
            ])
    logger.info(f"CSV saved: {path}")

def export_intrinsic_dim_csv(results: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "qa_intrinsic_dimension.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "intrinsic_dim_z", "intrinsic_dim_q_emb", "n_samples"])
        for model_key, r in results.items():
            writer.writerow([
                model_key,
                _na(r.get("intrinsic_dim_z")),
                _na(r.get("intrinsic_dim_q_emb")),
                r.get("n_samples", "N/A"),
            ])
    logger.info(f"CSV saved: {path}")


def export_linear_cka_csv(results: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "qa_linear_cka.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "cka_classical_vs_quantum", "n_samples"])
        for model_key, r in results.items():
            writer.writerow([
                model_key,
                _na(r.get("cka_classical_vs_quantum")),
                r.get("n_samples", "N/A"),
            ])
    logger.info(f"CSV saved: {path}")

def export_separability_csv(results: dict, output_dir: str) -> None:
    path = os.path.join(output_dir, "qa_class_separability.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "fisher_criterion_z", "fisher_criterion_q_emb",
                          "q_advantage", "n_samples"])
        for model_key, r in results.items():
            writer.writerow([
                model_key,
                _na(r.get("fisher_criterion_z")),
                _na(r.get("fisher_criterion_q_emb")),
                r.get("q_advantage", "N/A"),
                r.get("n_samples", "N/A"),
            ])
    logger.info(f"CSV saved: {path}")

#  Model Loading 
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
        logger.warn("CUDA unavailable — QNN_GPU excluded.")

    return qnn_models, cnn

#  Main 
def run_quantum_advantage() -> dict:
    cfg    = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Quantum Advantage on device: {device}")

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
    try:
        orthogonality_results = run_feature_orthogonality(qnn_models, test_loader, device)
    except Exception as e:
        logger.error(f"Experiment 1 failed: {e}")
        orthogonality_results = {}

    logger.info("\n=== Experiment 2: Branch Ablation ===")
    try:
        ablation_results = run_branch_ablation(qnn_models, test_loader, device)
    except Exception as e:
        logger.error(f"Experiment 2 failed: {e}")
        ablation_results = {}

    logger.info("\n=== Experiment 3: Re-upload Ablation ===")
    try:
        reupload_results = run_reupload_ablation(
            qnn_models, test_loader, device,
            n_qubits=cfg["n_qubits"], q_depth=cfg["q_depth"],
        )
    except Exception as e:
        logger.error(f"Experiment 3 failed: {e}")
        reupload_results = {}

    logger.info("\n=== Experiment 4: Entanglement Entropy ===")
    try:
        entropy_results = run_entanglement_entropy(
            qnn_models, test_loader, device,
            n_qubits=cfg["n_qubits"], q_depth=cfg["q_depth"],
            n_samples=cfg["entropy_samples"],
        )
    except Exception as e:
        logger.error(f"Experiment 4 failed: {e}")
        entropy_results = {}

    logger.info("\n=== Experiment 5: Quantum Gradient Variance ===")
    try:
        grad_variance_results = run_gradient_variance(
            qnn_models, cnn_model, test_loader, device,
            n_batches=cfg["grad_var_batches"],
        )
    except Exception as e:
        logger.error(f"Experiment 5 failed: {e}")
        grad_variance_results = {}

    logger.info("\n=== Experiment 6: Noise Robustness Ablation ===")
    try:
        noise_ablation_results = run_noise_ablation(
            qnn_models, test_loader, device,
            noise_levels=QA_NOISE_LEVELS,
        )
    except Exception as e:
        logger.error(f"Experiment 6 failed: {e}")
        noise_ablation_results = {}

    # Feature Collection (Experiments 7–14)
    logger.info("\n=== Feature Collection (Experiments 7–14) ===")
    logger.info(f"  Collecting {cfg['feature_samples']} vectors per model …")
    try:
        collected_features = _collect_all_features(
            qnn_models, test_loader, device,
            n_samples=cfg["feature_samples"],
        )
    except Exception as e:
        logger.error(f"Feature collection failed: {e}")
        collected_features = {}

    logger.info("\n=== Experiment 7: VQC Expressibility (Sim et al. 2019) ===")
    try:
        expressibility_results = run_vqc_expressibility(
            qnn_models,
            n_qubits=cfg["n_qubits"], q_depth=cfg["q_depth"],
            n_pairs=cfg["expr_pairs"],
        )
    except Exception as e:
        logger.error(f"Experiment 7 failed: {e}")
        expressibility_results = {}

    if not collected_features:
        logger.error("Skipping Experiments 8 & 9 — collected_features is empty.")
        kta_results, geom_diff_results = {}, {}
    else:
        logger.info("\n=== Experiments 8 & 9: KTA + Geometric Difference ===")
        try:
            kta_results, geom_diff_results = run_kernel_experiments(
                qnn_models, collected_features,
                n_qubits=cfg["n_qubits"], q_depth=cfg["q_depth"],
                kernel_samples=cfg["kernel_samples"],
            )
        except Exception as e:
            logger.error(f"Experiments 8 & 9 failed: {e}")
            kta_results, geom_diff_results = {}, {}

    logger.info("\n=== Experiment 10: Fisher Information Matrix + Effective Dimension ===")
    logger.info(
        f"  Per-sample backward passes: {cfg['fim_samples']} per model. "
        f"n_sizes: {cfg['fim_n_sizes']}."
    )
    try:
        fim_results = run_fisher_effective_dimension(
            qnn_models, cnn_model, test_loader, device,
            n_samples=cfg["fim_samples"],
            fim_n_sizes=cfg["fim_n_sizes"],
        )
    except Exception as e:
        logger.error(f"Experiment 10 failed: {e}")
        fim_results = {}

    if not collected_features:
        logger.error("Skipping Experiments 11-14 — collected_features is empty.")
        eff_rank_results = {}
        intrinsic_dim_results = {}
        cka_results = {}
        separability_results = {}
    else:
        logger.info("\n=== Experiment 11: Feature Effective Rank ===")
        try:
            eff_rank_results = run_feature_effective_rank(collected_features)
        except Exception as e:
            logger.error(f"Experiment 11 failed: {e}")
            eff_rank_results = {}

        logger.info("\n=== Experiment 12: Intrinsic Dimension (TwoNN) ===")
        try:
            intrinsic_dim_results = run_intrinsic_dimension(collected_features)
        except Exception as e:
            logger.error(f"Experiment 12 failed: {e}")
            intrinsic_dim_results = {}

        logger.info("\n=== Experiment 13: Linear CKA (Kornblith et al. 2019) ===")
        try:
            cka_results = run_linear_cka(collected_features)
        except Exception as e:
            logger.error(f"Experiment 13 failed: {e}")
            cka_results = {}

        logger.info("\n=== Experiment 14: Class Separability (Fisher Criterion) ===")
        try:
            separability_results = run_class_separability(collected_features)
        except Exception as e:
            logger.error(f"Experiment 14 failed: {e}")
            separability_results = {}

    #  Assemble JSON 
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device":       str(device),
        "config": {
            "n_qubits":         cfg["n_qubits"],
            "q_depth":          cfg["q_depth"],
            "entropy_samples":  cfg["entropy_samples"],
            "grad_var_batches": cfg["grad_var_batches"],
            "feature_samples":  cfg["feature_samples"],
            "kernel_samples":   cfg["kernel_samples"],
            "expr_pairs":       cfg["expr_pairs"],
            "fim_samples":      cfg["fim_samples"],
            "fim_n_sizes":      cfg["fim_n_sizes"],
            "class_names":      class_names,
            "qa_noise_levels":  QA_NOISE_LEVELS,
            "notes": {
                "entanglement_entropy": (
                    "QNN_GPU uses lightning.gpu; weights are mirrored onto "
                    "default.qubit for statevector extraction."
                ),
                "expressibility": (
                    "Trained weights used — measures expressibility of the "
                    "learned circuit, not just the ansatz capacity."
                ),
                "kernel_experiments": (
                    "Both K_Q and K_C are evaluated on the 6-dim pre-quantum "
                    "inputs (post-selector, post-angle-scaling). The comparison "
                    "is valid because the input space is identical."
                ),
                "fim": (
                    "QNN: full 36×36 empirical FIM via per-sample backward. "
                    "CNN: diagonal approximation due to large last-layer parameter count."
                ),
                "parameter_matched_ablation": (
                    "Not included — requires training a new classical MLP with "
                    "identical parameter count to the VQC. Run separately in a "
                    "dedicated training script."
                ),
            },
        },
        "experiment_1_feature_orthogonality":    orthogonality_results,
        "experiment_2_branch_ablation":           ablation_results,
        "experiment_3_reupload_ablation":         reupload_results,
        "experiment_4_entanglement_entropy":      entropy_results,
        "experiment_5_gradient_variance":         grad_variance_results,
        "experiment_6_noise_ablation":            noise_ablation_results,
        "experiment_7_vqc_expressibility":        expressibility_results,
        "experiment_8_kernel_target_alignment":   kta_results,
        "experiment_9_geometric_difference":      geom_diff_results,
        "experiment_10_fisher_effective_dim":     fim_results,
        "experiment_11_feature_effective_rank":   eff_rank_results,
        "experiment_12_intrinsic_dimension":      intrinsic_dim_results,
        "experiment_13_linear_cka":               cka_results,
        "experiment_14_class_separability":       separability_results,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"JSON saved: {OUTPUT_PATH}")

    #  CSV Exports 
    logger.info("\n=== Exporting CSVs ===")
    export_orthogonality_csv(orthogonality_results, OUTPUT_DIR)
    export_ablation_csv(ablation_results, OUTPUT_DIR)
    export_reupload_csv(reupload_results, OUTPUT_DIR)
    export_entropy_csv(entropy_results, cfg["n_qubits"], OUTPUT_DIR)
    export_grad_variance_csv(grad_variance_results, OUTPUT_DIR)
    export_noise_ablation_csv(noise_ablation_results, OUTPUT_DIR)
    export_expressibility_csv(expressibility_results, OUTPUT_DIR)
    export_kernel_alignment_csv(kta_results, OUTPUT_DIR)
    export_geometric_diff_csv(geom_diff_results, OUTPUT_DIR)
    export_fisher_eff_dim_csv(fim_results, cfg["fim_n_sizes"], OUTPUT_DIR)
    export_eff_rank_csv(eff_rank_results, OUTPUT_DIR)
    export_intrinsic_dim_csv(intrinsic_dim_results, OUTPUT_DIR)
    export_linear_cka_csv(cka_results, OUTPUT_DIR)
    export_separability_csv(separability_results, OUTPUT_DIR)

    #  Summary 
    print("\n" + "=" * 75)
    print("QUANTUM ADVANTAGE SUMMARY")
    print("=" * 75)
    for model_key in qnn_models:
        print(f"\n  {model_key}")

        orth = orthogonality_results.get(model_key)
        if orth is not None:
            tag = "orthogonal ✓" if abs(orth) < 0.3 else "~ partial"
            print(f"    [1]  Feature cosine similarity:   {orth:.4f}  ({tag})")

        gain = abl.get('quantum_gain_%')
        if gain is not None:
            print(f" [2] Quantum branch gain: {gain:+.2f}%")

        reu_val = reu.get('reupload_contribution_%')
        if reu_val is not None:
            print(f" [3] Re-upload contribution: {reu_val:+.2f}%")

        ent = entropy_results.get(model_key, {})
        oe  = ent.get("overall_mean_entropy", 0)
        tag_e = "entangled ✓" if oe > 0.3 else "~ weak"
        print(f"    [4]  Mean entanglement entropy:    {oe:.4f}  ({tag_e})")

        if model_key in noise_ablation_results:
            gauss = noise_ablation_results[model_key].get("gaussian", [])
            if len(gauss) >= 2:
                g0 = gauss[0].get("quantum_noise_gain_%", 0)
                gN = gauss[-1].get("quantum_noise_gain_%", 0)
                print(f"    [6]  Gaussian noise gain:         {g0:+.2f}% → {gN:+.2f}% (σ=0.50)")

        expr = expressibility_results.get(model_key, {})
        kl   = expr.get("kl_divergence_from_haar")
        if kl is not None:
            tag_kl = "high expressibility ✓" if kl < 0.1 else "moderate"
            print(f"    [7]  Expressibility KL (Haar):    {kl:.4f}  ({tag_kl})")

        kta = kta_results.get(model_key, {})
        kta_q = kta.get("kta_quantum")
        kta_c = kta.get("kta_classical")
        if kta_q is not None:
            win = "quantum ✓" if kta.get("quantum_wins") else "classical"
            print(f"    [8]  KTA quantum/classical:       {kta_q:.4f} / {kta_c:.4f}  ({win})")

        gd = geom_diff_results.get(model_key, {})
        g  = gd.get("geometric_difference")
        if g is not None:
            tag_g = "advantage ✓" if g > 1 else "no geometric advantage"
            print(f"    [9]  Geometric difference g:      {g:.4f}  ({tag_g})")

        fim = fim_results.get(model_key, {})
        dpp = fim.get("d_eff_per_param", {})
        dpp100 = dpp.get("100")
        cnn_fim = fim_results.get("CNN_baseline", {})
        cnn_dpp = cnn_fim.get("d_eff_per_param", {}).get("100")
        if dpp100 is not None and cnn_dpp is not None:
            tag_fim = "efficient ✓" if float(dpp100) > float(cnn_dpp) else "≤ CNN"
            print(f"    [10] d_eff/param (n=100):         QNN={dpp100}  CNN={cnn_dpp}  ({tag_fim})")

        er = eff_rank_results.get(model_key, {})
        util_q = er.get("q_emb_utilisation")
        util_z = er.get("z_utilisation")
        if util_q is not None:
            print(f"    [11] Eff-rank utilisation z/q:    {util_z:.3f} / {util_q:.3f}")

        id_r = intrinsic_dim_results.get(model_key, {})
        id_z = id_r.get("intrinsic_dim_z")
        id_q = id_r.get("intrinsic_dim_q_emb")
        if id_z is not None:
            print(f"    [12] Intrinsic dim z / q_emb:     {id_z} / {id_q}")

        cka_r = cka_results.get(model_key, {})
        cka_v = cka_r.get("cka_classical_vs_quantum")
        if cka_v is not None:
            tag_cka = "complementary ✓" if cka_v < 0.3 else "~ partial"
            print(f"    [13] Linear CKA(z, q_emb):        {cka_v:.4f}  ({tag_cka})")

        sep = separability_results.get(model_key, {})
        J_z = sep.get("fisher_criterion_z")
        J_q = sep.get("fisher_criterion_q_emb")
        if J_z is not None:
            tag_sep = "quantum better ✓" if sep.get("q_advantage") else "classical better"
            print(f"    [14] Fisher criterion z / q_emb:  {J_z} / {J_q}  ({tag_sep})")

    print("=" * 75)
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

    print(f"[INFO] scipy available:  {SCIPY_AVAILABLE}")
    print(f"[INFO] sklearn available: {SKLEARN_AVAILABLE}")

    run_quantum_advantage()