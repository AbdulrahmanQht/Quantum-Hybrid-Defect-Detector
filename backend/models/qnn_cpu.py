"""
HybridQnnCPU — Noise-robust hybrid quantum-classical model for industrial defect detection.

Architecture (6 qubits, depth 2):
    1. Split backbone   : early→256D (quantum input), late→512D (classical input)
    2. QuantumSelector  : sigmoid-gated MLP + Tanh projection, 256D→n_qubits
    3. Angle scales     : learnable per-qubit sigmoid-bounded scaling
    4. VQC              : Hadamard init | Y-axis AngleEmbedding | RX+RY+RZ rotations
                          even layers=CNOT ring | odd layers=CZ ladder+skip-1
                          Z-axis re-upload between layers | X+Y+Z measurement (18 outputs)
    5. Q-Residual       : Z_exp + q_in skip connection before post-quantum projection
    6. Q-Gate           : quantum embedding sigmoid-gates classical 512D features
    7. Fusion           : SE attention on cat[gated_classical, q_emb] (640D)
    8. Classifier       : Linear(640→256→num_classes) + BN + GELU + Dropout(0.45)
    9. Aux heads        : classical (512D) + quantum (128D), adaptive loss weight 0.05→0.40

Training:
    - Curriculum noise  : input noise 0.0→0.05 over first 60% of epochs
    - EMA               : quantum params only, decay=0.995, save/restore around val/test
    - Grad clip         : classical=5.0, VQC=2.5
    - Optimizer         : split Adam, quantum_lr_mult=3.5, cosine annealing schedule
    - Loss              : CrossEntropy + class weights + label smoothing=0.05
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from typing import Optional, Union

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.optim import Adam
from tqdm import tqdm

from backend.data.data_loader import DataLoaderManager
from backend.data.preprocessing import PreProcessing
from backend.models.cnn import ResidualBlock
from backend.utils.logger import Logger



# Quantum Feature Selector
class QuantumFeatureSelector(nn.Module):
    """
    Content-dependent soft masking before quantum encoding.

    A sigmoid scorer assigns a relevance weight to each feature dimension.
    Under noise, the scorer suppresses unreliable dimensions before the
    Tanh-bounded projector maps them to qubit angles.

    Args:
        input_dim : Width of the incoming feature vector (256 for mid backbone).
        n_qubits  : Number of output angles, one per qubit.
    """

    def __init__(self, input_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid(),           # per-dimension soft mask in (0, 1)
        )
        self.projector = nn.Sequential(
            nn.Linear(input_dim, n_qubits),
            nn.LayerNorm(n_qubits),
            nn.Tanh(),              # strict bound to (-1, 1) for angle stability
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.projector(z * self.scorer(z))   # (B, n_qubits)


# VQC
def vqc(dev: qml.Device, n_qubits: int, q_depth: int) -> qml.qnn.TorchLayer:
    """
    Build the VQC as a PennyLane TorchLayer.

    Circuit (per forward call):
        Hadamard on all qubits                    → superposition |+>^n
        AngleEmbedding (Y-axis, π-scaled)         → initial encoding
        IQP ZZ(i, i+1) with 0.5π scale           → input-correlated entanglement
        For each depth layer d:
            RX / RY / RZ per qubit                → full rotation set
            if d even:  CNOT ring + IQP ZZ 0.25π → nearest-neighbor + input coupling
            if d odd:   CZ ladder + skip-1 CZ     → longer-range correlations
            if d < q_depth-1:
                AngleEmbedding (Z-axis, 0.5π)     → orthogonal re-upload
        Measure PauliZ, PauliX, PauliY per qubit  → 18 output values

    Weight init: uniform in ±0.05π (near-zero to suppress barren plateaus).
    """

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # 1. Hadamard superposition initialisation
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        # 2. Initial Y-axis angle embedding
        qml.AngleEmbedding(inputs * math.pi, wires=range(n_qubits), rotation="Y")

        # 4. Variational layers
        for layer in range(q_depth):
            # Full rotation set: RX + RY + RZ
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

            # Orthogonal re-upload on Z-axis (different from initial Y embedding)
            if layer < q_depth - 1:
                qml.AngleEmbedding(
                    inputs * math.pi * 0.5, wires=range(n_qubits), rotation="Z"
                )

        # 5. Multi-basis measurement: Z, X, Y → 18 values for 6 qubits
        return (
            [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            + [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
            + [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
        )

    layer = qml.qnn.TorchLayer(circuit, {"weights": (q_depth, n_qubits, 3)})

    # Near-zero init: suppresses barren plateaus at the start of training
    with torch.no_grad():
        layer.weights.uniform_(-0.05 * math.pi, 0.05 * math.pi)

    return layer


def _quantum_forward(
    q_layer: nn.Module,
    q_input: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.amp.autocast(device_type=device.type, enabled=False):
        q_input_cpu = q_input.to("cpu").float()
        circuit = q_layer
        results = torch.stack([
            circuit(q_input_cpu[i]) for i in range(q_input_cpu.shape[0])
        ])
        return results.to(device)


# HybridQnnCPU
class HybridQnnCPU(nn.Module):
    """
    Noise-robust Hybrid QNN for industrial defect detection.
    See module docstring for full architecture and provenance notes.
    """

    mid_dim: int = 256      # intermediate backbone feature dim (quantum input)
    backbone_dim: int = 512 # final backbone feature dim (classical path)

    def __init__(
        self,
        num_classes: int,
        n_qubits: int = 6,
        q_depth: int = 2,
        quantum_embed_dim: int = 128,
        q_device_name: str = "default.qubit",
        ema_decay: float = 0.995,
    ) -> None:
        super().__init__()
        self.logger = Logger()
        self.num_classes = num_classes
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.quantum_embed_dim = quantum_embed_dim
        self.q_device_name = q_device_name
        self.ema_decay = ema_decay
        self._curriculum_noise: float = 0.0
        self._quantum_shadow: dict[str, torch.Tensor] = {}

        self.inference_transform = PreProcessing.get_transforms(
            img_width=384, img_height=384, is_training=False
        )

        # === 1. Split Backbone ===
        # early: 3-ch input → 256-ch intermediate (feeds quantum selector)
        self.backbone_early = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
        )
        # late: 256-ch → 512-ch semantic features (feeds classical head)
        self.backbone_late = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
        )

        # === 2. Quantum Path ===
        # Content-dependent gating before angle encoding
        self.quantum_selector = QuantumFeatureSelector(self.mid_dim, n_qubits)

        # Learnable per-qubit angle scaling as an external Parameter.
        # Using sigmoid(angle_scales) keeps the effective scale in (0, 1),
        # letting the model attenuate noise-sensitive qubit channels.
        self.angle_scales = nn.Parameter(torch.ones(n_qubits))

        # Quantum circuit
        self.q_device = qml.device(self.q_device_name, wires=n_qubits)
        self.q_layer = vqc(self.q_device, n_qubits, q_depth)

        # Post-quantum: 18 inputs (6 × 3 bases), residual on Z channel applied
        # before this layer so the skip already lives in q_residual.
        self.post_quantum = nn.Sequential(
            nn.Linear(n_qubits * 3, quantum_embed_dim),
            nn.LayerNorm(quantum_embed_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # === 3. Classical-Quantum Interaction ===
        # Quantum embedding sigmoid-projects to 512D and gates classical features.
        # Under noise, quantum can effectively zero-out unreliable classical dims.
        self.quantum_gate = nn.Sequential(
            nn.Linear(quantum_embed_dim, self.backbone_dim),
            nn.Sigmoid(),
        )

        # === 4. SE Attention Fusion ===
        fused_dim = self.backbone_dim + quantum_embed_dim  # 640
        self.fusion_gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 8),
            nn.GELU(),
            nn.Linear(fused_dim // 8, fused_dim),
            nn.Sigmoid(),
        )

        # Main classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.45),
            nn.Linear(256, num_classes),
        )

        # === 5. Auxiliary Heads ===
        # Force each branch to remain independently discriminative.
        self.c_aux_head = nn.Linear(self.backbone_dim, num_classes)
        self.q_aux_head = nn.Sequential(
            nn.Linear(quantum_embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

    # Forward
    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = x.device
        x = x.to(dtype=next(self.backbone_early.parameters()).dtype)

        # Split backbone
        h_mid = self.backbone_early(x)
        z_mid = F.adaptive_avg_pool2d(h_mid, (1, 1)).flatten(1)   # (B, 256)
        h_late = self.backbone_late(h_mid)
        z = F.adaptive_avg_pool2d(h_late, (1, 1)).flatten(1)      # (B, 512)

        # Quantum path
        q_in = self.quantum_selector(z_mid)                        # (B, n_qubits), Tanh-bounded

        # Per-qubit learned scaling; sigmoid keeps values in (0, 1)
        q_in_scaled = q_in * torch.sigmoid(self.angle_scales)

        # Curriculum noise injection (only during training)
        if self.training and self._curriculum_noise > 0.0:
            q_in_scaled = q_in_scaled + torch.randn_like(q_in_scaled) * self._curriculum_noise

        q_raw = _quantum_forward(self.q_layer, q_in_scaled, dev)   # (B, 18)

        # Post-quantum residual skip on Z-basis channel only.
        # Z expectations share a [-1,1] range with q_in, so the addition
        # is dimensionally consistent and preserves pre-circuit signal.
        z_exp  = q_raw[:, :self.n_qubits]                          # PauliZ  (B, 6)
        xy_exp = q_raw[:, self.n_qubits:]                          # PauliX+Y (B, 12)
        q_residual = torch.cat([z_exp + q_in, xy_exp], dim=1)      # (B, 18)

        q_emb = self.post_quantum(q_residual)                      # (B, quantum_embed_dim)

        # Quantum gates classical features
        gated_z = z * self.quantum_gate(q_emb)                    # (B, 512)

        # SE attention fusion
        fused = torch.cat([gated_z, q_emb], dim=1)                # (B, 640)
        fused = fused * self.fusion_gate(fused)

        main_logits = self.classifier(fused)

        if return_aux:
            return main_logits, self.c_aux_head(z), self.q_aux_head(q_emb)

        return main_logits


    # EMA helpers (correct save/restore pattern)
    def _update_ema(self) -> None:
        """Accumulate EMA shadow alongside live quantum weights each step."""
        for name, param in self.q_layer.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self._quantum_shadow:
                self._quantum_shadow[name] = param.data.clone()
            else:
                self._quantum_shadow[name].mul_(self.ema_decay).add_(
                    param.data, alpha=1.0 - self.ema_decay
                )

    def _apply_ema(self) -> dict[str, torch.Tensor]:
        """
        Overwrite live quantum weights with EMA shadow.
        Returns a snapshot of the live weights so the caller can restore them.
        """
        live: dict[str, torch.Tensor] = {}
        for name, param in self.q_layer.named_parameters():
            live[name] = param.data.clone()
            if name in self._quantum_shadow:
                param.data.copy_(self._quantum_shadow[name])
        return live

    def _restore_live(self, live: dict[str, torch.Tensor]) -> None:
        """Restore live quantum weights saved by _apply_ema."""
        for name, param in self.q_layer.named_parameters():
            if name in live:
                param.data.copy_(live[name])

    # This method helps model pause and resume training because models takes days to train
    def save_resume_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer,
        scheduler,
        scaler,
        best_acc: float,
        best_state: Optional[dict],
        best_shadow: Optional[dict],
    ) -> None:
        """
        Save a full training-state snapshot so training can be resumed later.
        Saved every epoch; the file is overwritten each time (no accumulation).
        """
        checkpoint = {
            "epoch":        epoch,          # last *completed* epoch (0-indexed)
            "model_state":  self.state_dict(),
            "opt_state":    optimizer.state_dict(),
            "sched_state":  scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "ema_shadow":   {k: v.cpu() for k, v in self._quantum_shadow.items()},
            "best_acc":     best_acc,
            "best_state":   best_state,
            "best_shadow":  best_shadow,
        }
        torch.save(checkpoint, path)
        print(f"Resume checkpoint saved → {path}  (epoch {epoch + 1} done)")
        self.logger.info(f"Resume checkpoint saved → {path}  (epoch {epoch + 1} done)")

    def load_resume_checkpoint(
        self,
        path: str,
        device: torch.device,
        optimizer,
        scheduler,
        scaler,
    ) -> tuple[int, float, Optional[dict], Optional[dict]]:
        """
        Restore a full training-state snapshot.
        Returns (start_epoch, best_acc, best_state, best_shadow).
        """
        try:
            ckpt = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location=device)

        self.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["opt_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        self._quantum_shadow = {
            k: v.to(device) for k, v in ckpt["ema_shadow"].items()
        }

        start_epoch  = ckpt["epoch"] + 1       # resume from the NEXT epoch
        best_acc     = ckpt["best_acc"]
        best_state   = ckpt.get("best_state")
        best_shadow  = ckpt.get("best_shadow")

        self.logger.info(
            f"Resumed from {path} — continuing from epoch {start_epoch + 1}"
            f"  (best val so far: {best_acc:.2f}%)"
        )
        return start_epoch, best_acc, best_state, best_shadow
    
    # Class weights
    @staticmethod
    def compute_class_weights(dataset) -> torch.Tensor:
        counts: dict[int, int] = {}
        for _, label in dataset.samples:
            counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values())
        k = len(counts)
        return torch.tensor(
            [total / (k * counts.get(i, 1)) for i in range(k)],
            dtype=torch.float32,
        )

    # Training step
    def train_model(
        self,
        train_loader,
        optimizer,
        criterion,
        device,
        epoch: int,
        num_epochs: int,
        scaler,
        *,
        classical_params: list,
        grad_clip_classical: float = 5.0,
        grad_clip_quantum: float = 2.5,
    ) -> tuple[float, float]:
        self.train()

        # Curriculum noise: 0.0 → 0.05 linearly over first 60 % of training
        ramp_end = int(num_epochs * 0.6)
        self._curriculum_noise = (
            0.05 * (epoch / max(1, ramp_end))
            if epoch < ramp_end
            else 0.05
        )

        running_loss = 0.0
        correct = 0
        total = 0
        is_cuda = device.type == "cuda"

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            colour="green",
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, enabled=is_cuda):
                main_logits, c_logits, q_logits = self(images, return_aux=True)

                main_loss  = criterion(main_logits, labels)
                c_aux_loss = criterion(c_logits, labels)
                q_aux_loss = criterion(q_logits, labels)

                # Auxiliary weight ramps from 0.05 to 0.40 over training.
                # Both branches are penalised equally; the main head dominates.
                aux_w = min(0.40, 0.05 + (epoch / num_epochs) * 0.40)
                loss = main_loss + aux_w * (c_aux_loss + q_aux_loss) * 0.5

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Tighter gradient clipping for quantum parameters
            nn.utils.clip_grad_norm_(classical_params, max_norm=grad_clip_classical)
            nn.utils.clip_grad_norm_(
                list(self.q_layer.parameters()),
                max_norm=grad_clip_quantum,
            )
            nn.utils.clip_grad_norm_(
                list(self.quantum_selector.parameters())
                + [self.angle_scales]
                + list(self.q_aux_head.parameters()),
                max_norm=grad_clip_classical,
            )

            # Log quantum gradient health on the very first batch
            if epoch == 0 and batch_idx == 0:
                wq = getattr(self.q_layer, "weights", None)
                g = (
                    "None"
                    if (wq is None or wq.grad is None)
                    else f"{wq.grad.abs().mean().item():.8f}"
                )
                print(
                    f"\nQuantum grad mean (epoch 1, batch 1): {g}"
                    f"\nCurriculum noise: {self._curriculum_noise:.4f}"
                    f"\nAux weight: {aux_w:.4f}\n"
                )

            scaler.step(optimizer)
            scaler.update()
            self._update_ema()  # accumulate EMA shadow after each step

            running_loss += loss.item()
            _, pred = main_logits.max(1)
            total   += labels.size(0)
            correct += pred.eq(labels).sum().item()
            pbar.set_postfix(
                acc=f"{100.0 * correct / total:.1f}%",
                q_aux=f"{q_aux_loss.item():.3f}",
                noise=f"{self._curriculum_noise:.3f}",
            )

        return running_loss / len(train_loader), 100.0 * correct / total

    # Validation step
    def validate_model(
        self,
        val_loader,
        criterion,
        device,
    ) -> tuple[float, float, dict[int, float]]:
        self.eval()

        # Temporarily apply EMA weights; save live weights so we can restore them
        live_backup = self._apply_ema()

        loss_sum = 0.0
        correct  = 0
        total    = 0
        class_correct: dict[int, int] = {}
        class_total:   dict[int, int] = {}

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = self(images)
                loss_sum += criterion(logits, labels).item()
                _, pred = logits.max(1)
                total   += labels.size(0)
                correct += pred.eq(labels).sum().item()
                for lbl, p in zip(labels, pred):
                    i = lbl.item()
                    class_total[i]   = class_total.get(i, 0) + 1
                    if i == p.item():
                        class_correct[i] = class_correct.get(i, 0) + 1

        # Restore live weights — training continues from where it left off
        self._restore_live(live_backup)

        acc = 100.0 * correct / total
        per_cls = {
            c: 100.0 * class_correct.get(c, 0) / class_total[c]
            for c in sorted(class_total.keys())
        }
        return loss_sum / len(val_loader), acc, per_cls

    # Test step
    def test_model(
        self,
        test_loader,
        criterion,
        device,
        path: str = "data/class_names.json",
    ) -> float:
        self.eval()

        # Apply EMA weights permanently for final evaluation
        self._apply_ema()

        loss_sum = 0.0
        correct  = 0
        total    = 0
        preds, y_true = [], []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="[Test]", colour="blue"):
                images, labels = images.to(device), labels.to(device)
                logits = self(images)
                loss_sum += criterion(logits, labels).item()
                _, pred = logits.max(1)
                total   += labels.size(0)
                correct += pred.eq(labels).sum().item()
                preds.extend(pred.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        acc  = 100.0 * correct / total
        loss = loss_sum / len(test_loader)
        print(f"\nTest | Loss: {loss:.4f} | Acc: {acc:.2f}%")

        try:
            with open(path, encoding="utf-8") as f:
                names = json.load(f)
        except FileNotFoundError:
            names = [f"Class {i}" for i in range(self.num_classes)]

        report = classification_report(y_true, preds, target_names=names, zero_division=0)
        print(report)
        self.logger.info(f"Test | Loss: {loss:.4f} | Acc: {acc:.2f}%")
        self.logger.info(f"Detailed Metrics:\n{report}")
        return acc

    # Full training loop
    def fit(
        self,
        device,
        train_loader,
        val_loader,
        test_loader,
        *,
        num_epochs: int = 75,
        learning_rate: float = 5e-4,
        quantum_lr_mult: float = 3.5,
        label_smoothing: float = 0.05,
        checkpoint_path: str = "models/qnn_cpu.pth",
        resume_checkpoint_path: str = "models/qnn_cpu_resume.pth",
        use_class_weights: bool = True,
        skip_prompt: bool = True,
    ) -> None:
        self.to(device)

        start_msg = "\n" + "=" * 45 + "\nHYBRID QNN-CPU BEGINS TRAINING\n" + "=" * 45
        print(start_msg)
        self.logger.info("HYBRID QNN-CPU BEGINS TRAINING")

        if use_class_weights:
            cw   = self.compute_class_weights(train_loader.dataset).to(device)
            crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
        else:
            crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Split optimizer: quantum parameters get a higher learning rate
        q_params = (
            list(self.q_layer.parameters())
            + list(self.quantum_selector.parameters())
            + [self.angle_scales]
            + list(self.q_aux_head.parameters())
        )
        q_ids     = {id(p) for p in q_params}
        classical = [p for p in self.parameters() if id(p) not in q_ids]

        opt = Adam([
            {"params": classical, "lr": learning_rate},
            {"params": q_params,  "lr": learning_rate * quantum_lr_mult},
        ])
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

        start_epoch  = 0
        best_acc     = 0.0
        best_state   = None
        best_shadow  = None
        
        if os.path.exists(resume_checkpoint_path):
            print(f"\nResume checkpoint found: {resume_checkpoint_path}")
            start_epoch, best_acc, best_state, best_shadow = self.load_resume_checkpoint(
                resume_checkpoint_path, device, opt, sched, scaler
            )
            print(f"Resuming from epoch {start_epoch + 1}/{num_epochs}  "
                f"(best val so far: {best_acc:.2f}%)\n")
        else:
            print("No resume checkpoint found — starting fresh.\n")

        for epoch in range(start_epoch, num_epochs):
            t0 = time.perf_counter()

            tr_loss, tr_acc = self.train_model(
                train_loader, opt, crit, device, epoch, num_epochs, scaler,
                classical_params=classical,
            )
            va_loss, va_acc, per_cls = self.validate_model(val_loader, crit, device)
            sched.step()

            dt  = time.perf_counter() - t0
            msg = (
                f"=== Epoch {epoch + 1}/{num_epochs} ===\n"
                f" Train | Loss: {tr_loss:.4f} | Acc: {tr_acc:.2f}%\n"
                f" Valid | Loss: {va_loss:.4f} | Acc: {va_acc:.2f}%\n"
                f" Per-class: {per_cls}\n"
                f" Time: {dt:.1f}s\n"
            )
            print(msg)
            self.logger.info(msg)

            if va_acc > best_acc:
                best_acc    = va_acc
                best_state  = copy.deepcopy(self.state_dict())
                best_shadow = copy.deepcopy(self._quantum_shadow)   # ← snapshot shadow at best epoch
                live_backup = self._apply_ema()
                torch.save(self.state_dict(), checkpoint_path)
                self._restore_live(live_backup)
                print(f"Saved best val {va_acc:.2f}% → {checkpoint_path}")
                
            self.save_resume_checkpoint(
            resume_checkpoint_path,
            epoch, opt, sched, scaler,
            best_acc, best_state, best_shadow,
        )

        print(f"\nBest val accuracy: {best_acc:.2f}%")
        if best_state is not None:
            self.load_state_dict(best_state)
            self._quantum_shadow = best_shadow                      # ← restore matching shadow before test
        self.test_model(test_loader, crit, device)

        if not skip_prompt and input("Save again? (y/n): ").strip().lower() == "y":
            torch.save(best_state, checkpoint_path)

    # Single-image inference
    def predict(
        self,
        image_tensor: torch.Tensor,
        device: Union[torch.device, str],
        class_names: Optional[list[str]] = None,
    ) -> dict:
        """
        Predict the defect class of a single pre-processed image tensor.
        Uses EMA weights. Also returns per-branch predictions for diagnostics.
        """
        self.to(device)
        self.eval()
        live_backup = self._apply_ema()

        start        = time.perf_counter()
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            main_logits, c_logits, q_logits = self(image_tensor, return_aux=True)
            probs      = F.softmax(main_logits, dim=1)
            conf, idx  = probs.max(1)

        latency_ms = (time.perf_counter() - start) * 1000
        i = idx.item()

        result = {
            "predicted_index":   i,
            "predicted_class":   class_names[i] if class_names else "Unknown",
            "confidence":        round(conf.item(), 4),
            "classical_pred":    c_logits.argmax(1).item(),
            "quantum_pred":      q_logits.argmax(1).item(),
            "heads_agree":       c_logits.argmax(1).item() == q_logits.argmax(1).item(),
            "all_class_scores": (
                {class_names[j]: round(s, 4) for j, s in enumerate(probs.flatten().tolist())}
                if class_names else [round(s, 4) for s in probs.flatten().tolist()]
            ),
            "inference_latency_ms": round(latency_ms, 3),
        }

        self.logger.log_results(
            model_name="QNN_CPU",
            latency=result["inference_latency_ms"],
            confidence=result["confidence"],
            result=str(result["predicted_class"]),
        )
        self._restore_live(live_backup)
        return result


    # Persistence
    def save_model(self, path: str) -> None:
        """Save model weights (EMA applied before saving)."""
        self._apply_ema()
        torch.save(self.state_dict(), path)
        self.logger.info(f"Saved {path}")

    def load_model(self, path: str, device: torch.device) -> None:
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=device)
        self.load_state_dict(state)
        self.to(device)
        # Initialise EMA shadow from loaded weights
        for name, param in self.q_layer.named_parameters():
            self._quantum_shadow[name] = param.data.clone()


if __name__ == "__main__":
    epochs = 75
    batch_size = 16
    lr = 5e-4
    quantum_lr_mult = 3.5
    n_qubits = 6
    q_depth = 2
    checkpoint = "models/qnn_cpu.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        _test_dev = qml.device("default.qubit", wires=n_qubits)
        print("PennyLane device: default.qubit ✓")
    except Exception as e:
        print(f"PennyLane init error: {e}")

    manager = DataLoaderManager(
        train_dir="data/train",
        val_dir="data/val",
        test_dir="data/test",
        img_width=384,
        img_height=384,
        batch_size=batch_size,
    ) 
    train_loader, val_loader, test_loader = manager.get_loaders()

    ds = train_loader.dataset
    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    names = [idx_to_class[i] for i in range(len(idx_to_class))]

    os.makedirs("data",   exist_ok=True)
    os.makedirs("models", exist_ok=True)
    with open("data/class_names.json", "w", encoding="utf-8") as f:
        json.dump(names, f)

    model = HybridQnnCPU(
        num_classes=len(names),
        n_qubits=n_qubits,
        q_depth=q_depth,
    )

    start_msg = (
        f"\n{'='*52}\n"
        f"HYBRID QNN-CPU  —  TRAINING START\n"
        f"{'-'*52}\n"
        f"Epochs: {epochs}\n"
        f"Batch Size: {batch_size}\n"
        f"Learning Rate: {lr}\n"
        f"Quantum LR Mult: {quantum_lr_mult}\n"
        f"N-Qubits: {n_qubits}\n"
        f"Q-Depth: {q_depth}\n"
        f"PyTorch Device: {device}\n"
        f"Checkpoint: {checkpoint}\n"
        f"{'='*52}"
    )
    print(start_msg)
    model.logger.info(start_msg)

    model.fit(
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=epochs,
        learning_rate=lr,
        quantum_lr_mult=quantum_lr_mult,
        checkpoint_path=checkpoint,
        skip_prompt=True,
    )