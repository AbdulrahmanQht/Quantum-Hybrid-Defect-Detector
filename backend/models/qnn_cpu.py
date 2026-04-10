"""
Strong Hybrid Quantum-Classical architecture for Industrial Defect Detection.

Inspired by:
    - ResNet: Shares the same ResidualBlock backbone as the CNN baseline,
      enabling a fair apples-to-apples comparison of classical vs. quantum features.
    - Data Re-uploading (Pérez-Salinas et al.): Re-embeds classical features
      between VQC layers to increase expressive power without adding qubits.
    - Strongly Entangling Layers: RY/RZ rotations + CNOT ring entanglement
      per depth layer to maximize quantum feature interaction.

Architecture:
    1. Classical Backbone: ResidualBlock stem → 512-D pooled feature vector.
    2. Pre-Quantum Projection: 512 → n_qubits via Linear + LayerNorm + Tanh
       (squashes input into [-1, 1] for stable angle embedding).
    3. VQC (Variational Quantum Circuit): AngleEmbedding + RY/RZ + CNOT ring,
       with data re-uploading between depth layers.
    4. Post-Quantum Projection: n_qubits → 128-D via Linear + LayerNorm + GELU.
    5. Fusion Classifier: Concatenates 512-D classical + 128-D quantum → num_classes.

Training Stabilization:
    - Near-zero quantum weight init (±0.05π) to avoid barren plateaus.
    - Split Adam optimizer with 8× higher LR for quantum parameters.
    - Gradient clipping at 5.0 to suppress quantum gradient spikes.
    - Cosine annealing LR schedule over full training run.
"""
from __future__ import annotations

import os
import copy
import json
import math
import time
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm
import pennylane as qml
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.metrics import classification_report

from backend.utils.logger import Logger
from backend.models.cnn import ResidualBlock
from backend.data.preprocessing import PreProcessing
from backend.data.data_loader import DataLoaderManager

def make_vqc_torch_layer(dev: qml.Device, n_qubits: int, q_depth: int, *, reupload: bool = True) -> qml.qnn.TorchLayer:
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(
            features=inputs * math.pi, wires=range(n_qubits), rotation="Y"
        )
        for layer in range(q_depth):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            if reupload and layer < q_depth - 1:
                qml.AngleEmbedding(
                    features=inputs * math.pi, wires=range(n_qubits), rotation="Y"
                )
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    layer = qml.qnn.TorchLayer(circuit, {"weights": (q_depth, n_qubits, 2)})
    if hasattr(layer, "weights") and layer.weights is not None:
        with torch.no_grad():
            s = 0.05 * math.pi
            layer.weights.uniform_(-s, s)
    return layer


def _quantum_tensor(q_layer: nn.Module, q_input: torch.Tensor, input_device: torch.device) -> torch.Tensor:
    with torch.amp.autocast(device_type=input_device.type, enabled=False):
        return q_layer(q_input.to("cpu").float()).to(input_device)


class HybridQnnCPU(nn.Module):
    """
    CPU variant of HybridQNN using PennyLane's 'default.qubit' simulator.
    Quantum gradients computed via backpropagation through the statevector.
    """
    backbone_dim = 512

    def __init__(self, num_classes: int, n_qubits: int = 6, q_depth: int = 2, q_device_name: str = "default.qubit", quantum_embed_dim: int = 128):
        super().__init__()
        self.logger = Logger()
        self.num_classes = num_classes
        self.n_qubits = n_qubits

        self.inference_transform = PreProcessing.get_transforms(img_width=384, img_height=384, is_training=False)
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
        )
        
        self.pre_quantum = nn.Sequential(
            nn.Linear(self.backbone_dim, n_qubits),
            nn.LayerNorm(n_qubits),
            nn.Tanh(),
        )
        
        self.q_device = qml.device(q_device_name, wires=n_qubits)
        self.q_layer = make_vqc_torch_layer(self.q_device, n_qubits, q_depth, reupload=True)
        
        self.post_quantum = nn.Sequential(
            nn.Linear(n_qubits, quantum_embed_dim),
            nn.LayerNorm(quantum_embed_dim),
            nn.GELU(),
        )
        
        fused = self.backbone_dim + quantum_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.45),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device
        
        x = x.to(dtype=next(self.backbone.parameters()).dtype)
        h = self.backbone(x)
        z = F.adaptive_avg_pool2d(h, (1, 1)).flatten(1)
        
        q_in = self.pre_quantum(z)
        q_raw = _quantum_tensor(self.q_layer, q_in, dev)
        q_emb = self.post_quantum(q_raw)
        
        return self.classifier(torch.cat([z, q_emb], dim=1))

    @staticmethod
    def compute_class_weights(dataset) -> torch.Tensor:
        class_counts: dict = {}
        for _, label in dataset.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        total = sum(class_counts.values())
        k = len(class_counts)
        w = [total / (k * class_counts.get(i, 1)) for i in range(k)]
        return torch.tensor(w, dtype=torch.float32)

    def train_model(self, train_loader, optimizer, criterion, device, epoch, num_epochs, *, grad_clip=5.0,):
        self.train()
        running_loss = 0.0
        correct = 0
        total = 0
        is_cuda = device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type, enabled=is_cuda)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", colour="green")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=is_cuda):
                logits = self(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            if epoch == 0 and batch_idx == 0:
                wq = getattr(self.q_layer, "weights", None)
                g = (
                    "None"
                    if wq is None or wq.grad is None
                    else f"{wq.grad.abs().mean().item():.8f}")
                print(f"\nQuantum grad mean (batch 1): {g}\n")
                
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            _, pred = logits.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            pbar.set_postfix(acc=100.0 * correct / total)
        return running_loss / len(train_loader), 100.0 * correct / total

    def validate_model(self, val_loader, criterion, device):
        self.eval()
        loss_sum = 0.0
        correct = 0
        total = 0
        class_correct: dict = {}
        class_total: dict = {}
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = self(images)
                loss_sum += criterion(logits, labels).item()
                _, pred = logits.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
                for lbl, p in zip(labels, pred):
                    i = lbl.item()
                    class_total[i] = class_total.get(i, 0) + 1
                    if i == p.item():
                        class_correct[i] = class_correct.get(i, 0) + 1
        acc = 100.0 * correct / total
        per_cls = {
            c: 100.0 * class_correct.get(c, 0) / class_total[c]
            for c in sorted(class_total.keys())
        }
        return loss_sum / len(val_loader), acc, per_cls

    def test_model(self, test_loader, criterion, device, path="data/class_names.json"):
        self.eval()
        loss_sum = 0.0
        correct = 0
        total = 0
        preds, y_true = [], []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="[Test]", colour="blue"):
                images, labels = images.to(device), labels.to(device)
                logits = self(images)
                loss_sum += criterion(logits, labels).item()
                _, pred = logits.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
                preds.extend(pred.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        acc = 100.0 * correct / total
        loss = loss_sum / len(test_loader)
        print(f"Clean test | Loss: {loss:.4f} | Acc: {acc:.2f}%")
        try:
            with open(path, encoding="utf-8") as f:
                names = json.load(f)
        except FileNotFoundError:
            names = [f"Class {i}" for i in range(self.num_classes)]

        report = classification_report(
            y_true, preds, target_names=names, zero_division=0
        )
        print(report)
        self.logger.info(f"Clean test | Loss: {loss:.4f} | Acc: {acc:.2f}%")
        self.logger.info(f"Detailed Metrics:\n{report}")
        return acc

    def fit(self, device, train_loader, val_loader, test_loader, *, num_epochs: int = 50, 
            learning_rate: float = 5e-4, quantum_lr_mult: float = 8.0, label_smoothing: float = 0.05,
            checkpoint_path: str = "models/qnn_cpu.pth", use_class_weights: bool = True, skip_prompt: bool = True):
        self.to(device)
        if use_class_weights:
            cw = self.compute_class_weights(train_loader.dataset).to(device)
            crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
        else:
            crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        q_params = list(self.q_layer.parameters())
        q_ids = {id(p) for p in q_params}
        classical = [p for p in self.parameters() if id(p) not in q_ids]
        opt = Adam(
            [
                {"params": classical, "lr": learning_rate},
                {"params": q_params, "lr": learning_rate * quantum_lr_mult}
            ])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
        
        best_acc, best_state = 0.0, None
        for epoch in range(num_epochs):
            t0 = time.perf_counter()
            tr_loss, tr_acc = self.train_model(
                train_loader, opt, crit, device, epoch, num_epochs
            )
            va_loss, va_acc, per_cls = self.validate_model(val_loader, crit, device)
            sched.step()
            dt = time.perf_counter() - t0
            msg = (
                f"=== Epoch {epoch + 1}/{num_epochs} ===\n"
                f" Train | Loss: {tr_loss:.4f} | Acc: {tr_acc:.2f}%\n"
                f" Valid | Loss: {va_loss:.4f} | Acc: {va_acc:.2f}%\n"
                f" Per-class val: {per_cls}\n Time: {dt:.1f}s\n"
            )
            print(msg)
            self.logger.info(msg)
            if va_acc > best_acc:
                best_acc = va_acc
                best_state = copy.deepcopy(self.state_dict())
                torch.save(best_state, checkpoint_path)
                print(f"Saved best val {va_acc:.2f}% -> {checkpoint_path}")
        print(f"\nBest val accuracy: {best_acc:.2f}%")
        if best_state is not None:
            self.load_state_dict(best_state)
        self.test_model(test_loader, crit, device)
        if not skip_prompt and input("Save again? (y/n): ").strip().lower() == "y":
            torch.save(best_state, checkpoint_path)


    def predict(self, image_tensor: torch.Tensor, device: Union[torch.device, str], class_names: Optional[list[str]] = None,) -> dict[str, Union[int, str, float, dict[str, float], list[float]]]:
        """
        Predicts the class of a single image.
        Accepts: torch.tensor.
        """
        self.to(device)
        self.eval()
        start = time.perf_counter()
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            logits = self(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
        latency_ms = (time.perf_counter() - start) * 1000
        idx: int = predicted_idx.item()
        conf: float = confidence.item()
        all_scores: list[float] = probabilities.flatten().tolist()
        result_dict: dict = {
            "predicted_index": idx,
            "predicted_class": class_names[idx] if class_names else "Unknown",
            "confidence": round(conf, 4),
            "all_class_scores": (
                {class_names[i]: round(s, 4) for i, s in enumerate(all_scores)}
                if class_names
                else [round(s, 4) for s in all_scores]
            ),
            "inference_latency_ms": round(latency_ms, 3),
        }
        self.logger.log_results(
            model_name="QNN_CPU_Single_Inference",
            latency=result_dict["inference_latency_ms"],
            confidence=result_dict["confidence"],
            result=str(result_dict["predicted_class"]),
        )
        return result_dict

    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        self.logger.info(f"Saved {path}")

    def load_model(self, path: str, device: torch.device) -> None:
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=device)
        self.load_state_dict(state)
        self.to(device)

if __name__ == "__main__":
    epochs = 50
    batch_size = 16
    lr = 5e-4
    quantum_lr_mult = 8.0
    n_qubits = 6
    q_depth = 2
    device_name = "default.qubit"
    checkpoint = "backend/models/qnn_cpu_noise_training_6_qubits_2_q_depth.pth"

    # PyTorch Device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch is using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # PennyLane Device check
    try:
        q_device = qml.device(device_name, wires=n_qubits)
        print(f"PennyLane device initialized: {device_name}")
    except Exception as e:
        print(f"Error initializing PennyLane device '{device_name}': {e}")
    manager = DataLoaderManager(
        train_dir="backend/data/train",
        val_dir="backend/data/val",
        test_dir="backend/data/test",
        img_width=384,
        img_height=384,
        batch_size=batch_size,
    )
    train_loader, val_loader, test_loader = manager.get_loaders()
    ds = train_loader.dataset
    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    names = [idx_to_class[i] for i in range(len(idx_to_class))]
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    with open("backend/data/class_names.json", "w", encoding="utf-8") as f:
        json.dump(names, f)

    model = HybridQnnCPU(
        num_classes=len(names),
        n_qubits=n_qubits,
        q_depth=q_depth,
        q_device_name=device_name,
    )

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
