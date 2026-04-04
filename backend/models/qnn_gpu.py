"""
Strong fully hybrid model: same ResidualBlock backbone as CnnModule + VQC on pooled
512-D features + fusion classifier. Train-time Gaussian noise for robustness.
"""

from __future__ import annotations

import copy
import json
import math
import time
import argparse
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
import pennylane as qml
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.metrics import classification_report

from utils.logger import Logger
from models.cnn import ResidualBlock
from data.preprocessing import PreProcessing
from data.data_loader import DataLoaderManager


def make_vqc_torch_layer(
    dev: qml.Device, n_qubits: int, q_depth: int, *, reupload: bool = True
) -> qml.qnn.TorchLayer:

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
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


def _quantum_tensor(
    q_layer: nn.Module, q_input: torch.Tensor, input_device: torch.device
) -> torch.Tensor:
    # REMOVE .to("cpu"). If q_layer is built on lightning.gpu and q_input is on cuda,
    # PennyLane handles the transfer or uses the GPU pointer directly.
    return q_layer(q_input.float())


class TrainTimeGaussianNoise(nn.Module):
    def __init__(self, sigma_min: float = 0.02, sigma_max: float = 0.08):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.sigma_max <= 0:
            return x
        lo, hi = min(self.sigma_min, self.sigma_max), max(
            self.sigma_min, self.sigma_max
        )
        sigma = torch.empty(1, device=x.device, dtype=x.dtype).uniform_(lo, hi).item()
        return (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)


class HybridQnnCPU(nn.Module):
    backbone_dim = 512

    def __init__(
        self,
        num_classes: int,
        n_qubits: int = 6,
        q_depth: int = 2,
        q_device_name: str = "lightning.gpu",
        quantum_embed_dim: int = 128,
        train_noise_sigma: Tuple[float, float] = (0.02, 0.08),
        use_train_noise: bool = True,
    ):
        super().__init__()
        self.logger = Logger()
        self.num_classes = num_classes
        self.n_qubits = n_qubits
        self.use_train_noise = use_train_noise
        self.noise = TrainTimeGaussianNoise(*train_noise_sigma)
        self.inference_transform = PreProcessing.get_transforms(
            img_width=384, img_height=384, is_training=False
        )
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
        self.q_device = qml.device(q_device_name, wires=n_qubits, batch_obs=True)
        self.q_layer = make_vqc_torch_layer(
            self.q_device, n_qubits, q_depth, reupload=False
        )
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

    def forward(self, x: torch.Tensor, precomputed_z: bool = False) -> torch.Tensor:
        if precomputed_z:
            # x is already pooled backbone features (shape: [B, 512])
            z = x.to(dtype=next(self.pre_quantum.parameters()).dtype)
        else:
            x = x.to(dtype=next(self.backbone.parameters()).dtype)
            if self.use_train_noise:
                x = self.noise(x)
            h = self.backbone(x)
            z = F.adaptive_avg_pool2d(h, (1, 1)).flatten(1)

        q_in = self.pre_quantum(z)
        q_raw = _quantum_tensor(self.q_layer, q_in, z.device)
        q_emb = self.post_quantum(q_raw)
        return self.classifier(torch.cat([z, q_emb], dim=1))

    @torch.no_grad()
    def _extract_features(self, loader, device) -> torch.utils.data.DataLoader:
        """
        Run all images through the frozen backbone once and return a DataLoader
        over (feature_vector, label) pairs. The backbone is NOT updated.
        """
        self.backbone.eval()
        all_z, all_labels = [], []
        dtype = next(self.backbone.parameters()).dtype

        for images, labels in loader:
            images = images.to(device=device, dtype=dtype)
            h = self.backbone(images)
            z = F.adaptive_avg_pool2d(h, (1, 1)).flatten(1)
            all_z.append(z.cpu())
            all_labels.append(labels.cpu())

        z_tensor = torch.cat(all_z)  # [N, 512]
        y_tensor = torch.cat(all_labels)  # [N]

        dataset = torch.utils.data.TensorDataset(z_tensor, y_tensor)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=loader.batch_size,
            shuffle=True,
            num_workers=0,  # tensors are already in RAM, no disk I/O needed
            pin_memory=True,
        )

    @staticmethod
    def compute_class_weights(dataset) -> torch.Tensor:
        class_counts: dict = {}
        for _, label in dataset.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        total = sum(class_counts.values())
        k = len(class_counts)
        w = [total / (k * class_counts.get(i, 1)) for i in range(k)]
        return torch.tensor(w, dtype=torch.float32)

    def train_model(
        self,
        train_loader,
        optimizer,
        criterion,
        device,
        epoch,
        num_epochs,
        *,
        grad_clip=5.0,
        precomputed_z: bool = False,
    ):
        self.train()
        running_loss = 0.0
        correct = 0
        total = 0
        # AMP is DISABLED — adjoint diff on lightning.gpu is incompatible with float16.
        # autocast would cast TorchLayer inputs to float16; adjoint NaNs on those.
        scaler = torch.amp.GradScaler(device.type, enabled=False)
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", colour="green"
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # No autocast context — run everything in float32
            logits = self(images, precomputed_z=precomputed_z)
            loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if epoch == 0 and batch_idx == 0:
                wq = getattr(self.q_layer, "weights", None)
                g = (
                    "None"
                    if wq is None or wq.grad is None
                    else f"{wq.grad.abs().mean().item():.8f}"
                )
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

    def validate_model(
        self, val_loader, criterion, device, precomputed_z: bool = False
    ):
        self.eval()
        loss_sum = 0.0
        correct = 0
        total = 0
        class_correct: dict = {}
        class_total: dict = {}
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = self(images, precomputed_z=precomputed_z)  # <-- pass flag
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
        print(f"Clean test | Loss: {loss_sum / len(test_loader):.4f} | Acc: {acc:.2f}%")
        try:
            with open(path, encoding="utf-8") as f:
                names = json.load(f)
        except FileNotFoundError:
            names = [f"Class {i}" for i in range(self.num_classes)]
        print(classification_report(y_true, preds, target_names=names, zero_division=0))
        self.logger.info(f"Clean test acc: {acc:.2f}%")
        return acc

    def fit(
        self,
        device,
        train_loader,
        val_loader,
        test_loader,
        *,
        num_epochs: int = 50,
        learning_rate: float = 5e-4,
        quantum_lr_mult: float = 8.0,
        label_smoothing: float = 0.05,
        checkpoint_path: str = "models/qnn_gpu.pth",
        use_class_weights: bool = True,
        skip_prompt: bool = True,
        cache_backbone_features: bool = True,  # <-- new toggle
    ):
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
                {"params": q_params, "lr": learning_rate * quantum_lr_mult},
            ]
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

        # ── Feature caching ──────────────────────────────────────────────────────
        # Extract backbone features once before training starts.
        # Val features never change (no augmentation), so cache once.
        # Train features are re-extracted each epoch only if use_train_noise=True
        # (so that noise sees fresh random draws). If noise is off, cache once.
        if cache_backbone_features:
            print("Pre-extracting val features (once)...")
            feat_val_loader = self._extract_features(val_loader, device)

            re_extract_train_each_epoch = self.use_train_noise
            if not re_extract_train_each_epoch:
                print("Pre-extracting train features (once, noise is off)...")
                feat_train_loader = self._extract_features(train_loader, device)
        # ─────────────────────────────────────────────────────────────────────────

        best_acc, best_state = 0.0, None
        for epoch in range(num_epochs):
            t0 = time.perf_counter()

            # Decide which loaders + mode to use this epoch
            if cache_backbone_features:
                if re_extract_train_each_epoch:
                    # Re-extract every epoch so noise is fresh
                    feat_train_loader = self._extract_features(train_loader, device)
                tr_loss, tr_acc = self.train_model(
                    feat_train_loader,
                    opt,
                    crit,
                    device,
                    epoch,
                    num_epochs,
                    precomputed_z=True,
                )
                va_loss, va_acc, per_cls = self.validate_model(
                    feat_val_loader,
                    crit,
                    device,
                    precomputed_z=True,
                )
            else:
                tr_loss, tr_acc = self.train_model(
                    train_loader,
                    opt,
                    crit,
                    device,
                    epoch,
                    num_epochs,
                )
                va_loss, va_acc, per_cls = self.validate_model(
                    val_loader,
                    crit,
                    device,
                )

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

    def predict(self, image_tensor, device, class_names: Optional[list] = None) -> dict:
        self.eval()
        t0 = time.perf_counter()
        x = image_tensor.to(device)
        with torch.no_grad():
            logits = self(x)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(1)
        return {
            "predicted_index": idx.item(),
            "predicted_class": class_names[idx.item()] if class_names else "Unknown",
            "confidence": round(conf.item(), 4),
            "inference_latency_ms": round((time.perf_counter() - t0) * 1000, 3),
        }

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


@torch.no_grad()
def accuracy_on_loader_with_noise(
    model: nn.Module, data_loader, device: torch.device, noise_sigma: float
) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        if noise_sigma > 0:
            images = (images + torch.randn_like(images) * noise_sigma).clamp(0.0, 1.0)
        pred = model(images).argmax(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return 100.0 * correct / total


def run_noise_sweep(
    model: nn.Module, test_loader, device: torch.device, sigmas: Sequence[float]
) -> List[Tuple[float, float]]:
    rows = []
    for s in sigmas:
        acc = accuracy_on_loader_with_noise(model, test_loader, device, s)
        rows.append((s, acc))
        print(f"  noise sigma={s:.3f} -> acc={acc:.2f}%")
    return rows


def main():
    import os

    parser = argparse.ArgumentParser(description="Strong hybrid QNN")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--quantum-lr-mult", type=float, default=8.0)
    parser.add_argument("--n-qubits", type=int, default=6)
    parser.add_argument("--q-depth", type=int, default=2)
    parser.add_argument(
        "--device-name", type=str, default="lightning.gpu"
    )  # Change to lightning.gpu for GPU training
    parser.add_argument("--no-train-noise", action="store_true")
    parser.add_argument("--noise-min", type=float, default=0.02)
    parser.add_argument("--noise-max", type=float, default=0.08)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="models/qnn_gpu.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manager = DataLoaderManager(
        train_dir="data/train",
        val_dir="data/val",
        test_dir="data/test",
        img_width=384,
        img_height=384,
        batch_size=args.batch_size,
    )
    train_loader, val_loader, test_loader = manager.get_loaders()
    ds = train_loader.dataset
    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    names = [idx_to_class[i] for i in range(len(idx_to_class))]
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    with open("data/class_names.json", "w", encoding="utf-8") as f:
        json.dump(names, f)

    model = HybridQnnCPU(
        num_classes=len(names),
        n_qubits=args.n_qubits,
        q_depth=args.q_depth,
        q_device_name=args.device_name,
        use_train_noise=False,  # To disable noise training make it False
        train_noise_sigma=(args.noise_min, args.noise_max),
    )

    model.fit(
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        quantum_lr_mult=args.quantum_lr_mult,
        checkpoint_path=args.checkpoint,
        skip_prompt=True,
    )
    
    if args.eval_only:
        model.load_model(args.checkpoint, device)
        print("Noise sweep on test set:")
        run_noise_sweep(model, test_loader, device, [0.0, 0.05, 0.10, 0.15, 0.20])
        return


if __name__ == "__main__":
    main()
