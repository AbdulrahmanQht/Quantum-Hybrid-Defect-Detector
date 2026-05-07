from __future__ import annotations

import os
import copy
import json
import time
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.metrics import classification_report

from backend.utils.logger import Logger
from backend.data.preprocessing import PreProcessing
from backend.data.data_loader import DataLoaderManager


class ResidualBlock(nn.Module):
    """
    A standard Residual Block with Depthwise Separable Convolutions.

    This block implements the 'Skip Connection' or 'Shortcut' pattern, which allows
    gradients to flow through the network more easily, preventing the vanishing
    gradient problem in deeper architectures.

    Architecture:
        1. Pointwise Expansion: 1x1 convolution expanding channel dimensions.
        2. Depthwise Conv: 7x7 spatial filtering per channel.
        3. Pointwise Reduction: 1x1 convolution projecting back to target dimensions.
        4. SE Attention: Channel-wise attention gating.
        5. Shortcut: Identity or 1x1 projection to match dimensions.
    Uses the formula: Y = F(X) + W(X) || output = (ConvBlock(x) * SE(ConvBlock(x))) + Shortcut(x)
    """
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        mid_channels = in_channels * expansion

        self.conv = nn.Sequential(
            # Expansion Phase
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            # Depthwise Phase (7x7 Context)
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=7,
                stride=stride,
                padding=3,
                groups=mid_channels,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            # Reduction Phase
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        # Squeeze-and-Excitation Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels // 8, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.conv(x)
        out = out * self.se(out)
        return out + self.shortcut(x)

class CNN(nn.Module):
    """
    Modernized CNN architecture made for Industrial Defect Detection.

    Inspired by:
        - ConvNeXt: Uses large kernels (7x7) and GELU for a Transformer-like field of view.
        - MobileNetV2: Employs Depthwise Separable Convolutions for extreme efficiency.
        - ResNet: Utilizes Skip Connections to maintain training stability over 50+ epochs.

    This model is designed to be lightweight enough for real-time inference via FastAPI
    while remaining robust enough to handle industrial image noise.
    """
    backbone_dim = 512

    def __init__(self, num_classes: int):
        super().__init__()
        self.logger = Logger()
        self.num_classes = num_classes
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

        # Input is 512 (backbone only)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.45),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=next(self.backbone.parameters()).dtype)
        h = self.backbone(x)
        z = F.adaptive_avg_pool2d(h, (1, 1)).flatten(1)  # (B, 512)
        return self.classifier(z)

    @staticmethod
    def compute_class_weights(dataset) -> torch.Tensor:
        class_counts: dict = {}
        for _, label in dataset.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
            
        total = sum(class_counts.values())
        k = len(class_counts)
        w = [total / (k * class_counts.get(i, 1)) for i in range(k)]
        return torch.tensor(w, dtype=torch.float32)

    def train_model(self, train_loader, optimizer, criterion, device, epoch, num_epochs, *, grad_clip=5.0):
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
        per_cls = {c: 100.0 * class_correct.get(c, 0) / class_total[c] for c in sorted(class_total.keys())}
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
        print(f"Clean test | Loss: {loss_sum / len(test_loader):.4f} | Acc: {acc:.2f}%")
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

    def fit(self, device, train_loader, val_loader, test_loader,
        *, num_epochs: int = 50, learning_rate: float = 5e-4, label_smoothing: float = 0.05,
        checkpoint_path: str = "models/cpu_new.pth", use_class_weights: bool = True, skip_prompt: bool = True):
        """Runs training, validation, testing and saves the best model."""
        self.to(device)
        
        if use_class_weights:
            cw = self.compute_class_weights(train_loader.dataset).to(device)
            crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
        else:
            crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        opt = Adam(self.parameters(), lr=learning_rate)
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

    def predict(self, image_tensor: torch.Tensor, device: Union[torch.device, str], 
                class_names: Optional[list[str]] = None,) -> dict[str, Union[int, str, float, dict[str, float], list[float]]]:
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
            model_name="CNN_Single_Inference",
            latency=result_dict["inference_latency_ms"],
            confidence=result_dict["confidence"],
            result=str(result_dict["predicted_class"]),)
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
    epochs = 75
    batch_size = 16
    lr = 5e-4
    checkpoint = "models/cnn_noise_training_75_epochs.pth"
    
    # PyTorch Device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch is using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

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
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    with open("data/class_names.json", "w", encoding="utf-8") as f:
        json.dump(names, f)

    model = CNN(num_classes=len(names))
    
    start_msg = (
        f"\n{'='*30}\n"
        f"CNN BEGINS TRAINING\n"
        f"Epochs: {epochs}\n"
        f"Batch Size: {batch_size}\n"
        f"Learning Rate: {lr}\n"
        f"Device: {device}\n"
        f"Checkpoint Path: {checkpoint}"
        f"{'='*30}"
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
            checkpoint_path=checkpoint,
            skip_prompt=True,
        )