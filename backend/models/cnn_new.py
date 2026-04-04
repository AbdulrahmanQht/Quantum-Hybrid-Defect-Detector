"""
Ablation baseline: identical backbone + training setup as HybridQnnCPU (qnn_cpu.py)
but without the entire quantum branch.
"""

from __future__ import annotations

import os
import copy
import json
import math
import time
import argparse
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.metrics import classification_report

from utils.logger import Logger
from models.cnn import ResidualBlock
from data.preprocessing import PreProcessing
from data.data_loader import DataLoaderManager


class CnnNew(nn.Module):
    backbone_dim = 512

    def __init__(self, num_classes: int):
        super().__init__()
        self.logger = Logger()
        self.num_classes = num_classes
        self.inference_transform = PreProcessing.get_transforms(
            img_width=384, img_height=384, is_training=False
        )
        # Backbone — byte-for-byte identical to HybridQnnCPU.backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
        )

        # Classifier head — same structure as HybridQnnCPU.classifier
        # Input is 512 (backbone only) instead of 512+128 (fused).
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
    ):
        self.train()
        running_loss = 0.0
        correct = 0
        total = 0
        is_cuda = device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type, enabled=is_cuda)
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", colour="green"
        )
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
        label_smoothing: float = 0.05,
        checkpoint_path: str = "models/cpu_new.pth",
        use_class_weights: bool = True,
        skip_prompt: bool = True,
    ):
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

    @staticmethod
    @torch.no_grad()
    def accuracy_on_loader_with_noise(
        model: nn.Module,
        data_loader,
        device: torch.device,
        noise_sigma: float,
    ) -> float:
        model.eval()
        correct = 0
        total = 0
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            if noise_sigma > 0:
                images = (images + torch.randn_like(images) * noise_sigma).clamp(
                    0.0, 1.0
                )
            pred = model(images).argmax(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
        return 100.0 * correct / total

    @staticmethod
    def run_noise_sweep(
        model: nn.Module,
        test_loader,
        device: torch.device,
        sigmas: Sequence[float],
    ) -> List[Tuple[float, float]]:
        rows = []
        for s in sigmas:
            acc = CnnNew.accuracy_on_loader_with_noise(model, test_loader, device, s)
            rows.append((s, acc))
            print(f"  noise sigma={s:.3f} -> acc={acc:.2f}%")
        return rows


def main():
    import os

    parser = argparse.ArgumentParser(
        description="Ablation CNN — same as HybridQnnCPU but without quantum branch"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="models/cnn_new.pth")
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

    model = CnnNew(num_classes=len(names))

    if args.eval_only:
        crit = nn.CrossEntropyLoss(label_smoothing=0.05)
        model.load_model(args.checkpoint, device)
        model.to(device)
        model.test_model(test_loader, crit, device)
        return
    else:
        model.fit(
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            checkpoint_path=args.checkpoint,
            skip_prompt=True,
        )

if __name__ == "__main__":
    main()
