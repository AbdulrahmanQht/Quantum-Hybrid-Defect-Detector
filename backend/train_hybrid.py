"""
Training script for the Hybrid Quantum-Classical Neural Network (QNN).

Run from the project root:
    python -m backend.train_hybrid

Saves:
    - Model checkpoints to backend/models/checkpoints/
    - Training logs via the existing Logger
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from backend.models.hybrid_qnn import HybridQNN
from backend.data.data_loader import DataLoaderManager
from backend.utils.logger import Logger


# ─── Configuration ───

CONFIG = {
    # Data
    "train_dir": "backend/data/train/Images",
    "val_dir": "backend/data/val/Images",
    "test_dir": "backend/data/test/Images",
    "img_width": 256,
    "img_height": 256,
    "batch_size": 16,

    # Model
    "num_classes": 6,
    "n_qubits": 4,
    "q_depth": 2,
    "q_device": "default.qubit",  # Change to "lightning.gpu" for cuQuantum

    # Training
    "epochs": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,

    # Output
    "checkpoint_dir": "backend/models/checkpoints",
    "model_name": "hybrid_qnn",
}


def compute_class_weights(dataset):
    """
    Compute inverse-frequency class weights to handle class imbalance.
    Classes range from 1568 to 2959 images - this balances the loss.
    """
    class_counts = {}
    for _, label in dataset.samples:
        class_counts[label] = class_counts.get(label, 0) + 1

    total = sum(class_counts.values())
    num_classes = len(class_counts)
    weights = []

    for i in range(num_classes):
        count = class_counts.get(i, 1)
        # Inverse frequency weighting
        weights.append(total / (num_classes * count))

    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device, logger):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Log progress every 20 batches
        if (batch_idx + 1) % 20 == 0:
            batch_acc = 100.0 * correct / total
            logger.info(
                f"  Batch [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {loss.item():.4f} | Acc: {batch_acc:.1f}%"
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset. Returns loss, accuracy, and per-class metrics."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # For per-class accuracy
    class_correct = {}
    class_total = {}

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Per-class tracking
        for label, pred in zip(labels, predicted):
            label_idx = label.item()
            class_total[label_idx] = class_total.get(label_idx, 0) + 1
            if label_idx == pred.item():
                class_correct[label_idx] = class_correct.get(label_idx, 0) + 1

    eval_loss = running_loss / total
    eval_acc = 100.0 * correct / total

    per_class_acc = {}
    for cls_idx in sorted(class_total.keys()):
        cls_correct = class_correct.get(cls_idx, 0)
        cls_total = class_total[cls_idx]
        per_class_acc[cls_idx] = 100.0 * cls_correct / cls_total

    return eval_loss, eval_acc, per_class_acc


def save_checkpoint(model, optimizer, epoch, val_acc, config, path):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "config": config,
    }, path)


def main():
    logger = Logger()
    config = CONFIG

    # ─── Setup ───
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Hybrid QNN Training | Device: {device}")
    logger.info(f"Config: {json.dumps(config, indent=2)}")

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # ─── Data ───
    logger.info("Loading datasets...")
    data = DataLoaderManager(
        train_dir=config["train_dir"],
        val_dir=config["val_dir"],
        test_dir=config["test_dir"],
        img_width=config["img_width"],
        img_height=config["img_height"],
        batch_size=config["batch_size"],
    )
    train_loader, val_loader, test_loader = data.get_loaders()

    class_names = data.train_dataset.classes
    logger.info(f"Classes: {class_names}")
    logger.info(
        f"Dataset sizes - Train: {len(data.train_dataset)} | "
        f"Val: {len(data.val_dataset)} | Test: {len(data.test_dataset)}"
    )

    # ─── Model ───
    model = HybridQNN(
        num_classes=config["num_classes"],
        n_qubits=config["n_qubits"],
        q_depth=config["q_depth"],
        q_device_name=config["q_device"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters - Total: {total_params} | Trainable: {trainable_params}")

    # ─── Loss with class weights for imbalanced data ───
    class_weights = compute_class_weights(data.train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info(f"Class weights: {class_weights.tolist()}")

    # ─── Optimizer + Scheduler ───
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # ─── Training Loop ───
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    logger.info(f"Starting training for {config['epochs']} epochs...")
    training_start = time.time()

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()
        logger.info(f"Epoch [{epoch}/{config['epochs']}]")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )

        # Validate
        val_loss, val_acc, val_per_class = evaluate(
            model, val_loader, criterion, device
        )

        # LR scheduling based on validation accuracy
        scheduler.step(val_acc)

        epoch_time = time.time() - epoch_start

        # Log results
        logger.info(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%\n"
            f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.1f}%\n"
            f"  Per-class Val Acc: {val_per_class}\n"
            f"  Epoch Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Track history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(
                config["checkpoint_dir"],
                f"{config['model_name']}_best.pth"
            )
            save_checkpoint(model, optimizer, epoch, val_acc, config, best_path)
            logger.info(f"  New best model saved! Val Acc: {val_acc:.1f}%")

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = os.path.join(
                config["checkpoint_dir"],
                f"{config['model_name']}_epoch{epoch}.pth"
            )
            save_checkpoint(model, optimizer, epoch, val_acc, config, ckpt_path)

    total_time = time.time() - training_start
    logger.info(f"Training complete in {total_time / 60:.1f} minutes. Best Val Acc: {best_val_acc:.1f}%")

    # ─── Final Test Evaluation ───
    logger.info("Running final evaluation on test set...")

    # Load best model for testing
    best_ckpt = torch.load(
        os.path.join(config["checkpoint_dir"], f"{config['model_name']}_best.pth"),
        map_location=device,
    )
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc, test_per_class = evaluate(
        model, test_loader, criterion, device
    )

    logger.info(
        f"Test Results:\n"
        f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.1f}%\n"
        f"  Per-class: {test_per_class}"
    )

    for idx, cls_name in enumerate(class_names):
        acc = test_per_class.get(idx, 0.0)
        logger.log_results(
            model_name="HybridQNN",
            latency=f"{total_time / 60:.1f}min",
            confidence=f"{acc:.1f}",
            result=f"Class '{cls_name}' accuracy",
        )

    # ─── Save training history ───
    history_path = os.path.join(config["checkpoint_dir"], f"{config['model_name']}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
