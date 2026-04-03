"""
Hybrid Quantum-Classical Neural Network (QNN) for Industrial Defect Detection.

Architecture:
    1. Classical Feature Extractor (Conv2d layers) - compresses 3x384x384 → n_qubits features
    2. Variational Quantum Circuit (PennyLane)     - quantum-enhanced feature processing
    3. Classical Classifier Head (Linear)           - maps quantum output → 6 defect classes

The VQC uses angle encoding to map classical features onto qubit rotations,
then applies parameterized gates (trainable weights) with entanglement.
PennyLane's TorchLayer makes the circuit behave like a standard nn.Module,
so the whole model trains end-to-end with normal PyTorch backprop.
"""

import copy
import time
import json
import torch
import colorama
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import pennylane as qml
from torch.optim import Adam
import torch.nn.functional as F
from utils.logger import Logger
from data.preprocessing import PreProcessing
from data.data_loader import DataLoaderManager
from sklearn.metrics import classification_report

colorama.init()


class HybridQNN(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network.

    Args:
        num_classes:    Number of output classes (default 6 for defect types).
        n_qubits:       Number of qubits in the VQC (default 4).
        q_depth:        Number of variational layers in the VQC (default 2).
        q_device_name:  PennyLane device backend. Use "default.qubit" for CPU
                        simulation or "lightning.gpu" for cuQuantum acceleration.
    """

    def __init__(
        self, num_classes=6, n_qubits=4, q_depth=2, q_device_name="default.qubit"
    ):
        super().__init__()
        self.logger = Logger()

        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.num_classes = num_classes

        # ─── Classical Feature Extractor ───
        # Compresses 3x384x384 grayscale image down to `n_qubits` features.
        # These features become the input to the quantum circuit.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 24 * 24, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),  # <-- Outputs 64 features, NOT 4
            nn.ReLU(),
        )

        # ─── Pre-Quantum Projection ───
        # Separate module: squeezes 64 → n_qubits for the VQC
        self.pre_quantum = nn.Sequential(
            nn.Linear(64, n_qubits),
            nn.Tanh(),  # Bounds to [-1, 1] for angle encoding
        )

        # ─── Quantum Layer (VQC) ───
        self.q_device = qml.device(q_device_name, wires=n_qubits)
        self.q_layer = self._build_quantum_layer()

        # ─── Post-Quantum Expansion ───
        # Expands quantum output back to 64 for fair concatenation
        self.post_quantum = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
        )

        # ─── Classifier Head ───
        # Receives 64 (classical) + 64 (quantum-enhanced) = 128 features
        # Skip connection: classical features ALWAYS reach the classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def _build_quantum_layer(self):
        n_qubits = self.n_qubits
        q_depth = self.q_depth

        @qml.qnode(self.q_device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(features=inputs * np.pi, wires=range(n_qubits), rotation='Y')
            for layer in range(q_depth):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (q_depth, n_qubits, 2)}
        return qml.qnn.TorchLayer(circuit, weight_shapes)

    def to(self, *args, **kwargs):
        self.feature_extractor = self.feature_extractor.to(*args, **kwargs)
        self.pre_quantum = self.pre_quantum.to(*args, **kwargs)
        self.post_quantum = self.post_quantum.to(*args, **kwargs)
        self.classifier = self.classifier.to(*args, **kwargs)
        # q_layer stays on CPU — do NOT move it
        return self

    def forward(self, x):
        x = x.to(dtype=next(self.feature_extractor.parameters()).dtype)
        input_device = x.device

        # Step 1: Classical feature extraction → (B, 64)
        classical_features = self.feature_extractor(x)

        # Step 2: Project to quantum interface → (B, n_qubits)
        q_input = self.pre_quantum(classical_features)

        # Step 3: Quantum circuit on CPU → (B, n_qubits)
        with torch.amp.autocast(device_type=input_device.type, enabled=False):
            q_output = self.q_layer(q_input.to("cpu").float()).to(input_device)

        # Step 4: Expand quantum output → (B, 64)
        q_expanded = self.post_quantum(q_output)

        # Step 5: Skip connection — concatenate classical + quantum
        combined = torch.cat([classical_features, q_expanded], dim=1)  # (B, 128)

        return self.classifier(combined)

    @staticmethod
    def compute_class_weights(dataset):
        """
        Compute inverse-frequency class weights to handle class imbalance.
        """
        class_counts = {}
        for _, label in dataset.samples:
            class_counts[label] = class_counts.get(label, 0) + 1

        total = sum(class_counts.values())
        num_classes = len(class_counts)
        weights = [
            total / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)
        ]
        return torch.tensor(weights, dtype=torch.float32)

    def train_model(
        self, train_loader, optimizer, criterion, device, epoch, num_epochs
    ):
        """Handles a single epoch of training."""
        self.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        # Scaler speeds up training and reduce GPU memory usage by using mixed precision (float16) where possible
        # Only use if device = cuda otherwise it would crash on CPU
        is_cuda = device.type == "cuda"
        scaler = torch.amp.GradScaler(device.type, enabled=is_cuda)

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", colour="green"
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            # images and labels are now correctly unpacked from the batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, enabled=is_cuda):
                outputs = self(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Barren Plateau Check
            if epoch == 0 and batch_idx == 0:
                q_weights = self.q_layer.weights
                print("\n" + "=" * 50)
                print("BARREN PLATEAU CHECK (Epoch 1, Batch 1)")
                if q_weights.grad is not None:
                    grad = q_weights.grad.detach().cpu()
                    mean_g = grad.abs().mean().item()
                    max_g = grad.abs().max().item()
                    std_g = grad.std().item()
                    print(f"  mean|grad| = {mean_g:.8f}")
                    print(f"  max|grad|  = {max_g:.8f}")
                    print(f"  std|grad|  = {std_g:.8f}")
                    print(f"  raw grad tensor:\n{grad}")
                    if mean_g < 1e-6:
                        print(
                            "CONFIRMED BARREN PLATEAU — quantum weights are not updating."
                        )
                    elif mean_g < 1e-4:
                        print("SEVERELY WEAK GRADIENTS — learning will be near zero.")
                    elif mean_g < 1e-2:
                        print("WEAK GRADIENTS — training will be very slow.")
                    else:
                        print("Gradients look healthy.")
                else:
                    print(
                        "  q_layer.weights.grad is None — backprop never reached the quantum layer."
                    )
                    print(
                        "  This means the quantum layer is FULLY DISCONNECTED from the computation graph."
                    )
                print("=" * 50 + "\n")

            torch.nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=1.0
            )  # Clipping restored
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            pbar.set_postfix(acc=100.0 * correct_train / total_train)

        return running_loss / len(train_loader), 100.0 * correct_train / total_train

    def validate_model(self, val_loader, criterion, device):
        """Handles a single epoch of validation."""
        self.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        # For per-class accuracy
        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

                for lbl, pred in zip(labels, predicted):
                    idx = lbl.item()
                    class_total[idx] = class_total.get(idx, 0) + 1
                    if idx == pred.item():
                        class_correct[idx] = class_correct.get(idx, 0) + 1

        val_acc = 100.0 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)

        val_per_class_acc = {}
        for cls_idx in sorted(class_total.keys()):
            cls_correct = class_correct.get(cls_idx, 0)
            cls_total = class_total[cls_idx]
            val_per_class_acc[cls_idx] = 100.0 * cls_correct / cls_total

        return avg_val_loss, val_acc, val_per_class_acc

    def test_model(self, test_loader, criterion, device):
        """Test best model."""
        self.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        # For per-class accuracy
        class_correct = {}
        class_total = {}

        # For detailed classification report
        all_preds = []
        all_labels = []

        print("\n--- Starting Final Evaluation on Unseen Test Set ---")
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="[Testing]", colour="blue"):
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()

                for lbl, pred in zip(labels, predicted):
                    idx = lbl.item()
                    class_total[idx] = class_total.get(idx, 0) + 1
                    if idx == pred.item():
                        class_correct[idx] = class_correct.get(idx, 0) + 1

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = 100.0 * correct_test / total_test
        avg_test_loss = test_loss / len(test_loader)
        test_per_class_acc = {}
        for cls_idx in sorted(class_total.keys()):
            cls_correct = class_correct.get(cls_idx, 0)
            cls_total = class_total[cls_idx]
            test_per_class_acc[cls_idx] = 100.0 * cls_correct / cls_total
        print(
            f"Final Test Results | Loss: {avg_test_loss:.4f} | Accuracy: {test_acc:.2f}%"
        )
        print(f"Per-Class Accuracy: {test_per_class_acc}")

        # Fetch class names for the classification report
        try:
            with open("data/class_names.json", "r") as f:
                class_names = json.load(f)
        except FileNotFoundError:
            class_names = [f"Class {i}" for i in range(len(set(all_labels)))]

            # Generate and print the comprehensive metrics report
        print("\n" + "=" * 45)
        print(" FINAL CLASSIFICATION REPORT (PRECISION/RECALL)")
        print("=" * 45)
        report = classification_report(
            all_labels, all_preds, target_names=class_names, zero_division=0
        )
        print(report)

        # Log the results
        self.logger.info(
            f"Test Evaluation - Loss: {avg_test_loss:.4f}, Accuracy: {test_acc:.2f}%"
        )
        self.logger.info(f"Per-Class Accuracy: {test_per_class_acc}")
        self.logger.info(f"Detailed Metrics:\n{report}")

        return avg_test_loss, test_acc, test_per_class_acc

    def fit(
        self,
        device,
        train_loader,
        val_loader,
        test_loader,
        num_epochs=30,
        learning_rate=0.001,
    ):
        """Runs training, validation, testing and saves the best model."""
        self.to(device)
        self.logger.info(f"Starting Training on {device} for {num_epochs} epochs.")

        class_weights = HybridQNN.compute_class_weights(train_loader.dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        optimizer = Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )

        best_val_acc = 0.0
        best_model_path = "models/qnn_cpu_model.pth"
        checkpoint_path = "models/qnn_cpu_best_checkpoint.pth"
        best_model_weights = None

        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()

            train_loss, train_acc = self.train_model(
                train_loader, optimizer, criterion, device, epoch, num_epochs
            )

            val_loss, val_acc, val_per_class_acc = self.validate_model(
                val_loader, criterion, device
            )

            scheduler.step()
            latency_sec = time.perf_counter() - epoch_start

            log_msg = (
                f"================ Epoch {epoch + 1}/{num_epochs} ================\n"
                f" Train | Loss: {train_loss:.4f}  | Accuracy: {train_acc:.2f}%\n"
                f" Valid | Loss: {val_loss:.4f}  | Accuracy: {val_acc:.2f}%\n"
                f" Per-Class Val: {val_per_class_acc}\n"
                f" Time  | {latency_sec:.2f}s\n"
                f"================================================"
            )
            print(log_msg)
            self.logger.info(log_msg)

            if val_acc > best_val_acc:
                print(
                    f"New best validation accuracy: {val_acc:.2f}%. Updating weights in memory..."
                )
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(self.state_dict())
                torch.save(best_model_weights, checkpoint_path)

        print(f"\nTraining complete. Best Validation Accuracy: {best_val_acc:.2f}%")

        print("Loading best model weights for the final test evaluation...")
        if best_model_weights is not None:
            self.load_state_dict(best_model_weights)

        self.test_model(test_loader, criterion, device)

        save_choice = (
            input(
                f"\nDo you want to save the best model to '{best_model_path}'? (y/n): "
            )
            .strip()
            .lower()
        )
        if save_choice in ["y", "yes"]:
            self.save_model(best_model_path)
        else:
            print("Model saving skipped.")

    def predict(self, image_tensor, device, class_names: list = None) -> dict:
        """
        Predicts the class of a single image.
        Accepts: A PIL Image object OR a string path to an image.
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

        idx = predicted_idx.item()
        conf = confidence.item()
        all_scores = probabilities.flatten().tolist()

        result_dict = {
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
            model_name="QNN_CPU__Single_Inference",
            latency=result_dict["inference_latency_ms"],
            confidence=result_dict["confidence"],
            result=str(result_dict["predicted_class"]),
        )

        return result_dict

    def save_model(self, path):
        try:
            torch.save(self.state_dict(), path)
            self.logger.info(f"Model successfully saved to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")

    def load_model(self, path, device):
        try:
            self.load_state_dict(torch.load(path, map_location=device))
            self.to(device)
            self.eval()
            self.logger.info(f"Model successfully loaded from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")


if __name__ == "__main__":
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing on device: {device}")

    # Load Data
    manager = DataLoaderManager(
        train_dir="data/train",
        val_dir="data/val",
        test_dir="data/test",
        img_width=384,
        img_height=384,
        batch_size=16,
    )
    train_loader, val_loader, test_loader = manager.get_loaders()

    # Extract and Save Class Names
    class_to_idx = manager.train_loader.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names_list = [idx_to_class[i] for i in range(len(idx_to_class))]

    with open("data/class_names.json", "w") as f:
        json.dump(class_names_list, f)
    print(
        f"Saved {len(class_names_list)} classes to class_names.json: {class_names_list}"
    )

    # InitializeModel
    num_classes = len(class_names_list)
    model = HybridQNN(
        num_classes=num_classes, n_qubits=8, q_depth=2, q_device_name="default.qubit"
    )

    model.logger.info(f"--- NEW EXPERIMENT STARTED ---")
    model.logger.info(
        f"Model: Hybrid QNN CPU | Classes: {num_classes} | Device: {device}"
    )

    # Train, Validate, and Test
    model.fit(
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=100,
        learning_rate=0.001,
    )
