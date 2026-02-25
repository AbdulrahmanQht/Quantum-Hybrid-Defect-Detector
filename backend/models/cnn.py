import copy
import time
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch.optim import Adam
import torch.nn.functional as F
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
            1. Depthwise Conv: 7x7 spatial filtering per channel.
            2. Pointwise Conv: 1x1 channel-wise mixing and expansion.
            3. Shortcut: An identity or 1x1 projection to match dimensions.
        Uses the formula: Y = F(X) + W(X) || output = Activation(ConvBlock(x) + Shortcut(x))
        """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=stride, padding=3, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class CnnModule(nn.Module):
    """
    Modernized CNN architecture made for Industrial Defect Detection.

    Inspired by:
        - ConvNeXt: Uses large kernels (7x7) and GELU for a Transformer-like field of view.
        - MobileNetV2: Employs Depthwise Separable Convolutions for extreme efficiency.
        - ResNet: Utilizes Skip Connections to maintain training stability over 50+ epochs.

    This model is designed to be lightweight enough for real-time inference via FastAPI
    while remaining robust enough to handle industrial image noise.
    """
    def __init__(self, num_classes):
        super(CnnModule, self).__init__()
        self.logger = Logger()

        # Initialize transformers once instead of everytime fastAPI calls predict()
        self.inference_transform = PreProcessing.get_transforms(img_width=256, img_height=256, is_training=False)

        self.model = nn.Sequential(
            # Input: [3, 256, 256]
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),  # Modern, smoother alternative to ReLU
            nn.MaxPool2d(2, 2),  # Input 256 -> 64

            # --- BODY (Residual Blocks) ---
            ResidualBlock(64, 128, stride=2),  # 64 -> 32
            ResidualBlock(128, 256, stride=2),  # 32 -> 16
            ResidualBlock(256, 512, stride=2),  # 16 -> 8

            # --- HEAD (Modern GAP) ---
            # Averages the remaining 8x8 spatial grids into a 1x1 single value per channel.
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),  # Helps reduce overfitting
            # Directly map the 512 features to the number of classes.
            # Skipping the massive middle Linear layer keeps the model fast and robust.
            nn.Linear(512, num_classes)
        )
        # Compile for faster training and inference
        # Comment it if you are using Windows
        # self.model = torch.compile(self.model)

    def forward(self, x):
        return self.model(x)

    def train_model(self, train_loader, optimizer, criterion, device, epoch, num_epochs):
        """Handles a single epoch of training."""
        self.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        # Scaler speeds up training and reduce GPU memory usage by using mixed precision (float16) where possible
        # Only use if device = cuda otherwise it would crash on CPU
        is_cuda = device.type == 'cuda'
        scaler = torch.amp.GradScaler(device.type, enabled=is_cuda)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", colour='green')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            # Running the forward pass in mixed precision mode to speed up training and reduce memory usage
            with torch.autocast(device_type=device.type, enabled=is_cuda):
                outputs = self(images)
                loss = criterion(outputs, labels)

            # Scale the loss and backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=loss.item(), acc=100. * correct_train / total_train)

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        return train_loss, train_acc

    def validate_model(self, val_loader, criterion, device):
        """Handles a single epoch of validation."""
        self.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        val_acc = 100. * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss, val_acc

    def test_model(self, test_loader, criterion, device):
        """Test best model."""
        self.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        print("\n--- Starting Final Evaluation on Unseen Test Set ---")
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="[Testing]", colour='blue'):
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()

        test_acc = 100. * correct_test / total_test
        avg_test_loss = test_loss / len(test_loader)
        print(f"Final Test Results | Loss: {avg_test_loss:.4f} | Accuracy: {test_acc:.2f}%")

        self.logger.info(f"Test Evaluation - Loss: {avg_test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        return avg_test_loss, test_acc

    def fit(self, device, train_loader, val_loader, test_loader, num_epochs=50, learning_rate=0.001):
        """Runs training, validation, testing and saves the best model."""
        self.to(device)
        self.logger.info(f"Starting Training on {device} for {num_epochs} epochs.")

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

        best_val_acc = 0.0
        best_model_path = "cnn_model.pth"
        best_model_weights = None

        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()

            # Train
            train_loss, train_acc = self.train_model(train_loader, optimizer, criterion, device, epoch, num_epochs)

            # Validate
            val_loss, val_acc = self.validate_model(val_loader, criterion, device)

            # Update scheduler based on validation accuracy
            scheduler.step(val_acc)

            latency_sec = (time.perf_counter() - epoch_start)

            # Log results
            log_msg = (f"================ Epoch {epoch + 1}/{num_epochs} ================\n"
                       f" Train | Loss: {train_loss:.4f}  | Accuracy: {train_acc:.2f}%\n"
                       f" Valid | Loss: {val_loss:.4f}  | Accuracy: {val_acc:.2f}%\n"
                       f" Time  | {latency_sec:.2f}s\n"
                       f"================================================")
            print(log_msg)
            self.logger.info(log_msg)

            # Store best weights in memory instead of saving to disk immediately
            if val_acc > best_val_acc:
                print(f"New best validation accuracy: {val_acc:.2f}%. Updating weights in memory...")
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(self.state_dict())

        print(f"\nTraining complete. Best Validation Accuracy: {best_val_acc:.2f}%")

        # Load weights from memory, not from disk, since we haven't saved to disk yet
        print("Loading best model weights for the final test evaluation...")
        if best_model_weights is not None:
            self.load_state_dict(best_model_weights)

        # Test model with the best weights
        self.test_model(test_loader, criterion, device)

        # After testing, ask user if they want to save the best model to disk
        save_choice = input(f"\nDo you want to save the best model to '{best_model_path}'? (y/n): ").strip().lower()
        if save_choice in ['y', 'yes']:
            self.save_model(best_model_path)
        else:
            print("Model saving skipped.")

    def predict(self, image_data, device, class_names: list = None) -> dict:
        """
        Predicts the class of a single image.
        Accepts: A PIL Image object OR a string path to an image.
        """
        self.to(device)
        self.eval()
        start = time.perf_counter()

        if isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        else:
            image = image_data

        # PIL -> Tensor [3, 256, 256] -> [1, 3, 256, 256]
        image_tensor = self.inference_transform(image).unsqueeze(0).to(device)

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
                if class_names else [round(s, 4) for s in all_scores]
            ),
            "inference_latency_ms": round(latency_ms, 3),
        }

        self.logger.log_results(
            model_name="Baseline_CNN_Single_Inference",
            latency=result_dict["inference_latency_ms"],
            confidence=result_dict["confidence"],
            result=str(result_dict["predicted_class"])
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
        train_dir="../data/train",
        val_dir="../data/val",
        test_dir="../data/test",
        img_width=256,
        img_height=256,
        batch_size=32,
    )
    train_loader, val_loader, test_loader = manager.get_loaders()

    # Extract and Save Class Names
    class_to_idx = manager.train_loader.dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names_list = [idx_to_class[i] for i in range(len(idx_to_class))]

    with open("../data/class_names.json", "w") as f:
        json.dump(class_names_list, f)
    print(f"Saved {len(class_names_list)} classes to class_names.json: {class_names_list}")

    # InitializeModel
    num_classes = len(class_names_list)
    model = CnnModule(num_classes=num_classes)

    model.logger.info(f"--- NEW EXPERIMENT STARTED ---")
    model.logger.info(f"Model: Classical CNN | Classes: {num_classes} | Device: {device}")

    # Train, Validate, and Test
    model.fit(
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=50,
        learning_rate=0.001)