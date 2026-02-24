import time
import json
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from itertools import chain
from tqdm import tqdm
from backend.data.preprocessing import PreProcessing
from backend.data.data_loader import DataLoaderManager
from backend.utils.logger import Logger
from PIL import Image

class CnnModule(nn.Module):
    def __init__(self, num_classes):
        super(CnnModule, self).__init__()
        self.logger = Logger()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),  # Modern, smoother alternative to ReLU
            nn.MaxPool2d(2, 2),  # Input 256 -> 64

            # --- STAGE 1 (Depthwise Separable) ---
            # Depthwise 7x7: Applies a single filter per input channel (groups=64).
            # This captures huge spatial relationships with very few parameters.
            nn.Conv2d(64, 64, kernel_size=7, padding=3, groups=64),
            nn.BatchNorm2d(64),
            # Pointwise 1x1: Mixes the channels and expands them.
            nn.Conv2d(64, 128, kernel_size=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),  # 64 -> 32

            # --- STAGE 2 (Depthwise Separable) ---
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            # --- STAGE 3 (Depthwise Separable) ---
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),  # 16 -> 8

            # --- HEAD (Modern GAP) ---
            # Averages the remaining 8x8 spatial grids into a 1x1 single value per channel.
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),  # Helps reduce overfitting
            # Directly map the 512 features to the number of classes.
            # Skipping the massive middle Linear layer keeps the model fast and robust.
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def train_and_validate(self, device, train_loader, val_loader, test_loader, num_epochs=30, learning_rate=0.001):
        self.to(device)
        self.logger.info(f"Starting Training on {device} for {num_epochs} epochs.")

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

        best_val_acc = 0.0

        for epoch in range(num_epochs):
            # --- TRAINING PHASE ---
            self.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            epoch_start = time.perf_counter()

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", colour='green')
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()

                pbar.set_postfix(loss=loss.item(), acc=100. * correct_train / total_train)

            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct_train / total_train

            # --- VALIDATION PHASE (Val + Test Combined) ---
            self.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            # Recreate the chain iterator EVERY epoch to avoid exhaustion
            combined_loader = chain(val_loader, test_loader)
            # Calculate combined length for loss averaging
            combined_len = len(val_loader) + len(test_loader)

            with torch.no_grad():
                for images, labels in combined_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += predicted.eq(labels).sum().item()

            val_acc = 100. * correct_val / total_val
            avg_val_loss = val_loss / combined_len
            epoch_latency = (time.perf_counter() - epoch_start) * 1000

            scheduler.step(val_acc)
            latency_sec = (time.perf_counter() - epoch_start)

            log_msg = (f"================ Epoch {epoch + 1}/{num_epochs} ================\n"
                       f" Train | Loss: {train_loss:.4f}  | Accuracy: {train_acc:.2f}%\n"
                       f" Valid | Loss: {avg_val_loss:.4f}  | Accuracy: {val_acc:.2f}%\n"
                       f" Time  | {latency_sec:.2f}s\n"
                       f"================================================")
            print(log_msg)
            self.logger.info(log_msg)

            # Save best model based on the combined validation accuracy
            if val_acc > best_val_acc:
                print(
                    f"Combined Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%. Saving model...")
                best_val_acc = val_acc
                self.save_model("best_cnn_model.pth")

        print(f"Training complete. Best Combined Validation Accuracy: {best_val_acc:.2f}%")

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

        transform = PreProcessing.get_transforms(img_width=256, img_height=256, is_training=False)

        # PIL -> Tensor [3, 256, 256] -> [1, 3, 256, 256]
        image_tensor = transform(image).unsqueeze(0).to(device)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    manager = DataLoaderManager(
        train_dir="../data/train",
        val_dir="../data/val",
        test_dir="../data/test",
        img_width=256,
        img_height=256,
        batch_size=32,
    )

    train_loader, val_loader, test_loader = manager.get_loaders()

    class_to_idx = manager.train_loader.dataset.class_to_idx

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names_list = [idx_to_class[i] for i in range(len(idx_to_class))]

    with open("../data/class_names.json", "w") as f:
        json.dump(class_names_list, f)
    print(f"Saved {len(class_names_list)} classes to class_names.json: {class_names_list}")

    num_classes = len(class_names_list)

    model = CnnModule(num_classes = num_classes)
    model.train_and_validate(device, train_loader, val_loader, test_loader, num_epochs = 30, learning_rate = 0.001)
    model.save_model("cnn_model.pth")