import time
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from itertools import chain
from backend.data.data_loader import DataLoaderManager
from backend.utils.logger import Logger

class CnnModule(nn.Module):
    def __init__(self, num_classes):
        super(CnnModule, self).__init__()
        self.logger = Logger()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 256 → 128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 → 64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 → 32

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 → 16

            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)

    def train_model(self, device, train_loader, num_epochs=10, learning_rate=0.001):
        self.to(device)
        self.logger.info(f"Starting Training on {device} for {num_epochs} epochs.")
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            epoch_start = time.perf_counter()
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            epoch_latency = (time.perf_counter() - epoch_start) * 1000
            log_msg = f"Epoch [{epoch + 1}/{num_epochs}] completed. Loss: {avg_loss:.4f} | Time: {epoch_latency:.2f}ms"
            print(log_msg)
            self.logger.info(log_msg)

    def validate_model(self, device, val_loader, test_loader):
        self.eval()
        correct = 0
        total = 0
        start_time = time.perf_counter()
        with torch.no_grad():
            for images, labels in chain(val_loader, test_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        latency = (time.perf_counter() - start_time) * 1000
        print(f"Validation Accuracy: {accuracy:.4f} | Time: {latency:.2f}ms")
        self.logger.log_results(
            model_name="Baseline_CNN",
            latency=round(latency, 2),
            confidence=round(accuracy, 2),
            result="Validation/Test Phase"
        )

    def predict(self, image: torch.Tensor, device, class_names: list = None) -> dict:
        self.eval()
        image = image.to(device)
        start = time.perf_counter()

        with torch.no_grad():
            logits = self(image)                        # raw scores
            probabilities = F.softmax(logits, dim=1)   # convert to 0–1 range
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        latency_ms = (time.perf_counter() - start) * 1000

        predicted_idx = predicted_idx.item()
        confidence = confidence.item()
        all_scores = probabilities.squeeze().tolist()   # one score per class

        result_dict = {
            "predicted_index": predicted_idx,
            "predicted_class": class_names[predicted_idx] if class_names else None,
            "confidence": round(confidence, 4),
            "all_class_scores": (
                {class_names[i]: round(s, 4) for i, s in enumerate(all_scores)}
                if class_names else all_scores
            ),
            "inference_latency_ms": round(latency_ms, 3),
        }

        self.logger.log_results(
            model_name="Baseline_CNN_Inference",
            latency=result_dict["inference_latency_ms"],
            confidence=result_dict["confidence"],
            result=result_dict["predicted_class"]
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

    num_classes = len(manager.train_loader.dataset.classes)
    print(manager.train_loader.dataset.class_to_idx)

    model = CnnModule(num_classes=num_classes)
    model.train_model(device, train_loader, num_epochs=10, learning_rate=0.001)
    model.validate_model(device, val_loader, test_loader)
    model.save_model("cnn_model.pth")