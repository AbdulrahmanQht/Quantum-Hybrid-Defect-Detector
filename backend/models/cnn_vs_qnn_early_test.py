import json
import time
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.cnn_new import CnnNew
from models.qnn_cpu import HybridQnnCPU
from data.data_loader import DataLoaderManager
from sklearn.metrics import classification_report, confusion_matrix


def run_final_tests():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Data & Names
    manager = DataLoaderManager(
        train_dir="data/train",
        val_dir="data/val",
        test_dir="data/test",
        img_width=384,
        img_height=384,
        batch_size=16,
    )
    _, _, test_loader = manager.get_loaders()

    with open("data/class_names.json", "r") as f:
        class_names = json.load(f)

    # 2. Load Both Models
    qnn = HybridQnnCPU(num_classes=len(class_names))
    qnn.load_model("models/qnn_cpu.pth", device)

    cnn = CnnNew(num_classes=len(class_names))
    cnn.load_model("models/cnn_new.pth", device)

    models = {"Hybrid QNN": qnn, "Ablation CNN": cnn}
    report_data = {}

    # --- TEST 1: CONFUSION MATRICES ---
    print("\n[1/3] Generating Confusion Matrices...")
    for name, model in models.items():
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, lbls in test_loader:
                outputs = model(imgs.to(device))
                y_pred.extend(outputs.argmax(1).cpu().numpy())
                y_true.extend(lbls.numpy())

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names
        )
        plt.title(f"Confusion Matrix: {name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.savefig(f"models/confusion_{name.replace(' ', '_')}.png")
        report_data[f"{name}_acc"] = 100.0 * np.sum(np.diag(cm)) / np.sum(cm)

    # --- TEST 2: NOISE ROBUSTNESS SWEEP (FR-7) ---
    print("\n[2/3] Running Noise Robustness Sweep (PR-2.1)...")
    sigmas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3,0.35, 0.4, 0.45, 0.5]
    noise_results = {"Sigma": sigmas}

    for name, model in models.items():
        acc_list = []
        for s in sigmas:
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    if s > 0:
                        # Adding synthetic noise per FR-7 [cite: 725]
                        images = (images + torch.randn_like(images) * s).clamp(0.0, 1.0)
                    preds = model(images).argmax(1)
                    correct += preds.eq(labels.to(device)).sum().item()
                    total += labels.size(0)
            acc_list.append(100.0 * correct / total)
        noise_results[name] = acc_list

    pd.DataFrame(noise_results).to_csv("models/noise_results.csv", index=False)


if __name__ == "__main__":
    run_final_tests()