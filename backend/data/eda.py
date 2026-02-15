import os
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

dataset_path = r"data\train\Images"
classes = os.listdir(dataset_path)

# 1. Class Distribution
counts = {c: len(os.listdir(os.path.join(dataset_path, c))) for c in classes}
plt.figure(figsize=(10, 5))
sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="viridis")
plt.title("Industrial Defect Class Distribution")
plt.ylabel("Number of Images")
plt.show()

# 2. Visual Sample Grid
fig, axes = plt.subplots(len(classes), 5, figsize=(15, 10))
for i, cls in enumerate(classes):
    img_names = os.listdir(os.path.join(dataset_path, cls))[:5]
    for j, img_name in enumerate(img_names):
        img_path = os.path.join(dataset_path, cls, img_name)
        img = Image.open(img_path)
        axes[i, j].imshow(img)
        axes[i, j].axis("off")
        if j == 0:
            axes[i, j].set_ylabel(cls, rotation=0, labelpad=50, fontweight="bold")
plt.tight_layout()
plt.show()

# 3. Pixel Intensity Distribution (Check for Noise/Contrast)
sample_img = Image.open(
    os.path.join(
        dataset_path, classes[0], os.listdir(os.path.join(dataset_path, classes[0]))[0]
    )
)
img_array = np.array(sample_img).flatten()
plt.figure(figsize=(8, 4))
plt.hist(img_array, bins=50, color="blue", alpha=0.7)
plt.title("Pixel Intensity Histogram (Detecting Contrast/Noise)")
plt.xlabel("Pixel Value (0-255)")
plt.show()
