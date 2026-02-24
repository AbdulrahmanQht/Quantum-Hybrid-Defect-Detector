import os
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset

class PreProcessing(Dataset):
    def __init__( self, root_dir, img_width, img_height, is_training = True, apply_augmentation = True):
        self.data_path = os.path.join(root_dir, 'Images')

        self.transform = PreProcessing.get_transforms(
            img_width, img_height, is_training, apply_augmentation
        )

        self.samples = []
        self.classes = sorted([d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for target_class in self.classes:
            class_path = os.path.join(self.data_path, target_class)
            for img_name in os.listdir(class_path):
                img_full_path = os.path.join(class_path, img_name)
                # Only add if it's a file, not a directory
                if os.path.isfile(img_full_path):
                    self.samples.append((img_full_path, self.class_to_idx[target_class]))

    @staticmethod
    def get_transforms(img_width, img_height, is_training=False, apply_augmentation=False):
        basic_transforms = [
            torchvision.transforms.Resize((img_width, img_height)),
            torchvision.transforms.ToTensor()
        ]

        if is_training and apply_augmentation:
            augmentation_transforms = [
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.RandomRotation(degrees=180),
                torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
            return torchvision.transforms.Compose(augmentation_transforms + basic_transforms)

        return torchvision.transforms.Compose(basic_transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Open as RGB then let transform handle Grayscale/Resize/Tensor
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label