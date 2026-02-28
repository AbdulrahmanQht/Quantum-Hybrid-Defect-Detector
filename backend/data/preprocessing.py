import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode

class PreProcessing(Dataset):
    """
        Custom Dataset class for industrial defect image loading and transformation.

        Responsibilities:
            - Crawls directory structures to map folder names to integer class labels.
            - Implements 'Get-on-the-fly' loading to keep memory usage low.
            - Applies domain-specific augmentations (flips, rotations) to simulate various camera angles on pipes.
        """
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
        """
            Defines the transformation pipeline for images.

            Training Pipeline: Resize -> Augment -> Tensor
            Inference Pipeline: Resize -> Tensor
        """
        # Coverts PIL to Tensor
        to_image = v2.ToImage()
        # Basic transformations: Resize and ToTensor
        resize_transform = v2.Resize((img_width, img_height), antialias=True)
        # Convert to tensor and normalize to [0, 1]
        to_dtype = v2.ToDtype(torch.float32, scale=True)

        if is_training and apply_augmentation:
            augmentation_transforms = [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=180),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
            ]
            return v2.Compose([to_image, resize_transform] + augmentation_transforms + [to_dtype])

        return v2.Compose([to_image, resize_transform, to_dtype])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Open as RGB then let transform handle Grayscale/Resize/Tensor
        image = read_image(img_path, mode=ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        return image, label