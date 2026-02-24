import os
import torch
from backend.utils.logger import Logger
from torch.utils.data import DataLoader
from backend.data.preprocessing import PreProcessing


class DataLoaderManager:
    def __init__(
        self, train_dir, val_dir, test_dir, img_width=256, img_height=256, batch_size=32
    ):
        self.batch_size = batch_size
        pin_memory = torch.cuda.is_available()
        cpu_count = os.cpu_count() or 1
        num_workers = max(1, cpu_count - 2)

        # NOTE: On Windows, using num_workers > 0 requires the
        if os.name == 'nt' and num_workers > 4:
            num_workers = 4

        # Initialize the PreProcessing datasets
        self.train_dataset = PreProcessing(
            train_dir, img_width, img_height, is_training=True, apply_augmentation=True
        )
        self.val_dataset = PreProcessing(
            val_dir, img_width, img_height, is_training=False, apply_augmentation=False
        )
        self.test_dataset = PreProcessing(
            test_dir, img_width, img_height, is_training=False, apply_augmentation=False
        )

        # Create the DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers = num_workers,
            pin_memory = pin_memory,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = pin_memory,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = pin_memory,
        )

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader
