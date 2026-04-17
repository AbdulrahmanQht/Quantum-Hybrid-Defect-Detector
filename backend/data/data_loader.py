import os
import torch
from torch.utils.data import DataLoader
from backend.data.preprocessing import PreProcessing


class DataLoaderManager:
    def __init__(
        self,
        train_dir,
        val_dir,
        test_dir,
        img_width=256,
        img_height=256,
        batch_size=32,
        normalize=False,
        drop_last=False,
    ):
        self.batch_size = batch_size
        pin_memory = torch.cuda.is_available()
        cpu_count = os.cpu_count() or 1
        num_workers = 16

        # NOTE: On Windows, using num_workers > 0 requires the
        if os.name == "nt" and num_workers > 4:
            num_workers = 4

        # For wsl2 (I'm training qnn GPU in wsl2 since lightining.gpu is not supported in windows for CUDA 13.2)
        wsl_interop = os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME")
        if wsl_interop and num_workers > 6:
            num_workers = min(8, cpu_count - 2)

        # Keep CPU workers alive between epochs to reduce overhead.
        keep_alive = num_workers > 0
        prefetch = 4 if num_workers > 0 else None

        # Initialize the PreProcessing datasets
        self.train_dataset = PreProcessing(
            train_dir,
            img_width,
            img_height,
            is_training=True,
            apply_augmentation=True,
            normalize=normalize,
        )
        self.val_dataset = PreProcessing(
            val_dir,
            img_width,
            img_height,
            is_training=False,
            apply_augmentation=False,
            normalize=normalize,
        )
        self.test_dataset = PreProcessing(
            test_dir,
            img_width,
            img_height,
            is_training=False,
            apply_augmentation=False,
            normalize=normalize,
        )

        # Create the DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=keep_alive,
            prefetch_factor=prefetch,
            drop_last=drop_last,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=keep_alive,
            prefetch_factor=prefetch,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=keep_alive,
            prefetch_factor=prefetch,
        )

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader
