import os
import io
import torch
import random
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode

class PreProcessing(Dataset):
    """
        Custom Dataset class for industrial defect image loading and transformation.

        Responsibilities:
            - Crawls directory structures to map folder names to integer class labels.
            - Implements 'Get-on-the-fly' loading to keep memory usage low.
            - Applies domain-specific geometric augmentations (flips, rotations) to
                simulate various robot camera angles on pipes.
            - Applies domain-specific noise augmentations to simulate real-world
                pipeline inspection degradation (sensor noise, motion blur, compression, 
                low-light contrast loss, lens occlusion, dead pixels).
                
            Pipeline order (training):
                [ToImage] -> [Resize] -> [Geometric augmentations] -> [ToDtype float32]
                -> [Noise augmentations] -> [Normalize (optional)]
        
            Noise transforms are placed AFTER ToDtype so they operate on float32 [0, 1].
        """
    def __init__( self, root_dir, img_width, img_height, is_training=True, apply_augmentation=True, normalize=False):
        self.data_path = os.path.join(root_dir, 'Images')

        self.transform = PreProcessing.get_transforms(img_width, img_height, is_training, apply_augmentation, normalize)

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
                    
    
    # Noise augmentation static methods
    # All accept a float32 tensor in [0, 1] and return one of the same shape.
    # Each has an internal probability (p) so not every sample is affected.
    # Wrapped with v2.Lambda in get_transforms below.
    @staticmethod
    def _gaussian_noise(img: torch.Tensor, sigma_range=(0.01, 0.08), p=0.5) -> torch.Tensor:
        """
        Additive Gaussian (white) noise with a random sigma sampled per image.
        Source: Electronic sensor noise from low-quality cameras operating in
        poorly-lit underground pipeline tunnels.
        """
        if random.random() > p:
            return img
        sigma = random.uniform(*sigma_range)
        return torch.clamp(img + torch.randn_like(img) * sigma, 0.0, 1.0)
 
    @staticmethod
    def _salt_and_pepper_noise(img: torch.Tensor, amount_range=(0.002, 0.02), p=0.3) -> torch.Tensor:
        """
        Randomly sets pixels to 0 (pepper) or 1 (salt).
        Source: Electrical interference and failing sensor pixels from vibration
        and EM fields produced by pipeline inspection equipment and pump motors.
        """
        if random.random() > p:
            return img
        amount = random.uniform(*amount_range)
        out = img.clone()
        n = int(amount * img.shape[-1] * img.shape[-2])
        salt_c   = [torch.randint(0, s, (n,)) for s in img.shape[1:]]
        pepper_c = [torch.randint(0, s, (n,)) for s in img.shape[1:]]
        out[:, salt_c[0],   salt_c[1]]   = 1.0
        out[:, pepper_c[0], pepper_c[1]] = 0.0
        return out
 
    @staticmethod
    def _motion_blur(img: torch.Tensor, kernel_range=(3, 9), p=0.3) -> torch.Tensor:
        """
        Horizontal or vertical linear motion blur via a 1-D box kernel.
        Source: The inspection robot moves continuously through the pipe while
        the shutter is open, smearing structural details like crack edges.
        """
        if random.random() > p:
            return img
        k = random.randrange(kernel_range[0], kernel_range[1] + 1, 2)  # must be odd
        kernel = torch.zeros(1, 1, k, k, dtype=img.dtype, device=img.device)
        if random.random() > 0.5:
            kernel[0, 0, k // 2, :] = 1.0 / k  # horizontal
        else:
            kernel[0, 0, :, k // 2] = 1.0 / k  # vertical
        c = img.shape[0]
        out = torch.nn.functional.conv2d(
            img.unsqueeze(0),
            kernel.expand(c, 1, k, k),
            padding=k // 2,
            groups=c,
        ).squeeze(0)
        return torch.clamp(out, 0.0, 1.0)
 
    @staticmethod
    def _jpeg_compression_noise(img: torch.Tensor, quality_range=(30, 75), p=0.3) -> torch.Tensor:
        """
        Simulates JPEG block-compression artifacts via encode/decode cycling.
        Source: Robot vehicles compress video on-board at low quality factors
        (QF 30-75) before RF transmission, introducing 8x8 block artifacts
        that can mask fine cracks or mimic surface texture.
        """
        if random.random() > p:
            return img
        try:
            from PIL import Image as PILImage
            import numpy as np
            quality = random.randint(*quality_range)
            uint8   = (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            pil_img = PILImage.fromarray(uint8, mode="RGB")
            buf     = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            decoded = PILImage.open(buf).convert("RGB")
            return torch.from_numpy(np.array(decoded)).permute(2, 0, 1).float().div(255.0).to(img.device)
        except Exception:
            return img
 
    @staticmethod
    def _contrast_reduction(img: torch.Tensor, factor_range=(0.4, 0.9), p=0.3) -> torch.Tensor:
        """
        Reduces contrast by blending toward the per-image mean intensity.
        Source: Pipeline interiors are dark enclosed spaces. The robot ring-light
        falls off with distance, leaving far-field pipe sections underexposed where
        deposition and early-stage corrosion are hardest to distinguish.
        """
        if random.random() > p:
            return img
        factor = random.uniform(*factor_range)
        mean   = img.mean(dim=(-2, -1), keepdim=True)
        return torch.clamp(mean + factor * (img - mean), 0.0, 1.0)
 
    @staticmethod
    def _lens_occlusion_erasing(img: torch.Tensor, num_patches_range=(1, 3), scale_range=(0.02, 0.12), p=0.25) -> torch.Tensor:
        """
        Randomly erases rectangular regions, filling them with dim noise.
        Source: Water condensation, mud splatter, and pipe sediment physically
        stick to the robot camera lens, creating patches that partially occlude
        the defect being classified.
        """
        if random.random() > p:
            return img
        out  = img.clone()
        _, h, w = img.shape
        for _ in range(random.randint(*num_patches_range)):
            area_frac = random.uniform(*scale_range)
            aspect    = random.uniform(0.5, 2.0)
            ph = max(1, min(int((h * w * area_frac / aspect) ** 0.5), h))
            pw = max(1, min(int(ph * aspect), w))
            top  = random.randint(0, h - ph)
            left = random.randint(0, w - pw)
            # dim noise (0.0–0.3) rather than pure black to simulate smear
            out[:, top:top + ph, left:left + pw] = torch.rand(img.shape[0], ph, pw, device=img.device) * 0.3
        return out
    
    @staticmethod
    def get_transforms(img_width, img_height, is_training=False, apply_augmentation=False, normalize=False):
        """
            Defines the transformation pipeline for images.

            Training Pipeline : Resize -> Geometric -> ToDtype -> Noise -> [Normalize]
            Inference Pipeline: Resize -> ToDtype -> [Normalize]
        """
        # Coverts PIL to Tensor
        to_image = v2.ToImage()
        # Basic transformations: Resize and ToTensor
        resize_transform = v2.Resize((img_width, img_height), antialias=True)
        # Convert to tensor and normalize to [0, 1]
        to_dtype = v2.ToDtype(torch.float32, scale=True)
        
        norm_transform = v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        ) if normalize else None

        if is_training and apply_augmentation:
            # Geometric augmentations (before to_dtype, operate on uint8 tensors).
            # Simulate robot camera orientation and position variation.
            geometric_augmentations = [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=180),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
            ]
            
            # Noise augmentations (after to_dtype, operate on float32 [0, 1]).
            # v2.Lambda wraps each static method; p is baked into each function.
            noise_augmentations = v2.RandomApply(
                [
                    v2.Lambda(PreProcessing._gaussian_noise),
                    v2.Lambda(PreProcessing._salt_and_pepper_noise),
                    v2.Lambda(PreProcessing._motion_blur),
                    v2.Lambda(PreProcessing._jpeg_compression_noise),
                    v2.Lambda(PreProcessing._contrast_reduction),
                    v2.Lambda(PreProcessing._lens_occlusion_erasing),
                ],
                p=0.6
            )
            steps = [to_image, resize_transform] + geometric_augmentations + [to_dtype] + [noise_augmentations]
        else:
            steps = [to_image, resize_transform, to_dtype]

        if norm_transform:
            steps.append(norm_transform)

        return v2.Compose(steps)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Open as RGB then let transform handle Grayscale/Resize/Tensor
        image = read_image(img_path, mode=ImageReadMode.RGB)

        if self.transform:
            image = self.transform(image)

        return image, label