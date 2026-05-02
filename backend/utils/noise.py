"""
backend/models/noise.py
======================
Shared noise-injection utilities for benchmark.py and quantum_advantage.py.

All public functions follow the same contract:
  - Input : float32 batch tensor in [0, 1], shape (B, C, H, W)
  - Output: float32 tensor, same shape, values clamped to [0, 1]
  - Deterministic at a given (noise_type, level) — no hidden random state
    beyond what is seeded externally — so both runners produce reproducible,
    directly comparable numbers.

Level semantics (documented once here, used everywhere):
  gaussian         – sigma of additive white Gaussian noise (0 = clean)
  blur             – Gaussian blur sigma; kernel auto-sized (0 = clean)
  contrast         – retention factor: 1.0 = original, 0.0 = flat per-image mean
  salt_pepper      – fraction of pixels corrupted (0 = clean)
  motion_blur      – half-width of horizontal box kernel in pixels (0 = clean)
  jpeg_compression – JPEG quality factor (100 = lossless, 10 = heavy artefacts)
  lens_occlusion   – fractional area of a central occlusion patch (0 = clean)
"""

from __future__ import annotations

import io
from typing import Union

import torch
import torch.nn.functional as F


#  Individual noise functions 
def noise_gaussian(images: torch.Tensor, level: float) -> torch.Tensor:
    """
    Additive white Gaussian noise.
    Aligned with PreProcessing._gaussian_noise (same additive formula).
    """
    if level == 0.0:
        return images
    return (images + torch.randn_like(images) * level).clamp(0.0, 1.0)


def noise_blur(images: torch.Tensor, level: float) -> torch.Tensor:
    """
    Isotropic Gaussian blur.
    Kernel = smallest odd integer >= 6*sigma, minimum 3.

    For very small images, cap the kernel to the largest odd size that fits
    within both spatial dimensions. Numerically tiny sigma values are treated
    as a no-op.
    """
    level = float(level)
    if level <= 1e-6:
        return images

    from torchvision.transforms import v2

    _, _, h, w = images.shape

    k = max(3, int(level * 6 + 1))
    k = k if k % 2 == 1 else k + 1

    max_k = min(h, w)
    if max_k % 2 == 0:
        max_k -= 1

    if max_k < 3:
        return images

    k = min(k, max_k)
    return v2.GaussianBlur(kernel_size=k, sigma=level)(images)




def noise_contrast(images: torch.Tensor, level: float) -> torch.Tensor:
    """
    Contrast reduction by blending toward the PER-IMAGE channel mean.
    Aligned with PreProcessing._contrast_reduction — blends toward
    img.mean(dim=(-2,-1)), NOT a fixed 0.5 midpoint.
    level = 1.0 → original;  level = 0.0 → flat grey.
    """
    if level >= 1.0:
        return images
    mean = images.mean(dim=(-2, -1), keepdim=True)   # (B, C, 1, 1)
    return (mean + level * (images - mean)).clamp(0.0, 1.0)


def noise_salt_pepper(images: torch.Tensor, level: float) -> torch.Tensor:
    """
    Salt-and-pepper noise.
    Half the corrupted pixels become 0 (pepper), half become 1 (salt).
    Single spatial mask broadcast over all channels for realistic corruption.
    """
    if level == 0.0:
        return images
    corrupted = images.clone()
    B, C, H, W = images.shape
    mask = torch.rand(B, 1, H, W, device=images.device).expand(B, C, H, W)
    corrupted[mask < level / 2]       = 0.0   # pepper
    corrupted[mask > 1.0 - level / 2] = 1.0   # salt
    return corrupted


def noise_motion_blur(images: torch.Tensor, level: Union[int, float]) -> torch.Tensor:
    """
    Horizontal motion blur via a 1-D box convolution.
    kernel size = 2 * level + 1.
    Aligned with PreProcessing._motion_blur (horizontal direction, box kernel).
    """
    level = int(level)
    if level == 0:
        return images
    k = 2 * level + 1
    B, C, H, W = images.shape
    kernel = torch.zeros(1, 1, k, k, dtype=images.dtype, device=images.device)
    kernel[0, 0, k // 2, :] = 1.0 / k
    blurred = F.conv2d(
        images.reshape(B * C, 1, H, W),
        kernel,
        padding=k // 2,
    ).reshape(B, C, H, W)
    return blurred.clamp(0.0, 1.0)


def noise_jpeg(images: torch.Tensor, level: Union[int, float]) -> torch.Tensor:
    """
    JPEG compression artefacts.
    Encodes each image with PIL at the given quality factor, then decodes.
    Falls back to returning the original batch if PIL/numpy are unavailable.
    Aligned with PreProcessing._jpeg_compression_noise.
    """
    level = int(level)
    if level >= 100:
        return images
    try:
        import numpy as np
        from PIL import Image as PILImage

        results = []
        for img in images.cpu():
            uint8   = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
            pil     = PILImage.fromarray(uint8, mode="RGB")
            buf     = io.BytesIO()
            pil.save(buf, format="JPEG", quality=level)
            buf.seek(0)
            decoded = PILImage.open(buf).convert("RGB")
            results.append(
                torch.from_numpy(np.array(decoded))
                .permute(2, 0, 1).float().div(255.0)
            )
        return torch.stack(results).to(images.device)
    except Exception:
        return images      # degrade gracefully if dependencies missing


def noise_lens_occlusion(images: torch.Tensor, level: float) -> torch.Tensor:
    """
    Rectangular central occlusion patch filled with dim noise in [0, 0.3].
    Patch area is approximately level * H * W, capped to image bounds.
    Deterministic centred placement for reproducible benchmarking.
    """
    if level <= 0.0:
        return images

    B, C, H, W = images.shape
    out = images.clone()

    target_area = max(1.0, float(H * W) * float(level))

    # Start from a square patch, then clamp to image bounds.
    ph = min(H, max(1, int(target_area ** 0.5)))
    pw = min(W, max(1, int(target_area / ph)))

    # Final safety clamp in case integer rounding overshoots.
    ph = min(ph, H)
    pw = min(pw, W)

    top = (H - ph) // 2
    left = (W - pw) // 2

    out[:, :, top: top + ph, left: left + pw] = (
        torch.rand(B, C, ph, pw, device=images.device, dtype=images.dtype) * 0.3
    )
    return out



#  Dispatch table 

_NOISE_FN: dict = {
    "gaussian":         noise_gaussian,
    "blur":             noise_blur,
    "contrast":         noise_contrast,
    "salt_pepper":      noise_salt_pepper,
    "motion_blur":      noise_motion_blur,
    "jpeg_compression": noise_jpeg,
    "lens_occlusion":   noise_lens_occlusion,
}


def apply_noise(images: torch.Tensor, noise_type: str, level: float) -> torch.Tensor:
    """
    Apply the requested noise type at the given intensity to a batch of images.

    Args:
        images    : float32 tensor [0, 1], shape (B, C, H, W), already on target device.
        noise_type: one of the keys in _NOISE_FN.
        level     : intensity value (semantics documented per function above).

    Returns:
        Noisy tensor, same shape and device as input, values in [0, 1].

    Raises:
        ValueError: if noise_type is not recognised.
    """
    fn = _NOISE_FN.get(noise_type)
    if fn is None:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. "
            f"Valid options: {list(_NOISE_FN)}"
        )
    return fn(images, level)


#  Canonical noise-level grids 

# Full grid — used by benchmarkpy
BENCHMARK_NOISE_LEVELS: dict[str, list] = {
    "gaussian":         [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75],
    "blur":             [0.00, 0.30, 0.60, 0.90, 1.20, 1.50, 2.00, 2.50, 3.00],
    "contrast":         [1.00, 0.85, 0.70, 0.55, 0.40, 0.25, 0.10],
    "salt_pepper":      [0.00, 0.005, 0.01, 0.02, 0.04, 0.07, 0.10],
    "motion_blur":      [0, 1, 2, 3, 4, 6, 8, 10],
    "jpeg_compression": [100, 85, 70, 55, 40, 25, 10],
    "lens_occlusion":   [0.00, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20],
}

# Reduced subset — used by quantum_advantage.py Experiment 6.
# Covers the same 5 types but at ~half the resolution to keep QA runtime reasonable.
QA_NOISE_LEVELS: dict[str, list] = {
    "gaussian":         [0.00, 0.10, 0.20, 0.30, 0.50],
    "blur":             [0.00, 1.50, 2.50, 3.00],
    "contrast":         [1.00, 0.55, 0.40, 0.25, 0.10],
    "salt_pepper":      [0.00, 0.02, 0.04, 0.07, 0.10],
    "motion_blur":      [0, 2, 4, 6, 8, 10],
    "jpeg_compression": [100, 40, 25, 10],
    "lens_occlusion":   [0.00, 0.08, 0.12, 0.16, 0.20],
}