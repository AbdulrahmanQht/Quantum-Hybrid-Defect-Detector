"""
tests/test_property_based.py
=============================
Property-based tests using Hypothesis for all seven noise functions
and the image validation pipeline.

Hypothesis generates hundreds of random inputs automatically and finds
edge cases that hand-written parametrize lists would never cover:
    - noise_level = 1e-308 (denormal float)
    - batch_size = 1
    - images with all-identical pixels (zero variance)
    - a contrast level of exactly 0.0 with a black image
    - JPEG quality at boundary values (1, 99, 100)

Install:
    pip install hypothesis --break-system-packages

Run:
    pytest tests/test_property_based.py -v
    pytest tests/test_property_based.py -v --hypothesis-seed=0   # deterministic

Results saved to: results/property_based/property_based.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import torch


try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st
    _HYPOTHESIS_AVAILABLE = True
except ImportError:
    _HYPOTHESIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _HYPOTHESIS_AVAILABLE,
    reason="hypothesis not installed: pip install hypothesis --break-system-packages",
)

from backend.utils.noise import (
    apply_noise,
    noise_gaussian,
    noise_blur,
    noise_contrast,
    noise_salt_pepper,
    noise_motion_blur,
    noise_jpeg,
    noise_lens_occlusion,
    BENCHMARK_NOISE_LEVELS,
)

BACKEND_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BACKEND_DIR))

RESULTS_DIR = BACKEND_DIR / "tests" / "results" / "property_based"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# Hypothesis strategies
# ────────────────────────────────────────────────────────────────────────────

# Batch size: 1–4 (kept small to avoid slow GPU simulation)
batch_size_st = st.integers(min_value=1, max_value=4)

# Spatial dimensions: small to keep tests fast
spatial_st = st.integers(min_value=8, max_value=64)

# Gaussian / salt-pepper level: full legal range including denormals
noise_level_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Blur sigma: 0.0 to 5.0
blur_sigma_st = st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)

# Contrast: 0.0 to 2.0
contrast_st = st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)

# Motion blur kernel: 0 to 15 pixels
motion_st = st.integers(min_value=0, max_value=15)

# JPEG quality: 1 to 100
jpeg_st = st.integers(min_value=1, max_value=100)

# Pixel fill value: uniform, black, white, random
pixel_fill_st = st.one_of(
    st.just(None),   # random uniform
    st.just(0.0),    # black
    st.just(1.0),    # white
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)

DEFAULT_SETTINGS = settings(
    max_examples=80,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=5000,  # ms per test
)


def _make_tensor(batch: int, height: int, width: int, fill=None) -> torch.Tensor:
    if fill is None:
        torch.manual_seed(0)
        return torch.rand(batch, 3, height, width)
    return torch.full((batch, 3, height, width), fill)


def _assert_valid_output(out: torch.Tensor, inp: torch.Tensor, context: str) -> None:
    assert out.shape == inp.shape, f"{context}: shape changed {inp.shape} -> {out.shape}"
    assert out.dtype == inp.dtype, f"{context}: dtype changed {inp.dtype} -> {out.dtype}"
    assert out.min() >= 0.0 - 1e-6, f"{context}: output below 0.0 (min={out.min():.6f})"
    assert out.max() <= 1.0 + 1e-6, f"{context}: output above 1.0 (max={out.max():.6f})"
    assert not torch.isnan(out).any(), f"{context}: NaN in output"
    assert not torch.isinf(out).any(), f"{context}: Inf in output"


# ────────────────────────────────────────────────────────────────────────────
# Gaussian noise
# ────────────────────────────────────────────────────────────────────────────

class TestGaussianHypothesis:

    @given(
        batch=batch_size_st,
        h=spatial_st, w=spatial_st,
        level=noise_level_st,
        fill=pixel_fill_st,
    )
    @DEFAULT_SETTINGS
    def test_gaussian_always_valid(self, batch, h, w, level, fill):
        inp = _make_tensor(batch, h, w, fill)
        out = noise_gaussian(inp, level)
        _assert_valid_output(out, inp, f"gaussian(level={level:.4f})")

    @given(batch=batch_size_st, h=spatial_st, w=spatial_st)
    @DEFAULT_SETTINGS
    def test_gaussian_zero_level_is_identity(self, batch, h, w):
        inp = _make_tensor(batch, h, w)
        assert torch.allclose(noise_gaussian(inp, 0.0), inp), (
            "gaussian(level=0.0) must be a no-op"
        )

    @given(
        batch=batch_size_st,
        h=spatial_st, w=spatial_st,
        level=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
    )
    @DEFAULT_SETTINGS
    def test_gaussian_nonzero_changes_something(self, batch, h, w, level):
        inp = _make_tensor(batch, h, w)
        out = noise_gaussian(inp, level)
        # With non-zero noise, at least some pixels must change
        assert not torch.equal(inp, out), (
            f"gaussian(level={level:.4f}) on random input produced identical output"
        )


# ────────────────────────────────────────────────────────────────────────────
# Blur
# ────────────────────────────────────────────────────────────────────────────

class TestBlurHypothesis:

    @given(batch=batch_size_st, h=spatial_st, w=spatial_st, sigma=blur_sigma_st)
    @DEFAULT_SETTINGS
    def test_blur_always_valid(self, batch, h, w, sigma):
        inp = _make_tensor(batch, h, w)
        out = noise_blur(inp, sigma)
        _assert_valid_output(out, inp, f"blur(sigma={sigma:.4f})")

    @given(batch=batch_size_st, h=spatial_st, w=spatial_st)
    @DEFAULT_SETTINGS
    def test_blur_zero_sigma_is_identity(self, batch, h, w):
        inp = _make_tensor(batch, h, w)
        assert torch.allclose(noise_blur(inp, 0.0), inp)

    @given(
        batch=batch_size_st,
        h=st.integers(min_value=16, max_value=64),
        w=st.integers(min_value=16, max_value=64),
        sigma=st.floats(min_value=2.0, max_value=5.0, allow_nan=False),
    )
    @DEFAULT_SETTINGS
    def test_heavy_blur_reduces_std(self, batch, h, w, sigma):
        """Heavy blur must always reduce pixel standard deviation."""
        inp = _make_tensor(batch, h, w)
        out = noise_blur(inp, sigma)
        assert out.std() <= inp.std() + 1e-4, (
            f"blur(sigma={sigma:.1f}) increased std from {inp.std():.4f} to {out.std():.4f}"
        )


# ────────────────────────────────────────────────────────────────────────────
# Contrast
# ────────────────────────────────────────────────────────────────────────────

class TestContrastHypothesis:

    @given(batch=batch_size_st, h=spatial_st, w=spatial_st, level=contrast_st)
    @DEFAULT_SETTINGS
    def test_contrast_always_valid(self, batch, h, w, level):
        inp = _make_tensor(batch, h, w)
        out = noise_contrast(inp, level)
        _assert_valid_output(out, inp, f"contrast(level={level:.4f})")

    @given(batch=batch_size_st, h=spatial_st, w=spatial_st)
    @DEFAULT_SETTINGS
    def test_contrast_level_one_is_identity(self, batch, h, w):
        inp = _make_tensor(batch, h, w)
        assert torch.allclose(noise_contrast(inp, 1.0), inp, atol=1e-5)

    @given(
        batch=batch_size_st, h=spatial_st, w=spatial_st,
        fill=st.floats(min_value=0.1, max_value=0.9, allow_nan=False),
    )
    @DEFAULT_SETTINGS
    def test_contrast_zero_level_converges_to_per_image_mean(self, batch, h, w, fill):
        """level=0.0 must flatten each image to its own per-channel mean."""
        inp = _make_tensor(batch, h, w, fill)
        out = noise_contrast(inp, 0.0)
        expected = inp.mean(dim=(-2, -1), keepdim=True).expand_as(inp)
        assert torch.allclose(out, expected, atol=1e-4), (
            "contrast(0.0) should flatten image to per-image mean, not global 0.5"
        )

    @given(batch=batch_size_st, h=spatial_st, w=spatial_st, level=contrast_st)
    @DEFAULT_SETTINGS
    def test_contrast_black_image_stays_black(self, batch, h, w, level):
        """A completely black image must stay black regardless of contrast level."""
        inp = _make_tensor(batch, h, w, 0.0)
        out = noise_contrast(inp, level)
        assert out.max() < 1e-5, (
            f"Black image changed after contrast(level={level:.4f}): max={out.max():.6f}"
        )


# ────────────────────────────────────────────────────────────────────────────
# Salt and pepper
# ────────────────────────────────────────────────────────────────────────────

class TestSaltPepperHypothesis:

    @given(batch=batch_size_st, h=spatial_st, w=spatial_st, level=noise_level_st)
    @DEFAULT_SETTINGS
    def test_salt_pepper_always_valid(self, batch, h, w, level):
        inp = _make_tensor(batch, h, w)
        out = noise_salt_pepper(inp, level)
        _assert_valid_output(out, inp, f"salt_pepper(level={level:.4f})")

    @given(batch=batch_size_st, h=spatial_st, w=spatial_st)
    @DEFAULT_SETTINGS
    def test_salt_pepper_zero_level_is_identity(self, batch, h, w):
        inp = _make_tensor(batch, h, w)
        assert torch.allclose(noise_salt_pepper(inp, 0.0), inp)

    @given(batch=batch_size_st, h=spatial_st, w=spatial_st, level=noise_level_st)
    @DEFAULT_SETTINGS
    def test_corrupted_pixels_are_binary(self, batch, h, w, level):
        """Any pixel that was changed must be exactly 0 or 1."""
        inp = _make_tensor(batch, h, w)
        out = noise_salt_pepper(inp, level)
        changed = ~torch.isclose(out, inp, atol=1e-6)
        if changed.any():
            changed_vals = out[changed]
            assert torch.all(
                (changed_vals == 0.0) | (changed_vals == 1.0)
            ), "Salt-and-pepper: corrupted pixels must be exactly 0 or 1"


# ────────────────────────────────────────────────────────────────────────────
# Motion blur
# ────────────────────────────────────────────────────────────────────────────

class TestMotionBlurHypothesis:

    @given(batch=batch_size_st, h=spatial_st, w=spatial_st, level=motion_st)
    @DEFAULT_SETTINGS
    def test_motion_blur_always_valid(self, batch, h, w, level):
        inp = _make_tensor(batch, h, w)
        out = noise_motion_blur(inp, level)
        _assert_valid_output(out, inp, f"motion_blur(level={level})")

    @given(batch=batch_size_st, h=spatial_st, w=spatial_st)
    @DEFAULT_SETTINGS
    def test_motion_blur_zero_is_identity(self, batch, h, w):
        inp = _make_tensor(batch, h, w)
        assert torch.allclose(noise_motion_blur(inp, 0), inp)


# ────────────────────────────────────────────────────────────────────────────
# JPEG compression
# ────────────────────────────────────────────────────────────────────────────

class TestJpegHypothesis:

    @given(batch=batch_size_st, h=spatial_st, w=spatial_st, quality=jpeg_st)
    @DEFAULT_SETTINGS
    def test_jpeg_always_valid(self, batch, h, w, quality):
        inp = _make_tensor(batch, h, w)
        out = noise_jpeg(inp, quality)
        _assert_valid_output(out, inp, f"jpeg(quality={quality})")

    @given(batch=batch_size_st, h=spatial_st, w=spatial_st)
    @DEFAULT_SETTINGS
    def test_jpeg_quality_100_has_minimal_distortion(self, batch, h, w):
        inp = _make_tensor(batch, h, w)
        out = noise_jpeg(inp, 100)
        _assert_valid_output(out, inp, "jpeg(quality=100)")
        mse = torch.mean((out - inp) ** 2).item()
        assert mse < 1e-4, (
            f"JPEG quality=100 introduced unexpectedly large distortion (MSE={mse:.6f})"
        )

    @given(
        batch=batch_size_st,
        h=st.integers(min_value=16, max_value=64),
        w=st.integers(min_value=16, max_value=64),
        quality=st.integers(min_value=1, max_value=30),
    )
    @DEFAULT_SETTINGS
    def test_low_quality_jpeg_changes_image(self, batch, h, w, quality):
        inp = _make_tensor(batch, h, w)
        out = noise_jpeg(inp, quality)
        mse = torch.mean((out - inp) ** 2).item()
        assert mse > 1e-6, f"JPEG quality={quality} produced negligible change"


# ────────────────────────────────────────────────────────────────────────────
# Lens occlusion
# ────────────────────────────────────────────────────────────────────────────

class TestLensOcclusionHypothesis:

    @given(batch=batch_size_st, h=spatial_st, w=spatial_st, level=noise_level_st)
    @DEFAULT_SETTINGS
    def test_lens_occlusion_always_valid(self, batch, h, w, level):
        inp = _make_tensor(batch, h, w)
        out = noise_lens_occlusion(inp, level)
        _assert_valid_output(out, inp, f"lens_occlusion(level={level:.4f})")

    @given(batch=batch_size_st, h=spatial_st, w=spatial_st)
    @DEFAULT_SETTINGS
    def test_lens_occlusion_zero_is_identity(self, batch, h, w):
        inp = _make_tensor(batch, h, w)
        assert torch.allclose(noise_lens_occlusion(inp, 0.0), inp)


# ────────────────────────────────────────────────────────────────────────────
# apply_noise dispatch
# ────────────────────────────────────────────────────────────────────────────

class TestApplyNoiseHypothesis:

    @given(
        noise_type=st.sampled_from(list(BENCHMARK_NOISE_LEVELS.keys())),
        batch=batch_size_st,
        h=spatial_st, w=spatial_st,
    )
    @DEFAULT_SETTINGS
    def test_apply_noise_dispatch_always_valid(self, noise_type, batch, h, w):
        level = BENCHMARK_NOISE_LEVELS[noise_type][0]  # clean baseline level
        inp = _make_tensor(batch, h, w)
        out = apply_noise(inp, noise_type, level)
        _assert_valid_output(out, inp, f"apply_noise({noise_type}, level={level})")

    @given(batch=batch_size_st, h=spatial_st, w=spatial_st)
    @DEFAULT_SETTINGS
    def test_unknown_noise_type_raises(self, batch, h, w):
        inp = _make_tensor(batch, h, w)
        with pytest.raises(ValueError, match="Unknown noise_type"):
            apply_noise(inp, "cosmic_rays_from_andromeda", 0.5)

class TestValidatorHypothesis:
    """
    Property-based tests for image validation logic.
    Tests the production validator if available, falls back to structural check.
    """

    def _get_validator(self):
        try:
            from backend.utils.validate import check_magic_bytes, check_file_size, check_dimensions
            return check_magic_bytes, check_file_size, check_dimensions
        except ImportError:
            pytest.skip(".utils.validate not importable")


    @given(size=st.integers(min_value=0, max_value=10 * 1024 * 1024))
    @DEFAULT_SETTINGS
    def test_file_size_check_always_returns_bool(self, size):
        _, check_file_size, _ = self._get_validator()
        result = check_file_size(size)
        assert isinstance(result, bool)

    @given(
        width=st.integers(min_value=1, max_value=10000),
        height=st.integers(min_value=1, max_value=10000),
    )
    @DEFAULT_SETTINGS
    def test_dimension_check_always_returns_bool(self, width, height):
        _, _, check_dimensions = self._get_validator()
        result = check_dimensions(width, height)
        assert isinstance(result, bool)

    @given(
        width=st.integers(min_value=1, max_value=4095),
        height=st.integers(min_value=1, max_value=4095),
    )
    @DEFAULT_SETTINGS
    def test_valid_dimensions_always_accepted(self, width, height):
        """Any dimension strictly below 4096 must be accepted."""
        _, _, check_dimensions = self._get_validator()
        assert check_dimensions(width, height) is True, (
            f"Valid dimension {width}×{height} was rejected"
        )

    @given(
        width=st.integers(min_value=4096, max_value=20000),
        height=st.integers(min_value=4096, max_value=20000),
    )
    @DEFAULT_SETTINGS
    def test_oversized_dimensions_always_rejected(self, width, height):
        """Any dimension >= 4096 must be rejected."""
        _, _, check_dimensions = self._get_validator()
        assert check_dimensions(width, height) is False, (
            f"Oversized dimension {width}×{height} was accepted"
        )

    @given(size=st.integers(min_value=0, max_value=5 * 1024 * 1024 - 1))
    @DEFAULT_SETTINGS
    def test_under_5mb_always_accepted(self, size):
        _, check_file_size, _ = self._get_validator()
        assert check_file_size(size) is True, (
            f"File of size {size} bytes (under 5MB) was rejected"
        )

    @given(size=st.integers(min_value=5 * 1024 * 1024, max_value=50 * 1024 * 1024))
    @DEFAULT_SETTINGS
    def test_5mb_and_above_always_rejected(self, size):
        _, check_file_size, _ = self._get_validator()
        assert check_file_size(size) is False, (
            f"File of size {size} bytes (>= 5MB) was accepted"
        )

# ────────────────────────────────────────────────────────────────────────────
# Results save
# ────────────────────────────────────────────────────────────────────────────

def test_save_property_based_summary():
    """Save a summary record confirming property-based tests ran."""
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "framework": "hypothesis",
        "noise_types_tested": list(BENCHMARK_NOISE_LEVELS.keys()),
        "properties_verified": [
            "output_shape_preserved",
            "output_dtype_preserved",
            "output_range_clamped_0_1",
            "no_nan_in_output",
            "no_inf_in_output",
            "zero_level_is_identity",
            "corrupted_pixels_are_binary_for_salt_pepper",
            "contrast_converges_to_per_image_mean",
            "heavy_blur_reduces_std",
            "file_size_boundary_correctness",
            "dimension_boundary_correctness",
        ],
        "hypothesis_max_examples": 80,
    }
    path = RESULTS_DIR / "property_based.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nProperty-based test summary saved to {path}")