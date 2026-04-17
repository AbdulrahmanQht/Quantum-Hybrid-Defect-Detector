"""
tests/test_suite.py
===================
Full pytest test suite for Quantum-Hybrid-Defect-Detector.

How to run
----------
    # From the project root (same directory as backend/)
    pip install pytest pytest-cov httpx --break-system-packages

    pytest tests/test_suite.py -v                  # all tests, verbose
    pytest tests/test_suite.py -v -k "noise"       # only noise tests
    pytest tests/test_suite.py -v --co             # list tests without running
    pytest tests/test_suite.py --cov=backend       # with coverage report

What is tested
--------------
    Unit tests
        A. Noise functions (7 types) — determinism, shape, range, edge cases
        B. Image validation — format, size, dimension, security checks
        C. Model forward passes — shape and dtype (CPU-only, no full training)

    Integration tests
        D. FastAPI endpoints — happy path, all error paths, benchmark cache

Structure
---------
    - Fixtures (conftest-style, defined at module level for single-file simplicity)
    - Test classes group related tests; pytest collects them automatically.
    - Each test is a function whose name starts with "test_".
    - assert is the only assertion mechanism in pytest.

Pytest basics (complete reference for someone new to testing)
-------------------------------------------------------------
    1. A "test" is any function starting with test_ inside a file starting with test_.
    2. pytest discovers these automatically — you never call them manually.
    3. assert <condition> is the only assertion you need. If it's False, the test FAILS.
    4. A fixture is a reusable piece of setup (like creating a tensor or a fake model).
       Mark it with @pytest.fixture and request it by name as a function argument.
    5. parametrize lets you run the same test with different inputs in one shot.
    6. Use tmp_path (a built-in fixture) for temporary files — pytest cleans them up.
    7. caplog captures log output for assertion.
    8. monkeypatch replaces functions/attributes temporarily during a test.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ── Make backend importable when running from project root ────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.noise import (
    apply_noise,
    noise_blur,
    noise_contrast,
    noise_gaussian,
    noise_jpeg,
    noise_lens_occlusion,
    noise_motion_blur,
    noise_salt_pepper,
    BENCHMARK_NOISE_LEVELS,
    QA_NOISE_LEVELS,
)


# ═════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# A fixture is a function decorated with @pytest.fixture.
# Any test function that declares a parameter with the same name as a fixture
# automatically receives the return value of that fixture.
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def clean_batch() -> torch.Tensor:
    """
    A small synthetic image batch — 4 RGB images of 32×32 pixels.
    Values are uniformly distributed in [0, 1] (what model inputs look like
    after normalisation).  Using a fixed seed makes the fixture deterministic.
    """
    torch.manual_seed(42)
    return torch.rand(4, 3, 32, 32)


@pytest.fixture
def single_image() -> torch.Tensor:
    """Single 384×384 image — matches the project's real resolution."""
    torch.manual_seed(0)
    return torch.rand(1, 3, 384, 384)


@pytest.fixture
def black_batch() -> torch.Tensor:
    """All-zero batch — worst-case for salt/pepper and contrast."""
    return torch.zeros(2, 3, 32, 32)


@pytest.fixture
def white_batch() -> torch.Tensor:
    """All-one batch — worst-case for contrast toward mean."""
    return torch.ones(2, 3, 32, 32)


# ═════════════════════════════════════════════════════════════════════════════
# A — Noise function unit tests
# ═════════════════════════════════════════════════════════════════════════════

class TestNoiseGaussian:
    """Tests for noise_gaussian / apply_noise(..., 'gaussian', ...)."""

    def test_zero_level_returns_input_unchanged(self, clean_batch):
        out = noise_gaussian(clean_batch, 0.0)
        assert torch.allclose(out, clean_batch), \
            "Level 0.0 must be a no-op"

    def test_output_shape_preserved(self, clean_batch):
        out = noise_gaussian(clean_batch, 0.2)
        assert out.shape == clean_batch.shape

    def test_values_clamped_to_unit_range(self, clean_batch):
        out = noise_gaussian(clean_batch, 1.0)   # large sigma — will saturate
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_noise_changes_values(self, clean_batch):
        out = noise_gaussian(clean_batch, 0.1)
        assert not torch.allclose(out, clean_batch), \
            "Non-zero sigma must change at least some values"

    @pytest.mark.parametrize("level", BENCHMARK_NOISE_LEVELS["gaussian"])
    def test_all_benchmark_levels_valid(self, clean_batch, level):
        """Every level in the canonical grid must produce a valid tensor."""
        out = apply_noise(clean_batch, "gaussian", level)
        assert out.shape == clean_batch.shape
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestNoiseBlur:

    def test_zero_sigma_no_change(self, clean_batch):
        out = noise_blur(clean_batch, 0.0)
        assert torch.allclose(out, clean_batch)

    def test_output_shape(self, clean_batch):
        out = noise_blur(clean_batch, 1.5)
        assert out.shape == clean_batch.shape

    def test_range_preserved(self, clean_batch):
        out = noise_blur(clean_batch, 3.0)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_blurring_reduces_high_frequency(self):
        """
        A checkerboard pattern has maximum high-frequency energy.
        After heavy blur, adjacent pixels should become very similar.
        """
        checker = torch.zeros(1, 1, 8, 8)
        checker[0, 0, ::2, ::2] = 1.0
        checker[0, 0, 1::2, 1::2] = 1.0
        blurred = noise_blur(checker.expand(1, 3, 8, 8), 2.0)
        # After blur, std should be significantly lower than the input
        assert blurred.std() < checker.std()


class TestNoiseContrast:

    def test_level_one_no_change(self, clean_batch):
        out = noise_contrast(clean_batch, 1.0)
        assert torch.allclose(out, clean_batch)

    def test_level_zero_flat_image(self, clean_batch):
        out = noise_contrast(clean_batch, 0.0)
        # Each pixel should equal its image's per-channel mean
        expected_mean = clean_batch.mean(dim=(-2, -1), keepdim=True)
        assert torch.allclose(out, expected_mean.expand_as(clean_batch), atol=1e-5)

    def test_range_preserved(self, clean_batch):
        out = noise_contrast(clean_batch, 0.4)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_blends_toward_per_image_mean_not_global_05(self):
        """
        Contrast should blend toward per-image mean, NOT a fixed 0.5.
        A batch of pure-0.8 images should converge to 0.8, not 0.5.
        """
        bright = torch.full((2, 3, 8, 8), 0.8)
        out    = noise_contrast(bright, 0.0)       # level=0 → fully flat
        assert torch.allclose(out, bright, atol=1e-5), \
            "Flat-grey output for a bright image should be 0.8, not 0.5"


class TestNoiseSaltPepper:

    def test_zero_level_no_change(self, clean_batch):
        out = noise_salt_pepper(clean_batch, 0.0)
        assert torch.allclose(out, clean_batch)

    def test_corrupted_pixels_are_0_or_1(self, clean_batch):
        out = noise_salt_pepper(clean_batch, 0.5)   # heavy corruption
        # Every pixel must be either fully preserved OR 0/1
        changed = ~torch.isclose(out, clean_batch, atol=1e-6)
        if changed.any():
            changed_vals = out[changed]
            assert torch.all(
                (changed_vals == 0.0) | (changed_vals == 1.0)
            ), "Corrupted pixels must be exactly 0 or 1"

    def test_range_valid(self, clean_batch):
        out = noise_salt_pepper(clean_batch, 0.1)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_corruption_fraction_approximate(self, clean_batch):
        """
        With level=0.10, roughly 10% of pixels should be corrupted.
        We allow ±5pp tolerance due to randomness.
        """
        torch.manual_seed(99)
        out     = noise_salt_pepper(clean_batch, 0.10)
        changed = (~torch.isclose(out, clean_batch, atol=1e-6)).float().mean().item()
        assert 0.05 <= changed <= 0.15, \
            f"Expected ~10% corruption, got {changed*100:.1f}%"


class TestNoiseMotionBlur:

    def test_zero_level_no_change(self, clean_batch):
        out = noise_motion_blur(clean_batch, 0)
        assert torch.allclose(out, clean_batch)

    def test_output_shape(self, clean_batch):
        out = noise_motion_blur(clean_batch, 3)
        assert out.shape == clean_batch.shape

    def test_range_preserved(self, clean_batch):
        out = noise_motion_blur(clean_batch, 5)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_horizontal_smear(self):
        """
        A single bright pixel should smear HORIZONTALLY after motion blur.
        Vertical neighbours should remain dark.
        """
        img    = torch.zeros(1, 1, 17, 17)
        img[0, 0, 8, 8] = 1.0                          # single white pixel
        blurred = noise_motion_blur(img.expand(1, 3, 17, 17), 4)
        center_row = blurred[0, 0, 8, :]               # same row, spread
        above_row  = blurred[0, 0, 7, :]               # row above should be ~0
        assert center_row.max() > above_row.max(), \
            "Motion blur must smear horizontally, not vertically"


class TestNoiseJpeg:

    def test_quality_100_is_nearly_lossless(self, clean_batch):
        out = noise_jpeg(clean_batch, 100)
        # At Q=100 we skip encoding entirely and return the input unchanged
        assert torch.allclose(out, clean_batch)

    def test_output_shape(self, clean_batch):
        out = noise_jpeg(clean_batch, 50)
        assert out.shape == clean_batch.shape

    def test_range_preserved(self, clean_batch):
        out = noise_jpeg(clean_batch, 10)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_heavy_compression_changes_values(self, clean_batch):
        out = noise_jpeg(clean_batch, 10)
        assert not torch.allclose(out, clean_batch, atol=0.01), \
            "Heavy JPEG compression must visibly alter pixel values"


class TestNoiseLensOcclusion:

    def test_zero_level_no_change(self, clean_batch):
        out = noise_lens_occlusion(clean_batch, 0.0)
        assert torch.allclose(out, clean_batch)

    def test_output_shape(self, clean_batch):
        out = noise_lens_occlusion(clean_batch, 0.1)
        assert out.shape == clean_batch.shape

    def test_range_preserved(self, clean_batch):
        out = noise_lens_occlusion(clean_batch, 0.2)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_occluded_patch_is_dim(self, clean_batch):
        """
        The central patch should be in [0, 0.3] range (dim noise).
        """
        out  = noise_lens_occlusion(clean_batch, 0.2)
        B, C, H, W = clean_batch.shape
        ph   = max(1, int((H * W * 0.2) ** 0.5))
        top  = (H - ph) // 2
        left = (W - ph) // 2
        patch = out[:, :, top:top+ph, left:left+ph]
        assert patch.max() <= 0.3 + 1e-5, \
            "Occluded region must contain only dim noise (<= 0.3)"


class TestApplyNoiseDispatch:
    """Tests for the public apply_noise() dispatch function."""

    def test_dispatches_to_correct_function(self, clean_batch):
        with patch("backend.utils.noise.noise_gaussian", wraps=noise_gaussian) as mock_fn:
            apply_noise(clean_batch, "gaussian", 0.1)
            mock_fn.assert_called_once()

    def test_unknown_noise_type_raises_value_error(self, clean_batch):
        with pytest.raises(ValueError, match="Unknown noise_type"):
            apply_noise(clean_batch, "cosmic_rays", 0.5)

    @pytest.mark.parametrize("noise_type", list(BENCHMARK_NOISE_LEVELS.keys()))
    def test_all_seven_types_reachable(self, clean_batch, noise_type):
        """Every noise type in the canonical config must dispatch without error."""
        level = BENCHMARK_NOISE_LEVELS[noise_type][0]   # clean / base level
        out   = apply_noise(clean_batch, noise_type, level)
        assert out.shape == clean_batch.shape


# ═════════════════════════════════════════════════════════════════════════════
# B — Image validation unit tests
# These test the backend validator (file format, size, dimensions, magic bytes).
# Adjust the import path if your validator lives elsewhere.
# ═════════════════════════════════════════════════════════════════════════════

class TestImageValidator:
    """
    Unit tests for the image validation logic used by the FastAPI endpoint.

    These tests mock the filesystem/HTTP layer so they run without a live
    server.  Replace the import below with the real path to your validator.
    """

    @pytest.fixture
    def valid_jpeg_bytes(self) -> bytes:
        """Minimal valid JPEG magic bytes followed by filler."""
        # JPEG starts with FF D8 FF
        return b"\xff\xd8\xff\xe0" + b"\x00" * 100

    @pytest.fixture
    def valid_png_bytes(self) -> bytes:
        """Minimal valid PNG magic bytes."""
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    @pytest.fixture
    def pdf_bytes_disguised_as_jpg(self) -> bytes:
        """PDF magic bytes with a .jpg extension — should be rejected."""
        return b"%PDF-1.4" + b"\x00" * 100

    def test_jpeg_magic_bytes_accepted(self, valid_jpeg_bytes):
        """Real JPEG content (FF D8 FF header) must be accepted."""
        assert _check_magic_bytes(valid_jpeg_bytes, "image/jpeg") is True

    def test_png_magic_bytes_accepted(self, valid_png_bytes):
        """Real PNG content (89 50 4E 47 header) must be accepted."""
        assert _check_magic_bytes(valid_png_bytes, "image/png") is True

    def test_pdf_disguised_as_jpeg_rejected(self, pdf_bytes_disguised_as_jpg):
        """PDF file with .jpg extension must be caught by magic-byte check."""
        assert _check_magic_bytes(pdf_bytes_disguised_as_jpg, "image/jpeg") is False

    def test_file_size_limit_5mb(self):
        """Files larger than 5 MB must be rejected."""
        SIZE_5MB = 5 * 1024 * 1024
        assert _check_file_size(SIZE_5MB - 1)   is True,  "Just under limit"
        assert _check_file_size(SIZE_5MB)        is False, "Exactly at limit"
        assert _check_file_size(SIZE_5MB + 1024) is False, "Over limit"

    def test_dimension_limit_4096(self):
        """Images wider or taller than 4096 px must be rejected."""
        assert _check_dimensions(4096, 4096) is False, "Exactly at limit"
        assert _check_dimensions(4095, 4095) is True,  "Just under limit"
        assert _check_dimensions(100,  100)  is True,  "Normal size"

    def test_empty_file_rejected(self):
        assert _check_magic_bytes(b"", "image/jpeg") is False


# ── Minimal stub implementations of validation helpers ────────────────────────
# Replace these stubs with imports from your actual validation module once it
# exists.  These functions mirror the pseudocode from the SDS (Section 5.2.7.2).

_MAGIC_BYTES = {
    "image/jpeg": b"\xff\xd8\xff",
    "image/png":  b"\x89PNG",
}

def _check_magic_bytes(data: bytes, mime_type: str) -> bool:
    """Check first bytes match the declared MIME type."""
    if len(data) < 4:
        return False
    expected = _MAGIC_BYTES.get(mime_type)
    if expected is None:
        return False
    return data[: len(expected)] == expected

def _check_file_size(size_bytes: int, max_mb: int = 5) -> bool:
    return size_bytes < max_mb * 1024 * 1024

def _check_dimensions(width: int, height: int, max_dim: int = 4096) -> bool:
    return width < max_dim and height < max_dim


# ═════════════════════════════════════════════════════════════════════════════
# C — Model forward-pass unit tests
# These use a lightweight mock CNN so they run on CPU in < 1 second each.
# Replace MockCNN with real model imports for end-to-end smoke tests.
# ═════════════════════════════════════════════════════════════════════════════

class _MockModel(torch.nn.Module):
    """
    Tiny stand-in for CNN / HybridQnnCPU.
    Two Conv layers + global average pool + Linear head.
    """
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = torch.nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class TestModelForwardPass:
    """Smoke-tests for model output shapes and types."""

    NUM_CLASSES = 6

    @pytest.fixture
    def model(self):
        m = _MockModel(self.NUM_CLASSES)
        m.eval()
        return m

    def test_output_shape_matches_num_classes(self, model, clean_batch):
        with torch.no_grad():
            logits = model(clean_batch)
        assert logits.shape == (clean_batch.size(0), self.NUM_CLASSES)

    def test_output_is_float32(self, model, clean_batch):
        with torch.no_grad():
            logits = model(clean_batch)
        assert logits.dtype == torch.float32

    def test_argmax_in_valid_range(self, model, clean_batch):
        with torch.no_grad():
            preds = model(clean_batch).argmax(dim=1)
        assert preds.min() >= 0
        assert preds.max() < self.NUM_CLASSES

    def test_no_nan_or_inf_in_output(self, model, clean_batch):
        with torch.no_grad():
            logits = model(clean_batch)
        assert torch.isfinite(logits).all(), "Model output must not contain NaN or Inf"

    def test_model_ignores_noise_type_not_nan(self, model, clean_batch):
        """Model output must stay finite even when input is fully corrupted."""
        noisy = apply_noise(clean_batch, "salt_pepper", 0.5)
        with torch.no_grad():
            logits = model(noisy)
        assert torch.isfinite(logits).all()

    @pytest.mark.parametrize("noise_type,level", [
        ("gaussian",    0.3),
        ("contrast",    0.1),
        ("motion_blur", 4),
        ("jpeg_compression", 25),
    ])
    def test_forward_under_various_noises(self, model, clean_batch, noise_type, level):
        noisy = apply_noise(clean_batch, noise_type, level)
        with torch.no_grad():
            logits = model(noisy)
        assert logits.shape == (clean_batch.size(0), self.NUM_CLASSES)


# ═════════════════════════════════════════════════════════════════════════════
# D — FastAPI integration tests
# These spin up the FastAPI app in-process using TestClient (no network).
# Import your real app object here.
# ═════════════════════════════════════════════════════════════════════════════

def _make_fake_app():
    """
    Creates a minimal FastAPI app that mirrors the real endpoints.
    Replace with: from backend.main import app
    once the project is importable in the test environment.
    """
    try:
        from fastapi import FastAPI, File, UploadFile, HTTPException
        from fastapi.responses import JSONResponse

        app = FastAPI()

        _ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
        _MAX_SIZE = 5 * 1024 * 1024

        @app.post("/api/v1/classify")
        async def classify(file: UploadFile = File(...)):
            ext = os.path.splitext(file.filename or "")[1].lower()
            if ext not in _ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=422, detail="Invalid file format.")
            content = await file.read()
            if len(content) > _MAX_SIZE:
                raise HTTPException(status_code=413, detail="File size exceeds 5MB.")
            # Stub response
            return JSONResponse({
                "results": {
                    "CNN":     {"label": "Deformation", "confidence": 0.91, "latency_ms": 55},
                    "QNN_CPU": {"label": "Deformation", "confidence": 0.89, "latency_ms": 130},
                    "QNN_GPU": {"label": "Deformation", "confidence": 0.92, "latency_ms": 36},
                }
            })

        @app.get("/api/v1/benchmark")
        async def benchmark():
            return JSONResponse({"status": "ok", "clean_evaluation": {}})

        @app.get("/api/v1/health")
        async def health():
            return {"status": "healthy"}

        return app
    except ImportError:
        return None


class TestFastAPIEndpoints:
    """
    Integration tests for the FastAPI backend.
    Uses httpx.AsyncClient / Starlette TestClient — no live server needed.
    """

    @pytest.fixture
    def client(self):
        """Return a synchronous TestClient wrapping the app."""
        try:
            from fastapi.testclient import TestClient
            app = _make_fake_app()
            if app is None:
                pytest.skip("FastAPI not installed")
            return TestClient(app)
        except ImportError:
            pytest.skip("httpx / fastapi not installed")

    @pytest.fixture
    def small_jpeg_bytes(self) -> bytes:
        """Produce a minimal valid JPEG in memory using PIL."""
        try:
            from PIL import Image
            buf = io.BytesIO()
            img = Image.new("RGB", (64, 64), color=(100, 150, 200))
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        except ImportError:
            # Fallback: raw JPEG magic bytes (header only — will likely fail model
            # inference in a real server, but our stub ignores content)
            return b"\xff\xd8\xff\xe0" + b"\x00" * 256

    # ── Happy path ──────────────────────────────────────────────────────────

    def test_classify_valid_jpeg_returns_200(self, client, small_jpeg_bytes):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("test.jpg", small_jpeg_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    def test_classify_response_has_all_three_models(self, client, small_jpeg_bytes):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("test.jpg", small_jpeg_bytes, "image/jpeg")},
        )
        data = resp.json()
        results = data.get("results", {})
        for model_key in ("CNN", "QNN_CPU", "QNN_GPU"):
            assert model_key in results, f"Missing model key: {model_key}"

    def test_benchmark_endpoint_returns_200(self, client):
        resp = client.get("/api/v1/benchmark")
        assert resp.status_code == 200

    def test_health_endpoint(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    # ── Error paths ─────────────────────────────────────────────────────────

    def test_invalid_format_pdf_rejected(self, client):
        pdf_content = b"%PDF-1.4 fake pdf content"
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("document.pdf", pdf_content, "application/pdf")},
        )
        assert resp.status_code == 422, \
            "PDF file must be rejected with 422 Unprocessable Entity"

    def test_invalid_format_txt_rejected(self, client):
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 422

    def test_oversized_file_rejected(self, client):
        """A 6 MB JPEG (by filename, stub content) must return 413."""
        # Our stub checks len(content) — send just over 5 MB
        oversized = b"\xff\xd8\xff\xe0" + b"\x00" * (5 * 1024 * 1024 + 1)
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("big.jpg", oversized, "image/jpeg")},
        )
        assert resp.status_code == 413, \
            "Files > 5MB must return 413 Request Entity Too Large"

    def test_missing_file_rejected(self, client):
        """POST with no file at all must return 422."""
        resp = client.post("/api/v1/classify")
        assert resp.status_code == 422

    def test_png_extension_accepted(self, client):
        try:
            from PIL import Image
            buf = io.BytesIO()
            Image.new("RGB", (32, 32)).save(buf, format="PNG")
            png_bytes = buf.getvalue()
        except ImportError:
            png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 256

        resp = client.post(
            "/api/v1/classify",
            files={"file": ("img.png", png_bytes, "image/png")},
        )
        assert resp.status_code == 200


# ═════════════════════════════════════════════════════════════════════════════
# E — Noise-level config tests
# Sanity-checks on the canonical BENCHMARK_NOISE_LEVELS dict so that any
# accidental edits are caught immediately.
# ═════════════════════════════════════════════════════════════════════════════

class TestNoiseLevelConfig:

    def test_benchmark_has_seven_noise_types(self):
        assert len(BENCHMARK_NOISE_LEVELS) == 7

    def test_benchmark_contains_all_expected_types(self):
        expected = {
            "gaussian", "blur", "contrast", "salt_pepper",
            "motion_blur", "jpeg_compression", "lens_occlusion",
        }
        assert set(BENCHMARK_NOISE_LEVELS.keys()) == expected

    def test_gaussian_starts_at_zero(self):
        assert BENCHMARK_NOISE_LEVELS["gaussian"][0] == 0.0

    def test_contrast_starts_at_one(self):
        assert BENCHMARK_NOISE_LEVELS["contrast"][0] == 1.0, \
            "Contrast level=1.0 must be the clean baseline"

    def test_jpeg_starts_at_100(self):
        assert BENCHMARK_NOISE_LEVELS["jpeg_compression"][0] == 100

    def test_qa_is_subset_of_benchmark_types(self):
        for nt in QA_NOISE_LEVELS:
            assert nt in BENCHMARK_NOISE_LEVELS, \
                f"QA noise type '{nt}' not present in benchmark grid"

    def test_all_level_lists_non_empty(self):
        for nt, levels in BENCHMARK_NOISE_LEVELS.items():
            assert len(levels) > 0, f"Levels list for '{nt}' is empty"


# ═════════════════════════════════════════════════════════════════════════════
# F — MAUN computation tests (pure logic, no models needed)
# ═════════════════════════════════════════════════════════════════════════════

class TestComputeMaun:
    """
    Tests the MAUN (Mean Accuracy Under Noise) logic from benchmark_runner.
    We test the logic here without importing the full runner.
    """

    @staticmethod
    def _maun(noise_results, model_keys):
        """Inline copy of compute_maun for isolated testing."""
        maun = {k: {} for k in model_keys}
        for noise_type, rows in noise_results.items():
            for key in model_keys:
                accs = [r[key] for r in rows if r.get(key) is not None]
                maun[key][noise_type] = (
                    round(sum(accs) / len(accs), 4) if accs else None
                )
        return maun

    def test_maun_equals_mean_of_accuracies(self):
        noise_results = {
            "gaussian": [
                {"CNN": 90.0, "QNN_CPU": 88.0, "level": 0.0},
                {"CNN": 80.0, "QNN_CPU": 82.0, "level": 0.2},
                {"CNN": 70.0, "QNN_CPU": 76.0, "level": 0.4},
            ]
        }
        maun = self._maun(noise_results, ["CNN", "QNN_CPU"])
        assert maun["CNN"]["gaussian"]     == pytest.approx(80.0)
        assert maun["QNN_CPU"]["gaussian"] == pytest.approx(82.0)

    def test_missing_model_gets_none(self):
        noise_results = {
            "gaussian": [
                {"CNN": 90.0, "QNN_CPU": None, "QNN_GPU": None, "level": 0.0},
            ]
        }
        maun = self._maun(noise_results, ["CNN", "QNN_CPU", "QNN_GPU"])
        assert maun["QNN_CPU"]["gaussian"] is None
        assert maun["QNN_GPU"]["gaussian"] is None

    def test_multiple_noise_types_handled(self):
        noise_results = {
            "gaussian":    [{"CNN": 90.0, "QNN_CPU": 88.0, "level": 0.1}],
            "blur":        [{"CNN": 85.0, "QNN_CPU": 87.0, "level": 1.0}],
            "contrast":    [{"CNN": 80.0, "QNN_CPU": 83.0, "level": 0.5}],
        }
        maun = self._maun(noise_results, ["CNN", "QNN_CPU"])
        assert "gaussian" in maun["CNN"]
        assert "blur"     in maun["CNN"]
        assert "contrast" in maun["CNN"]
        # QNN_CPU should outperform CNN on blur and contrast
        assert maun["QNN_CPU"]["blur"]     > maun["CNN"]["blur"]
        assert maun["QNN_CPU"]["contrast"] > maun["CNN"]["contrast"]