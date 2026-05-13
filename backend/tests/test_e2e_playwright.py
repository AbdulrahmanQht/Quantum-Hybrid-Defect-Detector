"""
tests/test_e2e_playwright.py
=============================
End-to-end browser tests for the Vue frontend served by the FastAPI backend.

Pages covered:
    /classify          - upload flow, validation, results cards, noise compare, exports, reset
    /benchmark         - benchmark sections, charts, tabs, metrics rendering
    /quantum-advantage - experiment sections, model tabs, chart rendering
    /contact           - form validation, submission, success/error states, persistence
    /404               - invalid routes, visual components, home navigation
    /                  - Home page hero, stats, slider, team, and highlights

Also covers:
    Theme toggle visibility and functionality
    Language toggle functionality and HTML attribute updates
    Mobile navigation menu
    Footer visibility and links
    Optional mocked API responses for deterministic UI testing

Install:
    pip install playwright pytest-playwright --break-system-packages
    playwright install chromium

Run:
    pytest tests/test_e2e_playwright.py -v --headed
    pytest tests/test_e2e_playwright.py -v
    pytest tests/test_e2e_playwright.py -v -k "Contact"
    pytest tests/test_e2e_playwright.py -v -k "Quantum"
    
    pytest tests/test_e2e_playwright.py -v --junitxml=tests/results/e2e/playwright.xml # To save results.

Environment variables:
    BASE_URL    App URL under test (default: http://127.0.0.1:8000)
    MOCK_API    1 = mock backend API responses in Playwright, 0 = hit the real backend

Requirements:
    Default mode:
        Start the backend only. It serves the built frontend from frontend/dist but you need to do cd frontend && npm run build.
            python -m app.main

    Optional frontend dev mode:
        If testing against a separate Vite dev server, set BASE_URL explicitly.
            BASE_URL=http://127.0.0.1:5173 pytest backend/tests/test_e2e_playwright.py -v
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
import csv
import io
import pytest

try:
    from playwright.sync_api import Page, expect
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _PLAYWRIGHT_AVAILABLE,
    reason="playwright not installed: pip install playwright pytest-playwright && playwright install chromium",
)

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
MOCK_API = os.getenv("MOCK_API", "1") == "1"


@pytest.fixture(scope="session")
def test_jpeg_path(tmp_path_factory) -> Path:
    try:
        from PIL import Image
        path = tmp_path_factory.mktemp("images") / "defect.jpg"
        Image.new("RGB", (384, 384), color=(80, 120, 160)).save(str(path), "JPEG")
        return path
    except ImportError:
        path = tmp_path_factory.mktemp("images") / "defect.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 4096)
        return path


@pytest.fixture(scope="session")
def oversized_jpeg_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("images") / "big.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * (6 * 1024 * 1024))
    return path


@pytest.fixture(scope="session")
def fake_pdf_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("images") / "trick.jpg"
    path.write_bytes(b"%PDF-1.4 fake content" + b"\x00" * 200)
    return path


@pytest.fixture(autouse=True)
def mock_api(page: Page) -> None:
    if not MOCK_API:
        return

    classify_payload = {
        "filename": "defect.jpg",
        "clean_image_base64": "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
        "clean": {
            "CNN": {
                "predicted_index": 5,
                "predicted_class": "Rupture",
                "confidence": 0.94,
                "all_class_scores": {
                    "Deformation": 0.01,
                    "Deposition": 0.01,
                    "Disconnect": 0.01,
                    "Misalignment": 0.01,
                    "Obstacle": 0.02,
                    "Rupture": 0.94,
                },
                "inference_latency_ms": 14.2,
            },
            "QNN_CPU": {
                "predicted_index": 5,
                "predicted_class": "Rupture",
                "confidence": 0.91,
                "all_class_scores": {
                    "Deformation": 0.01,
                    "Deposition": 0.01,
                    "Disconnect": 0.02,
                    "Misalignment": 0.01,
                    "Obstacle": 0.04,
                    "Rupture": 0.91,
                },
                "inference_latency_ms": 48.7,
            },
            "QNN_GPU": {
                "predicted_index": 5,
                "predicted_class": "Rupture",
                "confidence": 0.92,
                "all_class_scores": {
                    "Deformation": 0.01,
                    "Deposition": 0.01,
                    "Disconnect": 0.01,
                    "Misalignment": 0.01,
                    "Obstacle": 0.04,
                    "Rupture": 0.92,
                },
                "inference_latency_ms": 19.6,
            },
        },
        "noisy": {
            "noise_level": 0.65,
            "noisy_image_base64": "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
            "CNN": {
                "predicted_index": 2,
                "predicted_class": "Disconnect",
                "confidence": 0.96,
                "all_class_scores": {
                    "Deformation": 0.01,
                    "Deposition": 0.00,
                    "Disconnect": 0.96,
                    "Misalignment": 0.01,
                    "Obstacle": 0.01,
                    "Rupture": 0.01,
                },
                "inference_latency_ms": 15.3,
            },
            "QNN_CPU": {
                "predicted_index": 5,
                "predicted_class": "Rupture",
                "confidence": 0.78,
                "all_class_scores": {
                    "Deformation": 0.02,
                    "Deposition": 0.02,
                    "Disconnect": 0.08,
                    "Misalignment": 0.03,
                    "Obstacle": 0.07,
                    "Rupture": 0.78,
                },
                "inference_latency_ms": 50.1,
            },
            "QNN_GPU": {
                "predicted_index": 5,
                "predicted_class": "Rupture",
                "confidence": 0.82,
                "all_class_scores": {
                    "Deformation": 0.02,
                    "Deposition": 0.02,
                    "Disconnect": 0.06,
                    "Misalignment": 0.03,
                    "Obstacle": 0.05,
                    "Rupture": 0.82,
                },
                "inference_latency_ms": 21.8,
            },
        },
    }

    benchmark_payload = {
        "generated_at": "2026-05-08T00:00:00Z",
        "device": "cuda",
        "config": {
            "test_set_size": 100,
            "image_resolution": "384x384",
            "batch_size": 32,
            "training_epochs": 75,
            "n_qubits": 6,
            "q_depth": 2,
            "class_names": ["Deformation", "Deposition", "Disconnect", "Misalignment", "Obstacle", "Rupture"],
            "noise_levels": {"gaussian": [0.1, 0.3, 0.5]},
        },
        "clean_evaluation": {
            "CNN": {
                "accuracy": 93.2,
                "n_samples": 100,
                "averages": {
                    "macro": {"precision": 0.93, "recall": 0.93, "f1": 0.93},
                    "weighted": {"precision": 0.93, "recall": 0.93, "f1": 0.93},
                },
                "confusion_matrix": [[16, 0, 0, 0, 0, 0]] * 6,
                "per_class": {
                    k: {"accuracy": 0.93, "precision": 0.93, "recall": 0.93, "f1": 0.93, "support": 16}
                    for k in ["Deformation", "Deposition", "Disconnect", "Misalignment", "Obstacle", "Rupture"]
                },
            },
            "QNN_CPU": {
                "accuracy": 92.5,
                "n_samples": 100,
                "averages": {
                    "macro": {"precision": 0.92, "recall": 0.92, "f1": 0.92},
                    "weighted": {"precision": 0.92, "recall": 0.92, "f1": 0.92},
                },
                "confusion_matrix": [[16, 0, 0, 0, 0, 0]] * 6,
                "per_class": {
                    k: {"accuracy": 0.92, "precision": 0.92, "recall": 0.92, "f1": 0.92, "support": 16}
                    for k in ["Deformation", "Deposition", "Disconnect", "Misalignment", "Obstacle", "Rupture"]
                },
            },
            "QNN_GPU": {
                "accuracy": 93.7,
                "n_samples": 100,
                "averages": {
                    "macro": {"precision": 0.94, "recall": 0.94, "f1": 0.94},
                    "weighted": {"precision": 0.94, "recall": 0.94, "f1": 0.94},
                },
                "confusion_matrix": [[16, 0, 0, 0, 0, 0]] * 6,
                "per_class": {
                    k: {"accuracy": 0.94, "precision": 0.94, "recall": 0.94, "f1": 0.94, "support": 16}
                    for k in ["Deformation", "Deposition", "Disconnect", "Misalignment", "Obstacle", "Rupture"]
                },
            },
        },
        "noise_robustness": {
            "gaussian": [
                {"level": 0.1, "CNN": 92.0, "QNN_CPU": 91.5, "QNN_GPU": 92.8},
                {"level": 0.3, "CNN": 84.0, "QNN_CPU": 86.5, "QNN_GPU": 87.4},
                {"level": 0.5, "CNN": 72.0, "QNN_CPU": 79.2, "QNN_GPU": 80.5},
            ]
        },
        "noise_maun_summary": {},
        "inference_latency_ms": {"CNN": 14.2, "QNN_CPU": 48.7, "QNN_GPU": 19.6},
        "latency_data": [
            {"model": "CNN", "avg_latency_ms": 14.2},
            {"model": "QNN_CPU", "avg_latency_ms": 48.7},
            {"model": "QNN_GPU", "avg_latency_ms": 19.6},
        ],
        "maun_summary": [
            {"model": "CNN", "gaussian": 72.0, "blur": 75.0, "contrast": 70.0, "salt_pepper": 74.0, "motion_blur": 71.0, "jpeg_compression": 76.0, "lens_occlusion": 73.0, "overall_maun": 73.0},
            {"model": "QNN_CPU", "gaussian": 79.2, "blur": 80.0, "contrast": 78.0, "salt_pepper": 81.0, "motion_blur": 79.0, "jpeg_compression": 80.0, "lens_occlusion": 78.5, "overall_maun": 79.1},
            {"model": "QNN_GPU", "gaussian": 80.5, "blur": 81.0, "contrast": 79.0, "salt_pepper": 82.0, "motion_blur": 80.0, "jpeg_compression": 81.0, "lens_occlusion": 79.0, "overall_maun": 80.4},
        ],
        "extended_robustness": [
            {"noise_type": "gaussian", "level": 0.1, "CNN_accuracy": 92.0, "QNN_CPU_accuracy": 91.5, "QNN_GPU_accuracy": 92.8, "CNN_mean_conf": 94.0, "QNN_CPU_mean_conf": 90.0, "QNN_GPU_mean_conf": 91.0},
            {"noise_type": "gaussian", "level": 0.3, "CNN_accuracy": 84.0, "QNN_CPU_accuracy": 86.5, "QNN_GPU_accuracy": 87.4, "CNN_mean_conf": 93.0, "QNN_CPU_mean_conf": 84.0, "QNN_GPU_mean_conf": 85.0},
            {"noise_type": "gaussian", "level": 0.5, "CNN_accuracy": 72.0, "QNN_CPU_accuracy": 79.2, "QNN_GPU_accuracy": 80.5, "CNN_mean_conf": 92.0, "QNN_CPU_mean_conf": 77.0, "QNN_GPU_mean_conf": 78.0},
        ],
        "per_class_analysis": [],
    }

    qa_payload = {
        "generated_at": "2026-05-08T00:00:00Z",
        "device": "cuda",
        "config": {
            "n_qubits": 6,
            "q_depth": 2,
            "entropy_samples": 32,
            "grad_var_batches": 8,
            "feature_samples": 100,
            "kernel_samples": 100,
            "expr_pairs": 64,
            "fim_samples": 64,
            "fim_n_sizes": [1000],
            "class_names": ["Deformation", "Deposition", "Disconnect", "Misalignment", "Obstacle", "Rupture"],
            "qa_noise_levels": {"gaussian": [0.1, 0.3, 0.5]},
            "notes": {
                "entanglement_entropy": "note",
                "expressibility": "note",
                "kernel_experiments": "note",
                "fim": "note",
                "parameter_matched_ablation": "note",
            },
        },
        "experiment_1_feature_orthogonality": {"qnn_gpu": 0.82},
        "experiment_2_branch_ablation": {
            "qnn_gpu": {"full_accuracy": 93.7, "classical_only_accuracy": 88.1, "quantum_only_accuracy": 71.2, "quantum_gain_%": 5.6}
        },
        "experiment_3_reupload_ablation": {
            "qnn_gpu": {"with_reupload_accuracy": 93.7, "without_reupload_accuracy": 90.9, "reupload_contribution_%": 2.8}
        },
        "experiment_4_entanglement_entropy": {
            "qnn_gpu": {"mean_entropy_per_qubit": {"0": 0.61, "1": 0.58, "2": 0.60}, "overall_mean_entropy": 0.60, "interpretation": "stable"}
        },
        "experiment_5_gradient_variance": {
            "qnn_gpu": {"target": "quantum_block", "mean_grad_variance": 0.004, "mean_grad_abs_mean": 0.012, "n_batches": 8, "interpretation": "healthy"}
        },
        "experiment_6_noise_ablation": {
            "qnn_gpu": {
                "gaussian": [
                    {"level": 0.1, "full_accuracy_%": 92.8, "classical_only_%": 90.2, "quantum_only_%": 74.0, "quantum_noise_gain_%": 2.6},
                    {"level": 0.3, "full_accuracy_%": 87.4, "classical_only_%": 82.1, "quantum_only_%": 69.1, "quantum_noise_gain_%": 5.3},
                ]
            }
        },
        "experiment_7_vqc_expressibility": {"qnn_gpu": {"kl_divergence_from_haar": 0.041, "mean_fidelity": 0.50, "std_fidelity": 0.09, "haar_reference": "haar", "n_pairs": 64, "interpretation": "good"}},
        "experiment_8_kernel_target_alignment": {"qnn_gpu": {"kta_quantum": 0.71, "kta_classical": 0.66, "kta_difference": 0.05, "quantum_wins": True, "n_samples": 100, "q_in_global_std_raw": 0.1, "normalisation_applied": True, "interpretation": "better"}},
        "experiment_9_geometric_difference": {"qnn_gpu": {"geometric_difference": 1.24, "advantage": True, "n_samples": 100, "normalisation_applied": True, "interpretation": "advantage"}},
        "experiment_10_fisher_effective_dim": {"qnn_gpu": {"n_quantum_params": 24, "effective_dimension": {"1000": 18.2}, "d_eff_per_param": {"1000": 0.758}, "fim_trace": 1.0, "fim_rank": 24, "interpretation": "efficient"}},
        "experiment_11_feature_effective_rank": {"qnn_gpu": {"z_eff_rank": 12.0, "z_dim": 16, "z_utilisation": 0.75, "q_emb_eff_rank": 18.0, "q_emb_dim": 24, "q_emb_utilisation": 0.75, "interpretation": "good"}},
        "experiment_12_intrinsic_dimension": {"qnn_gpu": {"intrinsic_dim_z": 8.1, "intrinsic_dim_q_emb": 10.3, "n_samples": 100, "interpretation": "richer"}},
        "experiment_13_linear_cka": {"qnn_gpu": {"cka_classical_vs_quantum": 0.54, "n_samples": 100, "interpretation": "complementary"}},
        "experiment_14_class_separability": {"qnn_gpu": {"fisher_criterion_z": 1.5, "fisher_criterion_z_proj": 1.8, "fisher_criterion_q_emb": 2.2, "q_advantage": True, "n_samples": 100, "interpretation": "better"}},
    }

    page.route("**/api/v1/classify", lambda route: route.fulfill(status=200, json=classify_payload))
    page.route("**/api/v1/benchmark", lambda route: route.fulfill(status=200, json=benchmark_payload))
    page.route("**/api/v1/quantum-advantage", lambda route: route.fulfill(status=200, json=qa_payload))
    page.route("**/api/v1/contact", lambda route: route.fulfill(status=200, json={"status": "success"}))


class TestHomeView:
    def test_hero_section_renders(self, page: Page) -> None:
        page.goto(BASE_URL)
        expect(page.locator(".hero-badge")).to_be_visible()
        expect(page.locator(".hero-title")).to_be_visible()
        expect(page.locator(".hero-subtitle")).to_be_visible()
        expect(page.locator(".hero-description")).to_be_visible()
        
    def test_hero_action_buttons(self, page: Page) -> None:
        page.goto(BASE_URL)
        expect(page.locator(".hero-primary-btn")).to_be_visible()
        expect(page.get_by_role("button", name=re.compile(r"benchmark", re.IGNORECASE))).to_be_visible()
        expect(page.get_by_role("button", name=re.compile(r"quantum", re.IGNORECASE))).to_be_visible()

    def test_stat_cards_render(self, page: Page) -> None:
        page.goto(BASE_URL)
        expect(page.locator(".hero-stat-grid")).to_be_visible()
        # Should have multiple stat cards
        expect(page.locator(".hero-stat-card").first).to_be_visible()

    def test_slider_card_renders_and_rotates(self, page: Page) -> None:
        page.goto(BASE_URL)
        expect(page.locator(".slider-card")).to_be_visible()
        
        # Get the initial text
        initial_title = page.locator(".slider-card .section-title").inner_text()
        
        # Click to trigger rotation
        page.locator(".slider-card").click()
        
        # Use Playwright's auto-retrying expect to wait for the DOM text to actually change
        expect(page.locator(".slider-card .section-title")).not_to_have_text(initial_title)

    def test_highlights_and_team_sections_render(self, page: Page) -> None:
        page.goto(BASE_URL)
        expect(page.locator(".highlights-section")).to_be_visible()
        expect(page.locator(".team-section")).to_be_visible()
        # Check supervisors and researchers groups
        expect(page.locator(".team-grid--supervisors")).to_be_visible()
        # Check GitHub/LinkedIn link
        expect(page.locator(".team-link").first).to_be_visible()


class TestNavBar:
    def test_desktop_nav_links(self, page: Page) -> None:
        page.goto(BASE_URL)
        nav = page.get_by_role("navigation")
        expect(nav.get_by_role("link", name=re.compile(r"Home", re.IGNORECASE))).to_be_visible()
        expect(nav.get_by_role("link", name=re.compile(r"Classify", re.IGNORECASE))).to_be_visible()

    def test_logo_link_goes_to_home(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/classify")
        page.locator(".qnn-logo").click()
        expect(page).to_have_url(f"{BASE_URL}/")

    def test_theme_toggle(self, page: Page) -> None:
        page.goto(BASE_URL)
        html = page.locator("html")
        # Click toggle
        page.locator(".qnn-actions > .qnn-icon-btn").first.click()
        # Assuming dark mode defaults to off or cookie, test class toggle
        initial_class = html.get_attribute("class") or ""
        page.locator(".qnn-actions > .qnn-icon-btn").first.click()
        new_class = html.get_attribute("class") or ""
        assert initial_class != new_class

    def test_language_toggle_switches_locale(self, page: Page) -> None:
        page.goto(BASE_URL)
        page.locator(".qnn-lang-btn").click()
        expect(page.locator("html")).to_have_attribute("lang", "ar")
        page.locator(".qnn-lang-btn").click()
        expect(page.locator("html")).to_have_attribute("lang", "en")

    def test_mobile_burger_menu(self, page: Page) -> None:
        page.set_viewport_size({"width": 375, "height": 812})
        page.goto(BASE_URL)
        # Nav should be hidden via CSS transforms/opacity, but let's check class
        nav = page.locator("#qnn-nav")
        expect(nav).not_to_have_class(re.compile(r"qnn-nav--open"))
        # Click burger
        page.locator(".qnn-burger").click()
        expect(nav).to_have_class(re.compile(r"qnn-nav--open"))
        page.set_viewport_size({"width": 1280, "height": 720})


class TestFooter:
    def test_footer_renders_and_links(self, page: Page) -> None:
        page.goto(BASE_URL)
        footer = page.locator(".site-footer")
        expect(footer).to_be_visible()
        # Added .first to avoid strict mode violations (elements appearing multiple times)
        expect(footer.get_by_text(re.compile(r"Navigation", re.IGNORECASE)).first).to_be_visible()
        expect(footer.get_by_text(re.compile(r"Research Team", re.IGNORECASE)).first).to_be_visible()
        expect(footer.get_by_text(re.compile(r"Contact", re.IGNORECASE)).first).to_be_visible()
        expect(page.locator(".site-footer__bottom-inner")).to_be_visible()


class TestNotFoundPage:
    def test_404_page_renders_on_invalid_route(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/this-is-a-random-fake-route")
        
        # Verify glitch text is visible
        expect(page.locator(".glitch").first).to_have_text("404")
        
        # Verify suggestion links block exists
        expect(page.locator(".nf-nav-grid")).to_be_visible()
        
        # Verify the "Home" CTA button is visible
        expect(page.locator(".nf-home-link")).to_be_visible()

        # Verify footer note
        expect(page.locator(".nf-footer-note")).to_be_visible()
        expect(page.locator(".nf-footer-link")).to_be_visible()

    def test_404_home_link_navigates_to_home(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/this-is-a-random-fake-route")
        page.locator(".nf-home-link").click()
        expect(page).to_have_url(f"{BASE_URL}/")

class TestClassifyPage:
    def test_classify_page_loads(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/classify")
        expect(page.get_by_role("heading", name="Defect Detection")).to_be_visible()
        expect(page.get_by_role("button", name="Choose Image")).to_be_visible()

    def test_invalid_file_shows_error(self, page: Page, fake_pdf_path: Path) -> None:
        page.goto(f"{BASE_URL}/classify")
        page.locator("input[type='file']").set_input_files(str(fake_pdf_path))
        expect(page.locator(".p-message, .field-error, [role='alert']").first).to_be_visible()

    def test_oversized_file_shows_error(self, page: Page, oversized_jpeg_path: Path) -> None:
        page.goto(f"{BASE_URL}/classify")
        page.locator("input[type='file']").set_input_files(str(oversized_jpeg_path))
        expect(page.locator(".p-message, .field-error, [role='alert']").first).to_be_visible()

    def test_clean_classification_flow_renders_results(self, page: Page, test_jpeg_path: Path) -> None:
        page.goto(f"{BASE_URL}/classify")
        page.locator("input[type='file']").set_input_files(str(test_jpeg_path))
        page.get_by_role("button", name="Run Classification").click()

        expect(page.get_by_text("Top Prediction", exact=True)).to_be_visible()
        expect(page.get_by_text("Model Comparison", exact=True)).to_be_visible()
        expect(page.get_by_text("CNN", exact=True)).to_be_visible()
        expect(page.get_by_text("QNN CPU", exact=True)).to_be_visible()
        expect(page.get_by_text("QNN GPU", exact=True)).to_be_visible()
        expect(page.get_by_text("Rupture").first).to_be_visible()
        expect(page.get_by_text("Export Results", exact=True)).to_be_visible()

    def test_noise_compare_flow_renders_compare_view(self, page: Page, test_jpeg_path: Path) -> None:
        page.goto(f"{BASE_URL}/classify")
        page.locator("input[type='file']").set_input_files(str(test_jpeg_path))
        page.locator(".p-toggleswitch").click()
        expect(page.get_by_text("Severity", exact=True)).to_be_visible()
        page.get_by_role("button", name="Run Classification").click()

        expect(page.get_by_role("button", name="Compare")).to_be_visible()
        page.get_by_role("button", name="Compare").click()
        expect(page.get_by_text("Verdict", exact=True)).to_be_visible()
        
    def test_classify_reset_clears_state(self, page: Page, test_jpeg_path: Path) -> None:
        page.goto(f"{BASE_URL}/classify")
        page.locator("input[type='file']").set_input_files(str(test_jpeg_path))
        
        # Ensure preview pane appeared
        expect(page.locator(".preview-pane").first).to_be_visible()
        
        # Click reset button
        page.get_by_role("button", name="Reset", exact=False).click()
        
        # Ensure it went back to dropzone
        expect(page.locator(".upload-dropzone")).to_be_visible()

    def test_export_buttons_download_files(self, page: Page, test_jpeg_path: Path) -> None:
        page.goto(f"{BASE_URL}/classify")
        page.locator("input[type='file']").set_input_files(str(test_jpeg_path))
        page.get_by_role("button", name="Run Classification").click()

        with page.expect_download() as csv_dl:
            page.get_by_role("button", name="CSV").click()
        assert csv_dl.value.suggested_filename.endswith(".csv")

        with page.expect_download() as json_dl:
            page.get_by_role("button", name="JSON").click()
        assert json_dl.value.suggested_filename.endswith(".json")
    
    def test_classify_csv_export_content_is_valid(self, page: Page, test_jpeg_path: Path) -> None:
        page.goto(f"{BASE_URL}/classify")
        page.locator("input[type='file']").set_input_files(str(test_jpeg_path))
        page.get_by_role("button", name="Run Classification").click()
        expect(page.get_by_text("Export Results", exact=True)).to_be_visible()

        with page.expect_download() as dl:
            page.get_by_role("button", name="CSV").click()

        path = dl.value.path()
        content = Path(path).read_text()
        rows = list(csv.reader(io.StringIO(content)))

        assert len(rows) >= 2, "CSV must have header + at least one data row"
        assert rows[0][0] == "Condition", f"Unexpected CSV header: {rows[0]}"
        assert rows[1][0] in ("Clean", "Noisy"), f"Unexpected first data row: {rows[1][0]}"

    def test_classify_json_export_content_is_valid(self, page: Page, test_jpeg_path: Path) -> None:
        page.goto(f"{BASE_URL}/classify")
        page.locator("input[type='file']").set_input_files(str(test_jpeg_path))
        page.get_by_role("button", name="Run Classification").click()
        expect(page.get_by_text("Export Results", exact=True)).to_be_visible()

        with page.expect_download() as dl:
            page.get_by_role("button", name="JSON").click()

        path = dl.value.path()
        data = json.loads(Path(path).read_text())

        assert "timestamp" in data
        assert "fileName" in data
        assert "clean" in data
        assert "topPrediction" in data["clean"]
        assert "allResults" in data["clean"]
        assert len(data["clean"]["allResults"]) == 3, "Expected results for all 3 models"


class TestBenchmarkPage:
    def test_benchmark_page_renders_current_sections(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/benchmark")
        expect(page.get_by_role("heading", name="Benchmarking the hybrid defect detection stack")).to_be_visible()
        expect(page.get_by_text("Dataset description and representative samples")).to_be_visible()
        expect(page.get_by_text("Clean performance snapshot")).to_be_visible()
        expect(page.get_by_text("Noise robustness analysis")).to_be_visible()
        expect(page.get_by_text("Model diagnostics")).to_be_visible()

    def test_benchmark_tabs_switch(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/benchmark")
        page.get_by_role("tab", name="QNN GPU").first.click()
        expect(page.get_by_text("Confusion Matrix", exact=True)).to_be_visible()
        page.get_by_role("tab", name="Gaussian").first.click()
        expect(page.get_by_text("Accuracy Under Increasing Noise", exact=True)).to_be_visible()
    
    def test_benchmark_dataset_section(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/benchmark")
        expect(page.locator(".dataset-carousel")).to_be_visible()
        expect(page.locator(".fact-grid")).to_be_visible()
        expect(page.locator(".dataset-steps")).to_be_visible()

    def test_benchmark_latency_and_charts(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/benchmark")
        # Structural DOM targeting avoids all string/i18n matching issues!
        expect(page.locator(".performance-grid")).to_be_visible()
        
        # Radar & Scatter cards
        expect(page.locator(".chart-pair-grid .chart-card").first).to_be_visible()
        expect(page.locator(".chart-pair-grid .chart-card").nth(1)).to_be_visible()
        
        # Reliability full-width chart card
        expect(page.locator(".diag-inner-card--full")).to_be_visible()

    def test_benchmark_maun_robustness(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/benchmark")
        # Robustness Table
        expect(page.locator(".bm-datatable-robustness")).to_be_visible()
        # Robustness Line Chart
        expect(page.get_by_text("Accuracy Under Increasing Noise", exact=True)).to_be_visible()


class TestQuantumAdvantagePage:
    def test_quantum_advantage_page_renders(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/quantum-advantage")
        expect(page.locator(".qa-title")).to_be_visible()

        # .first on every get_by_text to avoid strict-mode violations when
        # the same text appears in both a heading and an accordion button
        expect(page.get_by_text(re.compile(r"Feature Orthogonality", re.IGNORECASE)).first).to_be_visible()
        expect(page.get_by_text(re.compile(r"Linear CKA", re.IGNORECASE)).first).to_be_visible()
        expect(page.get_by_text(re.compile(r"Entanglement Entropy", re.IGNORECASE)).first).to_be_visible()
        expect(page.get_by_text(re.compile(r"Gradient Variance", re.IGNORECASE)).first).to_be_visible()
        expect(page.get_by_text(re.compile(r"Expressibility", re.IGNORECASE)).first).to_be_visible()
        expect(page.get_by_text(re.compile(r"Kernel Target Alignment", re.IGNORECASE)).first).to_be_visible()
        expect(page.get_by_text(re.compile(r"Geometric Difference", re.IGNORECASE)).first).to_be_visible()
        expect(page.get_by_text(re.compile(r"Fisher Effective Dimension", re.IGNORECASE)).first).to_be_visible()
        expect(page.get_by_text(re.compile(r"Feature Effective Rank", re.IGNORECASE)).first).to_be_visible()
        expect(page.get_by_text(re.compile(r"Intrinsic Dimension", re.IGNORECASE)).first).to_be_visible()
        expect(page.get_by_text(re.compile(r"Class Separability", re.IGNORECASE)).first).to_be_visible()

    def test_quantum_advantage_noise_tabs_switch(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/quantum-advantage")
        page.get_by_role("tab", name="Gaussian").click()
        expect(page.get_by_role("heading", name="Quantum Advantage Report")).to_be_visible()
        

    def test_methodology_accordion(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/quantum-advantage")
        # Updated to PrimeVue v4 correct accordion class
        page.locator(".p-accordionheader").first.click()
        expect(page.locator(".p-accordioncontent").first).to_be_visible()


class TestContactPage:
    def test_contact_page_loads(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/contact")
        expect(page.locator(".form-title")).to_be_visible()

    def test_empty_submit_shows_validation(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/contact")
        page.locator(".submit-btn").click()
        expect(page.locator(".field-error").first).to_be_visible()

    def test_contact_form_persistence_and_clear(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/contact")
        page.locator("#contact-name").fill("Test Name")
        
        # Reload to test persistence
        page.reload()
        expect(page.locator("#contact-name")).to_have_value("Test Name")
        
        # Test clear button
        expect(page.locator(".clear-fab")).to_be_visible()
        page.locator(".clear-fab").click()
        expect(page.locator("#contact-name")).to_have_value("")

    def test_successful_submit_shows_toast(self, page: Page) -> None:
        page.goto(f"{BASE_URL}/contact")
        page.locator("#contact-name").fill("Abdulrahman Alqahtani")
        page.locator("#contact-subject").fill("Playwright E2E")
        page.locator("#contact-message").fill("This is a valid test submission.")
        page.locator(".submit-btn").click()
        expect(page.locator(".p-toast-message-success")).to_be_visible()

    def test_server_error_shows_toast(self, page: Page) -> None:
        if not MOCK_API:
            pytest.skip("Requires MOCK_API=1 to mock 500 error")
            
        page.route("**/api/v1/contact", lambda route: route.fulfill(status=500, json={"detail": "Error"}))
        page.goto(f"{BASE_URL}/contact")
        page.locator("#contact-name").fill("Error Name")
        page.locator("#contact-subject").fill("Error Subject")
        page.locator("#contact-message").fill("Error Message")
        page.locator(".submit-btn").click()
        expect(page.locator(".p-toast-message-error")).to_be_visible()
        
        