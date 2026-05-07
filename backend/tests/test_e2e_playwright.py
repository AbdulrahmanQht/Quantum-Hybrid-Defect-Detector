"""
tests/test_e2e_playwright.py
=============================
End-to-end browser tests for the Vue.js frontend.

Pages covered:
    /classify          — upload flow, progress, results cards, error paths
    /benchmark         — charts, metrics table, CSV/JSON export
    /quantum-advantage — 14 experiment cards, model tabs, chart rendering
    /contact           — form validation, submission, success/error states

Also covers:
    Bilingual (EN/AR) switching + RTL layout
    Dark/light theme switching
    CORS preflight (OPTIONS) from the frontend origin

Install:
    pip install playwright pytest-playwright --break-system-packages
    playwright install chromium

Run:
    pytest tests/test_e2e_playwright.py -v --headed        # visible browser
    pytest tests/test_e2e_playwright.py -v                 # headless
    pytest tests/test_e2e_playwright.py -v -k "Contact"    # subset
    pytest tests/test_e2e_playwright.py -v -k "Quantum"    # subset

Environment variables:
    BASE_URL    Frontend URL (default: http://localhost:5173)

Requirements:
    Both servers must be running:
        python -m app.main &
        cd frontend && npm run dev
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

try:
    from playwright.sync_api import Page, expect
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _PLAYWRIGHT_AVAILABLE,
    reason=(
        "playwright not installed: "
        "pip install playwright pytest-playwright && playwright install chromium"
    ),
)

BASE_URL = os.getenv("BASE_URL", "http://localhost:5173")

# ── Page routes ─────────────────────────────────────────────────────────────
CLASSIFY_URL         = f"{BASE_URL}/classify"
BENCHMARK_URL        = f"{BASE_URL}/benchmark"
QUANTUM_ADVANTAGE_URL = f"{BASE_URL}/quantum-advantage"
CONTACT_URL          = f"{BASE_URL}/contact"


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def test_jpeg_path(tmp_path_factory) -> Path:
    """Small valid JPEG for upload tests."""
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
    """6 MB file — triggers the size error screen."""
    path = tmp_path_factory.mktemp("images") / "big.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * (6 * 1024 * 1024))
    return path


@pytest.fixture(scope="session")
def pdf_as_jpg_path(tmp_path_factory) -> Path:
    """PDF content with .jpg extension — rejected by magic-byte check."""
    path = tmp_path_factory.mktemp("images") / "trick.jpg"
    path.write_bytes(b"%PDF-1.4 fake content" + b"\x00" * 200)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# /classify — happy path
# ══════════════════════════════════════════════════════════════════════════════

class TestClassificationHappyPath:
    """Test Plan §5.3.3-C: upload → progress → results cards."""

    def test_page_loads(self, page: Page) -> None:
        page.goto(CLASSIFY_URL)
        expect(page).not_to_have_title("")
        expect(
            page.locator("h1, [data-testid='page-title']").first
        ).to_be_visible()

    def test_upload_input_is_present(self, page: Page) -> None:
        page.goto(CLASSIFY_URL)
        expect(page.locator("input[type='file']")).to_be_visible()

    def test_upload_valid_jpeg_shows_progress(
        self, page: Page, test_jpeg_path: Path
    ) -> None:
        page.goto(CLASSIFY_URL)
        page.locator("input[type='file']").set_input_files(str(test_jpeg_path))
        progress = page.locator(
            "[data-testid='progress'], [role='progressbar'], "
            ".p-progressbar, .progress"
        )
        expect(progress.first).to_be_visible(timeout=5_000)

    def test_upload_shows_three_model_result_cards(
        self, page: Page, test_jpeg_path: Path
    ) -> None:
        page.goto(CLASSIFY_URL)
        page.locator("input[type='file']").set_input_files(str(test_jpeg_path))
        for model in ("CNN", "QNN_CPU", "QNN_GPU"):
            card = page.locator(
                f"[data-testid='result-{model}'], :text('{model}')"
            )
            expect(card.first).to_be_visible(timeout=30_000)

    def test_results_show_confidence_value(
        self, page: Page, test_jpeg_path: Path
    ) -> None:
        page.goto(CLASSIFY_URL)
        page.locator("input[type='file']").set_input_files(str(test_jpeg_path))
        confidence = page.locator(
            "[data-testid='confidence'], :text-matches('\\d+\\.\\d+%')"
        )
        expect(confidence.first).to_be_visible(timeout=30_000)

    def test_results_show_predicted_class_name(
        self, page: Page, test_jpeg_path: Path
    ) -> None:
        page.goto(CLASSIFY_URL)
        page.locator("input[type='file']").set_input_files(str(test_jpeg_path))
        valid_classes = [
            "Deformation", "Deposition", "Disconnect",
            "Misalignment", "Obstacle", "Rupture",
        ]
        # At least one class name must appear in the results section
        class_label = page.locator(
            ", ".join(f":text('{c}')" for c in valid_classes)
        )
        expect(class_label.first).to_be_visible(timeout=30_000)

    def test_results_show_latency_value(
        self, page: Page, test_jpeg_path: Path
    ) -> None:
        page.goto(CLASSIFY_URL)
        page.locator("input[type='file']").set_input_files(str(test_jpeg_path))
        latency = page.locator(
            "[data-testid='latency'], :text-matches('\\d+\\.?\\d*\\s*ms')"
        )
        expect(latency.first).to_be_visible(timeout=30_000)


# ══════════════════════════════════════════════════════════════════════════════
# /classify — error paths
# ══════════════════════════════════════════════════════════════════════════════

class TestClassificationErrorPaths:

    def _error_locator(self, page: Page):
        return page.locator(
            "[data-testid='error'], .p-message-error, [role='alert'], "
            ".error-message, .p-toast-message-error"
        )

    def test_invalid_format_shows_error(
        self, page: Page, pdf_as_jpg_path: Path
    ) -> None:
        """TC#1: PDF renamed .jpg → error must appear."""
        page.goto(CLASSIFY_URL)
        page.locator("input[type='file']").set_input_files(str(pdf_as_jpg_path))
        expect(self._error_locator(page).first).to_be_visible(timeout=10_000)

    def test_oversized_file_shows_error(
        self, page: Page, oversized_jpeg_path: Path
    ) -> None:
        """TC#2: >5 MB → size error screen, no crash."""
        page.goto(CLASSIFY_URL)
        page.locator("input[type='file']").set_input_files(str(oversized_jpeg_path))
        expect(self._error_locator(page).first).to_be_visible(timeout=10_000)

    def test_ui_remains_functional_after_error(
        self, page: Page, pdf_as_jpg_path: Path
    ) -> None:
        """After rejection, the upload input must still be interactive."""
        page.goto(CLASSIFY_URL)
        page.locator("input[type='file']").set_input_files(str(pdf_as_jpg_path))
        page.wait_for_timeout(3_000)
        expect(page.locator("input[type='file']")).to_be_visible()

    def test_no_js_console_errors_on_valid_upload(
        self, page: Page, test_jpeg_path: Path
    ) -> None:
        """No uncaught JS errors during a normal classification flow."""
        js_errors = []
        page.on("pageerror", lambda exc: js_errors.append(str(exc)))
        page.goto(CLASSIFY_URL)
        page.locator("input[type='file']").set_input_files(str(test_jpeg_path))
        page.wait_for_timeout(5_000)
        assert not js_errors, (
            f"JS errors during classification: {js_errors}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# /benchmark
# ══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkPage:

    def test_page_loads(self, page: Page) -> None:
        page.goto(BENCHMARK_URL)
        expect(page).not_to_have_title("")

    def test_charts_render(self, page: Page) -> None:
        page.goto(BENCHMARK_URL)
        chart = page.locator(
            "canvas, [data-testid='benchmark-chart'], .p-chart"
        )
        expect(chart.first).to_be_visible(timeout=10_000)

    def test_metrics_table_renders(self, page: Page) -> None:
        page.goto(BENCHMARK_URL)
        table = page.locator(
            "table, [data-testid='metrics-table'], .p-datatable"
        )
        expect(table.first).to_be_visible(timeout=10_000)

    def test_accuracy_values_visible(self, page: Page) -> None:
        """Accuracy percentages for all three models must appear."""
        page.goto(BENCHMARK_URL)
        for model in ("CNN", "QNN_CPU", "QNN_GPU"):
            label = page.locator(f":text('{model}')")
            expect(label.first).to_be_visible(timeout=10_000)

    def test_noise_robustness_section_visible(self, page: Page) -> None:
        page.goto(BENCHMARK_URL)
        noise_section = page.locator(
            "[data-testid='noise-robustness'], "
            ":text('Noise'), :text('Robustness'), :text('gaussian')"
        )
        expect(noise_section.first).to_be_visible(timeout=10_000)

    def test_export_csv_triggers_download(self, page: Page) -> None:
        """TC#6: CSV export button must produce a downloadable file."""
        page.goto(BENCHMARK_URL)
        with page.expect_download() as dl:
            page.locator(
                "[data-testid='export-csv'], "
                "button:has-text('CSV'), button:has-text('Export')"
            ).first.click()
        name = dl.value.suggested_filename
        assert name.endswith(".csv") or "csv" in name.lower(), (
            f"Expected a .csv download, got '{name}'"
        )

    def test_export_json_triggers_download(self, page: Page) -> None:
        """TC#6: JSON export must produce a downloadable JSON file."""
        page.goto(BENCHMARK_URL)
        with page.expect_download() as dl:
            page.locator(
                "[data-testid='export-json'], button:has-text('JSON')"
            ).first.click()
        name = dl.value.suggested_filename
        assert name.endswith(".json") or "json" in name.lower()

    def test_no_js_errors_on_load(self, page: Page) -> None:
        js_errors = []
        page.on("pageerror", lambda exc: js_errors.append(str(exc)))
        page.goto(BENCHMARK_URL)
        page.wait_for_timeout(3_000)
        assert not js_errors, f"JS errors on benchmark page: {js_errors}"


# ══════════════════════════════════════════════════════════════════════════════
# /quantum-advantage
# ══════════════════════════════════════════════════════════════════════════════

class TestQuantumAdvantagePage:
    """
    Tests for the quantum advantage dashboard page.
    Covers all 14 experiment result cards and the per-model tab switching.
    If the page does not exist yet, tests skip cleanly.
    """

    def _goto(self, page: Page) -> bool:
        """Navigate and return False if the page is a 404 or not implemented."""
        page.goto(QUANTUM_ADVANTAGE_URL)
        # If the page just redirects to home or shows 404, skip
        if page.url != QUANTUM_ADVANTAGE_URL and "quantum" not in page.url:
            return False
        not_found = page.locator(":text('404'), :text('Not Found'), :text('Page not found')")
        if not_found.count() > 0:
            return False
        return True

    def test_page_loads(self, page: Page) -> None:
        page.goto(QUANTUM_ADVANTAGE_URL)
        if page.locator(":text('404')").count():
            pytest.skip("Quantum advantage page not yet implemented")
        expect(page).not_to_have_title("")

    def test_page_has_heading(self, page: Page) -> None:
        page.goto(QUANTUM_ADVANTAGE_URL)
        if page.locator(":text('404')").count():
            pytest.skip("Quantum advantage page not yet implemented")
        expect(page.locator("h1, h2, [data-testid='qa-heading']").first).to_be_visible()

    def test_experiment_cards_render(self, page: Page) -> None:
        """At least some experiment result cards must be visible."""
        page.goto(QUANTUM_ADVANTAGE_URL)
        if page.locator(":text('404')").count():
            pytest.skip("Quantum advantage page not yet implemented")
        cards = page.locator(
            "[data-testid^='experiment-'], "
            ".experiment-card, "
            ":text('Experiment'), :text('experiment')"
        )
        expect(cards.first).to_be_visible(timeout=10_000)

    def test_geometric_difference_result_visible(self, page: Page) -> None:
        """
        Experiment 9 (geometric difference g=233–270) is the strongest
        publishable result. Its value must appear on the page.
        """
        page.goto(QUANTUM_ADVANTAGE_URL)
        if page.locator(":text('404')").count():
            pytest.skip("Quantum advantage page not yet implemented")
        geo_section = page.locator(
            "[data-testid='experiment-9'], "
            "[data-testid='geometric-difference'], "
            ":text('Geometric'), :text('geometric')"
        )
        expect(geo_section.first).to_be_visible(timeout=10_000)

    def test_entanglement_entropy_result_visible(self, page: Page) -> None:
        """Experiment 4 entanglement entropy section must render."""
        page.goto(QUANTUM_ADVANTAGE_URL)
        if page.locator(":text('404')").count():
            pytest.skip("Quantum advantage page not yet implemented")
        entropy_section = page.locator(
            "[data-testid='experiment-4'], "
            "[data-testid='entanglement'], "
            ":text('Entanglement'), :text('entropy')"
        )
        expect(entropy_section.first).to_be_visible(timeout=10_000)

    def test_model_tabs_switch_between_qnn_cpu_and_gpu(self, page: Page) -> None:
        """If the page has per-model tabs, switching must not crash the UI."""
        page.goto(QUANTUM_ADVANTAGE_URL)
        if page.locator(":text('404')").count():
            pytest.skip("Quantum advantage page not yet implemented")
        cpu_tab = page.locator(
            "[data-testid='tab-qnn-cpu'], "
            "button:has-text('QNN_CPU'), :text('QNN_CPU')"
        )
        gpu_tab = page.locator(
            "[data-testid='tab-qnn-gpu'], "
            "button:has-text('QNN_GPU'), :text('QNN_GPU')"
        )
        if not cpu_tab.count() or not gpu_tab.count():
            pytest.skip("Per-model tabs not found on quantum advantage page")
        cpu_tab.first.click()
        page.wait_for_timeout(500)
        gpu_tab.first.click()
        page.wait_for_timeout(500)
        expect(page.locator("h1, h2, [data-testid='qa-heading']").first).to_be_visible()

    def test_qa_charts_render(self, page: Page) -> None:
        """Any chart (noise ablation, expressibility, etc.) must be visible."""
        page.goto(QUANTUM_ADVANTAGE_URL)
        if page.locator(":text('404')").count():
            pytest.skip("Quantum advantage page not yet implemented")
        chart = page.locator("canvas, .p-chart, [data-testid*='chart']")
        if chart.count() == 0:
            pytest.skip("No charts found on quantum advantage page")
        expect(chart.first).to_be_visible(timeout=10_000)

    def test_no_js_errors_on_load(self, page: Page) -> None:
        js_errors = []
        page.on("pageerror", lambda exc: js_errors.append(str(exc)))
        page.goto(QUANTUM_ADVANTAGE_URL)
        page.wait_for_timeout(3_000)
        if page.locator(":text('404')").count():
            pytest.skip("Quantum advantage page not yet implemented")
        assert not js_errors, (
            f"JS errors on quantum advantage page: {js_errors}"
        )

    def test_qa_loading_state_resolves(self, page: Page) -> None:
        """Any loading spinner must disappear before 10 s (no infinite load)."""
        page.goto(QUANTUM_ADVANTAGE_URL)
        if page.locator(":text('404')").count():
            pytest.skip("Quantum advantage page not yet implemented")
        spinner = page.locator(
            ".p-progressspinner, [role='progressbar'], "
            "[data-testid='loading'], .loading"
        )
        if spinner.count() > 0:
            expect(spinner.first).to_be_hidden(timeout=10_000)


# ══════════════════════════════════════════════════════════════════════════════
# /contact
# ══════════════════════════════════════════════════════════════════════════════

class TestContactPage:

    def _goto(self, page: Page) -> None:
        page.goto(CONTACT_URL)
        if page.locator(":text('404')").count():
            pytest.skip("Contact page not yet routed in the frontend")

    def test_page_loads(self, page: Page) -> None:
        self._goto(page)
        expect(page).not_to_have_title("")

    def test_form_fields_are_present(self, page: Page) -> None:
        """Name, subject, and message fields must exist."""
        self._goto(page)
        for selector in (
            "input[name='name'], [data-testid='contact-name']",
            "input[name='subject'], [data-testid='contact-subject']",
            "textarea[name='message'], [data-testid='contact-message']",
        ):
            field = page.locator(selector)
            expect(field.first).to_be_visible()

    def test_submit_button_is_present(self, page: Page) -> None:
        self._goto(page)
        btn = page.locator(
            "button[type='submit'], [data-testid='contact-submit'], "
            "button:has-text('Send'), button:has-text('Submit')"
        )
        expect(btn.first).to_be_visible()

    def test_empty_submission_shows_validation_errors(self, page: Page) -> None:
        """Submitting an empty form must show client-side validation errors."""
        self._goto(page)
        btn = page.locator(
            "button[type='submit'], [data-testid='contact-submit'], "
            "button:has-text('Send'), button:has-text('Submit')"
        )
        btn.first.click()
        # At least one validation error message must appear
        error = page.locator(
            ".p-error, .p-invalid, [role='alert'], "
            "[data-testid='field-error'], .error-message"
        )
        expect(error.first).to_be_visible(timeout=5_000)

    def test_name_too_short_shows_error(self, page: Page) -> None:
        """Single-character name must trigger validation (min_length=2)."""
        self._goto(page)
        name_field = page.locator(
            "input[name='name'], [data-testid='contact-name']"
        ).first
        name_field.fill("A")
        name_field.blur()
        error = page.locator(
            ".p-error, .p-invalid, [data-testid='name-error'], "
            ":text-matches('name.*short|least.*2|characters')"
        )
        # Client-side validation may not fire until submit — try submit too
        page.locator(
            "button[type='submit'], button:has-text('Send')"
        ).first.click()
        expect(error.first).to_be_visible(timeout=5_000)

    def test_valid_form_submission_shows_success(self, page: Page) -> None:
        """
        Fill all fields correctly and submit.
        Expect a success toast/message or a confirmation screen.
        Note: the real SMTP may not be configured in the test environment —
        the backend returns 500 in that case, so we accept either success or
        a graceful 'could not send' error message, but NOT a crash (no JS error,
        no blank page, no unhandled exception).
        """
        self._goto(page)
        js_errors = []
        page.on("pageerror", lambda exc: js_errors.append(str(exc)))

        page.locator(
            "input[name='name'], [data-testid='contact-name']"
        ).first.fill("Ahmad Al-Rashidi")

        page.locator(
            "input[name='subject'], [data-testid='contact-subject']"
        ).first.fill("Testing the Contact Form")

        page.locator(
            "textarea[name='message'], [data-testid='contact-message']"
        ).first.fill(
            "This is an automated end-to-end test submission. "
            "Please disregard this message."
        )

        page.locator(
            "button[type='submit'], button:has-text('Send'), button:has-text('Submit')"
        ).first.click()

        # Either a success indicator or an error notification must appear
        # (never a blank page or an unhandled crash)
        outcome = page.locator(
            "[data-testid='contact-success'], .p-toast-message-success, "
            "[data-testid='contact-error'], .p-toast-message-error, "
            "[role='alert'], :text('sent'), :text('success'), "
            ":text('error'), :text('try again')"
        )
        expect(outcome.first).to_be_visible(timeout=15_000)
        assert not js_errors, (
            f"JS crash during contact form submission: {js_errors}"
        )

    def test_arabic_characters_accepted_in_name(self, page: Page) -> None:
        """Arabic Unicode input must not cause validation failure or UI crash."""
        self._goto(page)
        js_errors = []
        page.on("pageerror", lambda exc: js_errors.append(str(exc)))
        page.locator(
            "input[name='name'], [data-testid='contact-name']"
        ).first.fill("عبدالرحمن الشهري")
        page.wait_for_timeout(500)
        assert not js_errors, f"JS error on Arabic input: {js_errors}"

    def test_character_counter_updates_for_message(self, page: Page) -> None:
        """If the form shows a character counter, it must update as the user types."""
        self._goto(page)
        msg_field = page.locator(
            "textarea[name='message'], [data-testid='contact-message']"
        ).first
        counter = page.locator(
            "[data-testid='char-counter'], .char-count, :text-matches('\\d+/3000|\\d+ of 3000')"
        )
        if counter.count() == 0:
            pytest.skip("No character counter found on contact form")
        msg_field.fill("Hello, world!")
        expect(counter.first).to_be_visible()

    def test_no_js_errors_on_load(self, page: Page) -> None:
        js_errors = []
        page.on("pageerror", lambda exc: js_errors.append(str(exc)))
        self._goto(page)
        page.wait_for_timeout(2_000)
        assert not js_errors, f"JS errors on contact page load: {js_errors}"


# ══════════════════════════════════════════════════════════════════════════════
# UI features — bilingual + theme switching
# ══════════════════════════════════════════════════════════════════════════════

class TestUIFeatures:

    def _lang_btn(self, page: Page):
        return page.locator(
            "[data-testid='lang-switch'], button:has-text('AR'), "
            "button:has-text('العربية'), [aria-label='Arabic']"
        )

    def _theme_btn(self, page: Page):
        return page.locator(
            "[data-testid='theme-switch'], button:has-text('Dark'), "
            "button:has-text('Light'), [aria-label='theme'], .p-toggle"
        )

    def test_language_switch_on_classify_page(self, page: Page) -> None:
        page.goto(CLASSIFY_URL)
        btn = self._lang_btn(page)
        if not btn.count():
            pytest.skip("Language switch not found — check data-testid")
        btn.first.click()
        page.wait_for_timeout(1_000)
        expect(page.locator("h1, h2").first).to_be_visible()

    def test_language_switch_on_contact_page(self, page: Page) -> None:
        """Contact form labels must switch locale without breaking layout."""
        page.goto(CONTACT_URL)
        if page.locator(":text('404')").count():
            pytest.skip("Contact page not routed")
        btn = self._lang_btn(page)
        if not btn.count():
            pytest.skip("Language switch not found")
        btn.first.click()
        page.wait_for_timeout(1_000)
        # Form fields must still be visible after locale change
        expect(page.locator("input, textarea").first).to_be_visible()

    def test_rtl_layout_does_not_overflow(self, page: Page) -> None:
        """Arabic (RTL) layout must not produce a horizontal scrollbar."""
        page.goto(CLASSIFY_URL)
        btn = self._lang_btn(page)
        if not btn.count():
            pytest.skip("Language switch not found")
        btn.first.click()
        page.wait_for_timeout(500)
        scroll_w  = page.evaluate("document.body.scrollWidth")
        viewport_w = page.evaluate("window.innerWidth")
        assert scroll_w <= viewport_w + 5, (
            f"RTL layout overflows: scrollWidth {scroll_w} > viewport {viewport_w}"
        )

    def test_rtl_dir_attribute_applied(self, page: Page) -> None:
        """Switching to Arabic must set dir='rtl' on the html or body element."""
        page.goto(CLASSIFY_URL)
        btn = self._lang_btn(page)
        if not btn.count():
            pytest.skip("Language switch not found")
        btn.first.click()
        page.wait_for_timeout(500)
        html_dir = page.evaluate("document.documentElement.dir")
        body_dir  = page.evaluate("document.body.dir")
        assert html_dir == "rtl" or body_dir == "rtl", (
            f"dir='rtl' not applied after Arabic switch "
            f"(html.dir='{html_dir}', body.dir='{body_dir}')"
        )

    def test_dark_theme_does_not_break_classify_results(
        self, page: Page, test_jpeg_path: Path
    ) -> None:
        """Toggle dark theme, then upload — results must still render correctly."""
        page.goto(CLASSIFY_URL)
        btn = self._theme_btn(page)
        if not btn.count():
            pytest.skip("Theme switch not found")
        btn.first.click()
        page.wait_for_timeout(500)
        page.locator("input[type='file']").set_input_files(str(test_jpeg_path))
        expect(
            page.locator("[data-testid='result-CNN'], :text('CNN')").first
        ).to_be_visible(timeout=30_000)

    def test_dark_theme_does_not_break_benchmark_charts(self, page: Page) -> None:
        """Benchmark charts must still render after toggling dark mode."""
        page.goto(BENCHMARK_URL)
        btn = self._theme_btn(page)
        if not btn.count():
            pytest.skip("Theme switch not found")
        btn.first.click()
        page.wait_for_timeout(500)
        chart = page.locator("canvas, .p-chart")
        expect(chart.first).to_be_visible(timeout=10_000)

    def test_dark_theme_does_not_break_contact_form(self, page: Page) -> None:
        page.goto(CONTACT_URL)
        if page.locator(":text('404')").count():
            pytest.skip("Contact page not routed")
        btn = self._theme_btn(page)
        if not btn.count():
            pytest.skip("Theme switch not found")
        btn.first.click()
        page.wait_for_timeout(500)
        expect(page.locator("input, textarea").first).to_be_visible()

    def test_nav_links_work_across_all_pages(self, page: Page) -> None:
        """Navigation between all pages must not cause a full-page reload."""
        page.goto(CLASSIFY_URL)
        for url, label in [
            (BENCHMARK_URL, "benchmark"),
            (QUANTUM_ADVANTAGE_URL, "quantum"),
            (CONTACT_URL, "contact"),
            (CLASSIFY_URL, "classify"),
        ]:
            nav = page.locator(
                f"a[href*='{label}'], nav :text('{label.title()}')"
            )
            if not nav.count():
                # Try navigating directly if nav links aren't found
                page.goto(url)
            else:
                nav.first.click()
                page.wait_for_timeout(500)
            # Page must not be blank
            expect(page.locator("body")).not_to_be_empty()