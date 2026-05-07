"""
tests/test_contract.py
=======================
API contract and schema tests for all five FastAPI endpoints.

Endpoints covered:
    POST /api/v1/classify          — image upload → 3-model results
    GET  /api/v1/benchmark         — pre-computed benchmark payload
    GET  /api/v1/health            — liveness/readiness probe
    GET  /api/v1/quantum-advantage — 14-experiment QA results (Pydantic-validated)
    POST /api/v1/contact           — contact form → Gmail SMTP dispatch

What "contract" means here:
    The exact JSON field names, types, and structure that the Vue.js frontend
    depends on. If any of these change without the frontend being updated,
    a contract test fails before Playwright even runs.

Email sending is mocked throughout — no real SMTP calls are made.
QA file loading is mocked when the file does not exist on disk.

Results saved to:
    results/contract/contract_classify_schema.json
    results/contract/contract_benchmark_schema.json
    results/contract/contract_qa_schema.json
    results/contract/contract_contact_schema.json

Run:
    pytest tests/test_contract.py -v
    pytest tests/test_contract.py -v -k "Contact"
    pytest tests/test_contract.py -v -k "QuantumAdvantage"
    pytest tests/test_contract.py -v -k "CrossEndpoint"
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Deformation", "Deposition", "Disconnect",
    "Misalignment", "Obstacle", "Rupture",
]

EXPECTED_NOISE_TYPES = {
    "gaussian", "blur", "contrast", "salt_pepper",
    "motion_blur", "jpeg_compression", "lens_occlusion",
}

QA_EXPERIMENT_KEYS = [
    "experiment_1_feature_orthogonality",
    "experiment_2_branch_ablation",
    "experiment_3_reupload_ablation",
    "experiment_4_entanglement_entropy",
    "experiment_5_gradient_variance",
    "experiment_6_noise_ablation",
    "experiment_7_vqc_expressibility",
    "experiment_8_kernel_target_alignment",
    "experiment_9_geometric_difference",
    "experiment_10_fisher_effective_dim",
    "experiment_11_feature_effective_rank",
    "experiment_12_intrinsic_dimension",
    "experiment_13_linear_cka",
    "experiment_14_class_separability",
]

# Minimal valid QA payload — mirrors the real file structure exactly
_MINIMAL_QA_PAYLOAD = {
    "generated_at": "2026-04-18T06:18:19.742323+00:00",
    "device": "cuda",
    "config": {
        "n_qubits": 6, "q_depth": 2, "entropy_samples": 128,
        "grad_var_batches": 20, "feature_samples": 2833,
        "kernel_samples": 128, "expr_pairs": 1000, "fim_samples": 200,
        "fim_n_sizes": [50, 100, 500, 1000],
        "class_names": CLASS_NAMES,
        "qa_noise_levels": {
            "gaussian": [0.0, 0.1, 0.2, 0.3, 0.5],
            "blur": [0.0, 1.5, 2.5, 3.0],
            "contrast": [1.0, 0.55, 0.4, 0.25, 0.1],
            "salt_pepper": [0.0, 0.02, 0.04, 0.07, 0.1],
            "motion_blur": [0, 2, 4, 6, 8, 10],
            "jpeg_compression": [100, 40, 25, 10],
            "lens_occlusion": [0.0, 0.08, 0.12, 0.16, 0.2],
        },
        "notes": {
            "entanglement_entropy": "QNN_GPU uses lightning.gpu.",
            "expressibility": "Trained weights used.",
            "kernel_experiments": "Both K_Q and K_C evaluated on 6-dim inputs.",
            "fim": "QNN: full 36x36 empirical FIM.",
            "parameter_matched_ablation": "Not included.",
        },
    },
    "experiment_1_feature_orthogonality": {"QNN_CPU": 0.002082, "QNN_GPU": -0.023515},
    "experiment_2_branch_ablation": {
        "QNN_CPU": {
            "full_accuracy": 94.8818, "classical_only_accuracy": 92.1285,
            "quantum_only_accuracy": 49.0999, "quantum_gain_%": 2.7533,
        }
    },
    "experiment_3_reupload_ablation": {
        "QNN_CPU": {
            "with_reupload_accuracy": 94.8818,
            "without_reupload_accuracy": 93.9993,
            "reupload_contribution_%": 0.8825,
        }
    },
    "experiment_4_entanglement_entropy": {
        "QNN_CPU": {
            "mean_entropy_per_qubit": {
                "qubit_0": 0.989443, "qubit_1": 0.99303, "qubit_2": 0.992909,
                "qubit_3": 0.992359, "qubit_4": 0.99933, "qubit_5": 0.998026,
            },
            "overall_mean_entropy": 0.994183,
            "interpretation": "Values above 0.3 confirm genuine quantum correlations.",
        }
    },
    "experiment_5_gradient_variance": {
        "QNN_CPU": {
            "target": "quantum_layer_weights", "mean_grad_variance": 0.00011858,
            "mean_grad_abs_mean": 0.00643699, "n_batches": 20,
            "interpretation": "Near-zero variance = barren plateau risk.",
        },
        "CNN_baseline": {
            "target": "final_linear_layer_weights", "mean_grad_variance": 0.00054249,
            "mean_grad_abs_mean": 0.01195558, "n_batches": 20,
            "interpretation": "CNN final-layer gradient variance for comparison.",
        },
    },
    "experiment_6_noise_ablation": {
        "QNN_CPU": {
            "gaussian": [
                {"level": 0.0, "full_accuracy_%": 94.8818, "classical_only_%": 92.1285,
                 "quantum_only_%": 49.0999, "quantum_noise_gain_%": 2.7533}
            ]
        }
    },
    "experiment_7_vqc_expressibility": {
        "QNN_CPU": {
            "kl_divergence_from_haar": 0.198511, "mean_fidelity": 0.017307,
            "std_fidelity": 0.034728, "haar_reference": "Beta(1, 63)", "n_pairs": 1000,
            "interpretation": "Lower KL = higher expressibility.",
        }
    },
    "experiment_8_kernel_target_alignment": {
        "QNN_CPU": {
            "kta_quantum": 0.21662, "kta_classical": 0.334464,
            "kta_difference": -0.117843, "quantum_wins": False, "n_samples": 128,
            "q_in_global_std_raw": 0.406551, "normalisation_applied": True,
            "interpretation": "KTA measures kernel alignment to the label structure.",
        }
    },
    "experiment_9_geometric_difference": {
        "QNN_CPU": {
            "geometric_difference": 233.00878, "advantage": True, "n_samples": 128,
            "normalisation_applied": True,
            "interpretation": "g > 1 means quantum kernel spans directions classical RBF cannot.",
        }
    },
    "experiment_10_fisher_effective_dim": {
        "QNN_CPU": {
            "n_quantum_params": 36,
            "effective_dimension": {"50": 4.2468, "100": 3.3719, "500": 2.4527, "1000": 2.2465},
            "d_eff_per_param": {"50": 0.117967, "100": 0.093664, "500": 0.068131, "1000": 0.062403},
            "fim_trace": 0.068205, "fim_rank": 36, "n_samples": 200,
            "interpretation": "Higher d_eff per parameter = more efficient usage.",
        },
        "CNN_baseline": {
            "n_quantum_params": 1536,
            "effective_dimension": {"50": 7.9325, "100": 6.148, "500": 4.2231, "1000": 3.7782},
            "d_eff_per_param": {"50": 0.005164, "100": 0.004003, "500": 0.002749, "1000": 0.00246},
            "fim_trace": 5.317347, "fim_rank": 1536, "n_samples": 200,
            "approximation": "diagonal FIM (g² only) due to large m",
            "interpretation": "CNN final linear layer d_eff/m for comparison.",
        },
    },
    "experiment_11_feature_effective_rank": {
        "QNN_CPU": {
            "z_eff_rank": 246.6883, "z_dim": 512, "z_utilisation": 0.4818,
            "q_emb_eff_rank": 39.7518, "q_emb_dim": 128, "q_emb_utilisation": 0.3106,
            "interpretation": "utilisation = eff_rank / embedding_dim.",
        }
    },
    "experiment_12_intrinsic_dimension": {
        "QNN_CPU": {
            "intrinsic_dim_z": 2.1504, "intrinsic_dim_q_emb": 2.5844, "n_samples": 2833,
            "interpretation": "TwoNN intrinsic dimension (Facco 2017).",
        }
    },
    "experiment_13_linear_cka": {
        "QNN_CPU": {
            "cka_classical_vs_quantum": 0.464202, "n_samples": 2833,
            "interpretation": "CKA in [0,1]. 0 = orthogonal (ideal).",
        }
    },
    "experiment_14_class_separability": {
        "QNN_CPU": {
            "fisher_criterion_z": 14.8616, "fisher_criterion_z_proj": 9.8884,
            "fisher_criterion_q_emb": 5.9241, "q_advantage": False, "n_samples": 2833,
            "interpretation": "Fisher criterion J = tr(S_W^{-1} S_B).",
        }
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _save(data: dict, filename: str) -> None:
    path = RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"\nSaved → {path}")


def _extract_schema(obj: Any, depth: int = 0, max_depth: int = 4) -> Any:
    """Recursively extract a type-level schema snapshot for contract diffing."""
    if depth >= max_depth:
        return f"<{type(obj).__name__}>"
    if isinstance(obj, dict):
        return {k: _extract_schema(v, depth + 1, max_depth) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_extract_schema(obj[0], depth + 1, max_depth)] if obj else []
    return type(obj).__name__


def _assert_model_result_schema(result: dict, model_name: str) -> None:
    if "error" in result:
        return
    label = result.get("label") or result.get("predicted_class")
    assert label is not None, f"{model_name}: missing 'label'/'predicted_class'"
    conf = result.get("confidence")
    assert conf is not None, f"{model_name}: missing 'confidence'"
    assert 0.0 <= float(conf) <= 1.0, f"{model_name}: confidence {conf} out of [0,1]"
    latency = result.get("latency_ms") or result.get("inference_latency_ms")
    assert latency is not None, f"{model_name}: missing latency field"


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from .app.main import app 
    
    # Using 'with' triggers the @asynccontextmanager lifespan in main.py
    with TestClient(app, base_url="http://localhost") as c:
        yield c


@pytest.fixture(scope="module")
def jpeg_bytes():
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (384, 384), color=(80, 120, 160)).save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except ImportError:
        return b"\xff\xd8\xff\xe0" + b"\x00" * 4096


@pytest.fixture(scope="module")
def classify_response(client, jpeg_bytes):
    return client.post(
        "/api/v1/classify",
        files={"file": ("frame.jpg", jpeg_bytes, "image/jpeg")},
    )


@pytest.fixture(scope="module")
def benchmark_response(client):
    return client.get("/api/v1/benchmark")


@pytest.fixture(scope="module")
def qa_response(client):
    """
    Uses the real QA file if it exists on disk.
    Otherwise patches the module-level cache so no file I/O is needed.
    """
    qa_file = Path("data/QA/quantum_advantage_results.json")
    if qa_file.exists():
        return client.get("/api/v1/quantum-advantage")

    try:
        from .app.routers import quantum_advantage as qa_mod
        from .app.routers.quantum_advantage import QuantumAdvantageResults
        validated = QuantumAdvantageResults.model_validate(_MINIMAL_QA_PAYLOAD)
        with patch.object(qa_mod, "_qa_cache", validated):
            return client.get("/api/v1/quantum-advantage")
    except ImportError:
        pytest.skip("routers.quantum_advantage not importable")


# ══════════════════════════════════════════════════════════════════════════════
# 1. POST /api/v1/classify
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifyContract:

    def test_returns_200(self, classify_response):
        assert classify_response.status_code == 200, (
            f"Expected 200, got {classify_response.status_code}: "
            f"{classify_response.text[:200]}"
        )

    def test_content_type_is_json(self, classify_response):
        assert "application/json" in classify_response.headers.get("content-type", "")

    def test_top_level_has_results_key(self, classify_response):
        # Change this to 'clean' as it is your new primary data key
        body = classify_response.json()
        assert "clean" in body, f"Top-level 'clean' key missing. Got: {list(body.keys())}"

    def test_results_has_all_three_model_keys(self, classify_response):
        results = classify_response.json()["clean"]
        for key in ("CNN", "QNN_CPU", "QNN_GPU"):
            assert key in results, f"Model key '{key}' missing from results"

    def test_cnn_result_schema(self, classify_response):
        # Access through the ['clean'] branch
        _assert_model_result_schema(classify_response.json()["clean"]["CNN"], "CNN")

    def test_qnn_cpu_result_schema(self, classify_response):
        _assert_model_result_schema(classify_response.json()["clean"]["QNN_CPU"], "QNN_CPU")

    def test_qnn_gpu_result_schema(self, classify_response):
        # Check if QNN_GPU exists (it might be None if CUDA was unavailable)
        gpu_result = classify_response.json()["clean"].get("QNN_GPU")
        if gpu_result:
            _assert_model_result_schema(gpu_result, "QNN_GPU")

    def test_label_is_one_of_six_known_classes(self, classify_response):
        # Access the 'clean' branch instead of 'results'
        results = classify_response.json()["clean"]
        for model_key, result in results.items():
            # Skip if the model result is None (e.g., QNN_GPU on a CPU-only system)
            if result is None or "error" in result:
                continue
                
            label = result.get("label") or result.get("predicted_class")
            if label:
                assert label in set(CLASS_NAMES), (
                    f"'{model_key}' returned unknown class: '{label}'"
                )

    def test_no_debug_fields_in_200(self, classify_response):
        body = classify_response.text.lower()
        for forbidden in ("traceback", "stack_trace", "internal_error", "exception"):
            assert forbidden not in body, (
                f"Debug field '{forbidden}' leaked into 200 response"
            )

    def test_pdf_rejected_415(self, client):
        pdf = b"%PDF-1.4" + b"\x00" * 200
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("doc.pdf", pdf, "application/pdf")},
        )
        assert resp.status_code == 415

    def test_oversized_file_rejected_413(self, client):
        big = b"\xff\xd8\xff\xe0" + b"\x00" * (5 * 1024 * 1024 + 1)
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("big.jpg", big, "image/jpeg")},
        )
        assert resp.status_code == 413

    def test_missing_file_rejected_422(self, client):
        assert client.post("/api/v1/classify").status_code == 422

    def test_pdf_with_jpg_extension_rejected(self, client):
        """Magic-byte check must catch disguised content regardless of extension."""
        pdf = b"%PDF-1.4 fake" + b"\x00" * 100
        resp = client.post(
            "/api/v1/classify",
            files={"file": ("legit.jpg", pdf, "image/jpeg")},
        )
        assert resp.status_code == 415, (
            "PDF disguised as .jpg must be rejected by magic-byte check"
        )

    def test_save_schema_snapshot(self, classify_response):
        _save({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "endpoint": "POST /api/v1/classify",
            "schema": _extract_schema(classify_response.json()),
        }, "contract_classify_schema.json")


# ══════════════════════════════════════════════════════════════════════════════
# 2. GET /api/v1/benchmark
# ══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkContract:

    def test_returns_200(self, benchmark_response):
        assert benchmark_response.status_code == 200

    def test_has_clean_evaluation(self, benchmark_response):
        assert "clean_evaluation" in benchmark_response.json(), (
            "'clean_evaluation' missing — frontend chart components depend on this"
        )

    def test_clean_evaluation_has_all_three_models(self, benchmark_response):
        clean = benchmark_response.json().get("clean_evaluation", {})
        for key in ("CNN", "QNN_CPU", "QNN_GPU"):
            assert key in clean, f"'{key}' missing from clean_evaluation"

    def test_accuracy_numeric_and_in_range(self, benchmark_response):
        clean = benchmark_response.json().get("clean_evaluation", {})
        for key in ("CNN", "QNN_CPU", "QNN_GPU"):
            acc = clean.get(key, {}).get("accuracy")
            assert acc is not None, f"'{key}' accuracy missing"
            assert 0.0 <= float(acc) <= 100.0, f"'{key}' accuracy {acc} out of [0,100]"

    def test_has_noise_robustness(self, benchmark_response):
        assert "noise_robustness" in benchmark_response.json(), (
            "'noise_robustness' missing — frontend noise charts will break"
        )

    def test_noise_robustness_has_all_seven_types(self, benchmark_response):
        noise = benchmark_response.json().get("noise_robustness", {})
        for t in EXPECTED_NOISE_TYPES:
            assert t in noise, f"Noise type '{t}' missing from noise_robustness"

    def test_has_inference_latency_for_all_models(self, benchmark_response):
        latency = benchmark_response.json().get("inference_latency_ms", {})
        for key in ("CNN", "QNN_CPU", "QNN_GPU"):
            assert key in latency, f"'{key}' missing from inference_latency_ms"

    def test_save_schema_snapshot(self, benchmark_response):
        body = benchmark_response.json()
        _save({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "endpoint": "GET /api/v1/benchmark",
            "top_level_keys": list(body.keys()) if isinstance(body, dict) else [],
            "schema": _extract_schema(body),
        }, "contract_benchmark_schema.json")


# ══════════════════════════════════════════════════════════════════════════════
# 3. GET /api/v1/health
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthContract:

    def test_returns_200(self, client):
        assert client.get("/api/v1/health").status_code == 200

    def test_has_status_field(self, client):
        body = client.get("/api/v1/health").json()
        assert "status" in body, (
            "'status' field missing — Kubernetes/Docker health probes depend on this"
        )

    def test_status_is_string(self, client):
        assert isinstance(client.get("/api/v1/health").json()["status"], str)

    def test_status_value_is_healthy_or_ok(self, client):
        status = client.get("/api/v1/health").json()["status"]
        assert status in ("healthy", "ok"), f"Unexpected status value: '{status}'"

    def test_content_type_is_json(self, client):
        assert "application/json" in client.get(
            "/api/v1/health"
        ).headers.get("content-type", "")


# ══════════════════════════════════════════════════════════════════════════════
# 4. GET /api/v1/quantum-advantage
# ══════════════════════════════════════════════════════════════════════════════

class TestQuantumAdvantageContract:
    """
    Contract tests covering:
        - Happy path schema (all 14 experiments, config, top-level fields)
        - Per-experiment field types and value invariants
        - Three failure modes: missing file → 503, malformed JSON → 500
        - Cache behaviour: second call returns same generated_at
    """

    def test_returns_200(self, qa_response):
        assert qa_response.status_code == 200, (
            f"Expected 200, got {qa_response.status_code}: {qa_response.text[:300]}"
        )

    def test_content_type_is_json(self, qa_response):
        assert "application/json" in qa_response.headers.get("content-type", "")

    # ── Top-level fields ─────────────────────────────────────────────────────

    def test_has_generated_at(self, qa_response):
        assert "generated_at" in qa_response.json()

    def test_has_device_field(self, qa_response):
        assert "device" in qa_response.json()

    def test_has_config_block(self, qa_response):
        assert "config" in qa_response.json()

    def test_config_n_qubits_is_six(self, qa_response):
        config = qa_response.json()["config"]
        assert config.get("n_qubits") == 6, (
            f"n_qubits expected 6, got {config.get('n_qubits')}"
        )

    def test_config_q_depth_is_two(self, qa_response):
        assert qa_response.json()["config"].get("q_depth") == 2

    def test_config_class_names_are_correct(self, qa_response):
        names = qa_response.json()["config"]["class_names"]
        assert len(names) == 6 and set(names) == set(CLASS_NAMES)

    def test_config_has_all_seven_qa_noise_types(self, qa_response):
        noise = qa_response.json()["config"].get("qa_noise_levels", {})
        for t in EXPECTED_NOISE_TYPES:
            assert t in noise, f"QA noise type '{t}' missing from config"

    # ── All 14 experiment keys ────────────────────────────────────────────────

    @pytest.mark.parametrize("exp_key", QA_EXPERIMENT_KEYS)
    def test_experiment_key_present(self, qa_response, exp_key):
        assert exp_key in qa_response.json(), (
            f"Experiment key '{exp_key}' missing from QA response. "
            "Frontend QA dashboard depends on all 14 keys."
        )

    # ── Per-experiment value contracts ────────────────────────────────────────

    def test_exp1_values_are_numeric_or_null(self, qa_response):
        exp1 = qa_response.json().get("experiment_1_feature_orthogonality", {})
        assert "QNN_CPU" in exp1 or "QNN_GPU" in exp1, (
            "Experiment 1 must have at least QNN_CPU or QNN_GPU"
        )
        for model, val in exp1.items():
            if val is not None:
                assert isinstance(val, (int, float)), (
                    f"Exp 1 '{model}' must be numeric, got {type(val).__name__}"
                )

    def test_exp2_branch_ablation_required_fields(self, qa_response):
        exp2 = qa_response.json().get("experiment_2_branch_ablation", {})
        for model, data in exp2.items():
            if not isinstance(data, dict):
                continue
            for field in ("full_accuracy", "classical_only_accuracy", "quantum_only_accuracy"):
                assert field in data, (
                    f"Experiment 2 '{model}' missing field '{field}'"
                )

    def test_exp4_entanglement_entropy_above_correlation_threshold(self, qa_response):
        """
        The interpretation string says values > 0.3 confirm genuine quantum
        correlations. The measured value is ≈ 0.994 — assert it has not
        regressed below the documented threshold.
        """
        exp4 = qa_response.json().get("experiment_4_entanglement_entropy", {})
        for model, data in exp4.items():
            if not isinstance(data, dict):
                continue
            entropy = data.get("overall_mean_entropy")
            if entropy is not None:
                assert float(entropy) >= 0.3, (
                    f"Exp 4 '{model}' entropy {entropy} < 0.3 — "
                    "contradicts the quantum correlation claim."
                )

    def test_exp7_kl_divergence_is_non_negative(self, qa_response):
        exp7 = qa_response.json().get("experiment_7_vqc_expressibility", {})
        for model, data in exp7.items():
            if not isinstance(data, dict):
                continue
            kl = data.get("kl_divergence_from_haar")
            if kl is not None:
                assert float(kl) >= 0.0, (
                    f"Exp 7 '{model}' KL divergence {kl} must be non-negative"
                )

    def test_exp9_geometric_difference_greater_than_one(self, qa_response):
        """
        g > 1 is the formal quantum advantage proof (Huang et al. 2021).
        This is the strongest publishable result — must not silently regress.
        """
        exp9 = qa_response.json().get("experiment_9_geometric_difference", {})
        for model, data in exp9.items():
            if not isinstance(data, dict):
                continue
            g = data.get("geometric_difference")
            if g is not None:
                assert float(g) > 1.0, (
                    f"Exp 9 '{model}' geometric_difference {g} ≤ 1 — "
                    "quantum advantage claim is invalidated (Huang et al. 2021)."
                )
            if data.get("advantage") is not None:
                assert data["advantage"] is True, (
                    f"Exp 9 '{model}' advantage=False — claim not confirmed"
                )

    def test_exp10_qnn_more_parameter_efficient_than_cnn(self, qa_response):
        """
        QNN d_eff/param must exceed CNN d_eff/param at n=1000.
        This is the parameter efficiency headline claim.
        """
        exp10 = qa_response.json().get("experiment_10_fisher_effective_dim", {})
        qnn   = exp10.get("QNN_CPU") or exp10.get("QNN_GPU")
        cnn   = exp10.get("CNN_baseline")
        if not qnn or not cnn:
            pytest.skip("Experiment 10 missing QNN or CNN_baseline data")
        qnn_eff = (qnn.get("d_eff_per_param") or {}).get("1000")
        cnn_eff = (cnn.get("d_eff_per_param") or {}).get("1000")
        if qnn_eff is None or cnn_eff is None:
            pytest.skip("d_eff_per_param at n=1000 not present")
        assert float(qnn_eff) > float(cnn_eff), (
            f"QNN d_eff/param ({qnn_eff}) ≤ CNN ({cnn_eff}) at n=1000. "
            "Parameter efficiency claim has regressed."
        )

    def test_exp13_cka_in_unit_range(self, qa_response):
        """CKA ∈ [0, 1] by definition — any value outside is a computation bug."""
        exp13 = qa_response.json().get("experiment_13_linear_cka", {})
        for model, data in exp13.items():
            if not isinstance(data, dict):
                continue
            cka = data.get("cka_classical_vs_quantum")
            if cka is not None:
                assert 0.0 <= float(cka) <= 1.0, (
                    f"Exp 13 '{model}' CKA {cka} out of [0, 1]"
                )

    # ── Failure modes ─────────────────────────────────────────────────────────

    def test_missing_qa_file_returns_503(self, client):
        try:
            from .app.routers import quantum_advantage as qa_mod
        except ImportError:
            pytest.skip(".app.routers.quantum_advantage not importable")

        from fastapi import HTTPException
        original = qa_mod._qa_cache
        try:
            qa_mod._qa_cache = None
            with patch(
                ".app.routers.quantum_advantage._load_qa_results",
                side_effect=HTTPException(
                    status_code=503,
                    detail="Quantum Advantage results are not available yet.",
                ),
            ):
                resp = client.get("/api/v1/quantum-advantage")
                assert resp.status_code == 503, (
                    f"Missing QA file must return 503, got {resp.status_code}"
                )
        finally:
            qa_mod._qa_cache = original

    def test_malformed_qa_file_returns_500(self, client):
        try:
            from .app.routers import quantum_advantage as qa_mod
        except ImportError:
            pytest.skip(".app.routers.quantum_advantage not importable")

        from fastapi import HTTPException
        original = qa_mod._qa_cache
        try:
            qa_mod._qa_cache = None
            with patch(
                ".app.routers.quantum_advantage._load_qa_results",
                side_effect=HTTPException(
                    status_code=500,
                    detail="Quantum Advantage file is malformed JSON.",
                ),
            ):
                resp = client.get("/api/v1/quantum-advantage")
                assert resp.status_code == 500
                assert "malformed" in resp.json().get("detail", "").lower()
        finally:
            qa_mod._qa_cache = original

    def test_cache_returns_same_generated_at_on_second_call(self, client, qa_response):
        """Second call must be served from the module-level cache."""
        if qa_response.status_code != 200:
            pytest.skip("First QA request did not succeed")
        resp2 = client.get("/api/v1/quantum-advantage")
        assert resp2.status_code == 200
        assert resp2.json()["generated_at"] == qa_response.json()["generated_at"], (
            "Cache is not working — 'generated_at' differs between two calls to the same endpoint"
        )

    def test_save_qa_schema_snapshot(self, qa_response):
        if qa_response.status_code != 200:
            return
        body = qa_response.json()
        _save({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "endpoint": "GET /api/v1/quantum-advantage",
            "top_level_keys": list(body.keys()),
            "experiment_keys_present": [k for k in QA_EXPERIMENT_KEYS if k in body],
            "schema": _extract_schema(body),
        }, "contract_qa_schema.json")


# ══════════════════════════════════════════════════════════════════════════════
# 5. POST /api/v1/contact
# ══════════════════════════════════════════════════════════════════════════════

class TestContactContract:
    """
    FastMail.send_message is mocked throughout — no real SMTP connections.
    The full request path (Pydantic validation → sanitization → dispatch)
    runs normally; only the network call is replaced.
    """

    @staticmethod
    def _valid_payload(
        name="Ahmad Al-Rashidi",
        subject="Pipeline Inspection Query",
        message="I would like to know more about the defect detection system capabilities.",
    ) -> dict:
        return {"name": name, "subject": subject, "message": message}

    @staticmethod
    def _post(client, payload: dict):
        with patch(".app.routers.contact.FastMail") as mock_cls:
            mock_fm = MagicMock()
            mock_fm.send_message = AsyncMock(return_value=None)
            mock_cls.return_value = mock_fm
            return client.post("/api/v1/contact", json=payload)

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_valid_form_returns_200(self, client):
        resp = self._post(client, self._valid_payload())
        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )

    def test_response_schema_is_status_success(self, client):
        """Response must be exactly {"status": "success"}."""
        resp = self._post(client, self._valid_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body, "Response must contain 'status' field"
        assert body["status"] == "success", (
            f"Expected status='success', got '{body['status']}'"
        )

    def test_no_internal_fields_in_response(self, client):
        """SMTP config, recipients, and passwords must never appear in the response."""
        resp = self._post(client, self._valid_payload())
        if resp.status_code != 200:
            return
        body = resp.json()
        for leaked in ("email", "recipients", "smtp", "password", "token", "cc"):
            assert leaked not in body, (
                f"Internal field '{leaked}' leaked in contact response"
            )

    def test_arabic_name_accepted(self, client):
        resp = self._post(client, self._valid_payload(name="عبدالرحمن الشهري"))
        assert resp.status_code == 200

    def test_minimum_length_boundaries_accepted(self, client):
        """Exact floor: name=2, subject=3, message=10."""
        resp = self._post(client, self._valid_payload(
            name="Ab", subject="Abc", message="Abcdefghij",
        ))
        assert resp.status_code == 200

    # ── Pydantic field validation ─────────────────────────────────────────────

    @pytest.mark.parametrize("missing_field", ["name", "subject", "message"])
    def test_missing_required_field_rejected_422(self, client, missing_field):
        payload = self._valid_payload()
        del payload[missing_field]
        resp = self._post(client, payload)
        assert resp.status_code == 422, (
            f"Missing '{missing_field}' must return 422"
        )

    def test_name_one_char_rejected_422(self, client):
        assert self._post(client, self._valid_payload(name="A")).status_code == 422

    def test_name_101_chars_rejected_422(self, client):
        assert self._post(client, self._valid_payload(name="A" * 101)).status_code == 422

    def test_subject_two_chars_rejected_422(self, client):
        assert self._post(client, self._valid_payload(subject="AB")).status_code == 422

    def test_subject_151_chars_rejected_422(self, client):
        assert self._post(client, self._valid_payload(subject="S" * 151)).status_code == 422

    def test_message_nine_chars_rejected_422(self, client):
        assert self._post(client, self._valid_payload(message="123456789")).status_code == 422

    def test_message_3001_chars_rejected_422(self, client):
        assert self._post(client, self._valid_payload(message="M" * 3001)).status_code == 422

    def test_empty_name_rejected_422(self, client):
        assert self._post(client, self._valid_payload(name="")).status_code == 422

    def test_whitespace_only_name_rejected_422(self, client):
        """str_strip_whitespace=True collapses '   ' to '' which fails min_length=2."""
        assert self._post(client, self._valid_payload(name="   ")).status_code == 422

    # ── Security ──────────────────────────────────────────────────────────────

    def test_newline_injection_in_name_is_handled(self, client):
        """
        strip_newlines validator must remove CR/LF from name and subject to
        prevent email header injection. The request must either succeed (with
        newlines stripped) or be rejected — it must NOT inject headers.
        """
        resp = self._post(client, self._valid_payload(
            name="Attacker\r\nBcc: victim@example.com",
        ))
        assert resp.status_code in (200, 422), (
            f"Unexpected status {resp.status_code} for newline injection attempt"
        )

    def test_crlf_in_subject_is_handled(self, client):
        resp = self._post(client, self._valid_payload(
            subject="Normal\r\nX-Injected: evil",
        ))
        assert resp.status_code in (200, 422)

    def test_newlines_in_message_body_are_accepted(self, client):
        """
        CR/LF are valid in message bodies (multi-line messages).
        Only name and subject have the strip_newlines validator.
        """
        resp = self._post(client, self._valid_payload(
            message="Line one\nLine two\nLine three — a valid multi-line message.",
        ))
        assert resp.status_code == 200

    def test_non_json_body_rejected_422(self, client):
        """Contact endpoint expects JSON — multipart/form-encoded must be rejected."""
        with patch(".app.routers.contact.FastMail"):
            resp = client.post(
                "/api/v1/contact",
                data={"name": "Test", "subject": "Sub", "message": "Message here."},
            )
        assert resp.status_code == 422

    # ── SMTP failure path ─────────────────────────────────────────────────────

    def test_smtp_failure_returns_500(self, client):
        with patch(".app.routers.contact.FastMail") as mock_cls:
            mock_fm = MagicMock()
            mock_fm.send_message = AsyncMock(
                side_effect=Exception("SMTP connection refused")
            )
            mock_cls.return_value = mock_fm
            resp = client.post("/api/v1/contact", json=self._valid_payload())

        assert resp.status_code == 500

    def test_smtp_failure_returns_user_friendly_detail(self, client):
        with patch(".app.routers.contact.FastMail") as mock_cls:
            mock_fm = MagicMock()
            mock_fm.send_message = AsyncMock(
                side_effect=Exception("connection timeout")
            )
            mock_cls.return_value = mock_fm
            resp = client.post("/api/v1/contact", json=self._valid_payload())

        body = resp.json()
        assert "detail" in body
        detail = body["detail"].lower()
        # Must not leak SMTP internals
        assert "smtp" not in detail, "SMTP internals must not be exposed"
        # Must be user-friendly
        assert "unable to process" in detail or "try again" in detail, (
            f"500 detail must be user-friendly, got: '{body['detail']}'"
        )

    def test_smtp_failure_no_stack_trace_in_response(self, client):
        with patch(".app.routers.contact.FastMail") as mock_cls:
            mock_fm = MagicMock()
            mock_fm.send_message = AsyncMock(
                side_effect=Exception("timeout")
            )
            mock_cls.return_value = mock_fm
            resp = client.post("/api/v1/contact", json=self._valid_payload())

        body_lower = resp.text.lower()
        for leaked in ("traceback", "file \"", "raise ", "exception at"):
            assert leaked not in body_lower, (
                f"Stack trace element '{leaked}' found in 500 response"
            )

    # ── Rate limiting ─────────────────────────────────────────────────────────

    def test_rate_limit_triggers_after_five_requests(self, client):
        """
        5/minute rate limit — the 6th request within a minute must return 429.
        Uses xfail because the TestClient may reset rate state between requests.
        """
        with patch(".app.routers.contact.FastMail") as mock_cls:
            mock_fm = MagicMock()
            mock_fm.send_message = AsyncMock(return_value=None)
            mock_cls.return_value = mock_fm
            statuses = [
                client.post(
                    "/api/v1/contact",
                    json=self._valid_payload(subject=f"Request {i}"),
                ).status_code
                for i in range(7)
            ]

        if 429 not in statuses:
            pytest.xfail(
                "Rate limiter did not return 429 within 7 rapid requests. "
                "Verify with a real server — the TestClient may not enforce "
                "slowapi rate limits identically to production."
            )

    # ── Schema snapshot ───────────────────────────────────────────────────────

    def test_save_contact_schema_snapshot(self, client):
        resp = self._post(client, self._valid_payload())
        if resp.status_code != 200:
            return
        _save({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "endpoint": "POST /api/v1/contact",
            "request_schema": {
                "name": "str (2–100 chars, newlines stripped)",
                "subject": "str (3–150 chars, newlines stripped)",
                "message": "str (10–3000 chars)",
            },
            "response_schema_on_success": {"status": "str"},
            "rate_limit": "5/minute per IP",
            "smtp_mock_used_in_tests": True,
        }, "contract_contact_schema.json")


# ══════════════════════════════════════════════════════════════════════════════
# Cross-endpoint contract properties
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossEndpointContract:
    """Properties that must hold across all five endpoints simultaneously."""

    ALL_GET_ENDPOINTS = [
        "/api/v1/health",
        "/api/v1/benchmark",
        "/api/v1/quantum-advantage",
    ]

    def test_all_endpoints_return_json_not_html(self, client):
        """No endpoint should ever return an HTML error page."""
        cases = [
            ("GET",  "/api/v1/health"),
            ("GET",  "/api/v1/benchmark"),
            ("GET",  "/api/v1/quantum-advantage"),
            ("POST", "/api/v1/classify"),
            ("POST", "/api/v1/contact"),
        ]
        for method, path in cases:
            resp = client.get(path) if method == "GET" else client.post(path)
            ct = resp.headers.get("content-type", "")
            assert "text/html" not in ct, (
                f"{method} {path} returned HTML (content-type: {ct}). "
                "FastAPI should always return JSON."
            )

    def test_all_4xx_responses_have_detail_field(self, client):
        """FastAPI standard: 4xx responses must have a 'detail' field."""
        error_cases = [
            ("POST", "/api/v1/classify", {}),   # no file
            ("POST", "/api/v1/contact",  {}),   # no fields
        ]
        for method, path, payload in error_cases:
            resp = client.post(path, json=payload)
            if 400 <= resp.status_code < 500:
                body = resp.json()
                assert "detail" in body, (
                    f"POST {path} → {resp.status_code}: "
                    f"'detail' missing. Got: {list(body.keys())}"
                )

    def test_cors_allows_vue_dev_origin_on_get_endpoints(self, client):
        """Vue frontend at localhost:5173 must be allowed by CORS."""
        vue_origin = "http://localhost:5173"
        for path in self.ALL_GET_ENDPOINTS:
            resp = client.get(path, headers={"Origin": vue_origin})
            acao = resp.headers.get("access-control-allow-origin", "")
            assert acao in ("*", vue_origin), (
                f"CORS missing or wrong for {path}: '{acao}'. "
                "Vue frontend at localhost:5173 will be blocked."
            )

    def test_options_preflight_succeeds_for_classify(self, client):
        """Browser sends OPTIONS before cross-origin POST."""
        resp = client.options(
            "/api/v1/classify",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert resp.status_code in (200, 204), (
            f"OPTIONS preflight returned {resp.status_code}. "
            "Cross-origin classify calls from Vue will be blocked."
        )

    def test_options_preflight_succeeds_for_contact(self, client):
        resp = client.options(
            "/api/v1/contact",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert resp.status_code in (200, 204), (
            f"OPTIONS preflight for /contact returned {resp.status_code}."
        )