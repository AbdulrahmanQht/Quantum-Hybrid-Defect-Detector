"""
GET /api/v1/quantum-advantage
-----------------------------
Reads quantum_advantage_results.json once, validates it into a Pydantic model,
caches the result at module level, and serves subsequent requests from memory.

No refresh endpoint — the file is loaded lazily on the first request and
held for the lifetime of the process. Restart the server to pick up a new file.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.utils.logger import Logger

logger = Logger()
router = APIRouter(prefix="/api/v1", tags=["Quantum Advantage"])

QA_RESULTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "..", "..", "data", "QA", "quantum_advantage_results.json"
)

# Module-level cache — populated on the first request, never invalidated.
_qa_cache: Optional["QuantumAdvantageResults"] = None

# Pydantic models — mirror quantum_advantage_results.json exactly
class QANotes(BaseModel):
    """
    Descriptive strings stored in the config block for documentation purposes.
    FIX: noise_ablation is a new key written by Experiment 6.
    """
    entanglement_entropy: str
    branch_ablation:      str
    gradient_variance:    str
    noise_ablation:       str
 
 
class QAConfig(BaseModel):
    """
    Config block written at the top of quantum_advantage_results.json.
    """
    n_qubits: int
    q_depth: int
    entropy_samples: int
    grad_var_batches: int
    class_names: list[str]
    qa_noise_levels: dict[str, list]
    notes: QANotes
 
 
class BranchAblationMetrics(BaseModel):
    """Experiment 2 — clean branch ablation (no noise)."""
    model_config = ConfigDict(populate_by_name=True)
 
    full_accuracy: Optional[float] = None
    classical_only_accuracy: Optional[float] = None
    quantum_only_accuracy: Optional[float] = None
    quantum_gain_pct: Optional[float] = Field(None, alias="quantum_gain_%")

class ReuploadAblationMetrics(BaseModel):
    """Experiment 3 — re-upload ablation."""
    model_config = ConfigDict(populate_by_name=True)
    with_reupload_accuracy: Optional[float] = None
    without_reupload_accuracy: Optional[float] = None
    reupload_contribution_pct: Optional[float] = Field(None, alias="reupload_contribution_%")

class EntanglementMetrics(BaseModel):
    """Experiment 4 — von Neumann entropy per qubit."""
    mean_entropy_per_qubit: dict[str, Optional[float]]
    overall_mean_entropy: Optional[float] = None
    interpretation: str

class GradientVarianceMetrics(BaseModel):
    """Experiment 5 — quantum layer gradient variance."""
    target: str
    mean_grad_variance: float
    mean_grad_abs_mean: float
    n_batches: int
    interpretation: str
    
class NoiseAblationRow(BaseModel):
    """
    One row in the Experiment 6 noise ablation sweep.
 
    full_accuracy_%        — normal forward pass on noisy input
    classical_only_%       — quantum branch zeroed
    quantum_only_%         — classical branch zeroed
    quantum_noise_gain_%   — full_accuracy_% − classical_only_%
                             A positive and increasing value as noise rises
                             proves the quantum branch improves robustness.
    """
    model_config = ConfigDict(populate_by_name=True)
 
    level: float
 
    full_accuracy_pct: Optional[float] = Field(None, alias="full_accuracy_%")
    classical_only_pct: Optional[float] = Field(None, alias="classical_only_%")
    quantum_only_pct: Optional[float] = Field(None, alias="quantum_only_%")
    quantum_noise_gain_pct: Optional[float] = Field(None, alias="quantum_noise_gain_%")

class QuantumAdvantageResults(BaseModel):
    generated_at: str
    device: str
    config: QAConfig
    experiment_1_feature_orthogonality: dict[str, Optional[float]]
    experiment_2_branch_ablation: dict[str, BranchAblationMetrics]
    experiment_3_reupload_ablation: dict[str, ReuploadAblationMetrics]
    experiment_4_entanglement_entropy: dict[str, EntanglementMetrics]
    experiment_5_gradient_variance: dict[str, GradientVarianceMetrics]
    experiment_6_noise_ablation: dict[str, dict[str, list[NoiseAblationRow]]]

# Cache loader — called once
def _load_qa_results() -> QuantumAdvantageResults:
    """
    Read and validate quantum_advantage_results.json.
    Raises HTTPException on missing file or schema mismatch.
    """
    if not os.path.exists(QA_RESULTS_FILE):
        logger.error(f"Quantum Advantage file not found: {QA_RESULTS_FILE}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Quantum Advantage results are not available yet. "
                "Run quantum_advantage_runner.py to generate them."
            ),
        )

    try:
        with open(QA_RESULTS_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        exp6_raw = raw.get("experiment_6_noise_ablation", {})
        exp6_parsed: dict[str, dict[str, list[NoiseAblationRow]]] = {}
        for model_name, noise_dict in exp6_raw.items():
            exp6_parsed[model_name] = {}
            for noise_type, rows in noise_dict.items():
                exp6_parsed[model_name][noise_type] = [
                    NoiseAblationRow.model_validate(r) for r in rows
                ]
        raw["experiment_6_noise_ablation"] = exp6_parsed
 
        result = QuantumAdvantageResults.model_validate(raw)
        logger.info(
            f"Quantum Advantage results loaded and cached from {QA_RESULTS_FILE}"
        )
        return result

    except json.JSONDecodeError as exc:
        logger.error(f"quantum_advantage_results.json is malformed: {exc}")
        raise HTTPException(status_code=500, detail="Quantum Advantage file is malformed JSON.")

    except Exception as exc:
        logger.error(f"Failed to parse Quantum Advantage results: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to parse Quantum Advantage results: {exc}")


def _get_cached_qa_results() -> QuantumAdvantageResults:
    """Return the cached results, loading from disk on the very first call."""
    global _qa_cache
    if _qa_cache is None:
        _qa_cache = _load_qa_results()
    return _qa_cache


# Endpoint
@router.get(
    "/quantum-advantage",
    response_model=QuantumAdvantageResults,
    summary="Get quantum advantage results",
    description=(
        "Returns pre-computed quantum advantage metrics spanning feature "
        "orthogonality, branch ablation, re-upload ablation, entanglement "
        "entropy, gradient variance, and noise robustness ablation (Exp. 6). "
        "Results are loaded from disk once on the first request and served "
        "from memory thereafter. "
        "Re-generate by running quantum_advantage_runner.py and restarting "
        "the server."
    ),
)
def get_quantum_advantage() -> QuantumAdvantageResults:
    return _get_cached_qa_results()