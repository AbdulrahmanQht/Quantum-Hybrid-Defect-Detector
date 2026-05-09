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

from fastapi import APIRouter, HTTPException, Response, Request
from pydantic import BaseModel, Field, ConfigDict

from backend.app.limiter import limiter

from backend.utils.logger import Logger

logger = Logger()
router = APIRouter(prefix="/api/v1", tags=["Quantum Advantage"])

QA_RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "QA", "quantum_advantage_results.json")

# Module-level cache — populated on the first request, never invalidated.
_qa_cache: Optional["QuantumAdvantageResults"] = None

# Pydantic models — mirror quantum_advantage_results.json exactly
class QANotes(BaseModel):
    """
    Descriptive strings stored in the config block for documentation purposes.
    """
    entanglement_entropy: str
    expressibility: str
    kernel_experiments: str
    fim: str
    parameter_matched_ablation: str
    noise_ablation: Optional[str] = None
 
 
class QAConfig(BaseModel):
    """
    Config block written at the top of quantum_advantage_results.json.
    """
    n_qubits: int
    q_depth: int
    entropy_samples: int
    grad_var_batches: int
    feature_samples: int  
    kernel_samples: int   
    expr_pairs: int      
    fim_samples: int     
    fim_n_sizes: list[int] 
    class_names: list[str]
    qa_noise_levels: dict[str, list[float]]
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
    
class ExpressibilityMetrics(BaseModel):
    """Experiment 7 — VQC Expressibility (KL Divergence from Haar)."""
    kl_divergence_from_haar: float
    mean_fidelity: float
    std_fidelity: float
    haar_reference: str
    n_pairs: int
    interpretation: str

class KTAMetrics(BaseModel):
    """Experiment 8 — Kernel Target Alignment (KTA)."""
    kta_quantum: float
    kta_classical: float
    kta_difference: float
    quantum_wins: bool
    n_samples: int
    q_in_global_std_raw: float
    normalisation_applied: bool
    interpretation: str

class GeometricDifferenceMetrics(BaseModel):
    """Experiment 9 — Geometric Difference (Quantum Advantage Proof)."""
    geometric_difference: float
    advantage: bool
    n_samples: int
    normalisation_applied: bool
    interpretation: str

class FisherMetrics(BaseModel):
    """Experiment 10 — Fisher Effective Dimension (Parameter Efficiency)."""
    n_params: int = Field(..., alias="n_quantum_params")
    effective_dimension: dict[str, float]
    d_eff_per_param: dict[str, float]
    fim_trace: float
    fim_rank: int
    n_samples: int
    approximation: Optional[str] = None  # Specific to CNN baseline
    interpretation: str
    model_config = ConfigDict(populate_by_name=True)

class EffectiveRankMetrics(BaseModel):
    """Experiment 11 — Feature Effective Rank (Z vs Q-Emb)."""
    z_eff_rank: float
    z_dim: int
    z_utilisation: float
    q_emb_eff_rank: float
    q_emb_dim: int
    q_emb_utilisation: float
    interpretation: str

class IntrinsicDimensionMetrics(BaseModel):
    """Experiment 12 — TwoNN Intrinsic Dimension."""
    intrinsic_dim_z: float
    intrinsic_dim_q_emb: float
    n_samples: int
    interpretation: str

class CKAMetrics(BaseModel):
    """Experiment 13 — Linear CKA (Representational Complementarity)."""
    cka_classical_vs_quantum: float
    n_samples: int
    interpretation: str

class SeparabilityMetrics(BaseModel):
    """Experiment 14 — Class Separability (Fisher Criterion)."""
    fisher_criterion_z: float
    fisher_criterion_z_proj: float
    fisher_criterion_q_emb: float
    q_advantage: bool
    n_samples: int
    interpretation: str

class QuantumAdvantageResults(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='ignore')
    generated_at: str
    device: str
    config: QAConfig
    
    # Use Dict[str, ...] to handle CPU/GPU/Baseline sub-keys
    experiment_1_feature_orthogonality: dict[str, Optional[float]] = Field(default_factory=dict)
    experiment_2_branch_ablation: dict[str, BranchAblationMetrics] = Field(default_factory=dict)
    experiment_3_reupload_ablation: dict[str, ReuploadAblationMetrics] = Field(default_factory=dict)
    experiment_4_entanglement_entropy: dict[str, EntanglementMetrics] = Field(default_factory=dict)
    experiment_5_gradient_variance: dict[str, GradientVarianceMetrics] = Field(default_factory=dict)
    
    # Nested dict for noise ablation (Level -> Metrics)
    experiment_6_noise_ablation: dict[str, dict[str, list[NoiseAblationRow]]] = Field(default_factory=dict)
    
    experiment_7_vqc_expressibility: dict[str, ExpressibilityMetrics] = Field(default_factory=dict)
    experiment_8_kernel_target_alignment: dict[str, KTAMetrics] = Field(default_factory=dict)
    experiment_9_geometric_difference: dict[str, GeometricDifferenceMetrics] = Field(default_factory=dict)
    experiment_10_fisher_effective_dim: dict[str, FisherMetrics] = Field(default_factory=dict)
    experiment_11_feature_effective_rank: dict[str, EffectiveRankMetrics] = Field(default_factory=dict)
    experiment_12_intrinsic_dimension: dict[str, IntrinsicDimensionMetrics] = Field(default_factory=dict)
    experiment_13_linear_cka: dict[str, CKAMetrics] = Field(default_factory=dict)
    experiment_14_class_separability: dict[str, SeparabilityMetrics] = Field(default_factory=dict)

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
        
        # Inject the key if it's missing to satisfy the schema
        if "experiment_6_noise_ablation" not in raw:
            raw["experiment_6_noise_ablation"] = {}

        return QuantumAdvantageResults.model_validate(raw)

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

@router.get("/quantum-advantage", response_model=QuantumAdvantageResults)
@limiter.limit("300/minute")
def get_quantum_advantage(request: Request, response: Response) -> QuantumAdvantageResults:
    # Instruct the browser to cache this response for 10 hour (36000 seconds)
    response.headers["Cache-Control"] = "public, max-age=36000"
    return _get_cached_qa_results()