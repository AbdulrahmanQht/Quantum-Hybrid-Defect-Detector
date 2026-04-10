"""
GET /api/v1/benchmark
---------------------
Reads benchmark_results.json once, validates it into a Pydantic model,
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
router = APIRouter(prefix="/api/v1", tags=["Benchmark"])

BENCHMARK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "benchmark" ,"benchmark_results.json")

# Module-level cache — populated on the first request, never invalidated.
_benchmark_cache: Optional["BenchmarkResults"] = None

# Pydantic models — mirror benchmark_results.json exactly
class BenchmarkConfig(BaseModel):
    test_set_size:    int
    image_resolution: str
    batch_size:       int
    training_epochs:  int
    n_qubits:         int
    q_depth:          int
    class_names:      list[str]
    noise_sigmas:     list[float]

class AverageMetrics(BaseModel):
    precision: float
    recall:    float
    f1:        float

class Averages(BaseModel):
    macro:    AverageMetrics
    weighted: AverageMetrics

class PerClassMetrics(BaseModel):
    accuracy:  float
    precision: float
    recall:    float
    f1:        float
    support:   int

class ModelEvaluation(BaseModel):
    accuracy:         float
    per_class:        dict[str, PerClassMetrics]  # keyed by class name
    averages:         Averages
    confusion_matrix: list[list[int]]             # [true_class][predicted_class]
    n_samples:        int

class NoiseRobustnessRow(BaseModel):
    sigma:   float
    CNN:     Optional[float] = None
    QNN_CPU: Optional[float] = None
    QNN_GPU: Optional[float] = None

class InferenceLatency(BaseModel):
    CNN:     Optional[float] = None
    QNN_CPU: Optional[float] = None
    QNN_GPU: Optional[float] = None

class BenchmarkResults(BaseModel):
    generated_at:        str
    device:              str
    config:              BenchmarkConfig
    clean_evaluation:    dict[str, ModelEvaluation]   # keyed by model name
    noise_robustness:    list[NoiseRobustnessRow]
    inference_latency_ms: InferenceLatency

# Cache loader — called once
def _load_benchmark() -> BenchmarkResults:
    """
    Read and validate benchmark_results.json.
    Raises HTTPException on missing file or schema mismatch.
    """
    if not os.path.exists(BENCHMARK_FILE):
        logger.error(f"Benchmark file not found: {BENCHMARK_FILE}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Benchmark results are not available yet. "
                "Run benchmark_runner.py to generate them."
            ),
        )

    try:
        with open(BENCHMARK_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        result = BenchmarkResults.model_validate(raw)
        logger.info(f"Benchmark results loaded and cached from {BENCHMARK_FILE}")
        return result

    except json.JSONDecodeError as exc:
        logger.error(f"benchmark_results.json is malformed: {exc}")
        raise HTTPException(status_code=500, detail="Benchmark file is malformed JSON.")

    except Exception as exc:
        logger.error(f"Failed to parse benchmark results: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to parse benchmark results: {exc}")


def _get_cached_benchmark() -> BenchmarkResults:
    """Return the cached results, loading from disk on the very first call."""
    global _benchmark_cache
    if _benchmark_cache is None:
        _benchmark_cache = _load_benchmark()
    return _benchmark_cache


# Endpoint
@router.get(
    "/benchmark",
    response_model=BenchmarkResults,
    summary="Get benchmark results",
    description=(
        "Returns pre-computed benchmark results for all three models (CNN, QNN_CPU, QNN_GPU). "
        "Results are loaded from disk once on the first request and served from memory thereafter. "
        "Re-generate results by running `benchmark_runner.py` and restarting the server."
    ),
)

def get_benchmark() -> BenchmarkResults:
    return _get_cached_benchmark()