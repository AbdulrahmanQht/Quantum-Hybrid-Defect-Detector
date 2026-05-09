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
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Response, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel, Field
import pandas as pd

from backend.utils.logger import Logger

logger = Logger()
router = APIRouter(prefix="/api/v1", tags=["Benchmark"])
limiter = Limiter(key_func=get_remote_address)

BASE_BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "benchmark")
BENCHMARK_FILE = os.path.join(BASE_BENCHMARK_DIR, "benchmark_results.json")
INFERENCE_LATENCY_FILE = os.path.join(BASE_BENCHMARK_DIR, "inference_latency.csv")
NOISE_MAUN_SUMMARY_FILE = os.path.join(BASE_BENCHMARK_DIR, "noise_maun_summary.csv")
NOISE_ROBUSTNESS_FILE = os.path.join(BASE_BENCHMARK_DIR, "noise_robustness.csv")
PER_CLASS_METRICS_FILE = os.path.join(BASE_BENCHMARK_DIR, "per_class_metrics.csv")

# Module-level cache — populated on the first request, never invalidated.
_benchmark_cache: Optional["BenchmarkResults"] = None

# Pydantic models — mirror benchmark_results.json exactly
class BenchmarkConfig(BaseModel):
    test_set_size: int
    image_resolution: str
    batch_size: int
    training_epochs: int
    n_qubits: int
    q_depth: int
    class_names: list[str]
    noise_levels: dict[str, list]

class AverageMetrics(BaseModel):
    precision: float
    recall: float
    f1: float

class Averages(BaseModel):
    macro: AverageMetrics
    weighted: AverageMetrics

class PerClassMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int

class ModelEvaluation(BaseModel):
    accuracy: float
    per_class: dict[str, PerClassMetrics]
    averages: Averages
    confusion_matrix: list[list[int]] # [true_class][predicted_class]
    n_samples: int

class NoiseRobustnessRow(BaseModel):
    level: float
    CNN: Optional[float] = None
    QNN_CPU: Optional[float] = None
    QNN_GPU: Optional[float] = None
    CNN_mean_conf: Optional[float] = None
    QNN_CPU_mean_conf: Optional[float] = None
    QNN_GPU_mean_conf: Optional[float] = None

class InferenceLatency(BaseModel):
    CNN: Optional[float] = None
    QNN_CPU: Optional[float] = None
    QNN_GPU: Optional[float] = None

class LatencyEntry(BaseModel):
    model: str
    avg_latency_ms: float

class MaunSummary(BaseModel):
    model: str
    gaussian: float
    blur: float
    contrast: float
    salt_pepper: float
    motion_blur: float
    jpeg_compression: float
    lens_occlusion: float
    overall_maun: float

class RobustnessLevel(BaseModel):
    noise_type: str
    level: float
    CNN_accuracy: float
    QNN_CPU_accuracy: float
    QNN_GPU_accuracy: float
    CNN_mean_conf: float
    QNN_CPU_mean_conf: float
    QNN_GPU_mean_conf: float

class PerClassMetric(BaseModel):
    model: str
    class_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int

class BenchmarkResults(BaseModel):
    generated_at: str
    device: str
    config: BenchmarkConfig
    clean_evaluation: dict[str, ModelEvaluation]
    noise_robustness: dict[str, list[NoiseRobustnessRow]]
    noise_maun_summary: dict[str, dict[str, Optional[float]]]
    inference_latency_ms: InferenceLatency
    latency_data: List[LatencyEntry]
    maun_summary: List[MaunSummary]
    extended_robustness: List[RobustnessLevel]
    per_class_analysis: List[PerClassMetric]
    
    
def _normalise_col(col: str) -> str:
    """Normalise CSV column names to Pydantic-compatible snake_case.
    Converts spaces to underscores and strips trailing '_%' units.
    """
    col = col.strip().replace(' ', '_')
    if col.endswith('_%'):
        col = col[:-2]
    return col

# Cache loader — called once
def _load_benchmark() -> BenchmarkResults:
    """
    Read and validate benchmark_results.json.
    Raises HTTPException on missing file or schema mismatch.
    """
    required_files = [
        BENCHMARK_FILE,
        INFERENCE_LATENCY_FILE,
        NOISE_MAUN_SUMMARY_FILE,
        NOISE_ROBUSTNESS_FILE,
        PER_CLASS_METRICS_FILE,
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"Benchmark file not found: {file_path}")
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Required benchmark file is missing: "
                    f"{os.path.basename(file_path)}"
                ),
            )
 
    try:
        with open(BENCHMARK_FILE, encoding="utf-8") as f:
            full_data = json.load(f)

        df_latency = pd.read_csv(INFERENCE_LATENCY_FILE)
        full_data["latency_data"] = df_latency.to_dict(orient="records")

        df_maun = pd.read_csv(NOISE_MAUN_SUMMARY_FILE)
        full_data["maun_summary"] = df_maun.to_dict(orient="records")

        df_robust = pd.read_csv(NOISE_ROBUSTNESS_FILE)
        df_robust.columns = [_normalise_col(c) for c in df_robust.columns]
        full_data["extended_robustness"] = df_robust.to_dict(orient="records")

        df_per_class = pd.read_csv(PER_CLASS_METRICS_FILE)
        # Rename 'class' to 'class_name' to match Pydantic; strip % from metrics
        df_per_class.columns = [_normalise_col(c) for c in df_per_class.columns]
        df_per_class.rename(columns={'class': 'class_name'}, inplace=True)
        full_data["per_class_analysis"] = df_per_class.to_dict(orient="records")

        result = BenchmarkResults.model_validate(full_data)
        
        logger.info(
            "Benchmark results loaded and cached "
            f"(generated_at={result.generated_at}, "
            f"device={result.device}, "
            f"models={list(result.clean_evaluation.keys())}, "
            f"latency_entries={len(result.latency_data)}, "
            f"robustness_rows={len(result.extended_robustness)}, "
            f"per_class_rows={len(result.per_class_analysis)})"
        )
        return result
 
    except json.JSONDecodeError as exc:
        logger.error(f"benchmark_results.json is malformed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Benchmark file is malformed JSON.",
        )
 
    except Exception as exc:
        logger.error(f"Failed to parse benchmark results: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse benchmark results: {exc}",
        )
 
 
def _get_cached_benchmark() -> BenchmarkResults:
    """Return the cached results, loading from disk on the very first call."""
    global _benchmark_cache
    if _benchmark_cache is None:
        _benchmark_cache = _load_benchmark()
    return _benchmark_cache

@router.get("/benchmark", response_model=BenchmarkResults)
@limiter.limit("300/minute")
def get_benchmark(request: Request, response: Response) -> BenchmarkResults:
    # Instruct the browser to cache this response for 10 hour (36000 seconds)
    response.headers["Cache-Control"] = "public, max-age=36000"
    return _get_cached_benchmark()