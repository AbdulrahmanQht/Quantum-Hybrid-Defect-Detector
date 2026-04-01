from pydantic import BaseModel
from typing import Dict, Union, List

# Schema for prediction output for each model
class ModelPrediction(BaseModel):
    predicted_index: int
    predicted_class: str
    confidence: float
    # Matches your all_class_scores logic (Dict if names exist, else List)
    all_class_scores: Union[Dict[str, float], List[float]]
    inference_latency_ms: float

# The full response for /api/classify endpoint
class ClassificationResponse(BaseModel):
    filename: str
    Classical_CNN: ModelPrediction
    Hybrid_QNN: ModelPrediction
    GPU_Hybrid: ModelPrediction