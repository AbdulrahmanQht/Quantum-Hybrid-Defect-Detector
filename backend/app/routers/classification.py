import time
import io
import base64
import asyncio
import random
from PIL import Image
import torch
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms.functional import to_pil_image
from pydantic import BaseModel
from typing import Dict, Union, List, Optional

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException

from backend.app.limiter import limiter
from backend.utils.validate import (
    check_content_type,
    check_file_size,
    check_dimensions,
    check_magic_bytes
)
from backend.utils.noise import apply_noise, BENCHMARK_NOISE_LEVELS
from backend.utils.logger import Logger

# Schema for prediction output for each model
class ModelPrediction(BaseModel):
    predicted_index: int
    predicted_class: str
    confidence: float
    # Matches your all_class_scores logic (Dict if names exist, else List)
    all_class_scores: Union[Dict[str, float], List[float]]
    inference_latency_ms: float

# Helper for the model groups
class PredictionSet(BaseModel):
    CNN: ModelPrediction
    QNN_CPU: ModelPrediction
    QNN_GPU: Optional[ModelPrediction] = None

# Extension for noisy data
class NoisyPredictionSet(PredictionSet):
    noise_level: float
    noisy_image_base64: Optional[str] = None

# The full response for /api/v1/classify endpoint
class ClassificationResponse(BaseModel):
    filename: str
    clean_image_base64: Optional[str] = None
    clean: PredictionSet
    noisy: Optional[NoisyPredictionSet] = None
    

logger = Logger()
router = APIRouter(prefix="/api/v1", tags=["Classification"])

@router.post("/classify", response_model=ClassificationResponse)
@limiter.limit("60/minute")
async def classify_image(
    request: Request,
    file: UploadFile = File(...),
    compare_with_noise: bool = Form(False),
    noise_level: Optional[float] = Form(None),
    noise_type: Optional[str] = Form(None)
) -> ClassificationResponse:
    """
    Ingests an image payload, applies validation rules, and dispatches parallel inference jobs.

    Processing Steps:
        1. Validates file signature headers via content-types and magic bytes checks.
        2. Bypasses disk-bound bottlenecks by decoding raw bytes directly to a PyTorch tensor.
        3. Enforces a maximum resolution ceiling of 4096x4096px.
        4. Synthesizes a noisy image variant if compare_with_noise is active. If noise_type is 
           set to 'random', chains 1-3 distinct corruptions sequentially.
        5. Spins up parallel background threads to evaluate the image across the CNN, 
           QNN_CPU, and QNN_GPU (conditional on CUDA) tracks simultaneously.

    Returns:
        ClassificationResponse: Verified Pydantic output model containing filenames, 
                                base64 data URLs, scores, and execution latencies in ms.
    """
    # Validate noise requirements
    if compare_with_noise and (noise_level is None or not 0.0 <= noise_level <= 1.0):
        raise HTTPException(status_code=400, detail="noise_level must be between 0.0 and 1.0.")
    
    ml_models = request.app.state.ml_models
    inference_executor = request.app.state.inference_executor
    
    # Adding a counter to test response time before and after adding pydantic classes
    request_start = time.perf_counter()

    # Check availability of models and class names
    required_models = ["CNN", "QNN_CPU", "class_names"]
    if torch.cuda.is_available():
        required_models.append("QNN_GPU")

    missing_models = [m for m in required_models if m not in ml_models]
    if missing_models:
        logger.error(f"Classification failed: Missing models in registry: {missing_models}")
        raise HTTPException(status_code=503, detail=f"Server is not ready. Missing: {', '.join(missing_models)}",)
    
    # Fast fail checks for file type and size before processing to save resources
    if not check_content_type(file.content_type):
        logger.error("Unsupported media type. Only images are allowed.")
        raise HTTPException(status_code=415, detail="Unsupported media type. Only images are allowed.")

    # Reading all bytes at once for performance, reading 1 byte at a time was causing a bottleneck.
    file_bytes = await file.read()
    if not check_file_size(len(file_bytes)):
        logger.error("File too large")
        raise HTTPException(status_code=413, detail="File too large")
    
    # Magic bytes check: rejects files whose content doesn't match their claimed type
    if not check_magic_bytes(file_bytes, file.content_type):
        logger.error("File content does not match a supported image format.")
        raise HTTPException(status_code=415, detail="File content does not match a supported image format.")
    

    try:
        # Decode bytes DIRECTLY to a PyTorch Tensor (Bypasses PIL entirely)
        # Convert bytes to a 1D uint8 tensor, then decode to an image tensor in RGB format
        raw_tensor = torch.frombuffer(bytes(file_bytes), dtype=torch.uint8)
        img_tensor = decode_image(raw_tensor, mode=ImageReadMode.RGB)

        # Validate dimensions manually since we aren't using PIL
        _, height, width = img_tensor.shape
        if not check_dimensions(width, height):
            raise HTTPException(status_code=400, detail="Image dimensions exceed 4096x4096.")

    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        raise HTTPException(status_code=400, detail="Corrupted or invalid image file.")

    # Parallel Inference Setup
    class_names = ml_models["class_names"]
    cnn_setup = ml_models["CNN"]
    qnn_cpu_setup = ml_models["QNN_CPU"]
    qnn_gpu_setup = ml_models.get("QNN_GPU") # Safe access

    try:
        # Reuse CNN's transform — all models share the same preprocessing
        transform_pipeline = cnn_setup["model"].inference_transform
        clean_input_tensor = transform_pipeline(img_tensor).unsqueeze(0)  # [1, 3, 384, 384]
        
        # Squeeze to 3D for image conversion and noise application
        
        # Convert clean tensor to Base64
        clean_tensor_3d = clean_input_tensor.squeeze(0)
        clean_pil = to_pil_image(clean_tensor_3d.cpu())
        clean_buf = io.BytesIO()
        clean_pil.save(clean_buf, format="JPEG", quality=85)
        clean_b64_str = base64.b64encode(clean_buf.getvalue()).decode('utf-8')
        clean_base64 = f"data:image/jpeg;base64,{clean_b64_str}"
        
        # Generate Noisy Tensor (Logic from models/benchmark.py)
        noisy_input_tensor = None
        noisy_base64 = None
        if compare_with_noise:
            valid_noises = list(BENCHMARK_NOISE_LEVELS.keys())
            
            # 1. Determine which noises to apply
            if not noise_type or noise_type == "random":
                num_to_mix = random.randint(1, 3)
                selected_noises = random.sample(valid_noises, num_to_mix)
            else:
                selected_noises = [noise_type]
            
            # Start with the clean tensor
            noisy_input_tensor = clean_input_tensor.clone()
            
            # 2. Apply each noise sequentially
            for n_type in selected_noises:
                canonical_levels = BENCHMARK_NOISE_LEVELS.get(n_type)
                if not canonical_levels:
                    raise HTTPException(status_code=400, detail=f"Invalid noise type: {n_type}")

                # Map the 0.0 - 1.0 slider to an array index for this specific noise
                max_idx = len(canonical_levels) - 1
                mapped_idx = int(round(noise_level * max_idx))
                current_noise_value = canonical_levels[mapped_idx]
                
                # Pass the 4D tensor directly (B, C, H, W)
                noisy_input_tensor = apply_noise(noisy_input_tensor, n_type, current_noise_value)
            
            # Squeeze it to 3D (C, H, W) ONLY for the Base64 PIL conversion
            noisy_tensor_3d = noisy_input_tensor.squeeze(0)
            
            # Convert Tensor to Base64
            pil_img = to_pil_image(noisy_tensor_3d.cpu())
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG", quality=85)
            encoded_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            noisy_base64 = f"data:image/jpeg;base64,{encoded_str}"
            
        # Handles both clean and noisy submissions
        # Submit all inference jobs to the thread pool simultaneously.
        # PyTorch releases the GIL during C++/CUDA ops, so these genuinely run in parallel.
        futures = {}
        def submit_path(prefix, tensor):
            futures[f"{prefix}_CNN"] = inference_executor.submit(
                cnn_setup["model"].predict, tensor, cnn_setup["device"], class_names)
            
            futures[f"{prefix}_QNN_CPU"] = inference_executor.submit(
                qnn_cpu_setup["model"].predict, tensor, qnn_cpu_setup["device"], class_names)
            
            if qnn_gpu_setup:
                futures[f"{prefix}_QNN_GPU"] = inference_executor.submit(
                    qnn_gpu_setup["model"].predict, tensor, qnn_gpu_setup["device"], class_names)

        # Execute paths
        submit_path("clean", clean_input_tensor)
        if compare_with_noise:
            submit_path("noisy", noisy_input_tensor)

        # Collect results — .result() blocks until that future is done
        loop = asyncio.get_event_loop()
        keys = list(futures.keys())
        gathered = await asyncio.gather(
            *[loop.run_in_executor(None, futures[k].result) for k in keys]
        )
        results = dict(zip(keys, gathered))

        results_data = {
            "filename": file.filename,
            "clean_image_base64": clean_base64,
            "clean": {
                "CNN": results["clean_CNN"],
                "QNN_CPU": results["clean_QNN_CPU"],
                "QNN_GPU": results.get("clean_QNN_GPU")
            },
            "noisy": None
        }

        if compare_with_noise:
            results_data["noisy"] = {
                "noise_level": noise_level,
                "noisy_image_base64": noisy_base64,
                "CNN": results["noisy_CNN"],
                "QNN_CPU": results["noisy_QNN_CPU"],
                "QNN_GPU": results.get("noisy_QNN_GPU")
            }
            
        total_time_ms = (time.perf_counter() - request_start) * 1000
        logger.info(f"Full Request Processed: {total_time_ms:.2f}ms | Filename: {file.filename}")
        return results_data
    
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during classification. Please try again.")
