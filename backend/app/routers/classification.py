import time
import asyncio
from PIL import Image
import torch
from torchvision.io import decode_image, ImageReadMode
from pydantic import BaseModel
from typing import Dict, Union, List, Optional

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

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

# The full response for /api/v1/classify endpoint
class ClassificationResponse(BaseModel):
    filename: str
    clean: PredictionSet
    noisy: Optional[NoisyPredictionSet] = None
    

logger = Logger()
router = APIRouter(prefix="/api/v1", tags=["Classification"])
limiter = Limiter(key_func=get_remote_address)

# Constraints for images
Image.MAX_IMAGE_PIXELS = 16777216
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_DIMENSION = 4096  # 4096x4096px

def validate_magic_bytes(data: bytes) -> bool:
    if len(data) < 12:
        return False
    signatures = [
        data[:3] == b'\xff\xd8\xff',                               # JPEG
        data[:8] == b'\x89PNG\r\n\x1a\n',                         # PNG (full sig)
        data[:4] == b'RIFF' and data[8:12] == b'WEBP',             # WebP (correct)
        data[:2] == b'BM',                                          # BMP
        data[:4] in (b'\x49\x49\x2A\x00', b'\x4D\x4D\x00\x2A'),   # TIFF LE/BE
    ]
    return any(signatures)

def apply_inference_noise(tensor: torch.Tensor, noise_level: float) -> torch.Tensor:
    """
    Maps a single severity score [0.0, 1.0] to all noise augmentation parameters.
    Mirrors the training augmentations in PreProcessing so inference noise is
    consistent with what the models were trained to handle.

    Severity bands (approximate real-world equivalents):
        0.0 - 0.3 : Normal inspection conditions  (low EM, mild motion)
        0.3 - 0.6 : Degraded conditions            (fast robot movement, dirty lens)
        0.6 - 1.0 : Severe conditions              (heavy mud, RF interference, dark pipe)
    """
    t = tensor.clone()
    s = noise_level  # shorthand

    # 1. Gaussian — capped at sigma=0.10 (QNN: ~85% at max, was ~78% before)
    # Benchmark: sigma=0.10 → QNN 85.0%, sigma=0.15 → QNN 77.9%
    if s > 0.0:
        sigma = s * 0.10
        t = torch.clamp(t + torch.randn_like(t) * sigma, 0.0, 1.0)

    # 2. Salt & pepper — activates above mild severity
    # Benchmark: amount=0.04 → QNN 82.1% (safe), amount=0.07 → QNN 75.2% (too low)
    # Max amount here = (1.0 - 0.2) * 0.025 = 0.02 → QNN ~86.7% — conservative, keep as-is
    if s > 0.2:
        amount = (s - 0.2) * 0.025
        n = int(amount * t.shape[-1] * t.shape[-2])
        if n > 0:
            salt_r = torch.randint(0, t.shape[-2], (n,))
            salt_c = torch.randint(0, t.shape[-1], (n,))
            pepp_r = torch.randint(0, t.shape[-2], (n,))
            pepp_c = torch.randint(0, t.shape[-1], (n,))
            t[:, salt_r, salt_c] = 1.0
            t[:, pepp_r, pepp_c] = 0.0

    # 3. Motion blur — activates above mild severity
    # Benchmark: blur sigma~2.0 → QNN 82.9% (k=9 maps roughly to sigma~1.5 → QNN ~87%)
    # Keep as-is — conservative enough
    if s > 0.3:
        k = max(3, int(s * 9))
        k = k if k % 2 == 1 else k + 1
        kernel = torch.zeros(1, 1, k, k, dtype=t.dtype, device=t.device)
        kernel[0, 0, k // 2, :] = 1.0 / k
        c = t.shape[0]
        blurred = torch.nn.functional.conv2d(
            t.unsqueeze(0),
            kernel.expand(c, 1, k, k),
            padding=k // 2,
            groups=c
        ).squeeze(0)
        t = torch.clamp(blurred, 0.0, 1.0)

    # 4. Contrast reduction — activates above moderate severity
    # Benchmark: factor=0.55 → QNN 83.7%, factor=0.40 → QNN 74.0%
    # Old mapping: factor reaches 0.5 at s=1.0 → borderline
    # New mapping: floor raised to 0.55 so max degradation stays above 83%
    if s > 0.4:
        factor = 1.0 - (s - 0.4) * 0.75        # was * 0.833 → reached 0.5, now reaches 0.55
        mean = t.mean(dim=(-2, -1), keepdim=True)
        t = torch.clamp(mean + factor * (t - mean), 0.0, 1.0)

    # 5. Lens occlusion — only at high severity, keep as-is
    # Small dark patches don't correlate directly to benchmark noise types
    if s > 0.6:
        _, h, w = t.shape
        num_patches = int((s - 0.6) * 5)
        for _ in range(max(1, num_patches)):
            ph = max(1, int(h * s * 0.15))
            pw = max(1, int(w * s * 0.15))
            top  = torch.randint(0, h - ph, (1,)).item()
            left = torch.randint(0, w - pw, (1,)).item()
            t[:, top:top + ph, left:left + pw] = (
                torch.rand(t.shape[0], ph, pw, device=t.device) * 0.3
            )

    return t

@router.post("/classify", response_model=ClassificationResponse)
@limiter.limit("5/minute")
async def classify_image(
    request: Request,
    file: UploadFile = File(...),
    compare_with_noise: bool = Form(False),
    noise_level: Optional[float] = Form(None)
) -> ClassificationResponse:
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
    allowed_mimes = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"]
    if file.content_type not in allowed_mimes:
        raise HTTPException(status_code=415, detail="Unsupported media type. Only images are allowed.")

    # Reading all bytes at once for performance, reading 1 byte at a time was causing a bottleneck.
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    # Magic bytes check: rejects files whose content doesn't match their claimed type
    if not validate_magic_bytes(file_bytes):
        raise HTTPException(status_code=415, detail="File content does not match a supported image format.")

    try:
        # Decode bytes DIRECTLY to a PyTorch Tensor (Bypasses PIL entirely)
        # Convert bytes to a 1D uint8 tensor, then decode to an image tensor in RGB format
        raw_tensor = torch.frombuffer(bytes(file_bytes), dtype=torch.uint8)
        img_tensor = decode_image(raw_tensor, mode=ImageReadMode.RGB)

        # Validate dimensions manually since we aren't using PIL
        _, height, width = img_tensor.shape
        if height > MAX_DIMENSION or width > MAX_DIMENSION:
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
        # Generate Noisy Tensor (Logic from models/benchmark.py)
        noisy_input_tensor = None
        if compare_with_noise:
            noisy_input_tensor = apply_inference_noise(clean_input_tensor.squeeze(0), noise_level).unsqueeze(0)
            
        futures = {}
        
        # Handles both clean and noisy submissions
        # Submit all inference jobs to the thread pool simultaneously.
        # PyTorch releases the GIL during C++/CUDA ops, so these genuinely run in parallel.
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