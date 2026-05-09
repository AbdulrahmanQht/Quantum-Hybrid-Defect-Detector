import time
import io
import base64
import asyncio
from PIL import Image
import torch
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms.functional import to_pil_image
from pydantic import BaseModel
from typing import Dict, Union, List, Optional

from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.utils.validate import (
    check_content_type,
    check_file_size,
    check_dimensions,
    check_magic_bytes
)
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
    noisy_image_base64: str

# The full response for /api/v1/classify endpoint
class ClassificationResponse(BaseModel):
    filename: str
    clean_image_base64: str
    clean: PredictionSet
    noisy: Optional[NoisyPredictionSet] = None
    

logger = Logger()
router = APIRouter(prefix="/api/v1", tags=["Classification"])
limiter = Limiter(key_func=get_remote_address)

def apply_inference_noise(tensor: torch.Tensor, noise_level: float) -> torch.Tensor:
    """
    Revised mapping based on per-noise benchmark results.
    Accuracy targets (CNN, averaged across classes):
        0.0 – 0.3 : ~88–93%  (normal conditions)
        0.3 – 0.6 : ~70–88%  (degraded conditions)
        0.6 – 1.0 : ~40–70%  (severe conditions)

    Key changes vs prior version:
      - Gaussian uncapped: sigma now reaches 0.28 at s=1.0
        Benchmark: sigma=0.10→91%, 0.15→84%, 0.20→73%, 0.25→63%, 0.30→52%
      - Motion blur kernel reaches 13 at s=1.0 (was 9)
        Benchmark: k=6→84%, k=8→73%, k=10→61%
      - Contrast floor lowered to 0.20 (was 0.55)
        Benchmark: factor=0.40→88%, 0.25→71%, 0.10→31%
      - Salt & pepper scaled to amount=0.08 at s=1.0 (was 0.02)
        Benchmark: amount=0.04→91%, 0.07→88%, 0.10→83%
    """
    t = tensor.clone()
    s = noise_level

    # 1. Gaussian — extended to sigma=0.28 for visible high-end differentiation
    # sigma=0.10 → CNN ~91%, sigma=0.20 → CNN ~73%, sigma=0.28 → CNN ~55%
    if s > 0.0:
        sigma = s * 0.28
        t = torch.clamp(t + torch.randn_like(t) * sigma, 0.0, 1.0)

    # 2. Salt & pepper — scaled to amount=0.08 at s=1.0
    # amount=0.02 → CNN ~93%, amount=0.07 → CNN ~88%, amount=0.10 → CNN ~83%
    if s > 0.2:
        amount = (s - 0.2) * 0.10          # was 0.025, reaches 0.08 at s=1.0
        n = int(amount * t.shape[-1] * t.shape[-2])
        if n > 0:
            salt_r = torch.randint(0, t.shape[-2], (n,))
            salt_c = torch.randint(0, t.shape[-1], (n,))
            pepp_r = torch.randint(0, t.shape[-2], (n,))
            pepp_c = torch.randint(0, t.shape[-1], (n,))
            t[:, salt_r, salt_c] = 1.0
            t[:, pepp_r, pepp_c] = 0.0

    # 3. Motion blur — kernel reaches 13 at s=1.0 for severe range differentiation
    # k=3→93%, k=6→84%, k=9→77%, k=13→~65% (extrapolated from blur sigma benchmarks)
    if s > 0.3:
        k = max(3, int(s * 13))            # was s * 9
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

    # 4. Contrast reduction — floor lowered to 0.20 so high severity is visibly distinct
    # factor=0.70→CNN 94%, 0.55→92%, 0.40→88%, 0.25→71%, 0.20→~50%
    if s > 0.4:
        factor = 1.0 - (s - 0.4) * 1.333  # reaches 0.20 at s=1.0, was 0.75 → reached 0.55
        mean = t.mean(dim=(-2, -1), keepdim=True)
        t = torch.clamp(mean + factor * (t - mean), 0.0, 1.0)

    # 5. Lens occlusion — unchanged, activates at high severity only
    # Benchmark: coverage=0.08→CNN 88%, 0.12→84%, 0.16→80%, 0.20→76%
    if s > 0.6:
        _, h, w = t.shape
        hi = min(1.0, max(0.0, (s - 0.6) / 0.4))
        num_patches = 1 + int(hi * 4)
        patch_scale = 0.06 + hi * 0.14
        max_darkness = 0.45 - hi * 0.30

        for _ in range(num_patches):
            ph = max(1, min(h - 1, int(h * patch_scale)))
            pw = max(1, min(w - 1, int(w * patch_scale)))
            top = torch.randint(0, h - ph + 1, (1,)).item()
            left = torch.randint(0, w - pw + 1, (1,)).item()
            t[:, top:top + ph, left:left + pw] = (
                torch.rand(t.shape[0], ph, pw, device=t.device) * max_darkness
            )

    return t

@router.post("/classify", response_model=ClassificationResponse)
@limiter.limit("60/minute")
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
            # Squeeze to [3, H, W] for the transformation
            noisy_tensor_3d = apply_inference_noise(clean_input_tensor.squeeze(0), noise_level)
            noisy_input_tensor = noisy_tensor_3d.unsqueeze(0)
            
            # Convert Tensor to Base64
            pil_img = to_pil_image(noisy_tensor_3d.cpu())
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG", quality=85)
            encoded_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            noisy_base64 = f"data:image/jpeg;base64,{encoded_str}"
            
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
