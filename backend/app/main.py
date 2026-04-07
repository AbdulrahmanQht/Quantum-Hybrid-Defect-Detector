import os
import json
import time
import torch
import asyncio
from PIL import Image
from contextlib import asynccontextmanager
from torchvision.io import decode_image, ImageReadMode
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi import FastAPI, Request, UploadFile, File, HTTPException

from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler

from .routers.contact import router as contact_router
from .routers.benchmark import router as benchmark_router
from .routers.quantum_advantage import router as quantum_advantage
from .schemas.classification import ClassificationResponse

from backend.utils.logger import Logger
from backend.models.cnn import CNN
from backend.models.qnn_cpu import HybridQnnCPU
from backend.models.qnn_gpu import HybridQnnGPU


logger = Logger()

inference_executor = ThreadPoolExecutor(max_workers=3)

# Global dictionary to hold models
ml_models = {}

# Constraints for images
Image.MAX_IMAGE_PIXELS = 16777216
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_DIMENSION = 4096  # 4096x4096px
ALLOWED_FORMATS = ["JPEG", "PNG", "WEBP", "BMP", "TIFF"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up. Loading datasets and models...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_dir)
    class_names_path = os.path.join(backend_dir, "data", "class_names.json")

    # Load Class Names
    try:
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
        ml_models["class_names"] = class_names
    except FileNotFoundError:
        raise RuntimeError(f"Missing {class_names_path}. Please run training to generate it.")

    """ 
        Performance warm-up: First time predict is called it might take a long time to allocate GPU memory and load the model.
        To prevent this from causing a long delay on the first user request, we run a dummy prediction during startup to "warm up" the models.
    """
    
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models_dir = os.path.join(backend_dir, "models")

    # Load CNN
    cnn_path = os.path.join(models_dir, "cnn.pth")
    if not os.path.exists(cnn_path):
        raise RuntimeError(f"CNN checkpoint not found: {cnn_path}")
    try:
        cnn_model = CNN(num_classes=num_classes)
        cnn_model.load_model(cnn_path, device)
        ml_models["CNN"] = {"model": cnn_model, "device": device}
        logger.info("CNN loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load CNN: {e}")

    # Load Hybrid QNN-CPU
    qnn_cpu_path = os.path.join(models_dir, "qnn_cpu.pth")
    if not os.path.exists(qnn_cpu_path):
        raise RuntimeError(f"QNN-CPU checkpoint not found: {qnn_cpu_path}")
    try:
        qnn_cpu_model = HybridQnnCPU(num_classes=num_classes)
        qnn_cpu_model.load_model(qnn_cpu_path, device)
        ml_models["QNN_CPU"] = {"model": qnn_cpu_model, "device": device}
        logger.info("QNN-CPU loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load QNN-CPU: {e}")

    # Load Hybrid QNN-GPU (CUDA required)
    if torch.cuda.is_available():
        qnn_gpu_path = os.path.join(models_dir, "qnn_gpu_6_qubits.pth")
        if not os.path.exists(qnn_gpu_path):
            raise RuntimeError(f"QNN-GPU checkpoint not found: {qnn_gpu_path}")
        try:
            qnn_gpu_model = HybridQnnGPU(num_classes=num_classes)
            qnn_gpu_model.load_model(qnn_gpu_path, device)
            ml_models["QNN_GPU"] = {"model": qnn_gpu_model, "device": device}
            logger.info("QNN-GPU loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load QNN-GPU: {e}")
    else:
        logger.warning("CUDA not available. QNN-GPU will not be loaded.")

    logger.info("Warming up models to prevent cold-start latency.")
    dummy_tensor = torch.zeros((1, 3, 384, 384), dtype=torch.float32)

    for key in ["CNN", "QNN_CPU", "QNN_GPU"]:
        try:
            ml_models[key]["model"].predict(dummy_tensor, ml_models[key]["device"], class_names)
            logger.info(f"{key} warmed up.")
        except Exception as e:
            logger.warning(f"{key} warm-up failed (non-fatal): {str(e)}")

    logger.info("Server is warmed up!")
    yield

    logger.info("Shutting down. Clearing memory.")
    inference_executor.shutdown(wait=True)
    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

app.include_router(contact_router)
app.include_router(benchmark_router)
app.include_router(quantum_advantage)


@app.get("/api/v1/health")
def health_check():
    gpu_available = torch.cuda.is_available()
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "cuda_version": torch.version.cuda if gpu_available else None,
    }


@app.get("/api/v1/models")
def list_models():
    return {name: "loaded" for name in ml_models if name != "class_names"}


@app.post("/api/v1/classify")
@limiter.limit("5/minute")
async def classify_image(request: Request, file: UploadFile = File(...)) -> ClassificationResponse:
    # Adding a counter to test response time before and after adding pydantic classes
    request_start = time.perf_counter()

    # Check availability of models and class names
    required_models = ["CNN", "QNN_CPU", "class_names"]
    if torch.cuda.is_available():
        required_models.append("QNN_GPU")

    missing_models = [m for m in required_models if m not in ml_models]

    if missing_models:
        logger.error(f"Classification failed: Missing models in registry: {missing_models}")
        raise HTTPException(status_code=503, detail=f"Server is not ready. Missing: {', '.join(missing_models)}",
                            )
    # Fast fail checks for file type and size before processing to save resources
    allowed_mimes = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"]
    if file.content_type not in allowed_mimes:
        raise HTTPException(status_code=415, detail="Unsupported media type. Only images are allowed.")

    # Reading all bytes at once for performance, reading 1 byte at a time was causing a bottleneck.
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        # Decode bytes DIRECTLY to a PyTorch Tensor (Bypasses PIL entirely)
        # Convert bytes to a 1D uint8 tensor, then decode to an image tensor in RGB format
        raw_tensor = torch.frombuffer(file_bytes, dtype=torch.uint8)
        img_tensor = decode_image(raw_tensor, mode=ImageReadMode.RGB)

        # Validate dimensions manually since we aren't using PIL
        _, height, width = img_tensor.shape
        if height > MAX_DIMENSION or width > MAX_DIMENSION:
            raise HTTPException(status_code=400, detail="Image dimensions exceed 4096x4096.")

    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        raise HTTPException(status_code=400, detail="Corrupted or invalid image file.")

    class_names = ml_models["class_names"]

    # Parallel Inference Setup
    cnn_setup = ml_models["CNN"]
    qnn_cpu_setup = ml_models["QNN_CPU"]
    qnn_gpu_setup = ml_models["QNN_GPU"]

    try:
        # Reuse CNN's transform — all models share the same preprocessing
        transform_pipeline = cnn_setup["model"].inference_transform
        shared_input_tensor = transform_pipeline(img_tensor).unsqueeze(0)  # [1, 3, 384, 384]

        # Submit all inference jobs to the thread pool simultaneously.
        # PyTorch releases the GIL during C++/CUDA ops, so these genuinely run in parallel.
        futures = {
            "CNN": inference_executor.submit(
                cnn_setup["model"].predict, shared_input_tensor, cnn_setup["device"], class_names
            ),
            "QNN_CPU": inference_executor.submit(
                qnn_cpu_setup["model"].predict, shared_input_tensor, qnn_cpu_setup["device"], class_names
            ),
        }
        if "QNN_GPU" in ml_models:
            futures["QNN_GPU"] = inference_executor.submit(
                qnn_gpu_setup["model"].predict, shared_input_tensor, qnn_gpu_setup["device"], class_names
            )

        # Collect results — .result() blocks until that future is done
        loop = asyncio.get_event_loop()
        results = {
            key: await loop.run_in_executor(None, future.result)
            for key, future in futures.items()
        }

        total_time_ms = (time.perf_counter() - request_start) * 1000
        logger.info(f"Full Request Processed: {total_time_ms:.2f}ms | Filename: {file.filename}")

        results_data = {
            "filename": file.filename,
            "CNN": results["CNN"],
            "QNN_CPU": results["QNN_CPU"],
            "QNN_GPU": results.get("QNN_GPU"),  # None if CUDA unavailable
        }
        return ClassificationResponse(**results_data)

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error during classification. Please try again.")


# Frontend static files
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.abspath(os.path.join(current_dir, "..", "..", "frontend", "dist"))

if os.path.exists(frontend_path):
    logger.info(f"Frontend dist found. Serving from: {frontend_path}")
    assets_path = os.path.join(frontend_path, "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str, request: Request):
        if full_path.startswith("api/v1/"):
            return JSONResponse(status_code=404, content={"message": "API route not found"})
        file_path = os.path.join(frontend_path, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_path, "index.html"))

else:
    logger.error("Frontend dist not found. Static serving will not work.")

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server on http://127.0.0.1:8000")
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)