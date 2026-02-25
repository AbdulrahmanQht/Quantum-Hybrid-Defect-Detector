import os
import io
import json
import torch
import asyncio
from PIL import Image
from contextlib import asynccontextmanager
from backend.utils.logger import Logger
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from backend.models.cnn import CnnModule

logger = Logger()

# Global dictionary to hold models and configurations
ml_models = {}

# Constraints for images
Image.MAX_IMAGE_PIXELS = 16777216
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_DIMENSION = 4096  # 4096x4096px
ALLOWED_FORMATS = ['JPEG', 'PNG', 'WEBP', 'BMP', 'TIFF']

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up. Loading datasets and models...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # backend/api/main.py
    backend_dir = os.path.dirname(current_dir)
    class_names_path = os.path.join(backend_dir, "data", "class_names.json")

    # Load Class Names
    try:
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
        ml_models["class_names"] = class_names
    except FileNotFoundError:
        raise RuntimeError(f"Missing {class_names_path}. Please run training to generate it.")

    num_classes = len(class_names)

    # Load Classical CNN (PyTorch)
    device_cnn = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = CnnModule(num_classes = num_classes)
    cnn_model.load_model(os.path.join(backend_dir, "models", "cnn_model.pth"), device_cnn)
    cnn_model.to(device_cnn)
    ml_models["Classical_CNN"] = {"model": cnn_model, "device": device_cnn}

    # Load Hybrid QNN (PennyLane - CPU)
    device_qnn_cpu = torch.device("cpu")
    qnn_cpu_model = CnnModule(num_classes = num_classes)  # Placeholder: Replace with QnnModule
    # qnn_cpu_model.load_model(os.path.join(backend_dir, "models", "qnn_model.pth"), device_qnn_cpu)
    ml_models["Hybrid_QNN"] = {"model": qnn_cpu_model, "device": device_qnn_cpu}

    # 4. Load GPU-Accelerated Hybrid QNN (cuQuantum)
    device_qnn_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qnn_gpu_model = CnnModule(num_classes = num_classes)  # Placeholder: Replace with QnnModule
    # qnn_gpu_model.load_model(os.path.join(backend_dir, "models", "qnn_gpu_model.pth"), device_qnn_gpu)
    ml_models["GPU_Hybrid"] = {"model": qnn_gpu_model, "device": device_qnn_gpu}

    """ 
        Performance warm-up: First time predict is called it might take a long time to allocate GPU memory and load the model.
        To prevent this from causing a long delay on the first user request, we run a dummy prediction during startup to "warm up" the models.
    """
    logger.info("Warming up models to prevent cold-start latency.")
    dummy_image = Image.new('RGB', (256, 256), color='black')
    try:
        # Run a dummy prediction so CUDA allocates memory now, not during the first user request
        ml_models["Classical_CNN"]["model"].predict(
            dummy_image,
            ml_models["Classical_CNN"]["device"],
            class_names
        )
    except Exception as e:
        logger.warn(f"Model warm-up failed (non-fatal): {str(e)}")

    logger.info("Server is warmed up!")
    yield

    logger.info("Shutting down. Clearing memory.")
    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
app = FastAPI(lifespan = lifespan)

limiter = Limiter(key_func = get_remote_address)
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

@app.get("/api/health")
def health_check():
    gpu_available = torch.cuda.is_available()
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "cuda_version": torch.version.cuda if gpu_available else None,
    }


@app.post("/api/classify")
@limiter.limit("5/minute")
async def classify_image(request: Request, file: UploadFile = File(...)):
    # Fast fail checks for file type and size before processing to save resources
    allowed_mimes = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"]
    if file.content_type not in allowed_mimes:
        raise HTTPException(status_code=415, detail="Unsupported media type. Only images are allowed.")

    # Validate File Size by reading file in chunks to prevent memory issues with large files
    real_size = 0
    chunk_size = 1024 * 1024  # Read 1MB at a time
    file_bytes = b""

    while chunk := await file.read(chunk_size):
        real_size += len(chunk)
        if real_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File size exceeds the 5MB limit.")
        file_bytes += chunk

    try:
        # Validate Image Dimensions and Format
        image = Image.open(io.BytesIO(file_bytes))
        if image.format not in ALLOWED_FORMATS:
            raise HTTPException(status_code=400, detail="Invalid file format! Please upload a PNG, JPG, or JPEG image.")

        if image.width > MAX_DIMENSION or image.height > MAX_DIMENSION:
            raise HTTPException(status_code=400, detail="Image dimensions exceed 4096x4096.")

        image = image.convert("RGB")
    except Image.DecompressionBombError:
        logger.warn(f"Decompression bomb blocked from IP: {request.client.host}")
        raise HTTPException(status_code=400, detail="Image exceeds maximum allowed pixels.")
    except Exception:
        raise HTTPException(status_code=400, detail="Corrupted or invalid image file.")

    class_names = ml_models["class_names"]

    # Parallel Inference Setup
    cnn_setup = ml_models["Classical_CNN"]
    qnn_setup = ml_models["Hybrid_QNN"]
    gpu_qnn_setup = ml_models["GPU_Hybrid"]

    try:
        # asyncio.to_thread runs the blocking PyTorch operations in separate background threads
        cnn_task = asyncio.to_thread(cnn_setup["model"].predict, image.copy(), cnn_setup["device"], class_names)
        qnn_task = asyncio.to_thread(qnn_setup["model"].predict, image.copy(), qnn_setup["device"], class_names)
        gpu_qnn_task = asyncio.to_thread(gpu_qnn_setup["model"].predict, image.copy(), gpu_qnn_setup["device"], class_names)

        # Execute all three models simultaneously
        cnn_res, qnn_res, gpu_res = await asyncio.gather(cnn_task, qnn_task, gpu_qnn_task)

        # Aggregate results
        return {
            "filename": file.filename,
            "Classical_CNN": cnn_res,
            "Hybrid_QNN": qnn_res,
            "GPU_Hybrid": gpu_res
        }

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
        if full_path.startswith("api/"):
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