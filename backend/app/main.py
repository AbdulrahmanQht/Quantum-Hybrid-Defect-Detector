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

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from .routers.contact import router as contact_router
from .routers.benchmark import router as benchmark_router
from .routers.quantum_advantage import router as quantum_advantage
from .routers.classification import router as classification_router

from backend.utils.logger import Logger
from backend.models.cnn import CNN
from backend.models.qnn_cpu import HybridQnnCPU
from backend.models.qnn_gpu import HybridQnnGPU


logger = Logger()

# Increased to 6 because of noisy classifications runs 2 classifications 1 clean and 1 noisy per model
inference_executor = ThreadPoolExecutor(max_workers=6)

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
        logger.error(f"Failed to load CNN: {e}")
        raise RuntimeError("Failed to load CNN.")

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
        qnn_gpu_path = os.path.join(models_dir, "qnn_gpu.pth")
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

    for key, model_info in ml_models.items():
        if key == "class_names":
            continue
            
        try:
            # model_info is the dict containing {"model": ..., "device": ...}
            model_info["model"].predict(dummy_tensor, model_info["device"], class_names)
            logger.info(f"{key} warmed up.")
        except Exception as e:
            logger.warn(f"{key} warm-up failed (non-fatal): {str(e)}")

    app.state.ml_models = ml_models
    app.state.inference_executor = inference_executor
   
    logger.info("Server is warmed up!")
    yield

    logger.info("Shutting down. Clearing memory.")
    inference_executor.shutdown(wait=False, cancel_futures=True)
    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)
limiter = Limiter(key_func=get_remote_address, enabled=True)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        # Tighten CSP if you serve the frontend from FastAPI
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' http://127.0.0.1:8000;"
        )
        return response

app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0", "AbdulrahmanPC.local", "testserver"])

app.include_router(contact_router)
app.include_router(benchmark_router)
app.include_router(quantum_advantage)
app.include_router(classification_router)


@app.get("/api/v1/health")
@limiter.limit("10/minute")
def health_check(request: Request):
    try:
        import pennylane
        pennylane_ok = True
    except ImportError:
        pennylane_ok = False

    cnn_ok     = "CNN"     in ml_models
    qnn_cpu_ok = "QNN_CPU" in ml_models
    qnn_gpu_ok = "QNN_GPU" in ml_models

    status = "healthy" if all([cnn_ok, qnn_cpu_ok, qnn_gpu_ok, pennylane_ok]) else "degraded"

    return JSONResponse(
        status_code=200 if status == "healthy" else 503,
        content={
            "status": status,
            "models": {
                "CNN":     "ok" if cnn_ok     else "unavailable",
                "QNN_CPU": "ok" if qnn_cpu_ok else "unavailable",
                "QNN_GPU": "ok" if qnn_gpu_ok else "unavailable",
            },
            "pennylane": "ok" if pennylane_ok else "unavailable",
        }
    )
    
    
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
    from granian import Granian
    from granian.constants import Interfaces

    logger.info("Starting Granian server on http://127.0.0.1:8000")
    
    server = Granian(
        "backend.app.main:app",  # run from project root: python -m backend.app.main
        address="127.0.0.1",
        port=8000,
        interface=Interfaces.ASGI,
        workers=1,       # GPU app — multiple workers = duplicate VRAM per worker
        threads=2,       # Rust I/O threads; your bottleneck is inference not I/O
        preload=True,    # Load models once before worker forks
    )

    server.serve()