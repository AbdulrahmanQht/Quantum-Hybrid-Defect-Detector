import os
import torch
from fastapi import FastAPI, Request
from backend.utils.logger import Logger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app = FastAPI()
logger = Logger()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"]
)

@app.get("/api/health")
def health_check():
    gpu_available = torch.cuda.is_available()
    status_msg = f"Health check performed. GPU Available: {gpu_available}"
    logger.info(status_msg)

    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "cuda_version": torch.version.cuda if gpu_available else None
    }

# Frontend static files
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.abspath(os.path.join(current_dir, "..", "..", "frontend", "dist"))

if os.path.exists(frontend_path):
    logger.info(f"Frontend dist found. Serving from: {frontend_path}")

    # Mount the 'assets' folder specifically
    assets_path = os.path.join(frontend_path, "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")


    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str, request: Request):
        # Block invalid API calls
        if full_path.startswith("api/"):
            logger.warn(f"Invalid API request: {full_path} from {request.client.host}")
            return JSONResponse(status_code=404, content={"message": "API route not found"})

        # Serve actual files
        file_path = os.path.join(frontend_path, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)

        # Serve Vue index.html for all other routes
        if full_path:
            logger.info(f"Frontend route requested: /{full_path}")

        return FileResponse(os.path.join(frontend_path, "index.html"))
else:
    error_msg = f"Frontend dist not found at: {frontend_path}. Static serving will not work."
    logger.error(error_msg)
    print(f"ERROR: {error_msg}")

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server on http://127.0.0.1:8000")
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)