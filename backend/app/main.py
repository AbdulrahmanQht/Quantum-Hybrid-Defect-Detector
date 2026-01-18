import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Quantum Defect Detector API is Running!"}

@app.get("/health")
def health_check():
    gpu_available = torch.cuda.is_available()
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "cuda_version": torch.version.cuda if gpu_available else None
    }