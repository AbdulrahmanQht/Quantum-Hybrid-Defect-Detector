# Quantum-Hybrid Defect Detector

Quantum-Hybrid Defect Detector is a full-stack graduation project for industrial defect classification. It compares three inference paths on the same uploaded image:

- `CNN`: classical baseline
- `QNN_CPU`: hybrid quantum-classical model on PennyLane's default.qubit quantum simulation device
- `QNN_GPU`: hybrid quantum-classical model on PennyLane's lightning.gpu quantum simulation device

The application includes a Vue 3 frontend, a FastAPI backend, benchmark and quantum-advantage dashboards, contact form handling, and a test suite covering API contracts, model behavior, regression checks, and browser flows.

## Current Project Scope

The system supports:

- Defect classification from uploaded images
- Optional noisy-image comparison during inference
- Benchmark visualization for clean accuracy, latency, robustness, confusion matrices, and per-class metrics
- Quantum-advantage reporting from precomputed experiment artifacts
- English/Arabic UI switching
- Light/dark theme switching
- Static frontend serving from FastAPI after `frontend/dist` is built

Defect classes:

- `Deformation`
- `Deposition`
- `Disconnect`
- `Misalignment`
- `Obstacle`
- `Rupture`

## Architecture

### Frontend
- Vue 3
- Vue Router
- Vue I18n
- PrimeVue
- Chart.js / vue-chartjs
- Tailwind CSS
- Lucide icons

Source: `frontend/src/`

### Backend
- FastAPI
- Granian
- SlowAPI rate limiting
- Pydantic
- FastAPI Mail
- PyTorch / TorchVision
- PennyLane
- PennyLane Lightning / Lightning GPU

Source: `backend/app/`

### Model Runtime
At startup the backend:

1. Loads class names from `backend/data/class_names.json`
2. Loads `CNN`
3. Loads `QNN_CPU`
4. Loads `QNN_GPU` only if CUDA is available
5. Warms up loaded models with a dummy tensor

## Repository Layout

```text
backend/
  app/
    main.py
    routers/
      classification.py
      benchmark.py
      quantum_advantage.py
      contact.py
  data/
    benchmark/
    QA/
    train/
    val/
    test/
  models/
    cnn.py
    qnn_cpu.py
    qnn_gpu.py
    *.pth
  tests/
frontend/
  src/
    assets/
    components/
    router/
    views/
README.md
```
## API Routes

### Classification

- `POST /api/v1/classify`

Accepts multipart form data:

- `file`
- `compare_with_noise`
- `noise_level`

Returns clean predictions for all available models and optional noisy predictions.

### Benchmark

- `GET /api/v1/benchmark`

Returns cached benchmark artifacts loaded from `backend/data/benchmark/`.

### Quantum Advantage

- `GET /api/v1/quantum-advantage`

Returns cached experiment artifacts loaded from `backend/data/QA/`.

### Contact

- `POST /api/v1/contact`

Accepts JSON:

- `name`
- `subject`
- `message`

If mail environment variables are missing, the endpoint still accepts the request and returns a non-dispatch status.

### Health

- `GET /api/v1/health`

Returns backend, model, CUDA, and PennyLane readiness details.

## Requirements

### General

- Python `3.10+`
- Node.js `18+`
- npm `9+`

### For `QNN_GPU`

- NVIDIA GPU
- NVIDIA driver
- CUDA-compatible environment
- Linux is the intended deployment target for GPU serving

## Local Development

### 1. Backend Setup

Use the platform-specific requirements file.

#### Linux

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_linux.txt
```
#### Windows
```bash
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements_windows.txt
```
### 2. Frontend Setup
```bash
cd frontend
npm install
```
### 3. Run the Frontend Dev Server
```bash
cd frontend
npm run dev
```
### 4. Run the Backend
Development:
```bash
cd backend
fastapi dev app/main.py # uvicorn
python -m app.main # granian
```
Production-style local run:
```bash
cd backend
granian app.main:app --host 127.0.0.1 --port 8000 --interface asgi --workers 1
```
### Frontend Build
```bash
cd frontend
npm run build
```
If `frontend/dist` exists, FastAPI serves the built SPA and static assets.

# Docker
### 1. Build the Image
From the project root:
```bash
docker build --no-cache -t quantum-defect-detector:demo .
```
### 2. Run the Container
CPU-only run:
```bash
docker run --rm -it -p 8000:8000 quantum-defect-detector:demo
```
GPU-enabled run:
```bash
docker run --rm -it --gpus all -p 8000:8000 quantum-defect-detector:demo
```
### 3. Test the Running Container
```bash
# Health endpoint:
curl http://127.0.0.1:8000/api/v1/health

# Benchmark endpoint:
curl http://127.0.0.1:8000/api/v1/benchmark

# Quantum advantage endpoint:
curl http://127.0.0.1:8000/api/v1/quantum-advantage
```
#### Notes:

* The Docker image builds the frontend and serves it from the FastAPI backend.
* The production container uses Granian with workers=1 to avoid duplicating GPU memory across multiple workers.
* GPU support requires both CUDA-compatible drivers on the host and Docker GPU runtime support.
* The Docker image is large because it includes CUDA, Python, PyTorch, and PennyLane GPU dependencies.
