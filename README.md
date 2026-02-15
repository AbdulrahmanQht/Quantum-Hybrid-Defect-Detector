# Quantum Hybrid Defect Detector

A comparative benchmarking tool designed to detect industrial surface defects (such as cracks and corrosion) in oil and gas infrastructure. 

This project implements and analyzes the performance of three distinct AI architectures to determine if quantum-enhanced feature extraction offers superior noise robustness or efficiency compared to classical methods.

## Project Overview

This system allows users to upload industrial inspection images and receive parallel classification results from three models:
1.  **Classical CNN:** A standard deep learning baseline implemented in PyTorch.
2.  **Hybrid QNN:** A quantum-classical neural network using a Variational Quantum Circuit (VQC) via PennyLane.
3.  **GPU-Accelerated Hybrid:** A high-performance hybrid model optimizing quantum simulations using NVIDIA cuQuantum SDK.

The application features a **Vue.js** frontend for visualization and a **FastAPI** backend for model inference.

## Tech Stack

* **Frontend:** Vue.js 3, Tailwind CSS, PrimeVue
* **Backend:** Python 3.10+, FastAPI
* **Machine Learning:** PyTorch, Scikit-learn
* **Quantum Computing:** PennyLane, NVIDIA cuQuantum SDK, CUDA Toolkit

## Prerequisites

Before running the project, ensure you have the following installed:
* **Node.js** (v18 or higher)
* **Python** (v3.10 or higher)
* **NVIDIA CUDA Toolkit 12.x** (Required only for the GPU-Accelerated model)

## Installation & Setup

Follow these steps to set up the development environment.

### 1. Backend Setup (FastAPI)

Navigate to the backend directory and set up the Python environment.
```bash
# 1. Navigate to backend folder
cd backend

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup (Vue.js)

Open a new terminal window and navigate to the frontend directory.
```bash
# 1. Navigate to frontend folder
cd frontend

# 2. Install dependencies
npm install
```

## Usage

You need to run both the backend and frontend servers simultaneously in separate terminal windows.

### Terminal 1: Start Backend
```bash
cd backend
# Make sure venv is active
fastapi dev app/main.py
```

The API will run at http://127.0.0.1:8000

### Terminal 2: Start Frontend
```bash
cd frontend
npm run dev
```

The UI will run at http://localhost:5173

## Project Structure
```
Quantum-Hybrid-Defect-Detector/
├── backend/                # FastAPI application & Model logic
│   ├── app/                # API source code
│   ├── models/             # Saved PyTorch (.pth) model weights
│   ├── data/               # Dataset storage
│   └── requirements.txt    # Python dependencies
├── frontend/               # Vue.js application
│   ├── src/                # Vue components & views
│   └── package.json        # Node dependencies
└── README.md               # Project documentation
```
