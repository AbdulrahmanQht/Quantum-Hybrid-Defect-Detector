FROM node:20-bookworm-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    curl \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3 1

COPY backend/requirements_linux.txt /app/backend/requirements_linux.txt

RUN pip install --no-cache-dir "pip==24.2" "setuptools==72.1.0" "wheel==0.44.0" && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r /app/backend/requirements_linux.txt

COPY backend /app/backend
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

WORKDIR /app/backend
EXPOSE 8000
CMD ["granian", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--interface", "asgi", "--workers", "1"]