# syntax=docker/dockerfile:1.7
# =============================================================================
# TRIBE v2 Activation API — imagen de despliegue (CUDA + ffmpeg)
# =============================================================================
# Construir (los defaults ya apuntan al repo real de tribev2):
#   docker build -t tribe-activation-api .
#
# Ejecutar (requiere nvidia-container-toolkit en el host):
#   docker run --gpus all -p 8000:8000 \
#     --env-file .env \                    # contiene HF_TOKEN (LLaMA-3.2 es gated)
#     -v tribe-cache:/cache \              # persistir pesos (50-100 GB)
#     tribe-activation-api
# =============================================================================

# Base CUDA sobre Ubuntu 24.04 -> Python 3.12 (tribev2 exige Python >=3.11).
# Nota: nvidia/cuda solo publica imagenes ubuntu24.04 desde CUDA 12.6 (no hay 12.4.x).
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
# --platform=linux/amd64: la imagen corre en un servidor GPU NVIDIA x86_64, y
# ademas torchvision cu126 NO publica ruedas arm64. En un Mac Apple Silicon se
# construye por emulacion (mas lento pero correcto); en el servidor x86 es nativo.
FROM --platform=linux/amd64 ${CUDA_IMAGE}

# HF_HOME / TRIBE_CACHE: caches de pesos bajo /cache para montar UN solo volumen.
# VIRTUAL_ENV + PATH: activa el venv aislado (su Python queda primero en el PATH).
# (Los comentarios van AQUI, no dentro del ENV: un '#' dentro de una instruccion
#  multilinea se mal-interpreta y puede descartar variables.)
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    TRIBE_CACHE=/cache/tribe \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# --- Dependencias de sistema -------------------------------------------------
# ffmpeg: decodificacion de video y audio (moviepy/tribev2 lo usan).
# git:    para instalar tribev2 desde su repo de origen.
# python3-venv: para crear el venv (Ubuntu 24.04 bloquea pip al sistema, PEP 668).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-dev \
        ffmpeg \
        git \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# venv aislado: evita el bloqueo PEP 668 de Ubuntu 24.04 y mantiene limpio el sistema.
RUN python3 -m venv "${VIRTUAL_ENV}" && \
    pip install --upgrade pip setuptools wheel

WORKDIR /app

# --- 1) PyTorch con CUDA (PRIMERO, a proposito) ------------------------------
# Se fija antes que tribev2 para garantizar la build CUDA y evitar que una
# instalacion posterior arrastre la rueda CPU-only de PyPI.
# Versiones dentro del rango que pide tribev2: torch>=2.5.1,<2.7 / torchvision>=0.20,<0.22.
ARG TORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG TORCH_CUDA=cu126
RUN pip install --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

# --- 2) tribev2 (el modelo) --------------------------------------------------
# Instalacion segun el README del repo (pip install -e .; sin submodulos ni conda).
# tribev2 arrastra numpy==2.2.6, transformers, moviepy, soundfile, spacy, etc.
ARG TRIBE_REPO=https://github.com/facebookresearch/tribev2
ARG TRIBE_REF=main
RUN git clone --depth 1 --branch "${TRIBE_REF}" "${TRIBE_REPO}" /opt/tribe && \
    pip install /opt/tribe

# --- 3) Dependencias propias de la API ---------------------------------------
# Copiado aparte (antes del codigo) para aprovechar la cache de capas de Docker.
COPY requirements.txt .
RUN pip install -r requirements.txt

# --- 4) Codigo de la aplicacion ----------------------------------------------
COPY tribe_activation_api.py .

# Volumen para los pesos: evita re-descargar 50-100 GB en cada arranque.
VOLUME ["/cache"]

EXPOSE 8000

# start-period largo: el primer arranque puede descargar decenas de GB de pesos.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').getcode()==200 else 1)" || exit 1

# Un solo worker a proposito: el modelo es grande y GPU-bound. Cada worker
# cargaria el modelo entero en VRAM. Para escalar usa replicas de contenedor
# (una GPU por replica) + balanceador, no --workers > 1.
CMD ["python", "-m", "uvicorn", "tribe_activation_api:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
