# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Yeoun Persona Engine — FastAPI runtime image
#
# Target hardware: NVIDIA RTX PRO 4000 Blackwell (sm_120), 24GB VRAM.
# CUDA 12.8.1 + cuDNN runtime is the first stable line covering Blackwell.
# If pycuda fails to build with the runtime image (no `nvcc`), swap the
# base for `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# --- System dependencies ----------------------------------------------------
# Python 3.11 from deadsnakes PPA (Ubuntu 22.04 default is 3.10).
# ffmpeg / libsndfile1 are required by librosa/soundfile and Ditto inference.
# build-essential is kept for source-built wheels (e.g. pycuda fallback).
# curl powers the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl gnupg \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        build-essential \
        ffmpeg libsndfile1 \
        libsm6 libxext6 libgl1 libgles2-mesa libegl1 libglu1-mesa \
 && rm -rf /var/lib/apt/lists/*

# --- Non-root user ----------------------------------------------------------
RUN groupadd -r app && useradd -r -m -g app -u 1000 app

# --- Python virtual environment --------------------------------------------
RUN python3.11 -m venv "$VIRTUAL_ENV" \
 && pip install --upgrade pip setuptools wheel

WORKDIR /app

# --- Python dependencies (separate layer for cache) -------------------------
# `--extra-index-url` for CUDA 12.8 wheels is already declared inside
# requirements.txt, so a plain `pip install -r` is enough.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# --- Application code -------------------------------------------------------
COPY --chown=app:app . .

# --- Runtime directories (mounted as volumes in compose) -------------------
RUN mkdir -p /models /var/persona \
 && chown -R app:app /app /models /var/persona /opt/venv

USER app

EXPOSE 8000

# `$INTERNAL_TOKEN` is provided via env_file at runtime. Shell form on
# purpose so the variable expands at check-time.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS -H "Authorization: Bearer $INTERNAL_TOKEN" \
        http://127.0.0.1:8000/internal/health || exit 1

ENTRYPOINT ["uvicorn", "app.main:app", \
            "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
