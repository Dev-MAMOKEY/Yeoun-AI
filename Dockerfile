# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Yeoun Persona Engine — FastAPI 런타임 이미지
#
# 타겟 하드웨어: NVIDIA RTX PRO 4000 Blackwell (sm_120), 24GB VRAM.
# CUDA 12.8.1 + cuDNN runtime 은 Blackwell 을 포함하는 첫 안정 라인.
# Ditto 는 PyTorch 백엔드(`ditto_pytorch`) 를 사용하므로 nvcc 가 필요한
# pycuda/tensorrt 빌드는 없다. 향후 TRT 백엔드로 회귀 시에는 devel 이미지
# (`nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`) 로 교체.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# --- 시스템 의존성 ---------------------------------------------------------
# Python 3.11 은 deadsnakes PPA 에서 (Ubuntu 22.04 기본은 3.10).
# ffmpeg / libsndfile1 은 librosa·soundfile·Ditto inference 가 사용.
# build-essential 은 일부 휠(cython, numba 등) 의 소스 빌드 폴백용.
# libsm6 / libxext6 / libgl1 / libgles2-mesa / libegl1 / libglu1-mesa 는
#   opencv-python-headless 와 mediapipe 가 dlopen 으로 잡는 GLX/EGL 심볼.
#   Ditto inference.py 가 mediapipe·OpenCV 를 import 하므로 필수.
# curl 은 아래 HEALTHCHECK 가 사용.
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
