# Yeoun Persona Engine

여운(Yeoun) 서비스의 로컬 AI 서버. 고인의 사진·음성·인터뷰 답변으로 만든 페르소나와 대화할 수 있도록 LLM·TTS·talking head 추론을 단일 FastAPI 프로세스에서 수행한다.

## 구성

- **LLM**: `google/gemma-4-E4B-it` (AWQ INT4, 멀티모달 audio-in)
- **TTS**: `k2-fsa/OmniVoice` (zero-shot voice cloning)
- **Talking Head**: `antgroup/ditto-talkinghead` (TensorRT, 온디맨드)

## 요구사항

- NVIDIA Blackwell GPU 24GB VRAM (예: RTX PRO 4000 Blackwell)
- CUDA 12.8+, TensorRT 10.7+
- Python 3.11+
- Docker (배포용), `nvidia-container-toolkit`

## 부팅 절차 (개발)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cu128

cp .env.example .env
# .env 편집

uvicorn app.main:app --reload
```

Swagger UI: <http://127.0.0.1:8000/internal/docs>

## 부팅 절차 (Docker)

```bash
cp .env.example .env
docker compose up --build
```

## 테스트

```bash
pytest
```

## 환경 변수

`.env.example` 참고.
