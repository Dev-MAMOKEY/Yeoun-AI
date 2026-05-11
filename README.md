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

### 사전 요구사항
- Docker Engine 24+ 와 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) 설정
- 호스트의 NVIDIA 드라이버가 Blackwell(sm_120) 지원 (570.x 이상)

### 빌드 & 기동
```bash
cp .env.example .env
# .env 편집: INTERNAL_TOKEN, DATABASE_URL 등

docker compose build
docker compose up -d
```

### 운영 명령
```bash
# 로그 따라가기
docker compose logs -f yeoun-engine

# 헬스 확인 (이슈 #3 머지 이후 동작)
curl -H "Authorization: Bearer $INTERNAL_TOKEN" http://127.0.0.1:8000/internal/health

# 컨테이너 안 확인 (모델 볼륨, 페르소나 디렉토리)
docker compose exec yeoun-engine ls /models /var/persona

# 종료
docker compose down
```

### 모델 가중치 배치
모델은 이미지에 포함되지 않고 named volume `yeoun-models` 에 마운트된다. 초기 1회 `scripts/download_models.py` (이슈 #15) 로 가중치를 받아 `/models` 에 배치한다.

## 테스트

```bash
pytest
```

## 환경 변수

`.env.example` 참고.
