# Yeoun Persona Engine

여운(Yeoun) 서비스의 로컬 AI 서버. 고인의 사진·음성·인터뷰 답변으로 만든 페르소나와 대화할 수 있도록 LLM·TTS·talking head 추론을 단일 FastAPI 프로세스에서 수행한다.

## 구성

- **LLM**: `google/gemma-4-E4B-it` (BF16, 멀티모달 audio-in, 상주)
- **TTS**: `k2-fsa/OmniVoice` (zero-shot voice cloning, 상주)
- **Talking Head**: `antgroup/ditto-talkinghead` (PyTorch 백엔드, subprocess 온디맨드)

Blackwell sm_120 호환을 위해 vendor 가 빌드한 TRT 8.6.1 엔진 대신 PyTorch 백엔드 사용. TRT 재빌드는 별도 환경.

## 요구사항

- NVIDIA Blackwell GPU 24GB VRAM (예: RTX PRO 4000 Blackwell)
- CUDA 12.8+ (PyTorch wheel `cu128` 인덱스)
- Python 3.11+ (Docker 이미지), 로컬 venv 는 Python 3.10~3.14 호환
- Docker (배포용), `nvidia-container-toolkit`
- 시스템 `ffmpeg` (도커는 apt 로 자동 설치, 로컬 venv 는 별도)

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

# 헬스 확인 — 같은 머신(loopback)
curl -H "Authorization: Bearer $INTERNAL_TOKEN" http://127.0.0.1:8000/internal/health

# 헬스 확인 — Spring Boot 머신 등 LAN 다른 노드에서 (아래 IP 는 가정값 — 실제 AI 머신 LAN IP로 교체)
curl -H "Authorization: Bearer $INTERNAL_TOKEN" http://192.168.10.50:8000/internal/health

# 컨테이너 안 확인 (모델 볼륨, 페르소나 디렉토리)
docker compose exec yeoun-engine ls /models /var/persona

# 종료
docker compose down
```

### 네트워크 토폴로지
- 이 AI 머신엔 WireGuard 인터페이스가 없다. WG 는 다른 머신(Spring Boot 측 게이트웨이)에서 처리한다.
- Spring Boot 머신은 같은 LAN 의 다른 노드에서 본 AI 머신의 LAN 사설 IP 로 HTTP 직접 호출한다.
- 컨테이너는 host 의 모든 인터페이스(`0.0.0.0:8000`) 에서 listen 한다.

> **주의**: 위 바인딩만으로는 LAN 의 모든 기기가 접근 가능합니다. **반드시 아래 「방화벽 가이드」를 적용한 뒤 운영하십시오**. 미적용 시 LAN 내 임의의 기기가 토큰 brute-force 등을 시도할 수 있습니다.

### 방화벽 가이드 (LAN 접근 통제)

> Docker 는 `iptables` 의 `DOCKER-USER` 체인을 가로채므로, host 의 일반 `INPUT` 체인이나 `ufw` 기본 정책만으로는 컨테이너 포트 보호가 새는 경우가 있다. 반드시 `DOCKER-USER` 체인에 직접 룰을 넣는다.

> Docker 는 `DOCKER-USER` 체인 끝에 항상 `RETURN` 을 자동 삽입한다. 따라서 `-A` (append) 로 붙인 DROP 은 `RETURN` 다음에 위치해 절대 도달하지 않는다. **반드시 `-I` (insert) + 명시적 위치 번호로 ACCEPT 를 1번, DROP 을 2번에 둬서 `RETURN` 보다 앞에 위치시킨다.**

Spring Boot 머신의 IP 가 `192.168.10.30` 이라고 가정한 예시 (실제 환경 IP 로 교체):

```bash
# Spring Boot 머신만 :8000 허용 (위치 1), 그 외는 모두 DROP (위치 2). RETURN 은 그 뒤.
sudo iptables -I DOCKER-USER 1 -p tcp --dport 8000 -s 192.168.10.30 -j ACCEPT
sudo iptables -I DOCKER-USER 2 -p tcp --dport 8000 -j DROP
sudo iptables -L DOCKER-USER -n --line-numbers

# 룰 영속화 — 재부팅 후에도 유지 (Ubuntu/Debian 기본 iptables 는 휘발성)
sudo apt-get install -y iptables-persistent
sudo netfilter-persistent save
```

`ufw` 만 쓰는 경우 [chaifeng/ufw-docker](https://github.com/chaifeng/ufw-docker) 같은 보조 룰셋이 필요할 수 있다.

> **방화벽 미설정 시 같은 LAN 의 모든 기기가 `http://<이 머신 LAN IP>:8000` 에 접근 가능합니다.** 반드시 위 정책을 적용한 뒤 운영하십시오.

### 모델 가중치·vendor 자원 배치
Ditto-TalkingHead 의 소스 코드 + checkpoints(~6.5GB) 는 **Dockerfile 빌드 단계에서 이미지 안에 직접 clone** 한다 — 운영자가 호스트에서 별도 git clone 할 필요 없음. Gemma 4·OmniVoice 가중치는 첫 부팅 때 `transformers.from_pretrained` 가 HF 에서 자동 다운로드해 `HF_HOME` (named volume `yeoun-models` 하위) 에 캐시한다.

운영자 1회 셋업:

```bash
# 1) (선택) Gemma 4 gated 라이선스 동의 후 token 등록 — public 캐시 hit 시 불필요.
export HF_TOKEN=hf_xxx

# 2) 이미지 빌드 — Ditto vendor + checkpoints 가 함께 박혀 ~10-20 분 소요.
docker compose build

# 3) 기동 — 첫 부팅 시 Gemma·OmniVoice 가중치 (~10GB) 가 HF cache 볼륨에 채워진다.
docker compose up -d
docker compose logs -f yeoun-engine   # "ModelRegistry 시작" 로그 확인
```

이후 HF cache 는 named volume `yeoun-models` 에 영속화되어 컨테이너 재시작 시 재다운로드 안 함.

## 테스트

GPU 없는 개발 머신에서도 mock 기반 흐름이 모두 검증되도록 pytest 스위트 + 커버리지를 둔다.

```bash
# venv 활성화 후
pip install -r requirements-dev.txt

# 전체 스위트 — GPU_ENABLED=false / USE_DB_MOCK=true 자동 설정
pytest

# 커버리지 리포트 (라우터·세션·세이프티)
pytest --cov=app --cov-report=term-missing
```

스위트 구성:
- 단위: 모델 더미 로더 (`test_llm_gemma_mock.py`·`test_omnivoice_mock.py`·`test_ditto_mock.py`), 세이프티 가드 (`test_safety_guard.py`), 세션 스토어 (`test_session_store.py`)
- 라우터 통합: 헬스 (`test_health.py`), 페르소나 생성·삭제 (`test_persona_creation.py`·`test_persona_deletion.py`), 세션 (`test_sessions_flow.py`), 미디어 Range (`test_media_range.py`)
- E2E mock 흐름: `test_integration_flow.py` — 페르소나 생성 → 세션 start/end → 삭제 한 시퀀스

실 GPU 검증은 별도(가이드).

## 환경 변수

상세는 `.env.example`. 주요 변수:

| 변수 | 기본/예시 | 의미 |
|---|---|---|
| `INTERNAL_TOKEN` | (필수) | Spring Boot 와 공유하는 Bearer 토큰. `python -c "import secrets; print(secrets.token_urlsafe(48))"` 로 생성 |
| `DATABASE_URL` | `postgresql+asyncpg://...` | asyncpg DSN. `USE_DB_MOCK=true` 면 미사용 |
| `USE_DB_MOCK` | `false` | true 면 인메모리 mock repository |
| `GPU_ENABLED` | `true` | false 면 모델 로더가 더미 모드 (개발·테스트용) |
| `MODELS_DIR` | `/models` | HF 캐시·가중치 볼륨 마운트 경로 |
| `HF_HOME` | `/models/hf-cache` | HuggingFace 캐시 디렉토리. `MODELS_DIR` 볼륨 재사용을 위해 하위에 배치 |
| `PERSONA_DIR` | `/var/persona` | 페르소나 자원 (photo/voice/voice_ref/idle/speak) 루트 |
| `LLM_BF16_PATH` | `google/gemma-4-E4B-it` | HF repo ID 또는 로컬 경로 |
| `TTS_MODEL_PATH` | `k2-fsa/OmniVoice` | OmniVoice repo ID 또는 로컬 경로 |
| `DITTO_VENDOR_DIR` | `./vendor/ditto-talkinghead` | Ditto 소스 코드 디렉토리 |
| `DITTO_DATA_ROOT` | `.../checkpoints/ditto_pytorch` | PyTorch 백엔드 가중치. Ampere/Ada 인스턴스는 `ditto_trt_Ampere_Plus` 로 |
| `DITTO_CFG_PKL` | `.../v0.4_hubert_cfg_pytorch.pkl` | PyTorch 백엔드 cfg. TRT 백엔드는 `_trt.pkl` |
| `LOG_LEVEL` | `INFO` | DEBUG / INFO / WARNING / ERROR |

## 동시성·큐 정책

- uvicorn `--workers 1` 단일 워커. 멀티 워커 금지 (모델 중복 적재).
- GPU 추론 진입은 `gpu_semaphore` (LLM/TTS 공유, 1건) + `ditto_semaphore` (Ditto subprocess, 1건) 로 직렬화.
- 동시 SSE 메시지 처리 한도: **5건**. 초과 시 503 + `Retry-After: 30`.
- 세션 TTL: 비활성 **30분** 후 sweeper 가 메모리에서 폐기 (60초마다 sweep).
- 페르소나별 진행률은 워커 메모리(`PersonaProcessingStore`)에 보관 — 워커 재시작 시 사라지지만 `process_persona` 가 멱등이라 재호출 시 이미 완료 단계는 스킵.
