# Yeoun Persona Engine

여운(Yeoun) 서비스의 로컬 AI 서버. 고인의 사진·음성·인터뷰 답변으로 만든 페르소나와 대화할 수 있도록 LLM·TTS·talking head 추론을 단일 FastAPI 프로세스에서 수행한다.

## 구성

- **LLM**: `google/gemma-4-E4B-it` (BF16, 멀티모달 audio-in)
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

### 모델 가중치 배치
모델은 이미지에 포함되지 않고 named volume `yeoun-models` 에 마운트된다. 초기 1회 `scripts/download_models.py` (이슈 #15) 로 가중치를 받아 `/models` 에 배치한다.

## 테스트

```bash
pytest
```

## 환경 변수

`.env.example` 참고.
