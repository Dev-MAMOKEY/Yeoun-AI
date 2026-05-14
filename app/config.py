"""환경 변수 / `.env`에서 로드되는 런타임 설정.

모든 값은 `.env.example`에 문서화된 키와 1:1로 대응한다. `get_settings`
헬퍼는 `lru_cache`로 감싸져 있어 요청 라이프사이클 동안 같은 인스턴스를
재사용한다. 테스트에서는 환경 변수 변경 후 `get_settings.cache_clear()`를
호출하면 된다.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    internal_token: str = Field(
        ...,
        description="Spring Boot와 공유하는 내부 서비스 Bearer 토큰. WireGuard 내부망 인증에 사용.",
    )
    database_url: str = Field(
        "",
        description="PostgreSQL DSN (asyncpg 드라이버). `USE_DB_MOCK=true`일 때는 비워둬도 됨.",
    )
    use_db_mock: bool = Field(
        False,
        description="true이면 repository 함수가 인메모리 mock 데이터를 반환 (개발/테스트용).",
    )
    models_dir: str = Field(
        "/models",
        description="모델 가중치 디렉토리. Docker 볼륨 `yeoun-models` 마운트 지점.",
    )
    persona_dir: str = Field(
        "/var/persona",
        description="영구 페르소나 자원 디렉토리 (사진·음성·voice_ref·idle 클립).",
    )
    hf_home: str = Field(
        "/models/hf-cache",
        description="Hugging Face 캐시 디렉토리. 볼륨 재사용을 위해 MODELS_DIR 하위에 둔다.",
    )
    gpu_enabled: bool = Field(
        True,
        description="false이면 모델 로더가 단락 처리되어 더미 출력을 반환 (개발/테스트용).",
    )

    # --- LLM 가중치 경로 --------------------------------------------------
    llm_bf16_path: str | None = Field(
        None,
        description=(
            "Gemma 4 BF16 가중치 디렉토리 또는 HF repo ID. transformers `from_pretrained` 가 "
            "자동으로 다운로드·캐시한다. GPU_ENABLED=true 인데 None 이면 부팅 실패."
        ),
    )
    llm_gpu_max_memory: str = Field(
        "12GiB",
        description=(
            "Gemma 모델 GPU 점유 한도. accelerate `device_map=\"auto\"` 의 max_memory 인자로 "
            "전달돼 한도를 넘는 레이어는 CPU RAM 으로 자동 오프로드. Ditto subprocess 와 "
            "OmniVoice 가 공존하는 24GB GPU 에서 12GiB 가 보수적 안전마진. 단위: '12GiB', '13000MiB'."
        ),
    )
    llm_cpu_max_memory: str = Field(
        "32GiB",
        description=(
            "Gemma 모델 CPU RAM 오프로드 한도. 시스템 가용 RAM 의 절반 이내 권장."
        ),
    )

    # --- TTS 가중치 경로 --------------------------------------------------
    tts_model_path: str | None = Field(
        None,
        description=(
            "OmniVoice TTS 가중치 디렉토리 또는 HF repo ID (예: `k2-fsa/OmniVoice`). "
            "GPU_ENABLED=true 인데 None 이면 TTS 로드 실패."
        ),
    )

    # --- Ditto-TalkingHead 경로 ---------------------------------------------
    # Dockerfile 빌드 단계에서 vendor 소스 + checkpoints 를 `/app/vendor/ditto-talkinghead`
    # 로 직접 clone 한다. 추론은 그 안의 `inference.py` CLI 를 subprocess 로 호출.
    ditto_vendor_dir: str | None = Field(
        None,
        description=(
            "antgroup/ditto-talkinghead repo clone 위치. `inference.py` 가 들어 있는 디렉토리."
        ),
    )
    ditto_data_root: str | None = Field(
        None,
        description=(
            "TRT/PyTorch 모델 디렉토리. 기본은 `<vendor>/checkpoints/ditto_pytorch` "
            "(Blackwell 호환). Ampere/Ada 인스턴스에선 `ditto_trt_Ampere_Plus` 로 변경 가능."
        ),
    )
    ditto_cfg_pkl: str | None = Field(
        None,
        description=(
            "Ditto cfg pickle 경로. TRT 백엔드는 `v0.4_hubert_cfg_trt.pkl`, "
            "PyTorch 백엔드는 `v0.4_hubert_cfg_pytorch.pkl` 을 사용한다."
        ),
    )

    max_upload_bytes: int = Field(
        50 * 1024 * 1024,
        ge=1,
        description=(
            "페르소나 사진·음성 업로드 한 파일당 최대 크기(바이트). 기본 50MB. "
            "초과 시 라우터가 413 Request Entity Too Large 반환."
        ),
    )

    log_level: str = Field(
        "INFO",
        description="로그 레벨: DEBUG | INFO | WARNING | ERROR.",
    )


@lru_cache
def get_settings() -> Settings:
    """프로세스 전역 Settings 인스턴스 반환."""
    return Settings()
