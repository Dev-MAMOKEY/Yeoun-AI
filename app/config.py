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

    # --- LLM 가중치 경로 (이슈 #6) -----------------------------------------
    llm_bf16_path: str | None = Field(
        None,
        description=(
            "Gemma 4 BF16 가중치 디렉토리 또는 HF repo ID. transformers `from_pretrained` 가 "
            "자동으로 다운로드·캐시한다. GPU_ENABLED=true 인데 None 이면 부팅 실패."
        ),
    )

    # --- TTS 가중치 경로 (이슈 #7) -----------------------------------------
    tts_model_path: str | None = Field(
        None,
        description=(
            "OmniVoice TTS 가중치 디렉토리 또는 HF repo ID (예: `k2-fsa/OmniVoice`). "
            "GPU_ENABLED=true 인데 None 이면 TTS 로드 실패."
        ),
    )

    # --- Ditto-TalkingHead 경로 (이슈 #8) ----------------------------------
    # README 패턴 (https://huggingface.co/digital-avatar/ditto-talkinghead):
    #   git clone github.com/antgroup/ditto-talkinghead → vendor_dir
    #   git clone hf.co/digital-avatar/ditto-talkinghead vendor_dir/checkpoints → 가중치
    # 추론은 `python <vendor_dir>/inference.py --data_root ... --cfg_pkl ... \
    #         --audio_path ... --source_path ... --output_path ...` CLI 를 subprocess 로 호출.
    ditto_vendor_dir: str | None = Field(
        None,
        description=(
            "antgroup/ditto-talkinghead repo clone 위치. `inference.py` 가 들어 있는 디렉토리."
        ),
    )
    ditto_data_root: str | None = Field(
        None,
        description=(
            "TRT/PyTorch 모델 디렉토리. README 디폴트는 `<vendor>/checkpoints/ditto_trt_Ampere_Plus`. "
            "Blackwell(sm_120) 등 Ampere_Plus 미지원 환경에선 `<vendor>/checkpoints/ditto_pytorch` "
            "로 변경하거나 #15 의 ONNX→TRT 재변환 스크립트를 활용."
        ),
    )
    ditto_cfg_pkl: str | None = Field(
        None,
        description=(
            "Ditto cfg pickle 경로. TRT 백엔드는 `v0.4_hubert_cfg_trt.pkl`, "
            "PyTorch 백엔드는 `v0.4_hubert_cfg_pytorch.pkl` 을 사용한다."
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
