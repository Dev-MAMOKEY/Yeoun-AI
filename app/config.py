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
    llm_awq_path: str | None = Field(
        None,
        description=(
            "Gemma 4 AWQ INT4 가중치 디렉토리. 지정되면 우선 시도하고, Blackwell 등에서 "
            "AWQ 커널이 미지원이면 BF16 으로 폴백한다. None 이면 곧장 BF16 로드."
        ),
    )
    llm_bf16_path: str | None = Field(
        None,
        description=(
            "Gemma 4 BF16 가중치 디렉토리 (AWQ 폴백 또는 양자화 미사용 경로). "
            "GPU_ENABLED=true 인데 두 경로 모두 None 이면 부팅 실패."
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
