"""Runtime configuration loaded from environment variables / `.env`.

All values map 1:1 to the keys documented in `.env.example`. The
`get_settings` helper is `lru_cache`-d so the same instance is reused
across the request lifecycle; tests can call `get_settings.cache_clear()`
after mutating environment variables.
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
        description="Bearer token shared with Spring Boot for internal-network authentication.",
    )
    database_url: str = Field(
        "",
        description="PostgreSQL DSN (asyncpg driver). Empty when USE_DB_MOCK is true.",
    )
    use_db_mock: bool = Field(
        False,
        description="When true, repository functions return in-memory mock data (dev/test).",
    )
    models_dir: str = Field(
        "/models",
        description="Model weights directory (mounted Docker volume `yeoun-models`).",
    )
    persona_dir: str = Field(
        "/var/persona",
        description="Persistent persona assets directory (photos, voice, voice_ref, idle clips).",
    )
    hf_home: str = Field(
        "/models/hf-cache",
        description="Hugging Face cache directory; nested under MODELS_DIR for volume reuse.",
    )
    gpu_enabled: bool = Field(
        True,
        description="When false, the model loaders short-circuit and return dummy outputs (dev/test).",
    )
    log_level: str = Field(
        "INFO",
        description="Log level: DEBUG | INFO | WARNING | ERROR.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings instance."""
    return Settings()
