from functools import lru_cache
from pathlib import Path

from pydantic import AnyUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AIO"
    app_env: str = "local"
    database_url: str = "postgresql+asyncpg://aio:aio@localhost:5432/aio"
    redis_url: str = "redis://localhost:6379/0"
    cors_origins: list[AnyUrl] = Field(default_factory=lambda: ["http://localhost:3000"])
    cors_origin_regex: str | None = r"^http://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+):3000$"
    upload_dir: Path = Path("../data/uploads")
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    embedding_model_name: str = "intfloat/e5-small-v2"
    chunk_size: int = 1600
    chunk_overlap: int = 250
    retrieval_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
