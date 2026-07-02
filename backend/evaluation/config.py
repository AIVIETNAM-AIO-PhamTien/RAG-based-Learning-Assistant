from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    embedding_model_name: str = "intfloat/e5-small-v2"
    rerank_model_name: str = "BAAI/bge-reranker-base"
    output_dir: Path = Path("evaluation/results")
    default_num_samples: int = 100
    # Minimum interval between Gemini API call starts. Default assumes a 5
    # requests/minute quota (60/5=12s); override via .env if your quota differs.
    api_delay_seconds: float = 12.0
    api_max_retries: int = 5
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small"


@lru_cache
def get_eval_settings() -> EvalSettings:
    return EvalSettings()
