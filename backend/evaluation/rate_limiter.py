from __future__ import annotations

import logging
import random
import time
from functools import lru_cache

from google import genai
from google.genai import types

from evaluation.config import get_eval_settings

logger = logging.getLogger(__name__)

_RETRY_CONFIG = {
    429: {"base_delay": 5.0, "max_wait": 60.0, "label": "rate limited"},
    503: {"base_delay": 1.0, "max_wait": 10.0, "label": "server unavailable"},
}


class RateLimitExhausted(RuntimeError):
    pass


class RateLimitedClient:
    def __init__(
        self,
        api_key: str,
        delay_between_calls: float = 1.0,
        max_retries: int = 5,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._delay = delay_between_calls
        self._max_retries = max_retries

    def generate_content(
        self,
        model: str,
        contents: str,
        config: types.GenerateContentConfig | None = None,
    ):
        from google.genai.errors import APIError

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=model, contents=contents, config=config
                )
                time.sleep(self._delay)
                return response
            except APIError as exc:
                retry_cfg = _RETRY_CONFIG.get(exc.code)
                if retry_cfg is None:
                    raise
                last_exc = exc
                if attempt < self._max_retries:
                    backoff = retry_cfg["base_delay"] * (2**attempt)
                    wait = min(backoff + random.uniform(0, 1), retry_cfg["max_wait"])
                    logger.warning(
                        "%s %d (attempt %d/%d), retrying in %.1fs",
                        retry_cfg["label"],
                        exc.code,
                        attempt + 1,
                        self._max_retries,
                        wait,
                    )
                    time.sleep(wait)

        raise RateLimitExhausted(
            f"API failed after {self._max_retries} retries"
        ) from last_exc


@lru_cache
def get_rate_limited_client() -> RateLimitedClient:
    settings = get_eval_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required")
    return RateLimitedClient(
        api_key=settings.gemini_api_key,
        delay_between_calls=settings.api_delay_seconds,
        max_retries=settings.api_max_retries,
    )
