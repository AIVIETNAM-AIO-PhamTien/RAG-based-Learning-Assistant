from __future__ import annotations

import logging
import time
from functools import lru_cache

from google import genai
from google.genai import types

from evaluation.config import get_eval_settings

logger = logging.getLogger(__name__)

_RETRYABLE_CODES = frozenset({429, 503})


class RateLimitExhausted(RuntimeError):
    pass


class RateLimitedClient:
    def __init__(
        self,
        api_key: str,
        delay_between_calls: float = 1.0,
        max_retries: int = 5,
    ) -> None:
        retry = types.HttpRetryOptions(
            attempts=max_retries,
            initialDelay=2.0,
            maxDelay=60.0,
            expBase=2,
            jitter=1,
            httpStatusCodes=list(_RETRYABLE_CODES),
        )
        http_options = types.HttpOptions(retry_options=retry)
        self._client = genai.Client(api_key=api_key, http_options=http_options)
        self._delay = delay_between_calls
        self._last_call_start = 0.0

    def generate_content(
        self,
        model: str,
        contents: str,
        config: types.GenerateContentConfig | None = None,
    ):
        from google.genai.errors import APIError

        # Enforce a minimum interval between call *starts* (not a fixed sleep
        # after each call) — otherwise a fast response leaves the actual
        # request rate far above what `delay_between_calls` was meant to cap.
        elapsed = time.monotonic() - self._last_call_start
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_call_start = time.monotonic()

        try:
            return self._client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except APIError as exc:
            if exc.code in _RETRYABLE_CODES:
                raise RateLimitExhausted(
                    f"API failed with {exc.code} after SDK retries"
                ) from exc
            raise


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
