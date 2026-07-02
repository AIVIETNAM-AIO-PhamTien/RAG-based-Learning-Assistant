from unittest.mock import MagicMock, patch

import pytest
from google.genai.errors import ClientError, ServerError

from evaluation.rate_limiter import RateLimitedClient, RateLimitExhausted


@pytest.fixture()
def client():
    return RateLimitedClient(
        api_key="fake-key",
        delay_between_calls=0.0,
        max_retries=5,
    )


def test_successful_call(client):
    mock_response = MagicMock()
    mock_response.text = "hello"
    with patch.object(client, "_client") as mock_genai:
        mock_genai.models.generate_content.return_value = mock_response
        result = client.generate_content(model="gemini", contents="test")
    assert result.text == "hello"
    mock_genai.models.generate_content.assert_called_once()


def test_wraps_429_as_rate_limit_exhausted(client):
    exc = ClientError(429, {"error": {"message": "quota exceeded"}})
    with patch.object(client, "_client") as mock_genai:
        mock_genai.models.generate_content.side_effect = exc
        with pytest.raises(RateLimitExhausted, match="429"):
            client.generate_content(model="gemini", contents="test")


def test_wraps_503_as_rate_limit_exhausted(client):
    exc = ServerError(503, {"error": {"message": "unavailable"}})
    with patch.object(client, "_client") as mock_genai:
        mock_genai.models.generate_content.side_effect = exc
        with pytest.raises(RateLimitExhausted, match="503"):
            client.generate_content(model="gemini", contents="test")


def test_raises_non_retryable_error_directly(client):
    exc = ClientError(400, {"error": {"message": "bad request"}})
    with patch.object(client, "_client") as mock_genai:
        mock_genai.models.generate_content.side_effect = exc
        with pytest.raises(ClientError):
            client.generate_content(model="gemini", contents="test")


def test_raises_non_api_error_directly(client):
    with patch.object(client, "_client") as mock_genai:
        mock_genai.models.generate_content.side_effect = ValueError("bad input")
        with pytest.raises(ValueError, match="bad input"):
            client.generate_content(model="gemini", contents="test")


def test_no_sleep_when_enough_time_has_passed():
    """If the previous call finished long enough ago, don't sleep at all."""
    client = RateLimitedClient(api_key="fake-key", delay_between_calls=0.5, max_retries=5)
    mock_response = MagicMock()

    with (
        patch.object(client, "_client") as mock_genai,
        patch("evaluation.rate_limiter.time.sleep") as mock_sleep,
    ):
        mock_genai.models.generate_content.return_value = mock_response
        client.generate_content(model="gemini", contents="test")

    mock_sleep.assert_not_called()


def test_delay_between_call_starts_not_just_after_call():
    """The wait must be measured from when the previous call *started*, so a
    fast API response doesn't leave the real request rate above the cap."""
    client = RateLimitedClient(api_key="fake-key", delay_between_calls=0.5, max_retries=5)
    mock_response = MagicMock()

    with (
        patch.object(client, "_client") as mock_genai,
        patch("evaluation.rate_limiter.time.sleep") as mock_sleep,
        patch("evaluation.rate_limiter.time.monotonic") as mock_monotonic,
    ):
        mock_genai.models.generate_content.return_value = mock_response
        # Call 1 starts at t=10.0 (elapsed since init is huge -> no sleep,
        # _last_call_start becomes 10.0). Call 2 starts only 0.2s later.
        mock_monotonic.side_effect = [10.0, 10.0, 10.2, 10.2]
        client.generate_content(model="gemini", contents="test")
        client.generate_content(model="gemini", contents="test")

    mock_sleep.assert_called_once()
    slept_for = mock_sleep.call_args[0][0]
    assert slept_for == pytest.approx(0.3)


def test_no_pacing_delay_when_disabled():
    client = RateLimitedClient(api_key="fake-key", delay_between_calls=0.0, max_retries=5)
    mock_response = MagicMock()

    with (
        patch.object(client, "_client") as mock_genai,
        patch("evaluation.rate_limiter.time.sleep") as mock_sleep,
    ):
        mock_genai.models.generate_content.return_value = mock_response
        client.generate_content(model="gemini", contents="test")
        client.generate_content(model="gemini", contents="test")

    mock_sleep.assert_not_called()


def test_sdk_retry_config():
    with patch("evaluation.rate_limiter.genai.Client") as mock_cls:
        RateLimitedClient(api_key="fake", delay_between_calls=1.0, max_retries=3)

    call_kwargs = mock_cls.call_args[1]
    http_options = call_kwargs["http_options"]
    retry = http_options.retry_options
    assert retry.attempts == 3
    assert retry.initial_delay == 2.0
    assert retry.max_delay == 60.0
    assert 429 in retry.http_status_codes
    assert 503 in retry.http_status_codes
