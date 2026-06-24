from unittest.mock import MagicMock, patch

import pytest
from google.genai.errors import ClientError, ServerError

from evaluation.rate_limiter import RateLimitedClient, RateLimitExhausted


def _make_429():
    return ClientError(429, {"error": {"message": "quota exceeded"}})


def _make_503():
    return ServerError(503, {"error": {"message": "service unavailable"}})


@pytest.fixture()
def client():
    return RateLimitedClient(
        api_key="fake-key",
        delay_between_calls=0.0,
        max_retries=3,
    )


def test_successful_call(client):
    mock_response = MagicMock()
    mock_response.text = "hello"
    with patch.object(client, "_client") as mock_genai:
        mock_genai.models.generate_content.return_value = mock_response
        result = client.generate_content(model="gemini", contents="test")
    assert result.text == "hello"
    mock_genai.models.generate_content.assert_called_once()


def test_retry_on_429(client):
    mock_response = MagicMock()
    mock_response.text = "success"

    with patch.object(client, "_client") as mock_genai:
        mock_genai.models.generate_content.side_effect = [
            _make_429(),
            _make_429(),
            mock_response,
        ]
        result = client.generate_content(model="gemini", contents="test")

    assert result.text == "success"
    assert mock_genai.models.generate_content.call_count == 3


def test_retry_exhausted(client):
    with patch.object(client, "_client") as mock_genai:
        mock_genai.models.generate_content.side_effect = _make_429()
        with pytest.raises(RateLimitExhausted, match="after 3 retries"):
            client.generate_content(model="gemini", contents="test")

    assert mock_genai.models.generate_content.call_count == 4  # initial + 3 retries


def test_delay_between_calls(client):
    client._delay = 0.5
    mock_response = MagicMock()

    with (
        patch.object(client, "_client") as mock_genai,
        patch("evaluation.rate_limiter.time.sleep") as mock_sleep,
    ):
        mock_genai.models.generate_content.return_value = mock_response
        client.generate_content(model="gemini", contents="test")

    mock_sleep.assert_called_once_with(0.5)


def test_retry_on_503(client):
    mock_response = MagicMock()
    mock_response.text = "recovered"

    with patch.object(client, "_client") as mock_genai:
        mock_genai.models.generate_content.side_effect = [
            _make_503(),
            mock_response,
        ]
        result = client.generate_content(model="gemini", contents="test")

    assert result.text == "recovered"
    assert mock_genai.models.generate_content.call_count == 2


def test_retry_mixed_429_503(client):
    mock_response = MagicMock()
    mock_response.text = "ok"

    with patch.object(client, "_client") as mock_genai:
        mock_genai.models.generate_content.side_effect = [
            _make_429(),
            _make_503(),
            mock_response,
        ]
        result = client.generate_content(model="gemini", contents="test")

    assert result.text == "ok"
    assert mock_genai.models.generate_content.call_count == 3


def test_429_backoff_longer_than_503(client):
    mock_response = MagicMock()

    sleeps_429: list[float] = []
    sleeps_503: list[float] = []

    with (
        patch.object(client, "_client") as mock_genai,
        patch("evaluation.rate_limiter.time.sleep") as mock_sleep,
        patch("evaluation.rate_limiter.random.uniform", return_value=0.0),
    ):
        mock_genai.models.generate_content.side_effect = [_make_429(), mock_response]
        mock_sleep.reset_mock()
        client.generate_content(model="gemini", contents="test")
        sleeps_429 = [c.args[0] for c in mock_sleep.call_args_list]

    with (
        patch.object(client, "_client") as mock_genai,
        patch("evaluation.rate_limiter.time.sleep") as mock_sleep,
        patch("evaluation.rate_limiter.random.uniform", return_value=0.0),
    ):
        mock_genai.models.generate_content.side_effect = [_make_503(), mock_response]
        mock_sleep.reset_mock()
        client.generate_content(model="gemini", contents="test")
        sleeps_503 = [c.args[0] for c in mock_sleep.call_args_list]

    assert sleeps_429[0] > sleeps_503[0]


def test_no_retry_on_non_retryable_error(client):
    exc = ClientError(400, {"error": {"message": "bad request"}})
    with patch.object(client, "_client") as mock_genai:
        mock_genai.models.generate_content.side_effect = exc
        with pytest.raises(ClientError):
            client.generate_content(model="gemini", contents="test")

    assert mock_genai.models.generate_content.call_count == 1


def test_no_retry_on_other_errors(client):
    with patch.object(client, "_client") as mock_genai:
        mock_genai.models.generate_content.side_effect = ValueError("bad input")
        with pytest.raises(ValueError, match="bad input"):
            client.generate_content(model="gemini", contents="test")

    assert mock_genai.models.generate_content.call_count == 1
