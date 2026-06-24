from unittest.mock import MagicMock, patch

from evaluation.datasets.qa_generator import QAGenerator


def _make_generator():
    with patch("evaluation.datasets.qa_generator.get_eval_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            gemini_api_key="fake-key", gemini_model="gemini-test"
        )
        with patch("evaluation.datasets.qa_generator.get_rate_limited_client"):
            gen = QAGenerator()
    return gen


def test_generate_skips_none_response():
    gen = _make_generator()
    mock_response = MagicMock()
    mock_response.text = None
    gen._client.generate_content.return_value = mock_response

    chunks = [{"text": "some text", "page": 1, "source_pdf": "test.pdf"}]
    samples = gen.generate_from_chunks(chunks, num_questions_per_chunk=2)
    assert samples == []


def test_generate_skips_malformed_json():
    gen = _make_generator()
    mock_response = MagicMock()
    mock_response.text = "not valid json {"
    gen._client.generate_content.return_value = mock_response

    chunks = [{"text": "some text", "page": 1}]
    samples = gen.generate_from_chunks(chunks, num_questions_per_chunk=2)
    assert samples == []


def test_generate_skips_missing_fields():
    gen = _make_generator()
    mock_response = MagicMock()
    mock_response.text = '[{"question": "Q?"}]'
    gen._client.generate_content.return_value = mock_response

    chunks = [{"text": "some text", "page": 1}]
    samples = gen.generate_from_chunks(chunks)
    assert samples == []


def test_generate_valid_response():
    gen = _make_generator()
    mock_response = MagicMock()
    mock_response.text = (
        '[{"question": "What is AI?", "answer": "Artificial Intelligence", '
        '"context": "AI is artificial intelligence", "difficulty": "easy"}]'
    )
    gen._client.generate_content.return_value = mock_response

    chunks = [{"text": "AI is artificial intelligence", "page": 1, "source_pdf": "t.pdf"}]
    samples = gen.generate_from_chunks(chunks, num_questions_per_chunk=1)
    assert len(samples) == 1
    assert samples[0].question == "What is AI?"
    assert samples[0].ground_truth_answer == "Artificial Intelligence"
    assert samples[0].metadata["source_pdf"] == "t.pdf"
