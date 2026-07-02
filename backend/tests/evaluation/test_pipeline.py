import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from evaluation.pipeline import EvalPipeline, InMemoryRetriever
from evaluation.rate_limiter import RateLimitExhausted
from evaluation.schemas import EvalSample, ExperimentConfig, RetrievalResult


class FakeEmbedder:
    def embed_texts(self, texts):
        vectors = []
        for i, _ in enumerate(texts):
            v = np.zeros(384, dtype=np.float32)
            v[i % 384] = 1.0
            vectors.append(v.tolist())
        return vectors

    def embed_query(self, text):
        v = np.zeros(384, dtype=np.float32)
        v[0] = 1.0
        return v.tolist()


@pytest.fixture()
def fake_embedder():
    with patch("evaluation.pipeline.get_embedder") as mock:
        mock.return_value = FakeEmbedder()
        yield mock


def test_in_memory_retriever_index(fake_embedder):
    retriever = InMemoryRetriever()
    retriever.index(["doc A", "doc B", "doc C"])
    assert retriever.size == 3


def test_in_memory_retriever_empty(fake_embedder):
    retriever = InMemoryRetriever()
    retriever.index([])
    assert retriever.size == 0
    assert retriever.retrieve("query", 5) == []


def test_in_memory_retriever_retrieve_ranking(fake_embedder):
    retriever = InMemoryRetriever()
    retriever.index(["doc A", "doc B", "doc C"])

    results = retriever.retrieve("query", top_k=3)
    assert len(results) == 3
    idx, score, text = results[0]
    assert idx == 0
    assert score == pytest.approx(1.0)
    assert text == "doc A"


def test_in_memory_retriever_top_k_limit(fake_embedder):
    retriever = InMemoryRetriever()
    retriever.index(["a", "b", "c", "d", "e"])
    results = retriever.retrieve("q", top_k=2)
    assert len(results) == 2


def test_in_memory_retriever_with_rerank(fake_embedder):
    with patch("evaluation.pipeline.get_reranker") as mock_reranker:
        reranker = MagicMock()
        reranker.rerank.return_value = [(1, 0.95), (0, 0.80)]
        mock_reranker.return_value = reranker

        retriever = InMemoryRetriever()
        retriever.index(["doc A", "doc B", "doc C"])

        results = retriever.retrieve_with_rerank("query", top_k=2, candidate_k=3)
        assert len(results) == 2
        reranker.rerank.assert_called_once()
        assert results[0][1] == pytest.approx(0.95)
        assert results[1][1] == pytest.approx(0.80)


def test_generate_fallback_on_rate_limit():
    config = ExperimentConfig(name="test")
    pipeline = EvalPipeline(config)

    with (
        patch("evaluation.pipeline.get_eval_settings") as mock_settings,
        patch("evaluation.pipeline.get_rate_limited_client") as mock_client,
    ):
        mock_settings.return_value = MagicMock(gemini_api_key="fake-key")
        mock_client.return_value.generate_content.side_effect = RateLimitExhausted("rate limit")
        result = pipeline.generate("What is AI?", ["some context"])

    assert result.generated_answer == "[ERROR: rate limited]"
    assert result.latency_ms == 0.0


def test_warmup_calls_embedder_and_reranker():
    config = ExperimentConfig(name="test", rerank_enabled=True)
    pipeline = EvalPipeline(config)

    fake_emb = MagicMock()
    fake_rnk = MagicMock()
    fake_rnk.rerank.return_value = [(0, 0.9)]

    with (
        patch("evaluation.pipeline.get_embedder", return_value=fake_emb),
        patch("evaluation.pipeline.get_reranker", return_value=fake_rnk),
    ):
        pipeline.warmup()

    fake_emb.embed_query.assert_called_once_with("warmup")
    fake_rnk.rerank.assert_called_once_with("warmup", ["warmup document"], 1)


def test_warmup_skips_reranker_when_disabled():
    config = ExperimentConfig(name="test", rerank_enabled=False)
    pipeline = EvalPipeline(config)

    fake_emb = MagicMock()

    with (
        patch("evaluation.pipeline.get_embedder", return_value=fake_emb),
        patch("evaluation.pipeline.get_reranker") as mock_rnk,
    ):
        pipeline.warmup()

    fake_emb.embed_query.assert_called_once_with("warmup")
    mock_rnk.assert_not_called()


def test_retrieve_returns_k_fields(fake_embedder):
    config = ExperimentConfig(name="test", top_k=3, rerank_enabled=False)
    pipeline = EvalPipeline(config)
    pipeline.prepare_corpus(["context A", "context B"])

    result = pipeline.retrieve("test query")
    assert result.requested_k == 3
    assert result.effective_k == len(result.retrieved_contexts)


def test_embedding_model_mismatch_raises():
    config = ExperimentConfig(name="test", embedding_model="intfloat/e5-base-v2")

    with pytest.raises(ValueError, match="differs from app setting"):
        EvalPipeline(config)


def test_prepare_corpus_caching():
    config = ExperimentConfig(name="test", rerank_enabled=False)

    embed_call_count = 0
    original_fake = FakeEmbedder()

    class TrackingEmbedder:
        def embed_texts(self, texts):
            nonlocal embed_call_count
            embed_call_count += 1
            return original_fake.embed_texts(texts)

        def embed_query(self, text):
            return original_fake.embed_query(text)

    with patch("evaluation.pipeline.get_embedder", return_value=TrackingEmbedder()):
        pipeline = EvalPipeline(config)

        result1 = pipeline.prepare_corpus(["context A", "context B"])
        assert embed_call_count == 1

        result2 = pipeline.prepare_corpus(["context A", "context B"])
        assert result1 == result2
        assert embed_call_count == 1

        result3 = pipeline.prepare_corpus(["context C"])
        assert result3 != result1
        assert embed_call_count == 2


# ── batch generation ──────────────────────────────────────────────────


def _samples_and_retrievals(n: int) -> tuple[list[EvalSample], list[RetrievalResult]]:
    samples = [
        EvalSample(question=f"Question {i}?", ground_truth_answer=f"Answer {i}") for i in range(n)
    ]
    retrievals = [
        RetrievalResult(retrieved_contexts=[f"context {i}a", f"context {i}b"]) for i in range(n)
    ]
    return samples, retrievals


def test_build_batch_prompt_resets_citation_index_per_question():
    config = ExperimentConfig(name="test")
    pipeline = EvalPipeline(config)

    citations_q1 = pipeline._build_citations(["ctx 1a", "ctx 1b"])
    citations_q2 = pipeline._build_citations(["ctx 2a"])
    prompt = pipeline._build_batch_prompt(
        [("Question one?", citations_q1), ("Question two?", citations_q2)]
    )

    assert "Question one?" in prompt
    assert "Question two?" in prompt
    # Each question's own citation block starts at [1] again.
    assert prompt.count("[1] eval_doc") == 2


def test_generate_batch_grouped_parses_json_array():
    config = ExperimentConfig(name="test")
    pipeline = EvalPipeline(config)
    samples, retrievals = _samples_and_retrievals(3)

    fake_response = MagicMock()
    fake_response.text = json.dumps(["answer 0 [1]", "answer 1 [1][2]", "answer 2"])

    with (
        patch("evaluation.pipeline.get_eval_settings") as mock_settings,
        patch("evaluation.pipeline.get_rate_limited_client") as mock_client,
    ):
        mock_settings.return_value = MagicMock(gemini_api_key="fake-key")
        mock_client.return_value.generate_content.return_value = fake_response
        results = pipeline.generate_batch_grouped(samples, retrievals, batch_size=3)

    assert mock_client.return_value.generate_content.call_count == 1
    assert [r.generated_answer for r in results] == [
        "answer 0 [1]",
        "answer 1 [1][2]",
        "answer 2",
    ]
    assert results[0].citations_used == [1]
    assert results[1].citations_used == [1, 2]
    assert results[2].citations_used == []


def test_generate_batch_grouped_parses_combined_citation_bracket():
    # Model sometimes writes [1, 2] instead of [1][2] despite the prompt
    # asking for separate brackets — must still be counted correctly.
    config = ExperimentConfig(name="test")
    pipeline = EvalPipeline(config)
    samples, retrievals = _samples_and_retrievals(1)

    fake_response = MagicMock()
    fake_response.text = json.dumps(["answer 0 [1, 2]"])

    with (
        patch("evaluation.pipeline.get_eval_settings") as mock_settings,
        patch("evaluation.pipeline.get_rate_limited_client") as mock_client,
    ):
        mock_settings.return_value = MagicMock(gemini_api_key="fake-key")
        mock_client.return_value.generate_content.return_value = fake_response
        results = pipeline.generate_batch_grouped(samples, retrievals, batch_size=1)

    assert results[0].citations_used == [1, 2]


def test_generate_batch_grouped_splits_into_multiple_calls():
    config = ExperimentConfig(name="test")
    pipeline = EvalPipeline(config)
    samples, retrievals = _samples_and_retrievals(5)

    def fake_generate_content(**kwargs):
        response = MagicMock()
        # Echo back a plausible-sized array regardless of batch content.
        response.text = json.dumps(["ok"] * 2)
        return response

    with (
        patch("evaluation.pipeline.get_eval_settings") as mock_settings,
        patch("evaluation.pipeline.get_rate_limited_client") as mock_client,
    ):
        mock_settings.return_value = MagicMock(gemini_api_key="fake-key")
        mock_client.return_value.generate_content.side_effect = fake_generate_content
        results = pipeline.generate_batch_grouped(samples, retrievals, batch_size=2)

    # 5 samples / batch_size 2 -> groups of 2, 2, 1 -> 3 calls total.
    assert mock_client.return_value.generate_content.call_count == 3
    assert len(results) == 5


def test_generate_batch_grouped_marks_mismatched_answers_as_error():
    config = ExperimentConfig(name="test")
    pipeline = EvalPipeline(config)
    samples, retrievals = _samples_and_retrievals(3)

    fake_response = MagicMock()
    # Model only returned 2 answers for 3 questions.
    fake_response.text = json.dumps(["answer 0", "answer 1"])

    with (
        patch("evaluation.pipeline.get_eval_settings") as mock_settings,
        patch("evaluation.pipeline.get_rate_limited_client") as mock_client,
    ):
        mock_settings.return_value = MagicMock(gemini_api_key="fake-key")
        mock_client.return_value.generate_content.return_value = fake_response
        results = pipeline.generate_batch_grouped(samples, retrievals, batch_size=3)

    assert results[0].generated_answer == "answer 0"
    assert results[1].generated_answer == "answer 1"
    assert results[2].generated_answer == "[ERROR: batch parse mismatch]"


def test_generate_batch_grouped_handles_malformed_json():
    config = ExperimentConfig(name="test")
    pipeline = EvalPipeline(config)
    samples, retrievals = _samples_and_retrievals(2)

    fake_response = MagicMock()
    fake_response.text = "not valid json"

    with (
        patch("evaluation.pipeline.get_eval_settings") as mock_settings,
        patch("evaluation.pipeline.get_rate_limited_client") as mock_client,
    ):
        mock_settings.return_value = MagicMock(gemini_api_key="fake-key")
        mock_client.return_value.generate_content.return_value = fake_response
        results = pipeline.generate_batch_grouped(samples, retrievals, batch_size=2)

    assert all(r.generated_answer == "[ERROR: batch parse mismatch]" for r in results)


def test_generate_batch_grouped_rate_limit_marks_whole_group():
    config = ExperimentConfig(name="test")
    pipeline = EvalPipeline(config)
    samples, retrievals = _samples_and_retrievals(2)

    with (
        patch("evaluation.pipeline.get_eval_settings") as mock_settings,
        patch("evaluation.pipeline.get_rate_limited_client") as mock_client,
    ):
        mock_settings.return_value = MagicMock(gemini_api_key="fake-key")
        mock_client.return_value.generate_content.side_effect = RateLimitExhausted("rate limit")
        results = pipeline.generate_batch_grouped(samples, retrievals, batch_size=2)

    assert all(r.generated_answer == "[ERROR: rate limited]" for r in results)
