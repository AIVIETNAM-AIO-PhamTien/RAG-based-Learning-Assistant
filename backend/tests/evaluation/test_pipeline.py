from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from evaluation.pipeline import EvalPipeline, InMemoryRetriever
from evaluation.rate_limiter import RateLimitExhausted
from evaluation.schemas import ExperimentConfig


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
        mock_client.return_value.generate_content.side_effect = RateLimitExhausted(
            "rate limit"
        )
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
