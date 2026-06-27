from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from evaluation.metrics import compute_generation_metrics, compute_retrieval_metrics
from evaluation.pipeline import EvalPipeline
from evaluation.schemas import (
    EvalSample,
    ExperimentConfig,
    GenerationResult,
    RetrievalResult,
)


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


def _make_sample():
    return EvalSample(
        question="What is RAG?",
        ground_truth_answer="Retrieval-Augmented Generation",
        ground_truth_contexts=["RAG is Retrieval-Augmented Generation."],
    )


# ── Pipeline stage methods ───────────────────────────────────────────


def test_retrieve_sample(fake_embedder):
    config = ExperimentConfig(name="test", rerank_enabled=False)
    pipeline = EvalPipeline(config)
    sample = _make_sample()

    result = pipeline.retrieve_sample(sample)
    assert isinstance(result, RetrievalResult)
    assert len(result.retrieved_contexts) > 0
    assert result.latency_ms >= 0
    assert result.requested_k == config.top_k
    assert result.effective_k == len(result.retrieved_contexts)


def test_generate_sample():
    config = ExperimentConfig(name="test")
    pipeline = EvalPipeline(config)
    retrieval = RetrievalResult(
        retrieved_contexts=["Some context about RAG."],
        retrieved_scores=[0.9],
        latency_ms=10.0,
    )

    with (
        patch("evaluation.pipeline.get_eval_settings") as mock_settings,
        patch("evaluation.pipeline.get_rate_limited_client") as mock_client,
    ):
        mock_settings.return_value = MagicMock(gemini_api_key="fake-key")
        mock_response = MagicMock()
        mock_response.text = "RAG means Retrieval-Augmented Generation [1]."
        mock_client.return_value.generate_content.return_value = mock_response

        result = pipeline.generate_sample("What is RAG?", retrieval)

    assert isinstance(result, GenerationResult)
    assert "RAG" in result.generated_answer
    assert 1 in result.citations_used


def test_retrieve_batch(fake_embedder):
    config = ExperimentConfig(name="test", rerank_enabled=False, top_k=1)
    pipeline = EvalPipeline(config)
    samples = [_make_sample(), _make_sample()]

    results = pipeline.retrieve_batch(samples, show_progress=False)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, RetrievalResult)


def test_generate_batch():
    config = ExperimentConfig(name="test")
    pipeline = EvalPipeline(config)
    samples = [_make_sample()]
    retrievals = [
        RetrievalResult(
            retrieved_contexts=["context"],
            retrieved_scores=[0.9],
            latency_ms=10.0,
        )
    ]

    with (
        patch("evaluation.pipeline.get_eval_settings") as mock_settings,
        patch("evaluation.pipeline.get_rate_limited_client") as mock_client,
    ):
        mock_settings.return_value = MagicMock(gemini_api_key="fake-key")
        mock_response = MagicMock()
        mock_response.text = "Answer [1]."
        mock_client.return_value.generate_content.return_value = mock_response

        results = pipeline.generate_batch(samples, retrievals, show_progress=False)

    assert len(results) == 1
    assert isinstance(results[0], GenerationResult)


# ── Metric separation ────────────────────────────────────────────────


def test_compute_retrieval_metrics():
    sample = _make_sample()
    retrieval = RetrievalResult(
        retrieved_contexts=["RAG is Retrieval-Augmented Generation."],
        retrieved_scores=[0.95],
        latency_ms=10.0,
    )

    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [(0, 0.9)]

    with patch("evaluation.metrics.get_reranker", return_value=fake_reranker):
        scores = compute_retrieval_metrics(sample, retrieval)

    assert "recall" in scores
    assert "rr" in scores
    assert "rr" in scores
    assert "precision_at_k" in scores
    assert "hit_rate_at_k" in scores
    assert "ndcg_at_k" in scores
    assert "context_relevance" in scores
    assert scores["recall"] == 1.0
    assert scores["hit_rate_at_k"] == 1.0
    assert "answer_similarity" not in scores
    assert "rouge_l" not in scores


def test_compute_generation_metrics():
    sample = _make_sample()
    retrieval = RetrievalResult(
        retrieved_contexts=["RAG is Retrieval-Augmented Generation."],
        retrieved_scores=[0.95],
    )
    generation = GenerationResult(
        generated_answer="RAG stands for Retrieval-Augmented Generation [1].",
        latency_ms=200.0,
        citations_used=[1],
    )

    fake_embedder = MagicMock()
    v = np.ones(384, dtype=np.float32) / np.sqrt(384)
    fake_embedder.embed_query.return_value = v.tolist()

    fake_nli = MagicMock()
    fake_nli.predict.return_value = [0.9]

    with (
        patch("evaluation.metrics.get_embedder", return_value=fake_embedder),
        patch("evaluation.metrics._get_nli_model", return_value=(fake_nli, 1)),
    ):
        scores = compute_generation_metrics(sample, retrieval, generation)

    assert "answer_similarity" in scores
    assert "rouge_l" in scores
    assert "token_f1" in scores
    assert "citation_coverage" in scores
    assert "faithfulness_nli" in scores
    assert "recall" not in scores
    assert "rr" not in scores


def test_compute_generation_metrics_invalid_answer():
    sample = _make_sample()
    retrieval = RetrievalResult(retrieved_contexts=["ctx"])
    generation = GenerationResult(generated_answer="[ERROR: rate limited]")

    scores = compute_generation_metrics(sample, retrieval, generation)
    assert scores["answer_similarity"] == 0.0
    assert scores["rouge_l"] == 0.0
    assert scores["faithfulness_nli"] == 0.0
