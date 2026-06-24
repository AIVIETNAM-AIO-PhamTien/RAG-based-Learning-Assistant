from unittest.mock import MagicMock, patch

import numpy as np

from evaluation.local_metrics import (
    _lcs_length,
    _rouge_l_f1,
    _token_f1,
    compute_answer_relevancy,
    compute_context_relevance,
)
from evaluation.schemas import EvalResult, EvalSample, GenerationResult, RetrievalResult


def _make_result(
    question: str = "What is AI?",
    ground_truth: str = "Artificial intelligence",
    answer: str = "AI is artificial intelligence.",
    contexts: list[str] | None = None,
) -> EvalResult:
    return EvalResult(
        sample=EvalSample(
            question=question,
            ground_truth_answer=ground_truth,
            ground_truth_contexts=contexts or ["AI is artificial intelligence."],
        ),
        retrieval=RetrievalResult(
            retrieved_contexts=contexts or ["AI is artificial intelligence."],
            retrieved_scores=[0.9],
            latency_ms=10.0,
        ),
        generation=GenerationResult(
            generated_answer=answer,
            latency_ms=100.0,
            citations_used=[1],
        ),
    )


# --- Token F1 tests ---


def test_token_f1_exact_match():
    assert _token_f1("hello world", "hello world") == 1.0


def test_token_f1_no_overlap():
    assert _token_f1("hello world", "foo bar") == 0.0


def test_token_f1_partial():
    f1 = _token_f1("the cat sat", "the cat ran")
    assert 0.0 < f1 < 1.0


def test_token_f1_empty():
    assert _token_f1("", "hello") == 0.0
    assert _token_f1("hello", "") == 0.0


# --- ROUGE-L tests ---


def test_rouge_l_exact_match():
    assert _rouge_l_f1("hello world", "hello world") == 1.0


def test_rouge_l_no_overlap():
    assert _rouge_l_f1("hello world", "foo bar") == 0.0


def test_rouge_l_partial():
    score = _rouge_l_f1("the cat sat on the mat", "the cat on the mat")
    assert 0.5 < score < 1.0


def test_rouge_l_empty():
    assert _rouge_l_f1("", "hello") == 0.0


# --- LCS tests ---


def test_lcs_length():
    assert _lcs_length(["a", "b", "c"], ["a", "c"]) == 2
    assert _lcs_length(["a", "b"], ["c", "d"]) == 0
    assert _lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3


# --- Answer relevancy tests (mocked embedder) ---


def test_answer_relevancy_with_mock_embedder():
    result = _make_result(
        question="What is machine learning?",
        ground_truth="a subset of AI",
        answer="a subset of AI that learns from data",
    )

    fake_embedder = MagicMock()
    v1 = np.ones(384, dtype=np.float32) / np.sqrt(384)
    fake_embedder.embed_query.return_value = v1.tolist()

    with patch("evaluation.local_metrics.get_embedder", return_value=fake_embedder):
        metrics = compute_answer_relevancy([result])

    assert "answer_similarity" in metrics
    assert "rouge_l" in metrics
    assert "token_f1" in metrics
    assert metrics["answer_similarity"] >= 0.0
    assert metrics["rouge_l"] > 0.0
    assert metrics["token_f1"] > 0.0


def test_answer_relevancy_empty():
    metrics = compute_answer_relevancy([])
    assert metrics["answer_similarity"] == 0.0
    assert metrics["rouge_l"] == 0.0
    assert metrics["token_f1"] == 0.0


def test_answer_relevancy_skips_error_answers():
    result = _make_result(answer="[ERROR: rate limited]")
    fake_embedder = MagicMock()
    with patch("evaluation.local_metrics.get_embedder", return_value=fake_embedder):
        metrics = compute_answer_relevancy([result])
    assert metrics["answer_similarity"] == 0.0
    fake_embedder.embed_query.assert_not_called()


# --- Context relevance tests (mocked reranker) ---


def test_context_relevance_with_mock_reranker():
    result = _make_result(contexts=["relevant context", "another context"])

    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [(0, 0.85), (1, 0.65)]

    with patch("evaluation.local_metrics.get_reranker", return_value=fake_reranker):
        metrics = compute_context_relevance([result])

    assert metrics["context_relevance_avg"] == (0.85 + 0.65) / 2


def test_context_relevance_empty():
    metrics = compute_context_relevance([])
    assert metrics["context_relevance_avg"] == 0.0


def test_context_relevance_no_contexts():
    result = _make_result(contexts=[])
    result.retrieval.retrieved_contexts = []

    fake_reranker = MagicMock()
    with patch("evaluation.local_metrics.get_reranker", return_value=fake_reranker):
        metrics = compute_context_relevance([result])

    assert metrics["context_relevance_avg"] == 0.0
    fake_reranker.rerank.assert_not_called()
