from unittest.mock import MagicMock, patch

import numpy as np

from evaluation.metrics import (
    _context_overlap,
    _lcs_length,
    _rouge_l_f1,
    _token_f1,
    compute_sample_metrics,
)
from evaluation.schemas import EvalResult, EvalSample, GenerationResult, RetrievalResult


def _make_result(
    question: str = "Q?",
    ground_truth: str = "A",
    gt_contexts: list[str] | None = None,
    retrieved: list[str] | None = None,
    answer: str = "Answer [1]",
    ret_latency: float = 10.0,
    gen_latency: float = 100.0,
) -> EvalResult:
    gt = gt_contexts if gt_contexts is not None else ["default context"]
    ret = retrieved if retrieved is not None else ["default context"]
    return EvalResult(
        sample=EvalSample(
            question=question,
            ground_truth_answer=ground_truth,
            ground_truth_contexts=gt,
        ),
        retrieval=RetrievalResult(
            retrieved_contexts=ret,
            retrieved_scores=[0.9] * len(ret),
            latency_ms=ret_latency,
        ),
        generation=GenerationResult(
            generated_answer=answer,
            latency_ms=gen_latency,
            citations_used=[1],
        ),
    )


# --- Context overlap (recall + MRR) ---


def test_recall_perfect():
    recall, _ = _context_overlap(
        ["the answer is here", "other stuff"], ["the answer is here"]
    )
    assert recall == 1.0


def test_recall_zero():
    recall, _ = _context_overlap(["completely unrelated text"], ["the answer is here"])
    assert recall == 0.0


def test_recall_partial():
    recall, _ = _context_overlap(
        ["context A is present here", "unrelated"], ["context A", "context B"]
    )
    assert 0.0 < recall < 1.0


def test_mrr_first_position():
    _, rr = _context_overlap(["target text is here", "other"], ["target text"])
    assert rr == 1.0


def test_mrr_second_position():
    _, rr = _context_overlap(["unrelated", "target text is here"], ["target text"])
    assert rr == 0.5


def test_mrr_not_found():
    _, rr = _context_overlap(
        ["alpha beta gamma", "delta epsilon"], ["xyz completely different xyz"]
    )
    assert rr == 0.0


def test_context_overlap_empty_gt():
    recall, rr = _context_overlap(["some text"], [])
    assert recall == 0.0
    assert rr == 0.0


# --- Token F1 ---


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


# --- ROUGE-L ---


def test_rouge_l_exact_match():
    assert _rouge_l_f1("hello world", "hello world") == 1.0


def test_rouge_l_no_overlap():
    assert _rouge_l_f1("hello world", "foo bar") == 0.0


def test_rouge_l_partial():
    score = _rouge_l_f1("the cat sat on the mat", "the cat on the mat")
    assert 0.5 < score < 1.0


def test_rouge_l_empty():
    assert _rouge_l_f1("", "hello") == 0.0


# --- LCS ---


def test_lcs_length():
    assert _lcs_length(["a", "b", "c"], ["a", "c"]) == 2
    assert _lcs_length(["a", "b"], ["c", "d"]) == 0
    assert _lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3


# --- compute_sample_metrics ---


def test_compute_sample_metrics_returns_all_keys():
    result = _make_result(
        question="What is AI?",
        ground_truth="Artificial intelligence",
        answer="AI is artificial intelligence used in many fields.",
        gt_contexts=["AI is artificial intelligence."],
        retrieved=["AI is artificial intelligence."],
    )

    fake_embedder = MagicMock()
    v = np.ones(384, dtype=np.float32) / np.sqrt(384)
    fake_embedder.embed_query.return_value = v.tolist()

    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [(0, 0.9)]

    fake_nli = MagicMock()
    fake_nli.predict.return_value = [0.8]

    with (
        patch("evaluation.metrics.get_embedder", return_value=fake_embedder),
        patch("evaluation.metrics.get_reranker", return_value=fake_reranker),
        patch("evaluation.metrics._get_nli_model", return_value=fake_nli),
    ):
        scores = compute_sample_metrics(result)

    expected_keys = {
        "recall", "rr", "answer_similarity", "rouge_l", "token_f1",
        "citation_coverage", "faithfulness_nli", "context_relevance",
    }
    assert set(scores.keys()) == expected_keys
    for val in scores.values():
        assert isinstance(val, float)


def test_compute_sample_metrics_error_answer():
    result = _make_result(answer="[ERROR: rate limited]")

    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [(0, 0.5)]

    with patch("evaluation.metrics.get_reranker", return_value=fake_reranker):
        scores = compute_sample_metrics(result)

    assert scores["answer_similarity"] == 0.0
    assert scores["rouge_l"] == 0.0
    assert scores["faithfulness_nli"] == 0.0
    assert scores["context_relevance"] > 0.0


def test_compute_sample_metrics_citation_coverage():
    result = _make_result(
        retrieved=["ctx1", "ctx2"],
        answer="Based on [1] and [2], the answer is yes.",
    )

    fake_embedder = MagicMock()
    v = np.ones(384, dtype=np.float32) / np.sqrt(384)
    fake_embedder.embed_query.return_value = v.tolist()

    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [(0, 0.8), (1, 0.6)]

    fake_nli = MagicMock()
    fake_nli.predict.return_value = [0.9]

    with (
        patch("evaluation.metrics.get_embedder", return_value=fake_embedder),
        patch("evaluation.metrics.get_reranker", return_value=fake_reranker),
        patch("evaluation.metrics._get_nli_model", return_value=fake_nli),
    ):
        scores = compute_sample_metrics(result)

    assert scores["citation_coverage"] == 1.0


def test_compute_sample_metrics_no_citations():
    result = _make_result(
        retrieved=["ctx1"],
        answer="The answer with no citations.",
    )

    fake_embedder = MagicMock()
    v = np.ones(384, dtype=np.float32) / np.sqrt(384)
    fake_embedder.embed_query.return_value = v.tolist()

    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [(0, 0.5)]

    fake_nli = MagicMock()
    fake_nli.predict.return_value = [0.3]

    with (
        patch("evaluation.metrics.get_embedder", return_value=fake_embedder),
        patch("evaluation.metrics.get_reranker", return_value=fake_reranker),
        patch("evaluation.metrics._get_nli_model", return_value=fake_nli),
    ):
        scores = compute_sample_metrics(result)

    assert scores["citation_coverage"] == 0.0
