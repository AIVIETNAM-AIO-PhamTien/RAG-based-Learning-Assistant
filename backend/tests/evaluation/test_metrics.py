from unittest.mock import MagicMock, patch

import numpy as np

from evaluation.metrics import (
    _compute_relevance_vector,
    _context_overlap,
    _lcs_length,
    _ndcg_from_relevances,
    _rouge_l_f1,
    _token_f1,
    compute_all_metrics,
    compute_generation_metrics,
    compute_retrieval_metrics,
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
    recall, _ = _context_overlap(["the answer is here", "other stuff"], ["the answer is here"])
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
        patch("evaluation.metrics._get_nli_model", return_value=(fake_nli, 1)),
    ):
        scores = compute_sample_metrics(result)

    expected_keys = {
        "recall",
        "rr",
        "precision_at_k",
        "hit_rate_at_k",
        "ndcg_at_k",
        "answer_similarity",
        "rouge_l",
        "token_f1",
        "citation_coverage",
        "faithfulness_nli",
        "context_relevance",
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
        patch("evaluation.metrics._get_nli_model", return_value=(fake_nli, 1)),
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
        patch("evaluation.metrics._get_nli_model", return_value=(fake_nli, 1)),
    ):
        scores = compute_sample_metrics(result)

    assert scores["citation_coverage"] == 0.0


# --- New retrieval metrics ---


def test_precision_at_k_all_relevant():
    relevances = _compute_relevance_vector(["the answer is here"], ["the answer is here"])
    assert sum(relevances) / len(relevances) == 1.0


def test_precision_at_k_none_relevant():
    relevances = _compute_relevance_vector(["completely unrelated"], ["the answer is here"])
    assert sum(relevances) / len(relevances) == 0.0


def test_precision_at_k_partial():
    relevances = _compute_relevance_vector(
        ["target text is here", "unrelated stuff"], ["target text"]
    )
    assert sum(relevances) / len(relevances) == 0.5


def test_hit_rate_found():
    relevances = _compute_relevance_vector(["unrelated", "target text is here"], ["target text"])
    assert any(relevances)


def test_hit_rate_not_found():
    relevances = _compute_relevance_vector(
        ["alpha beta", "gamma delta"], ["xyz completely different"]
    )
    assert not any(relevances)


def test_ndcg_perfect_order():
    assert _ndcg_from_relevances([True, False, False]) == 1.0


def test_ndcg_no_relevant():
    assert _ndcg_from_relevances([False, False, False]) == 0.0


def test_ndcg_reversed():
    score = _ndcg_from_relevances([False, False, True])
    assert 0.0 < score < 1.0


def test_ndcg_empty():
    assert _ndcg_from_relevances([]) == 0.0


# --- Latency percentiles ---


def test_latency_percentiles_without_ragas():
    results = [
        _make_result(ret_latency=10.0, gen_latency=100.0),
        _make_result(ret_latency=20.0, gen_latency=200.0),
        _make_result(ret_latency=30.0, gen_latency=300.0),
    ]
    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [(0, 0.5)]

    fake_embedder = MagicMock()
    v = np.ones(384, dtype=np.float32) / np.sqrt(384)
    fake_embedder.embed_query.return_value = v.tolist()

    fake_nli = MagicMock()
    fake_nli.predict.return_value = [0.8]

    with (
        patch("evaluation.metrics.get_embedder", return_value=fake_embedder),
        patch("evaluation.metrics.get_reranker", return_value=fake_reranker),
        patch("evaluation.metrics._get_nli_model", return_value=(fake_nli, 1)),
    ):
        for r in results:
            r.metric_scores = compute_sample_metrics(r)

    agg = compute_all_metrics(results, use_ragas=False)
    assert "retrieval_latency_p50" in agg
    assert "retrieval_latency_p95" in agg
    assert "retrieval_latency_mean" in agg
    assert "generation_latency_p50" in agg
    assert "generation_latency_p95" in agg
    assert "generation_latency_mean" in agg


# --- NLI faithfulness entailment index ---


def test_faithfulness_high_entailment():
    """When entailment probability is high (index 1), faithfulness should be high."""
    result = _make_result(
        retrieved=["The sky is blue."],
        answer="The sky is blue.",
    )

    fake_embedder = MagicMock()
    v = np.ones(384, dtype=np.float32) / np.sqrt(384)
    fake_embedder.embed_query.return_value = v.tolist()

    fake_nli = MagicMock()
    fake_nli.predict.return_value = [np.array([0.05, 0.9, 0.05])]

    with (
        patch("evaluation.metrics.get_embedder", return_value=fake_embedder),
        patch("evaluation.metrics._get_nli_model", return_value=(fake_nli, 1)),
    ):
        scores = compute_generation_metrics(result.sample, result.retrieval, result.generation)

    assert scores["faithfulness_nli"] == 1.0


def test_faithfulness_high_contradiction():
    """When contradiction is high (index 0), faithfulness should be low."""
    result = _make_result(
        retrieved=["The sky is blue."],
        answer="The sky is green.",
    )

    fake_embedder = MagicMock()
    v = np.ones(384, dtype=np.float32) / np.sqrt(384)
    fake_embedder.embed_query.return_value = v.tolist()

    fake_nli = MagicMock()
    fake_nli.predict.return_value = [np.array([0.9, 0.05, 0.05])]

    with (
        patch("evaluation.metrics.get_embedder", return_value=fake_embedder),
        patch("evaluation.metrics._get_nli_model", return_value=(fake_nli, 1)),
    ):
        scores = compute_generation_metrics(result.sample, result.retrieval, result.generation)

    assert scores["faithfulness_nli"] == 0.0


def test_faithfulness_scored_per_chunk_not_joined():
    """A sentence supported by only one of several chunks should still count
    as faithful — entailment must be checked per-chunk, not against all
    retrieved chunks concatenated into one string (which could truncate away
    the supporting chunk)."""
    result = _make_result(
        retrieved=["Irrelevant chunk about cooking.", "The sky is blue."],
        answer="The sky is blue.",
    )

    fake_embedder = MagicMock()
    v = np.ones(384, dtype=np.float32) / np.sqrt(384)
    fake_embedder.embed_query.return_value = v.tolist()

    fake_nli = MagicMock()
    # 1 sentence x 2 chunks = 2 pairs: first chunk doesn't entail, second does.
    fake_nli.predict.return_value = [
        np.array([0.1, 0.05, 0.85]),
        np.array([0.05, 0.9, 0.05]),
    ]

    with (
        patch("evaluation.metrics.get_embedder", return_value=fake_embedder),
        patch("evaluation.metrics._get_nli_model", return_value=(fake_nli, 1)),
    ):
        scores = compute_generation_metrics(result.sample, result.retrieval, result.generation)

    assert scores["faithfulness_nli"] == 1.0
    # Verifies both chunks were checked individually (2 pairs, not 1 joined).
    call_args = fake_nli.predict.call_args[0][0]
    assert len(call_args) == 2


# --- answer_similarity with empty ground truth ---


def test_answer_similarity_empty_ground_truth():
    """answer_similarity should be NaN when ground_truth_answer is empty."""
    result = _make_result(
        ground_truth="",
        retrieved=["some context"],
        answer="Some answer.",
    )

    fake_embedder = MagicMock()
    v = np.ones(384, dtype=np.float32) / np.sqrt(384)
    fake_embedder.embed_query.return_value = v.tolist()

    fake_nli = MagicMock()
    fake_nli.predict.return_value = [0.5]

    with (
        patch("evaluation.metrics.get_embedder", return_value=fake_embedder),
        patch("evaluation.metrics._get_nli_model", return_value=(fake_nli, 1)),
    ):
        scores = compute_generation_metrics(result.sample, result.retrieval, result.generation)

    assert np.isnan(scores["answer_similarity"])


def test_answer_similarity_uses_ground_truth():
    """answer_similarity should embed ground_truth_answer, not question."""
    result = _make_result(
        question="What color is the sky?",
        ground_truth="The sky is blue.",
        retrieved=["The sky is blue."],
        answer="The sky is blue.",
    )

    call_log = []
    fake_embedder = MagicMock()

    def tracking_embed(text):
        call_log.append(text)
        v = np.ones(384, dtype=np.float32) / np.sqrt(384)
        return v.tolist()

    fake_embedder.embed_query.side_effect = tracking_embed

    fake_nli = MagicMock()
    fake_nli.predict.return_value = [0.8]

    with (
        patch("evaluation.metrics.get_embedder", return_value=fake_embedder),
        patch("evaluation.metrics._get_nli_model", return_value=(fake_nli, 1)),
    ):
        compute_generation_metrics(result.sample, result.retrieval, result.generation)

    assert "The sky is blue." in call_log
    assert "What color is the sky?" not in call_log


# --- compute_context_relevance parameter ---


def test_retrieval_metrics_without_context_relevance():
    """When compute_context_relevance=False, reranker should not be called."""
    sample = EvalSample(
        question="Q?",
        ground_truth_answer="A",
        ground_truth_contexts=["ctx"],
    )
    retrieval = RetrievalResult(
        retrieved_contexts=["ctx"],
        retrieved_scores=[0.9],
    )

    scores = compute_retrieval_metrics(sample, retrieval, compute_context_relevance=False)

    assert scores["context_relevance"] == 0.0
    assert "recall" in scores
    assert "precision_at_k" in scores


# --- nanmean aggregation ---


def test_aggregate_handles_nan_scores():
    """np.nanmean should exclude NaN values from averages."""
    r1 = _make_result()
    r2 = _make_result()
    r1.metric_scores = {"answer_similarity": 0.9, "recall": 1.0}
    r2.metric_scores = {"answer_similarity": float("nan"), "recall": 0.5}

    agg = compute_all_metrics([r1, r2], use_ragas=False)

    assert agg["answer_similarity"] == 0.9
    assert agg["recall"] == 0.75
