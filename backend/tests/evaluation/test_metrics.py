from evaluation.metrics import compute_generation_metrics, compute_retrieval_metrics
from evaluation.schemas import EvalResult, EvalSample, GenerationResult, RetrievalResult


def _make_result(
    gt_contexts: list[str],
    retrieved: list[str],
    answer: str = "Answer [1]",
    ret_latency: float = 10.0,
    gen_latency: float = 100.0,
) -> EvalResult:
    return EvalResult(
        sample=EvalSample(
            question="Q?",
            ground_truth_answer="A",
            ground_truth_contexts=gt_contexts,
        ),
        retrieval=RetrievalResult(
            retrieved_contexts=retrieved,
            retrieved_scores=[0.9] * len(retrieved),
            latency_ms=ret_latency,
        ),
        generation=GenerationResult(
            generated_answer=answer,
            latency_ms=gen_latency,
            citations_used=[1],
        ),
    )


def test_recall_at_5_perfect():
    result = _make_result(
        gt_contexts=["the answer is here"],
        retrieved=["the answer is here", "other stuff"],
    )
    metrics = compute_retrieval_metrics([result])
    assert metrics["recall_at_5"] == 1.0


def test_recall_at_5_zero():
    result = _make_result(
        gt_contexts=["the answer is here"],
        retrieved=["completely unrelated text"],
    )
    metrics = compute_retrieval_metrics([result])
    assert metrics["recall_at_5"] == 0.0


def test_recall_at_5_partial():
    result = _make_result(
        gt_contexts=["context A", "context B"],
        retrieved=["context A is present here", "unrelated"],
    )
    metrics = compute_retrieval_metrics([result])
    assert 0.0 < metrics["recall_at_5"] < 1.0


def test_mrr_first_position():
    result = _make_result(
        gt_contexts=["target text"],
        retrieved=["target text is here", "other"],
    )
    metrics = compute_retrieval_metrics([result])
    assert metrics["mrr"] == 1.0


def test_mrr_second_position():
    result = _make_result(
        gt_contexts=["target text"],
        retrieved=["unrelated", "target text is here"],
    )
    metrics = compute_retrieval_metrics([result])
    assert metrics["mrr"] == 0.5


def test_mrr_not_found():
    result = _make_result(
        gt_contexts=["xyz completely different content xyz"],
        retrieved=["alpha beta gamma", "delta epsilon", "zeta theta"],
    )
    metrics = compute_retrieval_metrics([result])
    assert metrics["mrr"] == 0.0


def test_generation_metrics_citation_coverage():
    result = _make_result(
        gt_contexts=["ctx"],
        retrieved=["ctx1", "ctx2"],
        answer="Based on [1] and [2], the answer is yes.",
    )
    metrics = compute_generation_metrics([result])
    assert metrics["citation_coverage_avg"] == 1.0


def test_generation_metrics_no_citations():
    result = _make_result(
        gt_contexts=["ctx"],
        retrieved=["ctx1"],
        answer="The answer with no citations.",
    )
    metrics = compute_generation_metrics([result])
    assert metrics["citation_coverage_avg"] == 0.0


def test_generation_metrics_latency():
    r1 = _make_result(gt_contexts=["c"], retrieved=["c"], ret_latency=10, gen_latency=100)
    r2 = _make_result(gt_contexts=["c"], retrieved=["c"], ret_latency=20, gen_latency=200)
    metrics = compute_generation_metrics([r1, r2])
    assert metrics["retrieval_latency_p50"] == 15.0
    assert metrics["generation_latency_p50"] == 150.0


def test_empty_results():
    ret = compute_retrieval_metrics([])
    assert ret["recall_at_5"] == 0.0
    assert ret["mrr"] == 0.0

    gen = compute_generation_metrics([])
    assert gen["citation_coverage_avg"] == 0.0
