from evaluation.schemas import (
    EvalResult,
    EvalSample,
    ExperimentConfig,
    ExperimentReport,
    GenerationResult,
    RetrievalResult,
)


def test_eval_sample_creation():
    sample = EvalSample(
        question="What is RAG?",
        ground_truth_answer="Retrieval-Augmented Generation",
        ground_truth_contexts=["RAG stands for Retrieval-Augmented Generation."],
        metadata={"source": "test", "id": 0},
    )
    assert sample.question == "What is RAG?"
    assert len(sample.ground_truth_contexts) == 1
    assert sample.metadata["source"] == "test"


def test_eval_sample_defaults():
    sample = EvalSample(question="Q?", ground_truth_answer="A")
    assert sample.ground_truth_contexts == []
    assert sample.metadata == {}


def test_retrieval_result_defaults():
    result = RetrievalResult()
    assert result.retrieved_contexts == []
    assert result.retrieved_scores == []
    assert result.latency_ms == 0.0


def test_generation_result_defaults():
    result = GenerationResult()
    assert result.generated_answer == ""
    assert result.citations_used == []


def test_eval_result_composition():
    sample = EvalSample(question="Q?", ground_truth_answer="A")
    retrieval = RetrievalResult(
        retrieved_contexts=["ctx1", "ctx2"],
        retrieved_scores=[0.9, 0.8],
        latency_ms=15.5,
    )
    generation = GenerationResult(
        generated_answer="Answer [1]",
        latency_ms=200.0,
        citations_used=[1],
    )
    result = EvalResult(
        sample=sample,
        retrieval=retrieval,
        generation=generation,
        metric_scores={"recall_at_5": 1.0},
    )
    assert result.metric_scores["recall_at_5"] == 1.0
    assert result.retrieval.latency_ms == 15.5


def test_experiment_config_defaults():
    config = ExperimentConfig(name="test")
    assert config.chunk_size == 1600
    assert config.chunk_overlap == 250
    assert config.top_k == 5
    assert config.rerank_enabled is True
    assert config.rerank_candidate_k == 20
    assert config.dataset_name == "pdf_qa"


def test_experiment_config_custom():
    config = ExperimentConfig(
        name="small_chunks",
        chunk_size=800,
        chunk_overlap=150,
        top_k=10,
        rerank_enabled=False,
        dataset_name="hotpotqa",
        num_samples=50,
    )
    assert config.chunk_size == 800
    assert config.rerank_enabled is False
    assert config.num_samples == 50


def test_experiment_report_serialization():
    config = ExperimentConfig(name="test")
    report = ExperimentReport(
        config=config,
        aggregate_metrics={"recall_at_5": 0.85, "mrr": 0.72},
        timestamp="2026-06-24T00:00:00Z",
        duration_seconds=42.5,
    )
    data = report.model_dump()
    assert data["config"]["name"] == "test"
    assert data["aggregate_metrics"]["recall_at_5"] == 0.85

    restored = ExperimentReport(**data)
    assert restored.config.name == "test"
    assert restored.duration_seconds == 42.5
