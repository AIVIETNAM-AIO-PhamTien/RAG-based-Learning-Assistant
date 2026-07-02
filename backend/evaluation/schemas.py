from __future__ import annotations

from pydantic import BaseModel, Field


class EvalSample(BaseModel):
    question: str
    ground_truth_answer: str
    ground_truth_contexts: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    retrieved_contexts: list[str] = Field(default_factory=list)
    retrieved_scores: list[float] = Field(default_factory=list)
    latency_ms: float = 0.0
    requested_k: int | None = None
    effective_k: int | None = None


class GenerationResult(BaseModel):
    generated_answer: str = ""
    latency_ms: float = 0.0
    citations_used: list[int] = Field(default_factory=list)


class EvalResult(BaseModel):
    sample: EvalSample
    retrieval: RetrievalResult
    generation: GenerationResult
    metric_scores: dict[str, float] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    name: str
    dataset_name: str = "pdf_qa"
    dataset_path: str | None = None
    num_samples: int | None = None
    chunk_size: int = 1600
    chunk_overlap: int = 250
    top_k: int = 5
    rerank_enabled: bool = True
    rerank_candidate_k: int = 20
    embedding_model: str = "intfloat/e5-small-v2"
    gemini_model: str = "gemini-2.5-flash"
    compute_context_relevance: bool = True
    batch_size: int = 1


class RetrievalArtifact(BaseModel):
    """Output of retrieval-only stage."""

    config: ExperimentConfig
    samples: list[EvalSample] = Field(default_factory=list)
    retrieval_results: list[RetrievalResult] = Field(default_factory=list)
    retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    latency_metrics: dict[str, float] = Field(default_factory=dict)
    timestamp: str = ""
    duration_seconds: float = 0.0


class GenerationArtifact(BaseModel):
    """Output of generation stage."""

    config: ExperimentConfig
    samples: list[EvalSample] = Field(default_factory=list)
    retrieval_results: list[RetrievalResult] = Field(default_factory=list)
    generation_results: list[GenerationResult] = Field(default_factory=list)
    timestamp: str = ""
    duration_seconds: float = 0.0


class ExperimentReport(BaseModel):
    config: ExperimentConfig
    results: list[EvalResult] = Field(default_factory=list)
    aggregate_metrics: dict[str, float] = Field(default_factory=dict)
    retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    generation_metrics: dict[str, float] = Field(default_factory=dict)
    latency_metrics: dict[str, float] = Field(default_factory=dict)
    timestamp: str = ""
    duration_seconds: float = 0.0
    stages_completed: list[str] = Field(default_factory=list)
