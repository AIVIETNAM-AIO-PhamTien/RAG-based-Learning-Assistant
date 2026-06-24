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


class ExperimentReport(BaseModel):
    config: ExperimentConfig
    results: list[EvalResult] = Field(default_factory=list)
    aggregate_metrics: dict[str, float] = Field(default_factory=dict)
    timestamp: str = ""
    duration_seconds: float = 0.0
