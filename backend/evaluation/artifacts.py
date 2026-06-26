from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from pydantic import BaseModel

from evaluation.schemas import (
    ExperimentConfig,
    GenerationArtifact,
    RetrievalArtifact,
)

logger = logging.getLogger(__name__)

_RETRIEVAL_RELEVANT_FIELDS = (
    "chunk_size",
    "chunk_overlap",
    "top_k",
    "rerank_enabled",
    "rerank_candidate_k",
    "embedding_model",
)


def get_experiment_dir(output_dir: Path, experiment_name: str) -> Path:
    """Return per-experiment directory, creating it if needed."""
    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_artifact(artifact: BaseModel, path: Path) -> None:
    """Atomic write: write to temp file then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(artifact.model_dump(), f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def load_retrieval_artifact(path: Path) -> RetrievalArtifact:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return RetrievalArtifact(**data)


def load_generation_artifact(path: Path) -> GenerationArtifact:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return GenerationArtifact(**data)


def resolve_retrieval_artifact(from_path: str) -> Path:
    """Resolve --from argument to a retrieval.json path.

    Accepts either a direct file path or an experiment directory.
    """
    p = Path(from_path)
    if p.is_file():
        return p
    candidate = p / "retrieval.json"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Cannot find retrieval artifact at '{from_path}'. "
        "Provide a path to retrieval.json or an experiment directory."
    )


def validate_config_compatibility(
    retrieval_config: ExperimentConfig,
    generation_config: ExperimentConfig,
) -> list[str]:
    """Compare retrieval-relevant fields between configs.

    Returns warning messages for mismatched fields.
    gemini_model is intentionally allowed to differ.
    """
    _MISSING = object()
    warnings: list[str] = []
    for field in _RETRIEVAL_RELEVANT_FIELDS:
        ret_val = getattr(retrieval_config, field, _MISSING)
        gen_val = getattr(generation_config, field, _MISSING)
        if ret_val is _MISSING or gen_val is _MISSING:
            warnings.append(f"Config field '{field}' not found on one or both configs")
            continue
        if ret_val != gen_val:
            warnings.append(
                f"Config mismatch: {field} = {ret_val} (retrieval) vs {gen_val} (generation)"
            )
    return warnings
