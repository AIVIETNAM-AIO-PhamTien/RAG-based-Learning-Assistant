from pathlib import Path

import pytest

from evaluation.artifacts import (
    get_experiment_dir,
    load_retrieval_artifact,
    resolve_retrieval_artifact,
    save_artifact,
    validate_config_compatibility,
)
from evaluation.schemas import (
    EvalSample,
    ExperimentConfig,
    RetrievalArtifact,
    RetrievalResult,
)


def _make_retrieval_artifact() -> RetrievalArtifact:
    config = ExperimentConfig(name="test_exp")
    sample = EvalSample(question="What is RAG?", ground_truth_answer="RAG answer")
    retrieval = RetrievalResult(
        retrieved_contexts=["context 1"],
        retrieved_scores=[0.95],
        latency_ms=12.5,
    )
    return RetrievalArtifact(
        config=config,
        samples=[sample],
        retrieval_results=[retrieval],
        retrieval_metrics={"recall": 1.0, "rr": 1.0},
        timestamp="2026-06-25T00:00:00Z",
        duration_seconds=5.0,
    )


def test_save_and_load_roundtrip(tmp_path: Path):
    artifact = _make_retrieval_artifact()
    path = tmp_path / "retrieval.json"
    save_artifact(artifact, path)

    loaded = load_retrieval_artifact(path)
    assert loaded.config.name == "test_exp"
    assert len(loaded.samples) == 1
    assert loaded.samples[0].question == "What is RAG?"
    assert len(loaded.retrieval_results) == 1
    assert loaded.retrieval_results[0].retrieved_scores == [0.95]
    assert loaded.retrieval_metrics["recall"] == 1.0


def test_save_artifact_creates_parent_dirs(tmp_path: Path):
    artifact = _make_retrieval_artifact()
    path = tmp_path / "deep" / "nested" / "retrieval.json"
    save_artifact(artifact, path)
    assert path.exists()


def test_save_artifact_atomic_no_tmp_left(tmp_path: Path):
    artifact = _make_retrieval_artifact()
    path = tmp_path / "retrieval.json"
    save_artifact(artifact, path)
    assert not path.with_suffix(".tmp").exists()


def test_get_experiment_dir(tmp_path: Path):
    exp_dir = get_experiment_dir(tmp_path, "my_experiment")
    assert exp_dir == tmp_path / "my_experiment"
    assert exp_dir.is_dir()


def test_resolve_retrieval_artifact_file(tmp_path: Path):
    artifact = _make_retrieval_artifact()
    path = tmp_path / "retrieval.json"
    save_artifact(artifact, path)
    assert resolve_retrieval_artifact(str(path)) == path


def test_resolve_retrieval_artifact_directory(tmp_path: Path):
    artifact = _make_retrieval_artifact()
    exp_dir = tmp_path / "exp1"
    exp_dir.mkdir()
    path = exp_dir / "retrieval.json"
    save_artifact(artifact, path)
    assert resolve_retrieval_artifact(str(exp_dir)) == path


def test_resolve_retrieval_artifact_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_retrieval_artifact(str(tmp_path / "nonexistent"))


def test_validate_config_compatibility_matching():
    c1 = ExperimentConfig(name="a", chunk_size=1600, top_k=5)
    c2 = ExperimentConfig(name="b", chunk_size=1600, top_k=5, gemini_model="gemini-2.0-flash")
    warnings = validate_config_compatibility(c1, c2)
    assert warnings == []


def test_validate_config_compatibility_mismatch():
    c1 = ExperimentConfig(name="a", chunk_size=1600, top_k=5)
    c2 = ExperimentConfig(name="b", chunk_size=800, top_k=10)
    warnings = validate_config_compatibility(c1, c2)
    assert len(warnings) == 2
    assert any("chunk_size" in w for w in warnings)
    assert any("top_k" in w for w in warnings)


def test_validate_config_compatibility_gemini_model_ignored():
    c1 = ExperimentConfig(name="a", gemini_model="gemini-2.5-flash")
    c2 = ExperimentConfig(name="b", gemini_model="gemini-2.0-flash")
    warnings = validate_config_compatibility(c1, c2)
    assert warnings == []
