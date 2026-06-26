import json
from pathlib import Path
from unittest.mock import patch

import pytest

from evaluation.datasets.base import columnar_to_rows, get_dataset_loader
from evaluation.datasets.pdf_qa import PdfQALoader
from evaluation.schemas import EvalSample


@pytest.fixture()
def sample_qa_json(tmp_path: Path) -> Path:
    data = [
        {
            "question": "What is machine learning?",
            "answer": "A subset of AI that learns from data.",
            "contexts": ["Machine learning is a subset of artificial intelligence."],
            "source_pdf": "ml_textbook.pdf",
            "page": 1,
        },
        {
            "question": "What is deep learning?",
            "answer": "Neural networks with many layers.",
            "contexts": ["Deep learning uses neural networks with multiple layers."],
            "source_pdf": "ml_textbook.pdf",
            "page": 5,
        },
        {
            "question": "What is NLP?",
            "answer": "Natural Language Processing.",
            "contexts": ["NLP is the study of computational linguistics."],
            "source_pdf": "nlp_book.pdf",
            "page": 2,
        },
    ]
    path = tmp_path / "test_qa.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_pdf_qa_loader_file(sample_qa_json: Path):
    loader = PdfQALoader(path=sample_qa_json)
    samples = loader.load()
    assert len(samples) == 3
    assert samples[0].question == "What is machine learning?"
    assert samples[0].ground_truth_contexts[0].startswith("Machine learning")
    assert samples[0].metadata["source_pdf"] == "ml_textbook.pdf"


def test_pdf_qa_loader_num_samples(sample_qa_json: Path):
    loader = PdfQALoader(path=sample_qa_json)
    samples = loader.load(num_samples=2)
    assert len(samples) == 2


def test_pdf_qa_loader_directory(tmp_path: Path):
    for i in range(2):
        data = [{"question": f"Q{i}?", "answer": f"A{i}", "contexts": [f"ctx{i}"]}]
        (tmp_path / f"file{i}.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

    loader = PdfQALoader(path=tmp_path)
    samples = loader.load()
    assert len(samples) == 2


def test_pdf_qa_loader_no_path():
    loader = PdfQALoader()
    with pytest.raises(ValueError, match="requires --dataset-path"):
        loader.load()


def test_pdf_qa_loader_missing_path_shows_hint():
    loader = PdfQALoader(path="/nonexistent/path.json")
    with pytest.raises(FileNotFoundError, match="Path not found"):
        loader.load()


def test_get_dataset_loader_pdf_qa(sample_qa_json: Path):
    loader = get_dataset_loader("pdf_qa", path=sample_qa_json)
    samples = loader.load(num_samples=1)
    assert len(samples) == 1
    assert isinstance(samples[0], EvalSample)


def test_get_dataset_loader_unknown():
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset_loader("nonexistent_dataset")


# ── columnar_to_rows helper ──────────────────────────────────────────


def test_columnar_to_rows_list_passthrough():
    data = [{"a": 1}, {"a": 2}]
    assert columnar_to_rows(data) == data


def test_columnar_to_rows_dict_conversion():
    data = {"title": ["T1", "T2"], "score": [0.9, 0.8]}
    result = columnar_to_rows(data)
    assert result == [{"title": "T1", "score": 0.9}, {"title": "T2", "score": 0.8}]


def test_columnar_to_rows_empty_dict():
    assert columnar_to_rows({}) == []


def test_columnar_to_rows_none():
    assert columnar_to_rows(None) == []


# ── NQ loader with columnar format ───────────────────────────────────


def _mock_nq_rows():
    """Simulate HF streaming rows in columnar format."""
    return [
        {
            "question": {"text": "What is Python?"},
            "document": {
                "tokens": {"token": ["Python", "is", "a", "programming", "language", "."]}
            },
            "annotations": {
                "id": ["ann1"],
                "short_answers": [[{"start_token": 0, "end_token": 1}]],
                "long_answer": [{"start_token": 0, "end_token": 6}],
                "yes_no_answer": [-1],
            },
        },
        {
            "question": {"text": "Who is Einstein?"},
            "document": {"tokens": {"token": ["Albert", "Einstein", "was", "a", "physicist"]}},
            "annotations": {
                "id": ["ann2"],
                "short_answers": [[]],
                "long_answer": [{"start_token": -1, "end_token": -1}],
                "yes_no_answer": [-1],
            },
        },
        {
            "question": {"text": "What is Java?"},
            "document": {"tokens": {"token": ["Java", "is", "a", "language", "."]}},
            "annotations": {
                "id": ["ann3"],
                "short_answers": [[{"start_token": 0, "end_token": 1}]],
                "long_answer": [{"start_token": 0, "end_token": 5}],
                "yes_no_answer": [-1],
            },
        },
    ]


def test_nq_loader_columnar_format():
    with patch(
        "datasets.load_dataset", return_value=_mock_nq_rows()
    ):
        from evaluation.datasets.natural_questions import NaturalQuestionsLoader

        loader = NaturalQuestionsLoader()
        samples = loader.load(num_samples=5)

    assert len(samples) == 2
    assert samples[0].question == "What is Python?"
    assert samples[0].ground_truth_answer == "Python"
    assert samples[1].question == "What is Java?"


def test_nq_loader_num_samples_counts_collected():
    """num_samples should count collected samples, not rows iterated."""
    with patch(
        "datasets.load_dataset", return_value=_mock_nq_rows()
    ):
        from evaluation.datasets.natural_questions import NaturalQuestionsLoader

        loader = NaturalQuestionsLoader()
        samples = loader.load(num_samples=1)

    assert len(samples) == 1


# ── ASQA loader with columnar format ─────────────────────────────────


def _mock_asqa_rows():
    return [
        {
            "ambiguous_question": "When did the US join WW2?",
            "qa_pairs": {
                "context": ["After Pearl Harbor...", "The US declared war..."],
                "question": ["When did the US enter?", "When was war declared?"],
                "short_answers": [["December 1941"], ["December 8, 1941"]],
                "wikipage": [None, None],
            },
            "annotations": {
                "knowledge": [[{"content": "...", "wikipage": "WW2"}]],
                "long_answer": ["The US joined WW2 in December 1941 after Pearl Harbor."],
            },
            "sample_id": "asqa_001",
        },
    ]


def test_asqa_loader_columnar_format():
    with patch("datasets.load_dataset", return_value=_mock_asqa_rows()):
        from evaluation.datasets.asqa import ASQALoader

        loader = ASQALoader()
        samples = loader.load()

    assert len(samples) == 1
    assert samples[0].question == "When did the US join WW2?"
    assert "December 1941" in samples[0].ground_truth_answer
    assert len(samples[0].ground_truth_contexts) == 2


# ── PopQA loader with JSON string answers ────────────────────────────


def _mock_popqa_rows():
    return [
        {
            "question": "What is the capital of France?",
            "possible_answers": '["Paris", "paris"]',
            "subj": "France",
            "prop": "capital",
            "obj": "Paris",
            "s_pop": 1000,
            "o_pop": 500,
        },
    ]


def test_popqa_loader_json_string_answers():
    with patch("datasets.load_dataset", return_value=_mock_popqa_rows()):
        from evaluation.datasets.popqa import PopQALoader

        loader = PopQALoader()
        samples = loader.load()

    assert len(samples) == 1
    assert samples[0].ground_truth_answer == "Paris"


# ── PubHealth loader with string label ───────────────────────────────


def _mock_pubhealth_rows():
    return [
        {
            "claim": "Vitamin C cures cancer",
            "explanation": "No scientific evidence supports this claim.",
            "label": "false",
            "main_text": "Studies show no significant effect...",
            "subjects": "health",
        },
    ]


def test_pubhealth_loader_string_label():
    with patch(
        "datasets.load_dataset", return_value=_mock_pubhealth_rows()
    ):
        from evaluation.datasets.pubhealth import PubHealthLoader

        loader = PubHealthLoader()
        samples = loader.load()

    assert len(samples) == 1
    assert "false" in samples[0].ground_truth_answer
    assert "unknown" not in samples[0].ground_truth_answer
