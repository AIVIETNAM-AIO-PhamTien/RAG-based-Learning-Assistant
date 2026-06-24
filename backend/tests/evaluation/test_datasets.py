import json
from pathlib import Path

import pytest

from evaluation.datasets.base import get_dataset_loader
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
