from __future__ import annotations

from typing import Protocol

from evaluation.schemas import EvalSample


class DatasetLoader(Protocol):
    def load(
        self, num_samples: int | None = None, split: str = "validation"
    ) -> list[EvalSample]: ...


def columnar_to_rows(data) -> list[dict]:
    """Convert HF columnar format {k: [v1, v2]} to [{k: v1}, {k: v2}]."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        keys = list(data.keys())
        if not keys:
            return []
        n = len(data[keys[0]])
        return [{k: data[k][i] for k in keys} for i in range(n)]
    return []


def get_dataset_loader(name: str, **kwargs) -> DatasetLoader:
    loaders = {
        "pdf_qa": _load_pdf_qa,
        "hotpotqa": _load_hotpotqa,
        "asqa": _load_asqa,
        "nq": _load_nq,
    }
    factory = loaders.get(name)
    if factory is None:
        raise ValueError(f"Unknown dataset: {name}. Available: {sorted(loaders)}")
    return factory(**kwargs)


def _load_pdf_qa(**kwargs) -> DatasetLoader:
    from evaluation.datasets.pdf_qa import PdfQALoader

    return PdfQALoader(**kwargs)


def _load_hotpotqa(**kwargs) -> DatasetLoader:
    from evaluation.datasets.hotpotqa import HotpotQALoader

    return HotpotQALoader(**kwargs)


def _load_asqa(**kwargs) -> DatasetLoader:
    from evaluation.datasets.asqa import ASQALoader

    return ASQALoader(**kwargs)


def _load_nq(**kwargs) -> DatasetLoader:
    from evaluation.datasets.natural_questions import NaturalQuestionsLoader

    return NaturalQuestionsLoader(**kwargs)
