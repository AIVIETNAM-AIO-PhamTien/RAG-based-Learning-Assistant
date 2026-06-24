from __future__ import annotations

from typing import Protocol

from evaluation.schemas import EvalSample


class DatasetLoader(Protocol):
    def load(
        self, num_samples: int | None = None, split: str = "validation"
    ) -> list[EvalSample]: ...


def get_dataset_loader(name: str, **kwargs) -> DatasetLoader:
    loaders = {
        "pdf_qa": _load_pdf_qa,
        "hotpotqa": _load_hotpotqa,
        "popqa": _load_popqa,
        "asqa": _load_asqa,
        "pubhealth": _load_pubhealth,
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


def _load_popqa(**kwargs) -> DatasetLoader:
    from evaluation.datasets.popqa import PopQALoader

    return PopQALoader(**kwargs)


def _load_asqa(**kwargs) -> DatasetLoader:
    from evaluation.datasets.asqa import ASQALoader

    return ASQALoader(**kwargs)


def _load_pubhealth(**kwargs) -> DatasetLoader:
    from evaluation.datasets.pubhealth import PubHealthLoader

    return PubHealthLoader(**kwargs)


def _load_nq(**kwargs) -> DatasetLoader:
    from evaluation.datasets.natural_questions import NaturalQuestionsLoader

    return NaturalQuestionsLoader(**kwargs)
