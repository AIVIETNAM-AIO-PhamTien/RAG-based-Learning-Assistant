from __future__ import annotations

import json
from pathlib import Path

from evaluation.schemas import EvalSample


class PdfQALoader:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None

    def load(
        self, num_samples: int | None = None, split: str = "validation"
    ) -> list[EvalSample]:
        if self.path is None:
            available = self._list_available()
            hint = f" Available files:\n{available}" if available else ""
            raise ValueError(
                f"PdfQALoader requires --dataset-path to a JSON file or directory.{hint}"
            )

        path = self.path
        if path.is_dir():
            samples = self._load_directory(path)
        elif path.is_file():
            samples = self._load_file(path)
        else:
            available = self._list_available()
            hint = f"\nAvailable files:\n{available}" if available else ""
            raise FileNotFoundError(f"Path not found: {path}{hint}")

        if num_samples is not None:
            samples = samples[:num_samples]
        return samples

    def _load_file(self, path: Path) -> list[EvalSample]:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        samples: list[EvalSample] = []
        for i, item in enumerate(raw):
            samples.append(
                EvalSample(
                    question=item["question"],
                    ground_truth_answer=item["answer"],
                    ground_truth_contexts=item.get("contexts", []),
                    metadata={
                        "source": "pdf_qa",
                        "id": i,
                        "source_pdf": item.get("source_pdf", ""),
                        "page": item.get("page"),
                    },
                )
            )
        return samples

    def _load_directory(self, directory: Path) -> list[EvalSample]:
        samples: list[EvalSample] = []
        for json_file in sorted(directory.glob("*.json")):
            samples.extend(self._load_file(json_file))
        return samples

    @staticmethod
    def _list_available() -> str:
        data_dir = Path(__file__).resolve().parent.parent / "data"
        if not data_dir.is_dir():
            return ""
        files = sorted(data_dir.glob("*.json"))
        if not files:
            return ""
        return "\n".join(f"  - {f.relative_to(data_dir.parent.parent)}" for f in files)
