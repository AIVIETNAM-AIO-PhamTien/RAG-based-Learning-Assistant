from __future__ import annotations

from evaluation.datasets.base import columnar_to_rows
from evaluation.schemas import EvalSample


class ASQALoader:
    def load(
        self, num_samples: int | None = None, split: str = "dev"
    ) -> list[EvalSample]:
        from datasets import load_dataset

        ds = load_dataset("din0s/asqa", split=split, streaming=True)

        samples: list[EvalSample] = []

        for i, row in enumerate(ds):
            if num_samples is not None and i >= num_samples:
                break

            question = row["ambiguous_question"]

            annotations = columnar_to_rows(row.get("annotations", []))
            if annotations:
                long_answer_text = annotations[0].get("long_answer", "")
            else:
                long_answer_text = ""

            short_answers: list[str] = []
            contexts: list[str] = []

            for qa in columnar_to_rows(row.get("qa_pairs", [])):
                sa = qa.get("short_answers", [])
                if isinstance(sa, list):
                    short_answers.extend(sa)

                ctx = qa.get("context", "")
                if ctx:
                    contexts.append(ctx)

            ground_truth = long_answer_text or "; ".join(short_answers)

            samples.append(
                EvalSample(
                    question=question,
                    ground_truth_answer=ground_truth,
                    ground_truth_contexts=contexts,
                    metadata={
                        "source": "asqa",
                        "id": i,
                        "short_answers": short_answers,
                    },
                )
            )
        return samples
