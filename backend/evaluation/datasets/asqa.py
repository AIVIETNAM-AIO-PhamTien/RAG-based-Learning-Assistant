from __future__ import annotations

from evaluation.datasets.base import columnar_to_rows
from evaluation.schemas import EvalSample

_NO_CONTEXT_PLACEHOLDER = "No context provided"


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
            long_answers = [a.get("long_answer", "") for a in annotations if a.get("long_answer")]
            long_answers = list(dict.fromkeys(long_answers))

            short_answers: list[str] = []
            contexts: list[str] = []

            for qa in columnar_to_rows(row.get("qa_pairs", [])):
                sa = qa.get("short_answers", [])
                if isinstance(sa, list):
                    short_answers.extend(sa)

                ctx = qa.get("context", "")
                if ctx and ctx != _NO_CONTEXT_PLACEHOLDER:
                    contexts.append(ctx)

            for annotation in annotations:
                for item in columnar_to_rows(annotation.get("knowledge", [])):
                    content = item.get("content", "")
                    if content:
                        contexts.append(content)

            short_answers = list(dict.fromkeys(short_answers))
            contexts = list(dict.fromkeys(contexts))
            all_answers = list(dict.fromkeys(long_answers + short_answers))

            ground_truth = max(long_answers, key=len) if long_answers else "; ".join(short_answers)

            samples.append(
                EvalSample(
                    question=question,
                    ground_truth_answer=ground_truth,
                    ground_truth_contexts=contexts,
                    metadata={
                        "source": "asqa",
                        "id": row.get("sample_id", i),
                        "all_short_answers": all_answers,
                    },
                )
            )
        return samples
