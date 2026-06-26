from __future__ import annotations

import json

from evaluation.schemas import EvalSample


class PopQALoader:
    def load(
        self, num_samples: int | None = None, split: str = "test"
    ) -> list[EvalSample]:
        from datasets import load_dataset

        ds = load_dataset("akariasai/PopQA", split=split, streaming=True)

        samples: list[EvalSample] = []

        for i, row in enumerate(ds):
            if num_samples is not None and i >= num_samples:
                break

            question = row["question"]
            answer = row["possible_answers"]
            if isinstance(answer, str):
                try:
                    parsed = json.loads(answer)
                    answer = parsed[0] if parsed else answer
                except (json.JSONDecodeError, IndexError):
                    pass
            elif isinstance(answer, list):
                answer = answer[0] if answer else ""

            subj = row.get("subj", "")
            prop = row.get("prop", "")
            obj = row.get("obj", "")
            context = f"{subj} {prop} {obj}".strip()

            samples.append(
                EvalSample(
                    question=question,
                    ground_truth_answer=str(answer),
                    ground_truth_contexts=[context] if context else [],
                    metadata={
                        "source": "popqa",
                        "id": i,
                        "s_pop": row.get("s_pop"),
                        "o_pop": row.get("o_pop"),
                    },
                )
            )
        return samples
