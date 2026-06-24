from __future__ import annotations

from evaluation.schemas import EvalSample


class PopQALoader:
    def load(
        self, num_samples: int | None = None, split: str = "test"
    ) -> list[EvalSample]:
        from datasets import load_dataset

        ds = load_dataset("akariasai/PopQA", split=split)

        samples: list[EvalSample] = []
        limit = num_samples or len(ds)

        for i, row in enumerate(ds):
            if i >= limit:
                break

            question = row["question"]
            answer = row["possible_answers"]
            if isinstance(answer, list):
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
