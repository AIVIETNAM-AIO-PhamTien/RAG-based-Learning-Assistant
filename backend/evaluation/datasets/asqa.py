from __future__ import annotations

from evaluation.schemas import EvalSample


class ASQALoader:
    def load(
        self, num_samples: int | None = None, split: str = "dev"
    ) -> list[EvalSample]:
        from datasets import load_dataset

        ds = load_dataset("din0s/asqa", split=split)

        samples: list[EvalSample] = []
        limit = num_samples or len(ds)

        for i, row in enumerate(ds):
            if i >= limit:
                break

            question = row["ambiguous_question"]
            long_answer = row.get("annotations", [{}])
            if isinstance(long_answer, list) and long_answer:
                long_answer_text = long_answer[0].get("long_answer", "")
            else:
                long_answer_text = ""

            short_answers: list[str] = []
            qa_pairs = row.get("qa_pairs", [])
            for qa in qa_pairs:
                sa = qa.get("short_answers", [])
                if isinstance(sa, list):
                    short_answers.extend(sa)

            contexts: list[str] = []
            for qa in qa_pairs:
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
