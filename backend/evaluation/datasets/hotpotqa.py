from __future__ import annotations

from evaluation.schemas import EvalSample


class HotpotQALoader:
    def load(
        self, num_samples: int | None = None, split: str = "validation"
    ) -> list[EvalSample]:
        from datasets import load_dataset

        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True)

        samples: list[EvalSample] = []

        for i, row in enumerate(ds):
            if num_samples is not None and i >= num_samples:
                break

            titles = row["context"]["title"]
            sentences = row["context"]["sentences"]
            supporting_titles = set(row["supporting_facts"]["title"])

            gold_contexts: list[str] = []
            distractor_contexts: list[str] = []

            for title, sents in zip(titles, sentences, strict=False):
                paragraph = " ".join(sents)
                if title in supporting_titles:
                    gold_contexts.append(paragraph)
                else:
                    distractor_contexts.append(paragraph)

            samples.append(
                EvalSample(
                    question=row["question"],
                    ground_truth_answer=row["answer"],
                    ground_truth_contexts=gold_contexts,
                    metadata={
                        "source": "hotpotqa",
                        "id": row["id"],
                        "type": row["type"],
                        "level": row["level"],
                        "distractor_contexts": distractor_contexts,
                    },
                )
            )
        return samples
