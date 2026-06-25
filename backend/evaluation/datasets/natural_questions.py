from __future__ import annotations

import re

from evaluation.schemas import EvalSample


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text).strip()


class NaturalQuestionsLoader:
    def load(
        self, num_samples: int | None = None, split: str = "validation"
    ) -> list[EvalSample]:
        from datasets import load_dataset

        ds = load_dataset(
            "google-research-datasets/natural_questions", split=split, streaming=True
        )

        samples: list[EvalSample] = []

        for i, row in enumerate(ds):
            if num_samples is not None and i >= num_samples:
                break

            try:
                question = row["question"]["text"]
                doc_tokens = row["document"]["tokens"]["token"]
            except (KeyError, TypeError):
                continue

            short_answers = []
            contexts = []

            for annotation in row.get("annotations", []):
                sa_list = annotation.get("short_answers", [])
                for sa in sa_list:
                    start = sa.get("start_token", -1)
                    end = sa.get("end_token", -1)
                    if start >= 0 and end > start:
                        answer_text = " ".join(doc_tokens[start:end])
                        if answer_text:
                            short_answers.append(answer_text)

                la = annotation.get("long_answer", {})
                la_start = la.get("start_token", -1)
                la_end = la.get("end_token", -1)
                if la_start >= 0 and la_end > la_start:
                    context = _strip_html(" ".join(doc_tokens[la_start:la_end]))
                    if context:
                        contexts.append(context)

            if not short_answers:
                continue

            samples.append(
                EvalSample(
                    question=question,
                    ground_truth_answer=short_answers[0],
                    ground_truth_contexts=contexts[:3],
                    metadata={
                        "source": "natural_questions",
                        "id": i,
                        "all_short_answers": short_answers,
                    },
                )
            )
        return samples
