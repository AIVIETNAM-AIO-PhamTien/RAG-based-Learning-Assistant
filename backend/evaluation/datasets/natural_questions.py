from __future__ import annotations

from evaluation.datasets.base import columnar_to_rows
from evaluation.schemas import EvalSample

MAX_DOC_CHARS = 50_000


def _span_text(tokens: list[str], is_html_flags: list[bool], start: int, end: int) -> str:
    return " ".join(
        tok for tok, is_html in zip(tokens[start:end], is_html_flags[start:end]) if not is_html
    ).strip()


class NaturalQuestionsLoader:
    def load(
        self, num_samples: int | None = None, split: str = "validation"
    ) -> list[EvalSample]:
        from datasets import load_dataset

        ds = load_dataset(
            "google-research-datasets/natural_questions", split=split, streaming=True
        )

        samples: list[EvalSample] = []

        for row in ds:
            if num_samples is not None and len(samples) >= num_samples:
                break

            try:
                question = row["question"]["text"]
                tokens_info = row.get("document", {}).get("tokens", {})
                doc_tokens = tokens_info["token"]
            except (KeyError, TypeError):
                continue

            is_html_flags = tokens_info.get("is_html", [False] * len(doc_tokens))

            short_answers = []
            contexts = []

            for annotation in columnar_to_rows(row.get("annotations", [])):
                for sa in columnar_to_rows(annotation.get("short_answers", [])):
                    start = sa.get("start_token", -1)
                    end = sa.get("end_token", -1)
                    if start >= 0 and end > start:
                        answer_text = _span_text(doc_tokens, is_html_flags, start, end)
                        if answer_text:
                            short_answers.append(answer_text)

                la = annotation.get("long_answer", {})
                la_start = la.get("start_token", -1)
                la_end = la.get("end_token", -1)
                if la_start >= 0 and la_end > la_start:
                    context = _span_text(doc_tokens, is_html_flags, la_start, la_end)
                    if context:
                        contexts.append(context)

            short_answers = list(dict.fromkeys(short_answers))
            contexts = list(dict.fromkeys(contexts))

            if not short_answers:
                continue

            # Use the full Wikipedia article (HTML tokens stripped) as the distractor
            # pool, same as a real PDF ingested by the app: the chunker slices it into
            # many chunks and the correct passage must be found among them, instead of
            # retrieving against a corpus made only of the correct passage.
            full_doc_text = _span_text(doc_tokens, is_html_flags, 0, len(doc_tokens))[
                :MAX_DOC_CHARS
            ]

            samples.append(
                EvalSample(
                    question=question,
                    ground_truth_answer=max(short_answers, key=len),
                    ground_truth_contexts=contexts[:3],
                    metadata={
                        "source": "natural_questions",
                        "id": row.get("id"),
                        "all_short_answers": short_answers,
                        "distractor_contexts": [full_doc_text] if full_doc_text else [],
                    },
                )
            )
        return samples
