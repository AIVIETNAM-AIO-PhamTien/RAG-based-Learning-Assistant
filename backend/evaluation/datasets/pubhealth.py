from __future__ import annotations

from evaluation.schemas import EvalSample


class PubHealthLoader:
    def load(
        self, num_samples: int | None = None, split: str = "validation"
    ) -> list[EvalSample]:
        from datasets import load_dataset

        ds = load_dataset("OpenMed/PubHealth-Processed", split=split, streaming=True)

        samples: list[EvalSample] = []

        for i, row in enumerate(ds):
            if num_samples is not None and i >= num_samples:
                break

            claim = row.get("claim", "")
            explanation = row.get("explanation", "")
            label = row.get("label")
            label_map = {0: "true", 1: "false", 2: "mixture", 3: "unproven"}
            label_str = label_map.get(label, "unknown")

            main_text = row.get("main_text", "")
            contexts = [main_text] if main_text else []

            question = f"Is the following health claim true or false? Claim: {claim}"
            ground_truth = f"{label_str}. {explanation}" if explanation else label_str

            samples.append(
                EvalSample(
                    question=question,
                    ground_truth_answer=ground_truth,
                    ground_truth_contexts=contexts,
                    metadata={
                        "source": "pubhealth",
                        "id": i,
                        "label": label_str,
                        "subjects": row.get("subjects", ""),
                    },
                )
            )
        return samples
