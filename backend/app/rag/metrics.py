import re

# Tolerates both the expected [1][2] format and a model occasionally combining
# citations into one bracket like [1, 2] — the prompt asks for the former, but
# LLM output isn't guaranteed to comply.
CITATION_PATTERN = re.compile(r"\[([\d,\s]+)]")


def citation_coverage(answer: str, available_indexes: set[int]) -> float:
    used: set[int] = set()
    for group in CITATION_PATTERN.findall(answer):
        used.update(int(n) for n in re.findall(r"\d+", group))
    if not used:
        return 0.0
    valid = used & available_indexes
    return len(valid) / len(used)


def hit_at_5(retrieved_chunk_ids: list[str], expected_chunk_id: str | None) -> bool:
    return expected_chunk_id is not None and expected_chunk_id in retrieved_chunk_ids[:5]
