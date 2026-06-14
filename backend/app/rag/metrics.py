import re

CITATION_PATTERN = re.compile(r"\[(\d+)]")


def citation_coverage(answer: str, available_indexes: set[int]) -> float:
    used = {int(match) for match in CITATION_PATTERN.findall(answer)}
    if not used:
        return 0.0
    valid = used & available_indexes
    return len(valid) / len(used)


def hit_at_5(retrieved_chunk_ids: list[str], expected_chunk_id: str | None) -> bool:
    return expected_chunk_id is not None and expected_chunk_id in retrieved_chunk_ids[:5]
