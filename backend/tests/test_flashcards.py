import json
import uuid

import pytest

from app.rag.flashcards import FlashcardGenerationError, parse_flashcards
from app.schemas.chat import Citation


def _citation(index: int = 1) -> Citation:
    return Citation(
        index=index,
        chunk_id=uuid.uuid4(),
        doc_id=uuid.uuid4(),
        doc_name="physics.pdf",
        page=4,
        text="Newton's second law describes force and acceleration.",
        snippet="Newton's second law",
    )


def test_parse_flashcards_attaches_citation_source() -> None:
    payload = json.dumps(
        {"flashcards": [{"question": "F = ma là gì?", "answer": "Định luật II Newton", "source_index": 1}]}
    )

    cards = parse_flashcards(payload, [_citation()], 1)

    assert cards[0]["question"] == "F = ma là gì?"
    assert cards[0]["source"].page == 4


def test_parse_flashcards_rejects_unknown_source() -> None:
    payload = json.dumps(
        {"flashcards": [{"question": "Q", "answer": "A", "source_index": 2}]}
    )

    with pytest.raises(FlashcardGenerationError, match="unavailable source"):
        parse_flashcards(payload, [_citation()], 1)


def test_parse_flashcards_requires_requested_count() -> None:
    with pytest.raises(FlashcardGenerationError, match="exactly 5"):
        parse_flashcards('{"flashcards": []}', [_citation()], 5)
