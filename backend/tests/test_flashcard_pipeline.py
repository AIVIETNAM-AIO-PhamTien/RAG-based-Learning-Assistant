import asyncio
import uuid

import pytest

from app.api.v1 import study
from app.schemas.chat import Citation
from app.schemas.study import Flashcard


def build_citation(
    *,
    index: int,
    doc_id: uuid.UUID,
    doc_name: str,
    page: int,
    text: str,
) -> Citation:
    return Citation(
        index=index,
        chunk_id=uuid.uuid4(),
        doc_id=doc_id,
        doc_name=doc_name,
        page=page,
        text=text,
        snippet=text,
    )


def build_document_notes(
    *, doc_id: uuid.UUID, doc_name: str, note_text: str, sources: list[Citation]
) -> study.DocumentNotes:
    return study.DocumentNotes(doc_id=doc_id, doc_name=doc_name, notes=note_text, sources=sources)


def test_split_into_batches_preserves_order() -> None:
    doc_id = uuid.uuid4()
    sources = [
        build_citation(index=1, doc_id=doc_id, doc_name="alpha.pdf", page=1, text="first"),
        build_citation(index=2, doc_id=doc_id, doc_name="alpha.pdf", page=2, text="second"),
        build_citation(index=3, doc_id=doc_id, doc_name="alpha.pdf", page=3, text="third"),
        build_citation(index=4, doc_id=doc_id, doc_name="alpha.pdf", page=4, text="fourth"),
        build_citation(index=5, doc_id=doc_id, doc_name="alpha.pdf", page=5, text="fifth"),
    ]

    batches = study._split_into_batches(sources, 2)

    assert [[item.index for item in batch] for batch in batches] == [[1, 2], [3, 4], [5]]


def test_allocate_flashcard_targets_handles_smaller_equal_and_larger_counts() -> None:
    doc_one_id = uuid.uuid4()
    doc_two_id = uuid.uuid4()
    doc_three_id = uuid.uuid4()
    document_notes = [
        build_document_notes(
            doc_id=doc_one_id,
            doc_name="alpha.pdf",
            note_text="notes",
            sources=[build_citation(index=1, doc_id=doc_one_id, doc_name="alpha.pdf", page=1, text="a")],
        ),
        build_document_notes(
            doc_id=doc_two_id,
            doc_name="beta.pdf",
            note_text="notes",
            sources=[build_citation(index=2, doc_id=doc_two_id, doc_name="beta.pdf", page=2, text="b")],
        ),
        build_document_notes(
            doc_id=doc_three_id,
            doc_name="gamma.pdf",
            note_text="notes",
            sources=[build_citation(index=3, doc_id=doc_three_id, doc_name="gamma.pdf", page=3, text="c")],
        ),
    ]

    assert study._allocate_flashcard_targets(document_notes, 2) == {
        doc_one_id: 1,
        doc_two_id: 1,
        doc_three_id: 0,
    }
    assert study._allocate_flashcard_targets(document_notes, 3) == {
        doc_one_id: 1,
        doc_two_id: 1,
        doc_three_id: 1,
    }
    assert study._allocate_flashcard_targets(document_notes, 5) == {
        doc_one_id: 2,
        doc_two_id: 2,
        doc_three_id: 1,
    }


def test_dedupe_flashcards_normalizes_whitespace_and_case() -> None:
    cards = [
        Flashcard(question="What is TCP?", answer="A transport protocol."),
        Flashcard(question="  what is tcp?  ", answer="A   transport protocol."),
        Flashcard(question="What is UDP?", answer="A connectionless protocol."),
    ]

    deduped = study._dedupe_flashcards(cards)

    assert [(card.question, card.answer) for card in deduped] == [
        ("What is TCP?", "A transport protocol."),
        ("What is UDP?", "A connectionless protocol."),
    ]


def test_fill_flashcards_returns_exact_count_with_unique_fallbacks() -> None:
    original_cards = [Flashcard(question="What is TCP?", answer="A transport protocol.")]
    fallback_cards = [
        Flashcard(question="What is in alpha?", answer="Alpha detail"),
        Flashcard(question="What is in beta?", answer="Beta detail"),
        Flashcard(question="What is in gamma?", answer="Gamma detail"),
    ]

    filled = study._fill_flashcards(original_cards, fallback_cards, 3)

    assert len(filled) == 3
    assert [(card.question, card.answer) for card in filled] == [
        ("What is TCP?", "A transport protocol."),
        ("What is in alpha?", "Alpha detail"),
        ("What is in beta?", "Beta detail"),
    ]


def test_build_coverage_hint_uses_document_order() -> None:
    doc_one_id = uuid.uuid4()
    doc_two_id = uuid.uuid4()
    document_notes = [
        build_document_notes(
            doc_id=doc_one_id,
            doc_name="alpha.pdf",
            note_text="alpha notes",
            sources=[build_citation(index=1, doc_id=doc_one_id, doc_name="alpha.pdf", page=1, text="a")],
        ),
        build_document_notes(
            doc_id=doc_two_id,
            doc_name="beta.pdf",
            note_text="beta notes",
            sources=[build_citation(index=2, doc_id=doc_two_id, doc_name="beta.pdf", page=2, text="b")],
        ),
    ]

    hint = study._build_coverage_hint({doc_one_id: 2, doc_two_id: 1}, document_notes)

    assert hint == "- alpha.pdf: target 2 card(s)\n- beta.pdf: target 1 card(s)"


def test_parse_flashcards_preserves_multiline_answers() -> None:
    generated = "Q: Explain TCP\nA: First line\nSecond line\n- bullet\nQ: Next question\nA: Next answer"

    cards = study._parse_flashcards(generated)

    assert [(card.question, card.answer) for card in cards] == [
        ("Explain TCP", "First line\nSecond line\n- bullet"),
        ("Next question", "Next answer"),
    ]


def test_compact_sources_for_response_replaces_full_text_with_snippet() -> None:
    source = build_citation(
        index=1,
        doc_id=uuid.uuid4(),
        doc_name="alpha.pdf",
        page=1,
        text="Long full text that should not be returned intact",
    )
    object.__setattr__(source, "snippet", "Short snippet")

    compact_sources = study._compact_sources_for_response([source])

    assert compact_sources[0].text == "Short snippet"
    assert compact_sources[0].snippet == "Short snippet"


def test_build_notes_context_truncates_long_document_notes() -> None:
    doc_id = uuid.uuid4()
    document_notes = [
        build_document_notes(
            doc_id=doc_id,
            doc_name="alpha.pdf",
            note_text="x" * 5000,
            sources=[build_citation(index=1, doc_id=doc_id, doc_name="alpha.pdf", page=1, text="a")],
        )
    ]

    notes_context = study._build_notes_context(document_notes)

    assert len(notes_context) < 3000
    assert notes_context.endswith("…")


@pytest.mark.asyncio
async def test_generate_batch_notes_falls_back_to_snippets_on_runtime_error() -> None:
    doc_id = uuid.uuid4()
    batch = [
        build_citation(index=1, doc_id=doc_id, doc_name="alpha.pdf", page=1, text="First snippet"),
        build_citation(index=2, doc_id=doc_id, doc_name="alpha.pdf", page=2, text="Second snippet"),
    ]
    semaphore = asyncio.Semaphore(1)

    original_generator = study.generate_flashcard_notes

    async def fake_generate_flashcard_notes(_batch: list[Citation]) -> str:
        raise RuntimeError("provider error")

    study.generate_flashcard_notes = fake_generate_flashcard_notes
    try:
        note_text = await study._generate_batch_notes(batch, semaphore)
    finally:
        study.generate_flashcard_notes = original_generator

    assert note_text == "- First snippet\n- Second snippet"
