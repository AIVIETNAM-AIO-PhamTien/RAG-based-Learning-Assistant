import uuid

import pytest
from fastapi import HTTPException

from app.api.v1 import study
from app.db.models import ChatSession
from app.schemas.chat import Citation
from app.schemas.study import StudyRequest


class FakeSession:
    def __init__(self, chat_session: ChatSession | None) -> None:
        self._chat_session = chat_session

    async def get(self, model, item_id):
        if model is ChatSession and self._chat_session and self._chat_session.id == item_id:
            return self._chat_session
        return None


def build_citation(
    index: int = 1,
    text: str = "Important study detail",
    doc_name: str = "notes.pdf",
    page: int | None = None,
) -> Citation:
    resolved_page = page or index
    return Citation(
        index=index,
        chunk_id=uuid.uuid4(),
        doc_id=uuid.uuid4(),
        doc_name=doc_name,
        page=resolved_page,
        text=text,
        snippet=text,
    )


@pytest.mark.asyncio
async def test_summary_raises_for_missing_session() -> None:
    with pytest.raises(HTTPException) as caught:
        await study.summary(uuid.uuid4(), StudyRequest(topic="TCP"), FakeSession(None))

    assert caught.value.status_code == 404
    assert caught.value.detail == "Chat session not found"


@pytest.mark.asyncio
async def test_summary_uses_topic_and_returns_generated_text(monkeypatch) -> None:
    session_id = uuid.uuid4()
    chat_session = ChatSession(id=session_id)
    fake_session = FakeSession(chat_session)
    captured: dict[str, object] = {}
    sources = [build_citation(text="Chunk about congestion control")]

    async def fake_retrieve(session, chat_session_id, topic=None, top_k=None):
        captured["session"] = session
        captured["chat_session_id"] = chat_session_id
        captured["topic"] = topic
        captured["top_k"] = top_k
        return sources

    async def fake_generate_summary(received_sources):
        captured["sources"] = received_sources
        return "Short study summary"

    monkeypatch.setattr(study, "retrieve_study_sources", fake_retrieve)
    monkeypatch.setattr(study, "generate_summary", fake_generate_summary)

    result = await study.summary(session_id, StudyRequest(topic="TCP", top_k=7), fake_session)

    assert result.summary == "Short study summary"
    assert result.sources == sources
    assert captured == {
        "session": fake_session,
        "chat_session_id": session_id,
        "topic": "TCP",
        "top_k": 7,
        "sources": sources,
    }


@pytest.mark.asyncio
async def test_summary_returns_fallback_when_no_sources(monkeypatch) -> None:
    session_id = uuid.uuid4()
    fake_session = FakeSession(ChatSession(id=session_id))

    async def fake_retrieve(*_args, **_kwargs):
        return []

    monkeypatch.setattr(study, "retrieve_study_sources", fake_retrieve)

    result = await study.summary(session_id, StudyRequest(), fake_session)

    assert result.summary == study.SUMMARY_FALLBACK
    assert result.sources == []


@pytest.mark.asyncio
async def test_summary_returns_fallback_when_generation_returns_empty_text(monkeypatch) -> None:
    session_id = uuid.uuid4()
    fake_session = FakeSession(ChatSession(id=session_id))
    sources = [build_citation(text="Congestion control")]

    async def fake_retrieve(*_args, **_kwargs):
        return sources

    async def fake_generate_summary(_sources):
        return "   "

    monkeypatch.setattr(study, "retrieve_study_sources", fake_retrieve)
    monkeypatch.setattr(study, "generate_summary", fake_generate_summary)

    result = await study.summary(session_id, StudyRequest(), fake_session)

    assert result.summary == study.SUMMARY_FALLBACK
    assert result.sources == sources


@pytest.mark.asyncio
async def test_summary_returns_service_unavailable_when_generation_is_unconfigured(monkeypatch) -> None:
    session_id = uuid.uuid4()
    fake_session = FakeSession(ChatSession(id=session_id))
    sources = [build_citation(text="Chunk about congestion control")]

    async def fake_retrieve(*_args, **_kwargs):
        return sources

    async def fake_generate_summary(_sources):
        raise study.GenerationConfigError("GEMINI_API_KEY is required for study generation")

    monkeypatch.setattr(study, "retrieve_study_sources", fake_retrieve)
    monkeypatch.setattr(study, "generate_summary", fake_generate_summary)

    with pytest.raises(HTTPException) as caught:
        await study.summary(session_id, StudyRequest(topic="TCP"), fake_session)

    assert caught.value.status_code == 503
    assert caught.value.detail == "GEMINI_API_KEY is required for study generation"


@pytest.mark.asyncio
async def test_summary_returns_service_unavailable_when_provider_fails(monkeypatch) -> None:
    session_id = uuid.uuid4()
    fake_session = FakeSession(ChatSession(id=session_id))
    sources = [build_citation(text="Chunk about congestion control")]

    async def fake_retrieve(*_args, **_kwargs):
        return sources

    async def fake_generate_summary(_sources):
        raise RuntimeError("provider timeout")

    monkeypatch.setattr(study, "retrieve_study_sources", fake_retrieve)
    monkeypatch.setattr(study, "generate_summary", fake_generate_summary)

    with pytest.raises(HTTPException) as caught:
        await study.summary(session_id, StudyRequest(topic="TCP"), fake_session)

    assert caught.value.status_code == 503
    assert caught.value.detail == study.STUDY_GENERATION_UNAVAILABLE_DETAIL


@pytest.mark.asyncio
async def test_flashcards_raises_for_missing_session() -> None:
    with pytest.raises(HTTPException) as caught:
        await study.flashcards(uuid.uuid4(), StudyRequest(topic="TCP"), FakeSession(None))

    assert caught.value.status_code == 404
    assert caught.value.detail == "Chat session not found"


@pytest.mark.asyncio
async def test_flashcards_reject_topic_and_top_k_scoping() -> None:
    session_id = uuid.uuid4()
    fake_session = FakeSession(ChatSession(id=session_id))

    with pytest.raises(HTTPException) as caught:
        await study.flashcards(
            session_id,
            StudyRequest(topic="networking", top_k=3, flashcard_count=2),
            fake_session,
        )

    assert caught.value.status_code == 400
    assert caught.value.detail == study.FLASHCARD_SCOPE_UNSUPPORTED_DETAIL


@pytest.mark.asyncio
async def test_flashcards_ignore_topic_and_return_exact_requested_count(monkeypatch) -> None:
    session_id = uuid.uuid4()
    fake_session = FakeSession(ChatSession(id=session_id))
    doc_id = uuid.uuid4()
    sources = [
        Citation(
            index=1,
            chunk_id=uuid.uuid4(),
            doc_id=doc_id,
            doc_name="networking.pdf",
            page=1,
            text="Chunk about sockets",
            snippet="Chunk about sockets",
        )
    ]
    captured: dict[str, object] = {}

    async def fake_retrieve(session, chat_session_id):
        captured["session"] = session
        captured["chat_session_id"] = chat_session_id
        return sources

    async def fake_generate_notes(received_sources):
        assert received_sources == sources
        return "- Socket basics"

    async def fake_generate_flashcards(notes_context, flashcard_count, coverage_hint):
        captured["notes_context"] = notes_context
        captured["flashcard_count"] = flashcard_count
        captured["coverage_hint"] = coverage_hint
        return "Q: What is a socket?\nA: A communication endpoint.\nQ: What is TCP?\nA: A transport protocol."

    monkeypatch.setattr(study, "retrieve_flashcard_sources", fake_retrieve)
    monkeypatch.setattr(study, "generate_flashcard_notes", fake_generate_notes)
    monkeypatch.setattr(study, "generate_flashcards_from_notes", fake_generate_flashcards)

    result = await study.flashcards(
        session_id,
        StudyRequest(flashcard_count=2),
        fake_session,
    )

    assert [(card.question, card.answer) for card in result.flashcards] == [
        ("What is a socket?", "A communication endpoint."),
        ("What is TCP?", "A transport protocol."),
    ]
    assert result.sources[0].text == result.sources[0].snippet
    assert captured["session"] is fake_session
    assert captured["chat_session_id"] == session_id
    assert captured["flashcard_count"] == 2
    assert "Document: networking.pdf" in captured["notes_context"]
    assert "networking.pdf: target 2 card(s)" in captured["coverage_hint"]


@pytest.mark.asyncio
async def test_flashcards_trim_to_exact_requested_count(monkeypatch) -> None:
    session_id = uuid.uuid4()
    fake_session = FakeSession(ChatSession(id=session_id))
    sources = [build_citation(text="Chunk about sockets")]

    async def fake_retrieve(*_args, **_kwargs):
        return sources

    async def fake_generate_notes(_sources):
        return "- Socket basics"

    async def fake_generate_flashcards(*_args, **_kwargs):
        return (
            "Q: What is a socket?\nA: A communication endpoint.\n"
            "Q: What is TCP?\nA: A transport protocol.\n"
            "Q: What is UDP?\nA: A connectionless protocol."
        )

    monkeypatch.setattr(study, "retrieve_flashcard_sources", fake_retrieve)
    monkeypatch.setattr(study, "generate_flashcard_notes", fake_generate_notes)
    monkeypatch.setattr(study, "generate_flashcards_from_notes", fake_generate_flashcards)

    result = await study.flashcards(session_id, StudyRequest(flashcard_count=2), fake_session)

    assert len(result.flashcards) == 2
    assert [(card.question, card.answer) for card in result.flashcards] == [
        ("What is a socket?", "A communication endpoint."),
        ("What is TCP?", "A transport protocol."),
    ]


@pytest.mark.asyncio
async def test_flashcards_preserve_multiline_answers(monkeypatch) -> None:
    session_id = uuid.uuid4()
    fake_session = FakeSession(ChatSession(id=session_id))
    sources = [build_citation(text="Chunk about sockets")]

    async def fake_retrieve(*_args, **_kwargs):
        return sources

    async def fake_generate_notes(_sources):
        return "- Socket basics"

    async def fake_generate_flashcards(*_args, **_kwargs):
        return "Q: Explain TCP\nA: First line\nSecond line\n- bullet"

    monkeypatch.setattr(study, "retrieve_flashcard_sources", fake_retrieve)
    monkeypatch.setattr(study, "generate_flashcard_notes", fake_generate_notes)
    monkeypatch.setattr(study, "generate_flashcards_from_notes", fake_generate_flashcards)

    result = await study.flashcards(session_id, StudyRequest(flashcard_count=1), fake_session)

    assert result.flashcards[0].answer == "First line\nSecond line\n- bullet"


@pytest.mark.asyncio
async def test_flashcards_dedupe_and_fill_to_exact_count(monkeypatch) -> None:
    session_id = uuid.uuid4()
    fake_session = FakeSession(ChatSession(id=session_id))
    first_doc_id = uuid.uuid4()
    second_doc_id = uuid.uuid4()
    sources = [
        Citation(
            index=1,
            chunk_id=uuid.uuid4(),
            doc_id=first_doc_id,
            doc_name="alpha.pdf",
            page=1,
            text="First chunk",
            snippet="First chunk",
        ),
        Citation(
            index=2,
            chunk_id=uuid.uuid4(),
            doc_id=second_doc_id,
            doc_name="beta.pdf",
            page=2,
            text="Second chunk",
            snippet="Second chunk",
        ),
    ]

    async def fake_retrieve(*_args, **_kwargs):
        return sources

    async def fake_generate_notes(received_sources):
        return f"- {received_sources[0].snippet}"

    async def fake_generate_flashcards(*_args, **_kwargs):
        return "Q: What is the key idea?\nA: Overview.\nQ:  What is the key idea?  \nA: Overview."

    monkeypatch.setattr(study, "retrieve_flashcard_sources", fake_retrieve)
    monkeypatch.setattr(study, "generate_flashcard_notes", fake_generate_notes)
    monkeypatch.setattr(study, "generate_flashcards_from_notes", fake_generate_flashcards)

    result = await study.flashcards(session_id, StudyRequest(flashcard_count=3), fake_session)

    assert len(result.flashcards) == 3
    assert result.flashcards[0].question == "What is the key idea?"
    assert result.flashcards[0].answer == "Overview."
    assert any(card.answer == "First chunk" for card in result.flashcards[1:])
    assert any(card.answer == "Second chunk" for card in result.flashcards[1:])


@pytest.mark.asyncio
async def test_flashcards_use_snippet_fallback_when_one_batch_fails(monkeypatch) -> None:
    session_id = uuid.uuid4()
    fake_session = FakeSession(ChatSession(id=session_id))
    doc_id = uuid.uuid4()
    sources = [
        Citation(
            index=1,
            chunk_id=uuid.uuid4(),
            doc_id=doc_id,
            doc_name="alpha.pdf",
            page=1,
            text="First chunk",
            snippet="First chunk",
        ),
        Citation(
            index=2,
            chunk_id=uuid.uuid4(),
            doc_id=doc_id,
            doc_name="alpha.pdf",
            page=2,
            text="Second chunk",
            snippet="Second chunk",
        ),
        Citation(
            index=3,
            chunk_id=uuid.uuid4(),
            doc_id=doc_id,
            doc_name="alpha.pdf",
            page=3,
            text="Third chunk",
            snippet="Third chunk",
        ),
        Citation(
            index=4,
            chunk_id=uuid.uuid4(),
            doc_id=doc_id,
            doc_name="alpha.pdf",
            page=4,
            text="Fourth chunk",
            snippet="Fourth chunk",
        ),
        Citation(
            index=5,
            chunk_id=uuid.uuid4(),
            doc_id=doc_id,
            doc_name="alpha.pdf",
            page=5,
            text="Fifth chunk",
            snippet="Fifth chunk",
        ),
    ]

    async def fake_retrieve(*_args, **_kwargs):
        return sources

    async def fake_generate_notes(received_sources):
        if received_sources[0].page == 5:
            raise RuntimeError("provider timeout")
        return "- Combined first batch"

    async def fake_generate_flashcards(notes_context, *_args, **_kwargs):
        assert "Fifth chunk" in notes_context
        return "Q: Summary?\nA: Covers all chunks."

    monkeypatch.setattr(study, "retrieve_flashcard_sources", fake_retrieve)
    monkeypatch.setattr(study, "generate_flashcard_notes", fake_generate_notes)
    monkeypatch.setattr(study, "generate_flashcards_from_notes", fake_generate_flashcards)

    result = await study.flashcards(session_id, StudyRequest(flashcard_count=1), fake_session)

    assert result.flashcards[0].answer == "Covers all chunks."


@pytest.mark.asyncio
async def test_flashcards_use_default_count_when_not_provided(monkeypatch) -> None:
    session_id = uuid.uuid4()
    fake_session = FakeSession(ChatSession(id=session_id))
    sources = [build_citation(text="Only chunk")]
    captured: dict[str, object] = {}

    async def fake_retrieve(*_args, **_kwargs):
        return sources

    async def fake_generate_notes(_sources):
        return "- Only chunk"

    async def fake_generate_flashcards(_notes_context, flashcard_count, _coverage_hint):
        captured["flashcard_count"] = flashcard_count
        return "Q: What matters?\nA: Only chunk."

    monkeypatch.setattr(study, "retrieve_flashcard_sources", fake_retrieve)
    monkeypatch.setattr(study, "generate_flashcard_notes", fake_generate_notes)
    monkeypatch.setattr(study, "generate_flashcards_from_notes", fake_generate_flashcards)

    result = await study.flashcards(session_id, StudyRequest(), fake_session)

    assert captured["flashcard_count"] == study.DEFAULT_FLASHCARD_COUNT
    assert len(result.flashcards) == study.DEFAULT_FLASHCARD_COUNT


@pytest.mark.asyncio
async def test_flashcards_return_empty_list_when_no_sources(monkeypatch) -> None:
    session_id = uuid.uuid4()
    fake_session = FakeSession(ChatSession(id=session_id))

    async def fake_retrieve(*_args, **_kwargs):
        return []

    monkeypatch.setattr(study, "retrieve_flashcard_sources", fake_retrieve)

    result = await study.flashcards(session_id, StudyRequest(), fake_session)

    assert result.flashcards == []
    assert result.sources == []


@pytest.mark.asyncio
async def test_flashcards_return_service_unavailable_when_generation_is_unconfigured(monkeypatch) -> None:
    session_id = uuid.uuid4()
    fake_session = FakeSession(ChatSession(id=session_id))
    sources = [build_citation(text="Chunk about sockets")]

    async def fake_retrieve(*_args, **_kwargs):
        return sources

    async def fake_generate_notes(_sources):
        return "- Socket basics"

    async def fake_generate_flashcards(*_args, **_kwargs):
        raise study.GenerationConfigError("GEMINI_API_KEY is required for study generation")

    monkeypatch.setattr(study, "retrieve_flashcard_sources", fake_retrieve)
    monkeypatch.setattr(study, "generate_flashcard_notes", fake_generate_notes)
    monkeypatch.setattr(study, "generate_flashcards_from_notes", fake_generate_flashcards)

    with pytest.raises(HTTPException) as caught:
        await study.flashcards(session_id, StudyRequest(flashcard_count=2), fake_session)

    assert caught.value.status_code == 503
    assert caught.value.detail == "GEMINI_API_KEY is required for study generation"
