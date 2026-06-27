import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatSession
from app.db.session import get_session
from app.rag.generator import (
    GenerationConfigError,
    generate_flashcard_notes,
    generate_flashcards_from_notes,
    generate_summary,
)
from app.rag.prompts import SUMMARY_FALLBACK
from app.rag.retriever import retrieve_flashcard_sources, retrieve_study_sources
from app.schemas.chat import Citation
from app.schemas.study import Flashcard, FlashcardsResponse, StudyRequest, SummaryResponse

router = APIRouter(prefix="/api/v1/sessions", tags=["study"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]
logger = logging.getLogger(__name__)
DEFAULT_FLASHCARD_COUNT = 6
FLASHCARD_BATCH_SIZE = 4
FLASHCARD_NOTES_CONCURRENCY = 3
MAX_FLASHCARD_NOTES_CONTEXT_CHARS = 9000
MAX_DOCUMENT_NOTES_CHARS = 1800
FLASHCARD_FALLBACK_QUESTION = "What is the key idea from this excerpt?"
STUDY_GENERATION_UNAVAILABLE_DETAIL = "Study generation is temporarily unavailable."
FLASHCARD_SCOPE_UNSUPPORTED_DETAIL = (
    "Flashcards always use all ready session chunks. Remove topic and top_k from the request."
)


@dataclass(frozen=True)
class DocumentNotes:
    doc_id: uuid.UUID
    doc_name: str
    notes: str
    sources: list[Citation]


@router.post("/{session_id}/summary", response_model=SummaryResponse)
async def summary(
    session_id: uuid.UUID, payload: StudyRequest, session: SessionDep
) -> SummaryResponse:
    chat_session = await session.get(ChatSession, session_id)
    if chat_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")

    sources = await retrieve_study_sources(session, session_id, topic=payload.topic, top_k=payload.top_k)
    if not sources:
        return SummaryResponse(summary=SUMMARY_FALLBACK, sources=[])

    try:
        summary_text = (await generate_summary(sources)).strip()
    except GenerationConfigError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=STUDY_GENERATION_UNAVAILABLE_DETAIL,
        ) from exc

    return SummaryResponse(summary=summary_text or SUMMARY_FALLBACK, sources=sources)


@router.post("/{session_id}/flashcards", response_model=FlashcardsResponse)
async def flashcards(
    session_id: uuid.UUID, payload: StudyRequest, session: SessionDep
) -> FlashcardsResponse:
    chat_session = await session.get(ChatSession, session_id)
    if chat_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")

    if payload.topic is not None or payload.top_k is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=FLASHCARD_SCOPE_UNSUPPORTED_DETAIL,
        )

    flashcard_count = payload.flashcard_count or DEFAULT_FLASHCARD_COUNT
    sources = await retrieve_flashcard_sources(session, session_id)
    if not sources:
        return FlashcardsResponse(flashcards=[], sources=[])

    try:
        cards = await _generate_session_flashcards(sources, flashcard_count)
    except GenerationConfigError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=STUDY_GENERATION_UNAVAILABLE_DETAIL,
        ) from exc

    response_sources = _compact_sources_for_response(sources)
    return FlashcardsResponse(flashcards=cards, sources=response_sources)


async def _generate_session_flashcards(
    sources: list[Citation], flashcard_count: int
) -> list[Flashcard]:
    document_notes = await _build_document_notes(sources)
    coverage_targets = _allocate_flashcard_targets(document_notes, flashcard_count)
    notes_context = _build_notes_context(document_notes)
    coverage_hint = _build_coverage_hint(coverage_targets, document_notes)

    try:
        generated = await generate_flashcards_from_notes(notes_context, flashcard_count, coverage_hint)
        logger.info("Flashcard generator raw output (%d chars): %s", len(generated), generated)
        parsed_cards = _parse_flashcards(generated)
        logger.info("Parsed %d flashcards from generator output", len(parsed_cards))
        unique_cards = _dedupe_flashcards(parsed_cards)
        logger.info("Kept %d unique flashcards after dedupe", len(unique_cards))
    except GenerationConfigError:
        raise
    except Exception:
        logger.exception("Flashcard generation failed; using fallback flashcards")
        unique_cards = []

    if len(unique_cards) >= flashcard_count:
        return unique_cards[:flashcard_count]

    fallback_cards = _build_fallback_flashcards(document_notes, coverage_targets)
    return _fill_flashcards(unique_cards, fallback_cards, flashcard_count)


async def _build_document_notes(sources: list[Citation]) -> list[DocumentNotes]:
    grouped_sources: dict[uuid.UUID, list[Citation]] = defaultdict(list)
    for source in sources:
        grouped_sources[source.doc_id].append(source)

    semaphore = asyncio.Semaphore(FLASHCARD_NOTES_CONCURRENCY)
    tasks = [
        _build_single_document_notes(doc_sources, semaphore)
        for doc_sources in grouped_sources.values()
    ]
    return await asyncio.gather(*tasks)


async def _build_single_document_notes(
    doc_sources: list[Citation], semaphore: asyncio.Semaphore
) -> DocumentNotes:
    batch_tasks = [
        _generate_batch_notes(batch, semaphore)
        for batch in _split_into_batches(doc_sources, FLASHCARD_BATCH_SIZE)
    ]
    batch_notes = await asyncio.gather(*batch_tasks)
    merged_notes = "\n\n".join(batch_notes).strip()

    first_source = doc_sources[0]
    return DocumentNotes(
        doc_id=first_source.doc_id,
        doc_name=first_source.doc_name,
        notes=merged_notes,
        sources=doc_sources,
    )


async def _generate_batch_notes(
    batch: list[Citation], semaphore: asyncio.Semaphore
) -> str:
    async with semaphore:
        try:
            note_text = (await generate_flashcard_notes(batch)).strip()
        except GenerationConfigError:
            raise
        except Exception:
            note_text = ""

    return note_text or _build_snippet_notes(batch)


def _build_snippet_notes(sources: list[Citation]) -> str:
    return "\n".join(f"- {source.snippet}" for source in sources)


def _split_into_batches(sources: list[Citation], batch_size: int) -> list[list[Citation]]:
    return [sources[index : index + batch_size] for index in range(0, len(sources), batch_size)]


def _allocate_flashcard_targets(
    document_notes: list[DocumentNotes], flashcard_count: int
) -> dict[uuid.UUID, int]:
    targets = {document.doc_id: 0 for document in document_notes}
    if not document_notes or flashcard_count <= 0:
        return targets

    if flashcard_count >= len(document_notes):
        for document in document_notes:
            targets[document.doc_id] = 1
        remaining = flashcard_count - len(document_notes)
        for index in range(remaining):
            document = document_notes[index % len(document_notes)]
            targets[document.doc_id] += 1
        return targets

    for document in document_notes[:flashcard_count]:
        targets[document.doc_id] = 1
    return targets


def _build_notes_context(document_notes: list[DocumentNotes]) -> str:
    document_count = max(1, len(document_notes))
    header_budget = 80 * document_count
    available_notes_budget = max(1200, MAX_FLASHCARD_NOTES_CONTEXT_CHARS - header_budget)
    per_document_budget = min(
        MAX_DOCUMENT_NOTES_CHARS,
        max(300, available_notes_budget // document_count),
    )

    sections: list[str] = []
    for document in document_notes:
        start_page = document.sources[0].page
        end_page = document.sources[-1].page
        sections.append(
            f"Document: {document.doc_name}\n"
            f"Pages: {start_page}-{end_page}\n"
            f"Notes:\n{_truncate_text(document.notes, per_document_budget)}"
        )
    return "\n\n".join(sections)


def _truncate_text(value: str, limit: int) -> str:
    normalized = value.strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1].rstrip()}…"


def _build_coverage_hint(
    coverage_targets: dict[uuid.UUID, int], document_notes: list[DocumentNotes]
) -> str:
    return "\n".join(
        f"- {document.doc_name}: target {coverage_targets.get(document.doc_id, 0)} card(s)"
        for document in document_notes
    )


def _parse_flashcards(generated: str) -> list[Flashcard]:
    cards: list[Flashcard] = []
    question: str | None = None
    answer_lines: list[str] = []

    for raw_line in generated.splitlines():
        line = raw_line.strip()
        if not line:
            if answer_lines:
                answer_lines.append("")
            continue

        if line.startswith("Q:"):
            if question and answer_lines:
                cards.append(Flashcard(question=question, answer=_join_answer_lines(answer_lines)))
            question = line[2:].strip()
            answer_lines = []
            continue

        if line.startswith("A:") and question:
            answer_lines = [line[2:].strip()]
            continue

        if question and answer_lines:
            answer_lines.append(line)

    if question and answer_lines:
        cards.append(Flashcard(question=question, answer=_join_answer_lines(answer_lines)))

    return cards


def _join_answer_lines(answer_lines: list[str]) -> str:
    return "\n".join(line for line in answer_lines if line).strip()


def _dedupe_flashcards(cards: list[Flashcard]) -> list[Flashcard]:
    seen: set[tuple[str, str]] = set()
    unique_cards: list[Flashcard] = []
    for card in cards:
        normalized = (_normalize_flashcard_text(card.question), _normalize_flashcard_text(card.answer))
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_cards.append(card)
    return unique_cards


def _normalize_flashcard_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _build_fallback_flashcards(
    document_notes: list[DocumentNotes], coverage_targets: dict[uuid.UUID, int]
) -> list[Flashcard]:
    fallback_cards: list[Flashcard] = []
    for document in document_notes:
        target_count = coverage_targets.get(document.doc_id, 0)
        if target_count <= 0:
            continue
        doc_sources = document.sources
        for index in range(target_count):
            source = doc_sources[index % len(doc_sources)]
            fallback_cards.append(
                Flashcard(
                    question=f"{FLASHCARD_FALLBACK_QUESTION} ({document.doc_name}, page {source.page})",
                    answer=source.snippet,
                )
            )
    if fallback_cards:
        return fallback_cards

    return [
        Flashcard(
            question=f"{FLASHCARD_FALLBACK_QUESTION} ({document.doc_name}, page {document.sources[0].page})",
            answer=document.sources[0].snippet,
        )
        for document in document_notes
        if document.sources
    ]


def _fill_flashcards(
    cards: list[Flashcard], fallback_cards: list[Flashcard], flashcard_count: int
) -> list[Flashcard]:
    combined_cards = list(cards)
    seen = {
        (_normalize_flashcard_text(card.question), _normalize_flashcard_text(card.answer))
        for card in combined_cards
    }
    variant_number = 2
    for fallback_card in fallback_cards:
        normalized = (
            _normalize_flashcard_text(fallback_card.question),
            _normalize_flashcard_text(fallback_card.answer),
        )
        if normalized in seen:
            continue
        combined_cards.append(fallback_card)
        seen.add(normalized)
        if len(combined_cards) == flashcard_count:
            return combined_cards

    while combined_cards and len(combined_cards) < flashcard_count:
        seed_card = (
            fallback_cards[(len(combined_cards) - len(cards)) % len(fallback_cards)]
            if fallback_cards
            else combined_cards[-1]
        )
        variant_card = Flashcard(
            question=f"{seed_card.question} (review {variant_number})",
            answer=seed_card.answer,
        )
        variant_number += 1
        normalized = (
            _normalize_flashcard_text(variant_card.question),
            _normalize_flashcard_text(variant_card.answer),
        )
        if normalized in seen:
            continue
        combined_cards.append(variant_card)
        seen.add(normalized)

    return combined_cards[:flashcard_count]


def _compact_sources_for_response(sources: list[Citation]) -> list[Citation]:
    return [
        Citation(
            index=source.index,
            chunk_id=source.chunk_id,
            doc_id=source.doc_id,
            doc_name=source.doc_name,
            page=source.page,
            text=source.snippet,
            snippet=source.snippet,
        )
        for source in sources
    ]
