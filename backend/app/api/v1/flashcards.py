import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatSession, Flashcard
from app.db.session import get_session
from app.rag.flashcards import FlashcardGenerationError, generate_flashcards
from app.rag.generator import GenerationConfigError
from app.rag.retriever import retrieve_top_k
from app.schemas.flashcard import (
    FlashcardGenerateRequest,
    FlashcardRead,
    FlashcardsResponse,
    FlashcardStats,
    FlashcardUpdateRequest,
)

router = APIRouter(prefix="/api/v1/sessions", tags=["flashcards"])


def _response(cards: list[Flashcard]) -> FlashcardsResponse:
    stats = {"not_reviewed": 0, "learning": 0, "known": 0}
    for card in cards:
        stats[card.status] += 1
    return FlashcardsResponse(
        flashcards=[FlashcardRead.model_validate(card, from_attributes=True) for card in cards],
        stats=FlashcardStats(total=len(cards), **stats),
    )


async def _session_or_404(session: AsyncSession, session_id: uuid.UUID) -> ChatSession:
    chat_session = await session.get(ChatSession, session_id)
    if chat_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")
    return chat_session


@router.get("/{session_id}/flashcards", response_model=FlashcardsResponse)
async def list_flashcards(
    session_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> FlashcardsResponse:
    await _session_or_404(session, session_id)
    cards = list(
        (
            await session.scalars(
                select(Flashcard)
                .where(Flashcard.session_id == session_id)
                .order_by(Flashcard.created_at)
            )
        ).all()
    )
    return _response(cards)


@router.post("/{session_id}/flashcards/generate", response_model=FlashcardsResponse)
async def create_flashcards(
    session_id: uuid.UUID,
    payload: FlashcardGenerateRequest,
    session: AsyncSession = Depends(get_session),
) -> FlashcardsResponse:
    await _session_or_404(session, session_id)
    citations = await retrieve_top_k(session, session_id, payload.topic)
    try:
        generated = await generate_flashcards(payload.topic, payload.count, citations)
    except (FlashcardGenerationError, GenerationConfigError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    await session.execute(delete(Flashcard).where(Flashcard.session_id == session_id))
    cards = [
        Flashcard(
            session_id=session_id,
            question=item["question"],
            answer=item["answer"],
            source_doc_name=item["source"].doc_name,
            source_page=item["source"].page,
        )
        for item in generated
    ]
    session.add_all(cards)
    await session.commit()
    for card in cards:
        await session.refresh(card)
    return _response(cards)


@router.patch("/{session_id}/flashcards/{flashcard_id}", response_model=FlashcardRead)
async def update_flashcard(
    session_id: uuid.UUID,
    flashcard_id: uuid.UUID,
    payload: FlashcardUpdateRequest,
    session: AsyncSession = Depends(get_session),
) -> FlashcardRead:
    await _session_or_404(session, session_id)
    card = await session.scalar(
        select(Flashcard).where(Flashcard.id == flashcard_id, Flashcard.session_id == session_id)
    )
    if card is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Flashcard not found")
    card.status = payload.status
    await session.commit()
    await session.refresh(card)
    return FlashcardRead.model_validate(card, from_attributes=True)
