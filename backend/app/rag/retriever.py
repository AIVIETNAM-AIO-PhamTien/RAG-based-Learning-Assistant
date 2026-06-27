from collections import defaultdict, deque
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import ChatSessionDocument, Chunk, Document
from app.rag.embedder import get_embedder
from app.rag.reranker import get_reranker
from app.schemas.chat import Citation


def _snippet(text: str, limit: int = 500) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else f"{normalized[: limit - 1]}…"


def _build_citation(index: int, chunk: Chunk, document: Document) -> Citation:
    return Citation(
        index=index,
        chunk_id=chunk.id,
        doc_id=document.id,
        doc_name=document.name,
        page=chunk.page,
        text=chunk.text,
        snippet=_snippet(chunk.text),
    )


async def retrieve_top_k(
    session: AsyncSession, chat_session_id: UUID, query: str, top_k: int | None = None
) -> list[Citation]:
    settings = get_settings()
    resolved_top_k = top_k or settings.retrieval_top_k
    query_embedding = get_embedder().embed_query(query)
    distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
    candidate_limit = settings.rerank_candidate_k if settings.rerank_enabled else resolved_top_k
    statement = (
        select(Chunk, Document, distance)
        .join(Document, Document.id == Chunk.doc_id)
        .join(ChatSessionDocument, ChatSessionDocument.document_id == Document.id)
        .where(ChatSessionDocument.session_id == chat_session_id)
        .order_by(distance)
        .limit(candidate_limit)
    )
    rows = (await session.execute(statement)).all()

    if settings.rerank_enabled and rows:
        ranking = get_reranker().rerank(
            query, [chunk.text for chunk, _document, _distance in rows], resolved_top_k
        )
        rows = [rows[original_index] for original_index, _score in ranking]
    else:
        rows = rows[:resolved_top_k]

    return [
        _build_citation(index, chunk, document)
        for index, (chunk, document, _distance) in enumerate(rows, start=1)
    ]


async def retrieve_study_sources(
    session: AsyncSession,
    chat_session_id: UUID,
    topic: str | None = None,
    top_k: int | None = None,
) -> list[Citation]:
    normalized_topic = (topic or "").strip()
    if normalized_topic:
        return await retrieve_top_k(session, chat_session_id, normalized_topic, top_k=top_k)

    settings = get_settings()
    limit = top_k or settings.rerank_candidate_k
    statement = (
        select(Chunk, Document)
        .join(Document, Document.id == Chunk.doc_id)
        .join(ChatSessionDocument, ChatSessionDocument.document_id == Document.id)
        .where(ChatSessionDocument.session_id == chat_session_id)
        .where(Document.status == "ready")
        .order_by(Document.created_at.asc(), Chunk.page.asc(), Chunk.created_at.asc())
    )
    rows = (await session.execute(statement)).all()
    selected_rows = _select_fair_rows(rows, limit)

    return [
        _build_citation(index, chunk, document)
        for index, (chunk, document) in enumerate(selected_rows, start=1)
    ]


async def retrieve_flashcard_sources(session: AsyncSession, chat_session_id: UUID) -> list[Citation]:
    statement = (
        select(Chunk, Document)
        .join(Document, Document.id == Chunk.doc_id)
        .join(ChatSessionDocument, ChatSessionDocument.document_id == Document.id)
        .where(ChatSessionDocument.session_id == chat_session_id)
        .where(Document.status == "ready")
        .order_by(Document.created_at.asc(), Chunk.page.asc(), Chunk.created_at.asc())
    )
    rows = (await session.execute(statement)).all()

    return [
        _build_citation(index, chunk, document)
        for index, (chunk, document) in enumerate(rows, start=1)
    ]


def _select_fair_rows(rows: list[tuple[Chunk, Document]], limit: int) -> list[tuple[Chunk, Document]]:
    if limit <= 0 or not rows:
        return []

    grouped_rows: dict[UUID, deque[tuple[Chunk, Document]]] = defaultdict(deque)
    ordered_doc_ids: list[UUID] = []
    seen_doc_ids: set[UUID] = set()

    for chunk, document in rows:
        grouped_rows[document.id].append((chunk, document))
        if document.id not in seen_doc_ids:
            seen_doc_ids.add(document.id)
            ordered_doc_ids.append(document.id)

    selected_rows: list[tuple[Chunk, Document]] = []
    while len(selected_rows) < limit:
        progressed = False
        for doc_id in ordered_doc_ids:
            doc_queue = grouped_rows[doc_id]
            if not doc_queue:
                continue
            selected_rows.append(doc_queue.popleft())
            progressed = True
            if len(selected_rows) == limit:
                break
        if not progressed:
            break

    return selected_rows
