from collections import defaultdict, deque
import random
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import ChatSessionDocument, Chunk, Document
from app.rag.embedder import get_embedder
from app.rag.reranker import get_reranker
from app.rag.vector_store import search_chunk_ids
from app.schemas.chat import Citation

FLASHCARD_DEFAULT_COUNT = 10
FLASHCARD_MIN_SELECTED_CHUNKS = 8
FLASHCARD_MAX_SELECTED_CHUNKS = 18
FLASHCARD_MIN_DOCUMENTS = 2
FLASHCARD_MAX_DOCUMENTS = 6


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
    candidate_limit = settings.rerank_candidate_k if settings.rerank_enabled else resolved_top_k

    # The session -> documents relationship stays in Postgres; use it to scope
    # the vector search to this session's documents.
    doc_ids = list(
        (
            await session.execute(
                select(ChatSessionDocument.document_id).where(
                    ChatSessionDocument.session_id == chat_session_id
                )
            )
        ).scalars()
    )
    if not doc_ids:
        return []

    query_embedding = get_embedder().embed_query(query)
    hits = search_chunk_ids(query_embedding, doc_ids, candidate_limit)
    if not hits:
        return []

    ordered_chunk_ids = [chunk_id for chunk_id, _score in hits]
    rows_by_id = {
        chunk.id: (chunk, document)
        for chunk, document in (
            await session.execute(
                select(Chunk, Document)
                .join(Document, Document.id == Chunk.doc_id)
                .where(Chunk.id.in_(ordered_chunk_ids))
            )
        ).all()
    }
    # Preserve Qdrant's similarity order; drop any hit whose Postgres row is gone.
    rows = [rows_by_id[chunk_id] for chunk_id in ordered_chunk_ids if chunk_id in rows_by_id]

    if settings.rerank_enabled and rows:
        ranking = get_reranker().rerank(
            query, [chunk.text for chunk, _document in rows], resolved_top_k
        )
        rows = [rows[original_index] for original_index, _score in ranking]
    else:
        rows = rows[:resolved_top_k]

    return [
        _build_citation(index, chunk, document)
        for index, (chunk, document) in enumerate(rows, start=1)
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


async def retrieve_flashcard_sources(
    session: AsyncSession, chat_session_id: UUID, flashcard_count: int | None = None
) -> list[Citation]:
    statement = (
        select(Chunk, Document)
        .join(Document, Document.id == Chunk.doc_id)
        .join(ChatSessionDocument, ChatSessionDocument.document_id == Document.id)
        .where(ChatSessionDocument.session_id == chat_session_id)
        .where(Document.status == "ready")
        .order_by(Document.created_at.asc(), Chunk.page.asc(), Chunk.created_at.asc())
    )
    rows = (await session.execute(statement)).all()
    selected_rows = _select_flashcard_rows(rows, flashcard_count)

    return [
        _build_citation(index, chunk, document)
        for index, (chunk, document) in enumerate(selected_rows, start=1)
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


def _select_flashcard_rows(
    rows: list[tuple[Chunk, Document]], flashcard_count: int | None, rng: random.Random | None = None
) -> list[tuple[Chunk, Document]]:
    if not rows:
        return []

    resolved_count = flashcard_count or FLASHCARD_DEFAULT_COUNT
    selected_chunk_limit = min(
        len(rows),
        min(FLASHCARD_MAX_SELECTED_CHUNKS, max(FLASHCARD_MIN_SELECTED_CHUNKS, resolved_count)),
    )
    grouped_rows = _group_rows_by_document(rows)
    ordered_doc_ids = list(grouped_rows)
    if not ordered_doc_ids:
        return []

    random_source = rng or random.SystemRandom()
    selected_document_limit = min(
        len(ordered_doc_ids),
        selected_chunk_limit,
        FLASHCARD_MAX_DOCUMENTS,
        max(FLASHCARD_MIN_DOCUMENTS, (resolved_count + 2) // 3),
    )
    selected_doc_ids = list(ordered_doc_ids)
    if len(selected_doc_ids) > selected_document_limit:
        selected_doc_ids = random_source.sample(selected_doc_ids, selected_document_limit)
    random_source.shuffle(selected_doc_ids)

    per_document_limit = max(1, -(-selected_chunk_limit // len(selected_doc_ids)))
    primary_queues: dict[UUID, deque[tuple[Chunk, Document]]] = {}
    supplemental_queues: dict[UUID, deque[tuple[Chunk, Document]]] = {}

    for doc_id in selected_doc_ids:
        doc_rows = grouped_rows[doc_id]
        representative_rows = _select_document_representative_rows(
            doc_rows, per_document_limit, random_source
        )
        representative_ids = {chunk.id for chunk, _ in representative_rows}
        remaining_rows = [row for row in doc_rows if row[0].id not in representative_ids]
        random_source.shuffle(remaining_rows)
        primary_queues[doc_id] = deque(representative_rows)
        supplemental_queues[doc_id] = deque(remaining_rows)

    selected_rows = _select_round_robin_rows(primary_queues, selected_doc_ids, selected_chunk_limit)
    if len(selected_rows) >= selected_chunk_limit:
        return selected_rows

    supplemental_rows = _select_round_robin_rows(
        supplemental_queues, selected_doc_ids, selected_chunk_limit - len(selected_rows)
    )
    selected_rows.extend(supplemental_rows)
    return selected_rows


def _group_rows_by_document(rows: list[tuple[Chunk, Document]]) -> dict[UUID, list[tuple[Chunk, Document]]]:
    grouped_rows: dict[UUID, list[tuple[Chunk, Document]]] = {}
    for chunk, document in rows:
        grouped_rows.setdefault(document.id, []).append((chunk, document))
    return grouped_rows


def _group_document_rows_by_page(
    doc_rows: list[tuple[Chunk, Document]],
) -> dict[int, list[tuple[Chunk, Document]]]:
    grouped_rows: dict[int, list[tuple[Chunk, Document]]] = {}
    for chunk, document in doc_rows:
        grouped_rows.setdefault(chunk.page, []).append((chunk, document))
    return grouped_rows


def _select_document_representative_rows(
    doc_rows: list[tuple[Chunk, Document]], limit: int, rng: random.Random
) -> list[tuple[Chunk, Document]]:
    if limit <= 0 or not doc_rows:
        return []

    rows_by_page = _group_document_rows_by_page(doc_rows)
    ordered_pages = sorted(rows_by_page)
    selected_pages = _select_spread_pages(ordered_pages, min(limit, len(ordered_pages)), rng)
    return [rng.choice(rows_by_page[page]) for page in selected_pages]


def _select_spread_pages(pages: list[int], limit: int, rng: random.Random) -> list[int]:
    if limit <= 0 or not pages:
        return []
    if limit >= len(pages):
        return list(pages)
    if limit == 1:
        return [rng.choice(pages)]

    last_index = len(pages) - 1
    selected_indices: set[int] = set()
    for offset in range(limit):
        center = round(offset * last_index / (limit - 1))
        start = max(0, center - 1)
        end = min(last_index, center + 1)
        candidates = [index for index in range(start, end + 1) if index not in selected_indices]
        if not candidates:
            candidates = [index for index in range(len(pages)) if index not in selected_indices]
        selected_indices.add(rng.choice(candidates))

    return [pages[index] for index in sorted(selected_indices)]


def _select_round_robin_rows(
    grouped_rows: dict[UUID, deque[tuple[Chunk, Document]]],
    ordered_doc_ids: list[UUID],
    limit: int,
) -> list[tuple[Chunk, Document]]:
    selected_rows: list[tuple[Chunk, Document]] = []
    if limit <= 0:
        return selected_rows

    while len(selected_rows) < limit:
        progressed = False
        for doc_id in ordered_doc_ids:
            doc_queue = grouped_rows[doc_id]
            if not doc_queue:
                continue
            selected_rows.append(doc_queue.popleft())
            progressed = True
            if len(selected_rows) == limit:
                return selected_rows
        if not progressed:
            break

    return selected_rows
