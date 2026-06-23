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


async def retrieve_top_k(
    session: AsyncSession, chat_session_id: UUID, query: str
) -> list[Citation]:
    settings = get_settings()
    query_embedding = get_embedder().embed_query(query)
    distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
    candidate_limit = (
        settings.rerank_candidate_k if settings.rerank_enabled else settings.retrieval_top_k
    )
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
            query, [chunk.text for chunk, _document, _distance in rows], settings.retrieval_top_k
        )
        rows = [rows[original_index] for original_index, _score in ranking]
    else:
        rows = rows[: settings.retrieval_top_k]

    return [
        Citation(
            index=index,
            chunk_id=chunk.id,
            doc_id=document.id,
            doc_name=document.name,
            page=chunk.page,
            text=chunk.text,
            snippet=_snippet(chunk.text),
        )
        for index, (chunk, document, _distance) in enumerate(rows, start=1)
    ]
