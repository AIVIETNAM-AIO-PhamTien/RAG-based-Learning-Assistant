"""Qdrant-backed vector store for chunk embeddings.

Postgres stays the source of truth for chunk content and relationships; this
module holds only the embedding vectors, plus a ``doc_id`` payload used to scope
searches to a chat session's documents. See
``explain/embedding-storage-migration.md`` for the reasoning.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from uuid import UUID

from qdrant_client import QdrantClient, models

from app.config import get_settings

logger = logging.getLogger(__name__)

DOC_ID_PAYLOAD_KEY = "doc_id"


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """Return a process-wide Qdrant client.

    Uses Qdrant Cloud when ``QDRANT_URL`` is configured; otherwise falls back to
    an ephemeral in-memory instance so the app still boots for offline dev/tests
    (vectors are NOT persisted in that mode).
    """
    settings = get_settings()
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    logger.warning(
        "QDRANT_URL is not set; using an in-memory Qdrant instance. "
        "Vectors will NOT persist across restarts."
    )
    return QdrantClient(location=":memory:")


def ensure_collection(client: QdrantClient | None = None) -> None:
    """Create the chunk collection and its ``doc_id`` payload index if missing."""
    settings = get_settings()
    client = client or get_qdrant_client()
    collection = settings.qdrant_collection
    if client.collection_exists(collection):
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=settings.embedding_dim, distance=models.Distance.COSINE
        ),
    )
    client.create_payload_index(
        collection_name=collection,
        field_name=DOC_ID_PAYLOAD_KEY,
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def upsert_chunk_vectors(
    items: list[tuple[UUID, UUID, list[float]]], client: QdrantClient | None = None
) -> None:
    """Upsert ``(chunk_id, doc_id, vector)`` triples as Qdrant points.

    The Qdrant point id equals the Postgres chunk id, so retrieval can map a hit
    straight back to its ``chunks`` row.
    """
    if not items:
        return
    settings = get_settings()
    client = client or get_qdrant_client()
    points = [
        models.PointStruct(
            id=str(chunk_id),
            vector=vector,
            payload={DOC_ID_PAYLOAD_KEY: str(doc_id)},
        )
        for chunk_id, doc_id, vector in items
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points, wait=True)


def delete_document_vectors(doc_id: UUID, client: QdrantClient | None = None) -> None:
    """Delete every vector belonging to ``doc_id`` (used for idempotent re-ingest)."""
    settings = get_settings()
    client = client or get_qdrant_client()
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=DOC_ID_PAYLOAD_KEY,
                        match=models.MatchValue(value=str(doc_id)),
                    )
                ]
            )
        ),
        wait=True,
    )


def search_chunk_ids(
    query_vector: list[float],
    doc_ids: list[UUID],
    limit: int,
    client: QdrantClient | None = None,
) -> list[tuple[UUID, float]]:
    """Return ``(chunk_id, score)`` for the top ``limit`` vectors within ``doc_ids``.

    Results are ordered by similarity score (highest first). Scoping is done with
    a ``doc_id`` payload filter, mirroring the old SQL join on
    ``chat_session_documents``.
    """
    if not doc_ids or limit <= 0:
        return []
    settings = get_settings()
    client = client or get_qdrant_client()
    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key=DOC_ID_PAYLOAD_KEY,
                    match=models.MatchAny(any=[str(doc_id) for doc_id in doc_ids]),
                )
            ]
        ),
        limit=limit,
        with_payload=False,
    )
    return [(UUID(str(point.id)), point.score) for point in response.points]
