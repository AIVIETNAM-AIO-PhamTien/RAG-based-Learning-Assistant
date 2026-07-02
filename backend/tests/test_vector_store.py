import uuid

from qdrant_client import QdrantClient

from app.config import get_settings
from app.rag import vector_store


def _client() -> QdrantClient:
    return QdrantClient(location=":memory:")


def _vec(index: int, dim: int | None = None) -> list[float]:
    dim = dim or get_settings().embedding_dim
    vector = [0.0] * dim
    vector[index] = 1.0
    return vector


def test_ensure_collection_is_idempotent() -> None:
    client = _client()
    collection = get_settings().qdrant_collection

    vector_store.ensure_collection(client)
    vector_store.ensure_collection(client)  # second call must not raise

    assert client.collection_exists(collection)


def test_upsert_search_scopes_by_doc_and_orders_by_score() -> None:
    client = _client()
    vector_store.ensure_collection(client)

    doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
    chunk_a, chunk_b = uuid.uuid4(), uuid.uuid4()

    vector_store.upsert_chunk_vectors(
        [(chunk_a, doc_a, _vec(0)), (chunk_b, doc_b, _vec(1))],
        client=client,
    )

    # Filter restricts results to the requested document only.
    scoped = vector_store.search_chunk_ids(_vec(0), [doc_a], limit=5, client=client)
    assert [chunk_id for chunk_id, _ in scoped] == [chunk_a]

    # With both docs in scope, the nearest vector to the query ranks first.
    ranked = vector_store.search_chunk_ids(_vec(1), [doc_a, doc_b], limit=5, client=client)
    assert ranked[0][0] == chunk_b


def test_delete_document_vectors_removes_only_that_doc() -> None:
    client = _client()
    vector_store.ensure_collection(client)

    doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
    chunk_a, chunk_b = uuid.uuid4(), uuid.uuid4()
    vector_store.upsert_chunk_vectors(
        [(chunk_a, doc_a, _vec(0)), (chunk_b, doc_b, _vec(1))],
        client=client,
    )

    vector_store.delete_document_vectors(doc_a, client=client)

    remaining = vector_store.search_chunk_ids(_vec(0), [doc_a, doc_b], limit=5, client=client)
    remaining_ids = {chunk_id for chunk_id, _ in remaining}
    assert chunk_a not in remaining_ids
    assert chunk_b in remaining_ids


def test_search_returns_empty_without_docs() -> None:
    client = _client()
    vector_store.ensure_collection(client)

    assert vector_store.search_chunk_ids(_vec(0), [], limit=5, client=client) == []
