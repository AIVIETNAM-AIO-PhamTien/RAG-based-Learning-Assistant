from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Chunk, Document
from app.rag.chunker import chunk_pages
from app.rag.embedder import get_embedder
from app.rag.parser import PdfParseError, parse_pdf_text
from app.rag.vector_store import (
    delete_document_vectors,
    ensure_collection,
    upsert_chunk_vectors,
)


async def ingest_document(session: AsyncSession, document_id: UUID, path: Path) -> None:
    document = await session.scalar(select(Document).where(Document.id == document_id))
    if document is None:
        raise ValueError("Document not found")

    document.status = "processing"
    document.error_message = None
    await session.commit()

    try:
        settings = get_settings()
        pages, page_count = parse_pdf_text(path)
        chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise PdfParseError("No chunks produced from PDF text")

        embeddings = get_embedder().embed_texts([chunk.text for chunk in chunks])

        # Postgres keeps chunk content/metadata; Qdrant keeps only the vectors.
        # Rebuild both stores for this document so re-ingest is idempotent. The
        # chunk id is generated here so it can double as the Qdrant point id.
        await session.execute(delete(Chunk).where(Chunk.doc_id == document_id))
        chunk_rows = [
            Chunk(
                id=uuid4(),
                doc_id=document_id,
                page=chunk.page,
                text=chunk.text,
                parent_text=chunk.parent_text,
            )
            for chunk in chunks
        ]
        session.add_all(chunk_rows)

        ensure_collection()
        delete_document_vectors(document_id)
        upsert_chunk_vectors(
            [
                (row.id, document_id, embedding)
                for row, embedding in zip(chunk_rows, embeddings, strict=True)
            ]
        )

        document.page_count = page_count
        document.status = "ready"
        await session.commit()
    except Exception as exc:
        # Discard the staged chunk changes, then record the failure on a fresh
        # read of the document so the Postgres row stays consistent.
        await session.rollback()
        document = await session.scalar(select(Document).where(Document.id == document_id))
        if document is not None:
            document.status = "failed"
            document.error_message = str(exc)
            await session.commit()
        raise
