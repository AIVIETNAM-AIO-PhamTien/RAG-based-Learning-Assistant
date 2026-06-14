from pathlib import Path
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Chunk, Document
from app.rag.chunker import chunk_pages
from app.rag.embedder import get_embedder
from app.rag.parser import PdfParseError, parse_pdf_text


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
        await session.execute(delete(Chunk).where(Chunk.doc_id == document_id))
        session.add_all(
            Chunk(
                doc_id=document_id,
                page=chunk.page,
                text=chunk.text,
                parent_text=chunk.parent_text,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        )
        document.page_count = page_count
        document.status = "ready"
        await session.commit()
    except Exception as exc:
        document.status = "failed"
        document.error_message = str(exc)
        await session.commit()
        raise
