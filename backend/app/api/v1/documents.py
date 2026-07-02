import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatSession, ChatSessionDocument, Document
from app.db.session import get_session
from app.rag.ingest import ingest_document
from app.schemas.document import DocumentRead, DocumentUploadResponse
from app.storage import StorageError, get_storage

router = APIRouter(prefix="/api/v1", tags=["documents"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]


def _safe_pdf_name(filename: str | None) -> str:
    name = Path(filename or "upload.pdf").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported"
        )
    return name


@router.post("/sessions/{session_id}/documents", response_model=DocumentUploadResponse)
async def upload_document(
    session_id: uuid.UUID,
    session: SessionDep,
    file: Annotated[UploadFile, File()],
) -> Document:
    chat_session = await session.get(ChatSession, session_id)
    if chat_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")
    if file.content_type and file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported"
        )

    safe_name = _safe_pdf_name(file.filename)
    document_id = uuid.uuid4()
    data = await file.read()

    # Persist the original PDF to the configured storage backend (local disk or R2).
    # Only the opaque key is kept on the Document row; the backend owns the layout.
    storage_key = f"documents/{document_id}/{safe_name}"
    get_storage().save(storage_key, data, "application/pdf")

    document = Document(id=document_id, name=safe_name, storage_path=storage_key)
    session.add(document)
    session.add(ChatSessionDocument(session_id=session_id, document_id=document_id))
    await session.commit()

    try:
        await ingest_document(session, document_id, data)
    except Exception:
        await session.rollback()
        await session.refresh(document)
        return document

    await session.refresh(document)
    return document


@router.get("/documents/{document_id}", response_model=DocumentRead)
async def get_document(document_id: uuid.UUID, session: SessionDep) -> Document:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document


@router.get("/documents/{document_id}/file")
async def download_document(document_id: uuid.UUID, session: SessionDep) -> Response:
    """Serve the original PDF (streamed from storage) so the viewer can open it."""
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    try:
        data = get_storage().load(document.storage_path)
    except StorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document file not found"
        ) from exc
    return Response(
        content=data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{document.name}"'},
    )
