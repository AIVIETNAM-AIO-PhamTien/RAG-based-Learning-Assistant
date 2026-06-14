from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatSession, ChatSessionDocument, Document
from app.db.session import get_session
from app.schemas.chat import ChatSessionCreate, ChatSessionRead
from app.schemas.document import DocumentRead

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]


@router.post("", response_model=ChatSessionRead)
async def create_chat_session(payload: ChatSessionCreate, session: SessionDep) -> ChatSession:
    chat_session = ChatSession(title=payload.title)
    session.add(chat_session)
    await session.commit()
    await session.refresh(chat_session)
    return chat_session


@router.get("/{session_id}/documents", response_model=list[DocumentRead])
async def list_session_documents(session_id: str, session: SessionDep) -> list[Document]:
    statement = (
        select(Document)
        .join(ChatSessionDocument, ChatSessionDocument.document_id == Document.id)
        .where(ChatSessionDocument.session_id == session_id)
        .order_by(Document.created_at.desc())
    )
    return list((await session.scalars(statement)).all())
