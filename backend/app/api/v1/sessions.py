import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatMessage, ChatSession, ChatSessionDocument, Document
from app.db.session import get_session
from app.schemas.chat import ChatMessageRead, ChatSessionCreate, ChatSessionRead, ChatSessionUpdate
from app.schemas.document import DocumentRead

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]


@router.post("", response_model=ChatSessionRead)
@router.post("/", response_model=ChatSessionRead, include_in_schema=False)
async def create_chat_session(payload: ChatSessionCreate, session: SessionDep) -> ChatSession:
    chat_session = ChatSession(title=payload.title)
    session.add(chat_session)
    await session.commit()
    await session.refresh(chat_session)
    return chat_session


@router.get("", response_model=list[ChatSessionRead])
@router.get("/", response_model=list[ChatSessionRead], include_in_schema=False)
async def list_chat_sessions(session: SessionDep) -> list[ChatSession]:
    statement = select(ChatSession).order_by(ChatSession.created_at.desc(), ChatSession.id.desc())
    return list((await session.scalars(statement)).all())


@router.patch("/{session_id}", response_model=ChatSessionRead)
async def rename_chat_session(
    session_id: uuid.UUID, payload: ChatSessionUpdate, session: SessionDep
) -> ChatSession:
    chat_session = await session.get(ChatSession, session_id)
    if chat_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")

    chat_session.title = payload.title
    await session.commit()
    await session.refresh(chat_session)
    return chat_session


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(session_id: uuid.UUID, session: SessionDep) -> Response:
    chat_session = await session.get(ChatSession, session_id)
    if chat_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")

    await session.delete(chat_session)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{session_id}/documents", response_model=list[DocumentRead])
async def list_session_documents(session_id: uuid.UUID, session: SessionDep) -> list[Document]:
    chat_session = await session.get(ChatSession, session_id)
    if chat_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")

    statement = (
        select(Document)
        .join(ChatSessionDocument, ChatSessionDocument.document_id == Document.id)
        .where(ChatSessionDocument.session_id == session_id)
        .order_by(Document.created_at.desc())
    )
    return list((await session.scalars(statement)).all())


@router.get("/{session_id}/messages", response_model=list[ChatMessageRead])
async def list_session_messages(
    session_id: uuid.UUID, session: SessionDep
) -> list[ChatMessage]:
    chat_session = await session.get(ChatSession, session_id)
    if chat_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")

    statement = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
    )
    return list((await session.scalars(statement)).all())
