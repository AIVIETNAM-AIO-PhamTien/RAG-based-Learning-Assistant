from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ChatSessionCreate(BaseModel):
    title: str | None = None


class ChatSessionRead(BaseModel):
    id: UUID
    title: str | None = None
    created_at: datetime


class ChatSessionUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


class Citation(BaseModel):
    index: int
    chunk_id: UUID
    doc_id: UUID
    doc_name: str
    page: int
    text: str
    snippet: str


class ChatMessageRead(BaseModel):
    id: UUID
    session_id: UUID
    role: str
    content: str
    citations: list[Citation] | None = None
    created_at: datetime
