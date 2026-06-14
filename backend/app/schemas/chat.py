from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ChatSessionCreate(BaseModel):
    title: str | None = None


class ChatSessionRead(BaseModel):
    id: UUID
    title: str | None = None
    created_at: datetime


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
