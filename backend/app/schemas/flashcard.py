from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

FlashcardStatus = Literal["not_reviewed", "learning", "known"]


class FlashcardGenerateRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=500)
    count: Literal[5, 10, 15] = 10


class FlashcardUpdateRequest(BaseModel):
    status: FlashcardStatus


class FlashcardRead(BaseModel):
    id: UUID
    question: str
    answer: str
    status: FlashcardStatus
    source_doc_name: str
    source_page: int
    created_at: datetime


class FlashcardStats(BaseModel):
    total: int
    not_reviewed: int
    learning: int
    known: int


class FlashcardsResponse(BaseModel):
    flashcards: list[FlashcardRead]
    stats: FlashcardStats
