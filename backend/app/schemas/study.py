from pydantic import BaseModel, Field

from app.schemas.chat import Citation


class StudyRequest(BaseModel):
    topic: str | None = Field(default=None, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=20)
    flashcard_count: int | None = Field(default=None, ge=1, le=20)


class SummaryResponse(BaseModel):
    summary: str
    sources: list[Citation]


class Flashcard(BaseModel):
    question: str
    answer: str


class FlashcardsResponse(BaseModel):
    flashcards: list[Flashcard]
    sources: list[Citation]
