from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    id: str
    source: str
    page: int
    text: str
    score: float = 0.0


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int | None = Field(default=None, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class IndexResponse(BaseModel):
    chunks: int
    documents: int
    index_dir: str
