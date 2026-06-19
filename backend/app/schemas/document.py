from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class DocumentRead(BaseModel):
    id: UUID
    name: str
    status: str
    page_count: int | None = None
    error_message: str | None = None
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    id: UUID
    name: str
    status: str
    page_count: int | None = None
    error_message: str | None = None
