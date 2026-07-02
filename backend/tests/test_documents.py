import uuid
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from app.api.v1 import documents
from app.db.models import ChatSession, Document


class FakeSession:
    def __init__(self) -> None:
        self.document = Document(
            id=uuid.uuid4(),
            name="upload.pdf",
            storage_path="uploads/upload.pdf",
        )
        self.rollback_called = False
        self.refresh_called = False
        self.added = []

    async def get(self, model, item_id):
        if model is ChatSession:
            return ChatSession(id=item_id)
        return None

    def add(self, value) -> None:
        self.added.append(value)
        if isinstance(value, Document):
            self.document = value

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        self.rollback_called = True

    async def refresh(self, document) -> None:
        self.refresh_called = True


def test_safe_pdf_name_rejects_non_pdf() -> None:
    with pytest.raises(HTTPException):
        documents._safe_pdf_name("notes.txt")


class FakeStorage:
    def __init__(self) -> None:
        self.saved: list[tuple[str, bytes]] = []

    def save(self, key: str, data: bytes, content_type: str = "application/pdf") -> None:
        self.saved.append((key, data))


@pytest.mark.asyncio
async def test_upload_document_rolls_back_failed_ingest(monkeypatch) -> None:
    fake_session = FakeSession()
    fake_storage = FakeStorage()
    upload_file = Mock()
    upload_file.filename = "upload.pdf"
    upload_file.content_type = "application/pdf"
    upload_file.read = AsyncMock(return_value=b"%PDF-1.4")

    async def fail_ingest(*_args) -> None:
        raise RuntimeError("ingest failed")

    monkeypatch.setattr(documents, "get_storage", lambda: fake_storage)
    monkeypatch.setattr(documents, "ingest_document", fail_ingest)

    result = await documents.upload_document(uuid.uuid4(), fake_session, upload_file)

    assert result is fake_session.document
    assert fake_session.rollback_called is True
    assert fake_session.refresh_called is True
    # PDF was persisted to storage before ingest ran.
    assert fake_storage.saved and fake_storage.saved[0][1] == b"%PDF-1.4"
