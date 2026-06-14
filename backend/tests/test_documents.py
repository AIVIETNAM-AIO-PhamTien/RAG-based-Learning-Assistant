import uuid
from pathlib import Path
from unittest.mock import Mock

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


@pytest.mark.asyncio
async def test_upload_document_rolls_back_failed_ingest(monkeypatch, tmp_path: Path) -> None:
    fake_session = FakeSession()
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"%PDF-1.4")
    upload_file = Mock()
    upload_file.filename = "upload.pdf"
    upload_file.content_type = "application/pdf"
    upload_file.file = source_path.open("rb")

    async def fail_ingest(*_args) -> None:
        raise RuntimeError("ingest failed")

    settings = Mock()
    settings.upload_dir = tmp_path
    monkeypatch.setattr(documents, "get_settings", lambda: settings)
    monkeypatch.setattr(documents, "ingest_document", fail_ingest)

    try:
        result = await documents.upload_document(uuid.uuid4(), fake_session, upload_file)
    finally:
        upload_file.file.close()

    assert result is fake_session.document
    assert fake_session.rollback_called is True
    assert fake_session.refresh_called is True
