import uuid
from datetime import UTC, datetime

import pytest
from fastapi import HTTPException

from app.api.v1.sessions import (
    create_chat_session,
    delete_chat_session,
    list_chat_sessions,
    list_session_documents,
    list_session_messages,
    rename_chat_session,
)
from app.db.models import ChatMessage, ChatSession
from app.schemas.chat import ChatSessionCreate, ChatSessionUpdate


class FakeScalarResult:
    def __init__(self, values):
        self._values = values

    def all(self):
        return self._values


class FakeSession:
    def __init__(
        self,
        chat_session: ChatSession | None,
        messages: list[ChatMessage],
        sessions: list[ChatSession] | None = None,
        documents: list[object] | None = None,
    ) -> None:
        self._chat_session = chat_session
        self._messages = messages
        self._sessions = sessions or []
        self._documents = documents or []
        self.added: list[ChatSession] = []
        self.deleted: list[ChatSession] = []
        self.committed = False
        self.refreshed: list[ChatSession] = []

    def add(self, value: ChatSession) -> None:
        self.added.append(value)

    async def commit(self) -> None:
        self.committed = True

    async def refresh(self, value: ChatSession) -> None:
        self.refreshed.append(value)

    async def delete(self, value: ChatSession) -> None:
        self.deleted.append(value)

    async def get(self, model, item_id):
        if model is ChatSession and self._chat_session and self._chat_session.id == item_id:
            return self._chat_session
        return None

    async def scalars(self, statement):
        statement_text = str(statement)
        if "FROM chat_sessions" in statement_text and "chat_messages" not in statement_text:
            return FakeScalarResult(self._sessions)
        if "chat_session_documents" in statement_text:
            return FakeScalarResult(self._documents)
        return FakeScalarResult(self._messages)


@pytest.mark.asyncio
async def test_create_chat_session_creates_and_returns_session() -> None:
    fake_session = FakeSession(None, [])

    result = await create_chat_session(ChatSessionCreate(title="New session"), fake_session)

    assert fake_session.committed is True
    assert fake_session.added == [result]
    assert fake_session.refreshed == [result]
    assert result.title == "New session"


@pytest.mark.asyncio
async def test_create_chat_session_allows_null_title() -> None:
    fake_session = FakeSession(None, [])

    result = await create_chat_session(ChatSessionCreate(title=None), fake_session)

    assert fake_session.committed is True
    assert fake_session.added == [result]
    assert fake_session.refreshed == [result]
    assert result.title is None


@pytest.mark.asyncio
async def test_list_chat_sessions_returns_sessions_in_order() -> None:
    older_session = ChatSession(
        id=uuid.uuid4(),
        title="Older session",
        created_at=datetime(2026, 6, 19, tzinfo=UTC),
    )
    newer_session = ChatSession(
        id=uuid.uuid4(),
        title="Newer session",
        created_at=datetime(2026, 6, 20, tzinfo=UTC),
    )

    result = await list_chat_sessions(FakeSession(None, [], sessions=[newer_session, older_session]))

    assert result == [newer_session, older_session]


@pytest.mark.asyncio
async def test_list_chat_sessions_returns_empty_list() -> None:
    result = await list_chat_sessions(FakeSession(None, [], sessions=[]))

    assert result == []


@pytest.mark.asyncio
async def test_rename_chat_session_updates_title_and_returns_session() -> None:
    session_id = uuid.uuid4()
    chat_session = ChatSession(id=session_id, title="Before")
    fake_session = FakeSession(chat_session, [])

    result = await rename_chat_session(session_id, ChatSessionUpdate(title="After"), fake_session)

    assert result is chat_session
    assert result.title == "After"
    assert fake_session.committed is True
    assert fake_session.refreshed == [chat_session]


@pytest.mark.asyncio
async def test_rename_chat_session_raises_for_missing_session() -> None:
    with pytest.raises(HTTPException) as caught:
        await rename_chat_session(uuid.uuid4(), ChatSessionUpdate(title="After"), FakeSession(None, []))

    assert caught.value.status_code == 404
    assert caught.value.detail == "Chat session not found"


@pytest.mark.asyncio
async def test_delete_chat_session_deletes_existing_session() -> None:
    session_id = uuid.uuid4()
    chat_session = ChatSession(id=session_id, title="To delete")
    fake_session = FakeSession(chat_session, [])

    response = await delete_chat_session(session_id, fake_session)

    assert response.status_code == 204
    assert fake_session.deleted == [chat_session]
    assert fake_session.committed is True


@pytest.mark.asyncio
async def test_delete_chat_session_raises_for_missing_session() -> None:
    with pytest.raises(HTTPException) as caught:
        await delete_chat_session(uuid.uuid4(), FakeSession(None, []))

    assert caught.value.status_code == 404
    assert caught.value.detail == "Chat session not found"


@pytest.mark.asyncio
async def test_list_session_documents_raises_for_missing_session() -> None:
    with pytest.raises(HTTPException) as caught:
        await list_session_documents(uuid.uuid4(), FakeSession(None, []))

    assert caught.value.status_code == 404
    assert caught.value.detail == "Chat session not found"


@pytest.mark.asyncio
async def test_list_session_messages_returns_messages_for_existing_session() -> None:
    session_id = uuid.uuid4()
    chat_session = ChatSession(id=session_id)
    created_at = datetime.now(UTC)
    messages = [
        ChatMessage(
            id=uuid.uuid4(),
            session_id=session_id,
            role="user",
            content="hello",
            created_at=created_at,
        ),
        ChatMessage(
            id=uuid.uuid4(),
            session_id=session_id,
            role="assistant",
            content="hi",
            created_at=created_at,
        ),
    ]

    result = await list_session_messages(session_id, FakeSession(chat_session, messages))

    assert result == messages


@pytest.mark.asyncio
async def test_list_session_messages_raises_for_missing_session() -> None:
    with pytest.raises(HTTPException) as caught:
        await list_session_messages(uuid.uuid4(), FakeSession(None, []))

    assert caught.value.status_code == 404
    assert caught.value.detail == "Chat session not found"
