import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.api.v1.chat import _build_session_title, _recent_messages
from app.db.models import ChatMessage
from app.rag.prompts import build_prompt
from app.schemas.chat import Citation


class FakeScalarResult:
    def __init__(self, values):
        self._values = values

    def all(self):
        return self._values


class FakeSession:
    def __init__(self, messages: list[ChatMessage]) -> None:
        self._messages = messages

    async def scalars(self, _statement):
        return FakeScalarResult(self._messages)


@pytest.mark.asyncio
async def test_recent_messages_returns_last_three_in_chronological_order() -> None:
    session_id = uuid.uuid4()
    base_time = datetime(2026, 6, 20, tzinfo=UTC)
    messages = [
        ChatMessage(
            id=uuid.uuid4(),
            session_id=session_id,
            role="user",
            content="first",
            created_at=base_time,
        ),
        ChatMessage(
            id=uuid.uuid4(),
            session_id=session_id,
            role="assistant",
            content="second",
            created_at=base_time + timedelta(seconds=1),
        ),
        ChatMessage(
            id=uuid.uuid4(),
            session_id=session_id,
            role="user",
            content="third",
            created_at=base_time + timedelta(seconds=2),
        ),
    ]

    result = await _recent_messages(FakeSession(list(reversed(messages))), session_id)

    assert [message.content for message in result] == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_recent_messages_returns_all_available_messages() -> None:
    session_id = uuid.uuid4()
    messages = [
        ChatMessage(
            id=uuid.uuid4(),
            session_id=session_id,
            role="user",
            content="latest question",
            created_at=datetime(2026, 6, 20, tzinfo=UTC),
        )
    ]

    result = await _recent_messages(FakeSession(messages), session_id)

    assert [message.content for message in result] == ["latest question"]


def test_build_prompt_includes_recent_messages_before_question() -> None:
    session_id = uuid.uuid4()
    citations = [
        Citation(
            index=1,
            chunk_id=uuid.uuid4(),
            doc_id=uuid.uuid4(),
            doc_name="notes.pdf",
            page=4,
            text="TCP slow start increases the congestion window exponentially.",
            snippet="TCP slow start increases...",
        )
    ]
    recent_messages = [
        ChatMessage(
            id=uuid.uuid4(),
            session_id=session_id,
            role="user",
            content="Explain TCP slow start",
            created_at=datetime(2026, 6, 20, tzinfo=UTC),
        ),
        ChatMessage(
            id=uuid.uuid4(),
            session_id=session_id,
            role="assistant",
            content="It increases the congestion window quickly at the start.",
            created_at=datetime(2026, 6, 20, tzinfo=UTC),
        ),
        ChatMessage(
            id=uuid.uuid4(),
            session_id=session_id,
            role="user",
            content="When does it stop?",
            created_at=datetime(2026, 6, 20, tzinfo=UTC),
        ),
    ]

    prompt = build_prompt("What is it?", citations, recent_messages)

    assert "Recent conversation:" in prompt
    assert "- user: Explain TCP slow start" in prompt
    assert "- assistant: It increases the congestion window quickly at the start." in prompt
    assert "- user: When does it stop?" in prompt
    assert prompt.index("Recent conversation:") < prompt.index("Context chunks:")
    assert "Question: What is it?" in prompt
    assert "[1] notes.pdf, page 4" in prompt


def test_build_prompt_omits_recent_conversation_when_history_is_empty() -> None:
    citations = [
        Citation(
            index=1,
            chunk_id=uuid.uuid4(),
            doc_id=uuid.uuid4(),
            doc_name="notes.pdf",
            page=2,
            text="A socket is an endpoint for communication.",
            snippet="A socket is an endpoint...",
        )
    ]

    prompt = build_prompt("What is a socket?", citations, [])

    assert "Recent conversation:" not in prompt
    assert "Context chunks:" in prompt
    assert "Question: What is a socket?" in prompt


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Explain TCP slow start", "Explain TCP slow start"),
        ("   \n  ", "New chat"),
        (
            "This is a very long first message that should become a short session title without cutting in the middle of a word or growing too large for the sidebar",
            "This is a very long first message that should become a short...",
        ),
    ],
)
def test_build_session_title_normalizes_and_truncates_message(message: str, expected: str) -> None:
    assert _build_session_title(message) == expected


def test_build_session_title_caps_length_to_two_hundred_characters() -> None:
    result = _build_session_title("word " * 60)

    assert len(result) <= 200
    assert result.endswith("...")
    assert "\n" not in result
    assert "  " not in result