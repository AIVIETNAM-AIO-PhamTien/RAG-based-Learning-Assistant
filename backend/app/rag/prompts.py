from app.db.models import ChatMessage
from app.schemas.chat import Citation

SYSTEM_PROMPT = """You answer questions using only the provided document chunks.
Cite every factual claim that uses a chunk with its citation index like [1].
Only use citation indexes that appear in the context.
If the answer is not in the chunks, say you could not find it in the uploaded documents.
"""


def build_context(citations: list[Citation]) -> str:
    return "\n\n".join(
        f"[{item.index}] {item.doc_name}, page {item.page}\n{item.text}" for item in citations
    )


def build_recent_conversation(recent_messages: list[ChatMessage] | None) -> str:
    if not recent_messages:
        return ""

    lines = [f"- {message.role}: {message.content}" for message in recent_messages]
    return "Recent conversation:\n" + "\n".join(lines) + "\n\n"


def build_prompt(
    question: str,
    citations: list[Citation],
    recent_messages: list[ChatMessage] | None = None,
) -> str:
    context = build_context(citations)
    recent_conversation = build_recent_conversation(recent_messages)
    return f"""{recent_conversation}Context chunks:
{context}

Question: {question}

Answer with inline citations like [1], [2]."""
