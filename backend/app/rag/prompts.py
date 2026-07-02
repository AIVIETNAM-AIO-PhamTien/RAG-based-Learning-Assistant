from app.db.models import ChatMessage
from app.schemas.chat import Citation

SYSTEM_PROMPT = """You answer questions using only the provided document chunks.
Cite every factual claim that uses a chunk with its citation index like [1].
When a claim draws on multiple chunks, cite each index separately right after
another, like [1][2] — never combine them into one bracket like [1, 2].
Only use citation indexes that appear in the context.
If the answer is not in the chunks, say you could not find it in the uploaded documents.
"""

SUMMARY_SYSTEM_PROMPT = """You create concise study summaries using only the provided document chunks.
Do not introduce facts that are not present in the chunks.
If the chunks are insufficient, say so clearly.
"""

FLASHCARD_NOTES_SYSTEM_PROMPT = """You create short study notes using only the provided document chunks.
Return concise factual notes.
Prefer short bullets.
Preserve the source language when possible.
Do not introduce facts that are not present in the chunks.
"""

FLASHCARDS_SYSTEM_PROMPT = """You create study flashcards using only the provided document notes.
Return only Q/A pairs in the exact format:
Q: ...
A: ...
Do not add numbering, bullets, or extra commentary.
"""

SUMMARY_FALLBACK = "I could not find relevant context in the uploaded documents."


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

Answer with inline citations like [1] or, when citing multiple sources for one
claim, [1][2] (never [1, 2])."""
Answer with inline citations like [1], [2]."""


def build_summary_prompt(citations: list[Citation]) -> str:
    context = build_context(citations)
    return f"""Context chunks:
{context}

Summarize the material into concise study notes.
Prefer short bullets and preserve the source language when possible."""


def build_flashcard_notes_prompt(citations: list[Citation]) -> str:
    context = build_context(citations)
    return f"""Context chunks:
{context}

Write short study notes for this ordered batch.
Use 3-6 short bullets when possible.
Preserve the source language when possible.
Return only the notes."""


def build_flashcards_from_notes_prompt(
    notes_context: str,
    flashcard_count: int,
    coverage_hint: str,
) -> str:
    return f"""Document study notes:
{notes_context}

Coverage targets:
{coverage_hint}

Create exactly {flashcard_count} study flashcards from the material.
Keep coverage fair across documents.
Preserve the source language when possible.
Return only lines in this exact format:
Q: ...
A: ..."""
