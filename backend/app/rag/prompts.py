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


def build_prompt(question: str, citations: list[Citation]) -> str:
    context = build_context(citations)
    return f"""Context chunks:
{context}

Question: {question}

Answer with inline citations like [1], [2]."""


def build_flashcards_prompt(topic: str, count: int, citations: list[Citation]) -> str:
    return f"""Create exactly {count} concise Vietnamese study flashcards about: {topic}.
Use only the context below. Return JSON only in this shape:
{{"flashcards":[{{"question":"...","answer":"...","source_index":1}}]}}
Each source_index must refer to exactly one context index. Keep answers under 80 words.

Context:
{build_context(citations)}"""
