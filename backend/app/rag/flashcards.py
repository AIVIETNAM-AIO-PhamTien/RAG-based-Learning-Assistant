import json

from google import genai
from google.genai import types

from app.config import get_settings
from app.rag.generator import GenerationConfigError
from app.rag.prompts import build_flashcards_prompt
from app.schemas.chat import Citation


class FlashcardGenerationError(RuntimeError):
    pass


def parse_flashcards(
    payload: str, citations: list[Citation], count: int
) -> list[dict[str, object]]:
    try:
        generated = json.loads(payload).get("flashcards", [])
    except json.JSONDecodeError as exc:
        raise FlashcardGenerationError("Model returned invalid flashcard JSON") from exc
    if not isinstance(generated, list) or len(generated) != count:
        raise FlashcardGenerationError(f"Model must return exactly {count} flashcards")

    sources = {citation.index: citation for citation in citations}
    cards: list[dict[str, object]] = []
    for item in generated:
        if not isinstance(item, dict):
            raise FlashcardGenerationError("Each flashcard must be an object")
        question, answer, source_index = (
            item.get("question"),
            item.get("answer"),
            item.get("source_index"),
        )
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(answer, str)
            or not answer.strip()
        ):
            raise FlashcardGenerationError("Flashcards require a question and answer")
        source = sources.get(source_index)
        if source is None:
            raise FlashcardGenerationError("Flashcard references an unavailable source")
        cards.append({"question": question.strip(), "answer": answer.strip(), "source": source})
    return cards


async def generate_flashcards(
    topic: str, count: int, citations: list[Citation]
) -> list[dict[str, object]]:
    if not citations:
        raise FlashcardGenerationError("No relevant document context was found for this topic")
    settings = get_settings()
    if not settings.gemini_api_key:
        raise GenerationConfigError("GEMINI_API_KEY is required for flashcard generation")
    client = genai.Client(api_key=settings.gemini_api_key)
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=build_flashcards_prompt(topic, count, citations),
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return parse_flashcards(response.text or "", citations, count)
