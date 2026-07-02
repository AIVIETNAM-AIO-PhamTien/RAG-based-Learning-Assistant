from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from app.config import get_settings
from app.db.models import ChatMessage
from app.rag.prompts import (
    FLASHCARDS_SYSTEM_PROMPT,
    FLASHCARD_NOTES_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_flashcard_notes_prompt,
    build_flashcards_from_notes_prompt,
    build_prompt,
    build_summary_prompt,
)
from app.schemas.chat import Citation


class GenerationConfigError(RuntimeError):
    pass


async def stream_answer(
    question: str,
    citations: list[Citation],
    recent_messages: list[ChatMessage] | None = None,
) -> AsyncIterator[str]:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise GenerationConfigError("GEMINI_API_KEY is required for chat generation")

    client = genai.Client(api_key=settings.gemini_api_key)
    prompt = build_prompt(question, citations, recent_messages)
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
    stream = await client.aio.models.generate_content_stream(
        model=settings.gemini_model,
        contents=prompt,
        config=config,
    )
    async for chunk in stream:
        if chunk.text:
            yield chunk.text


async def generate_summary(citations: list[Citation]) -> str:
    prompt = build_summary_prompt(citations)
    return await _generate_text(prompt, SUMMARY_SYSTEM_PROMPT)


async def generate_flashcard_notes(citations: list[Citation]) -> str:
    prompt = build_flashcard_notes_prompt(citations)
    return await _generate_text(prompt, FLASHCARD_NOTES_SYSTEM_PROMPT)


async def generate_flashcards_from_notes(
    notes_context: str,
    flashcard_count: int,
    coverage_hint: str,
) -> str:
    prompt = build_flashcards_from_notes_prompt(notes_context, flashcard_count, coverage_hint)
    return await _generate_text(prompt, FLASHCARDS_SYSTEM_PROMPT)


async def _generate_text(prompt: str, system_prompt: str) -> str:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise GenerationConfigError("GEMINI_API_KEY is required for study generation")

    client = genai.Client(api_key=settings.gemini_api_key)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=config,
    )
    return response.text or ""
