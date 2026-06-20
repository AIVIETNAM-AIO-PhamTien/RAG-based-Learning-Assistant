from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from app.config import get_settings
from app.db.models import ChatMessage
from app.rag.prompts import SYSTEM_PROMPT, build_prompt
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
