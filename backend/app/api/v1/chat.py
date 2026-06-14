import json
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatMessage, ChatSession
from app.db.session import async_session_factory, get_session
from app.rag.generator import stream_answer
from app.rag.metrics import citation_coverage
from app.rag.retriever import retrieve_top_k
from app.schemas.chat import ChatRequest

router = APIRouter(prefix="/api/v1/sessions", tags=["chat"])
SessionDep = Annotated[AsyncSession, Depends(get_session)]


def _sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/{session_id}/chat")
async def chat(
    session_id: uuid.UUID, payload: ChatRequest, session: SessionDep
) -> StreamingResponse:
    chat_session = await session.get(ChatSession, session_id)
    if chat_session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")

    session.add(ChatMessage(session_id=session_id, role="user", content=payload.message))
    await session.commit()

    async def events() -> AsyncIterator[str]:
        answer_parts: list[str] = []
        citations_payload: list[dict[str, object]] = []
        try:
            async with async_session_factory() as stream_session:
                citations = await retrieve_top_k(stream_session, session_id, payload.message)
                citations_payload = [citation.model_dump(mode="json") for citation in citations]

            if not citations:
                text = "I could not find relevant context in the uploaded documents."
                answer_parts.append(text)
                yield _sse("token", {"text": text})
            else:
                async for token in stream_answer(payload.message, citations):
                    answer_parts.append(token)
                    yield _sse("token", {"text": token})

            answer = "".join(answer_parts)
            coverage = citation_coverage(answer, {item["index"] for item in citations_payload})
            async with async_session_factory() as write_session:
                write_session.add(
                    ChatMessage(
                        session_id=session_id,
                        role="assistant",
                        content=answer,
                        citations=citations_payload,
                    )
                )
                await write_session.commit()

            yield _sse("citations", {"citations": citations_payload, "citation_coverage": coverage})
            yield _sse("done", {})
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(events(), media_type="text/event-stream")
