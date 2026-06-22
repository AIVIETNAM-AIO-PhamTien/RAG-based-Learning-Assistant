from fastapi import APIRouter

from app.api.v1.chat import router as chat_router
from app.api.v1.documents import router as documents_router
from app.api.v1.flashcards import router as flashcards_router
from app.api.v1.health import router as health_router
from app.api.v1.sessions import router as sessions_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(sessions_router)
api_router.include_router(documents_router)
api_router.include_router(chat_router)
api_router.include_router(flashcards_router)
