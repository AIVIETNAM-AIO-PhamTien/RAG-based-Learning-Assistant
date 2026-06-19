from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import get_settings
from src.indexing import build_index
from src.learning import answer_question, generate_flashcards, get_retriever, summarize
from src.schemas import AskRequest, AskResponse, FlashcardsResponse, IndexResponse, StudyRequest, SummaryResponse


settings = get_settings()
app = FastAPI(title=settings.app_name)
app.mount("/static", StaticFiles(directory="src/interface/static"), name="static")
templates = Jinja2Templates(directory="src/interface/templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "app_name": settings.app_name})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/index", response_model=IndexResponse)
def index_documents():
    try:
        result = build_index()
        get_retriever.cache_clear()
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    try:
        return answer_question(payload.question, top_k=payload.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/summary", response_model=SummaryResponse)
def summary(payload: StudyRequest):
    return summarize(payload.topic, top_k=payload.top_k)


@app.post("/api/flashcards", response_model=FlashcardsResponse)
def flashcards(payload: StudyRequest):
    return generate_flashcards(payload.topic, top_k=payload.top_k)
