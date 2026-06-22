from functools import lru_cache

from src.llm import LearningLLM
from src.retriever import Retriever
from src.schemas import AskResponse, Flashcard, FlashcardsResponse, SummaryResponse


@lru_cache
def get_retriever() -> Retriever:
    return Retriever()


@lru_cache
def get_llm() -> LearningLLM:
    return LearningLLM()


def answer_question(question: str, top_k: int | None = None) -> AskResponse:
    sources = get_retriever().search(question, top_k=top_k)
    answer = get_llm().answer(question, sources)
    return AskResponse(answer=answer, sources=sources)


def summarize(topic: str, top_k: int | None = None) -> SummaryResponse:
    sources = get_retriever().search(topic, top_k=top_k)
    return SummaryResponse(summary=get_llm().summarize(sources), sources=sources)


def generate_flashcards(topic: str, top_k: int | None = None) -> FlashcardsResponse:
    sources = get_retriever().search(topic, top_k=top_k)
    generated = get_llm().flashcards(sources)
    cards: list[Flashcard] = []
    question: str | None = None
    for line in generated.splitlines():
        if line.startswith("Q:"):
            question = line[2:].strip()
        elif line.startswith("A:") and question:
            cards.append(Flashcard(question=question, answer=line[2:].strip()))
            question = None
    if not cards and sources:
        cards = [Flashcard(question="Ý chính của đoạn này là gì?", answer=source.text[:300]) for source in sources]
    return FlashcardsResponse(flashcards=cards, sources=sources)
