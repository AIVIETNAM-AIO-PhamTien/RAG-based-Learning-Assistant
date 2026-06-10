from functools import lru_cache

from src.llm import LearningLLM
from src.retriever import Retriever
from src.schemas import AskResponse


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
