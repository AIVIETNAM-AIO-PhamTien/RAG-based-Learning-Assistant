from collections.abc import Sequence
from functools import lru_cache


class RerankError(RuntimeError):
    pass


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
        except Exception as exc:
            raise RerankError(f"Failed to load reranker model {model_name}: {exc}") from exc

    def rerank(self, query: str, documents: Sequence[str], top_k: int) -> list[tuple[int, float]]:
        """Return (original_index, score) pairs sorted by relevance, truncated to top_k."""
        if not documents:
            return []
        try:
            pairs = [(query, document) for document in documents]
            scores = self.model.predict(pairs)
        except Exception as exc:
            raise RerankError(f"Failed to rerank with {self.model_name}: {exc}") from exc

        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [(index, float(score)) for index, score in ranked[:top_k]]


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    from app.config import get_settings

    return CrossEncoderReranker(get_settings().rerank_model_name)
