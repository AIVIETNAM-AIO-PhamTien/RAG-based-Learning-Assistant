from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from app.config import get_settings

EMBEDDING_DIMENSIONS = 384


class EmbeddingError(RuntimeError):
    pass


class E5SmallV2Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingError(f"Failed to load embedding model {model_name}: {exc}") from exc

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            passages = [f"passage: {text}" for text in texts]
            result = self.model.encode(passages, batch_size=12, normalize_embeddings=False)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed text with {self.model_name}: {exc}") from exc

        return self._normalize(result)

    def embed_query(self, text: str) -> list[float]:
        try:
            result = self.model.encode([f"query: {text}"], batch_size=1, normalize_embeddings=False)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed query with {self.model_name}: {exc}") from exc

        return self._normalize(result)[0]

    def _normalize(self, embeddings: object) -> list[list[float]]:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != EMBEDDING_DIMENSIONS:
            message = (
                f"Expected {EMBEDDING_DIMENSIONS}d embeddings from {self.model_name}, "
                f"got {vectors.shape[1]}d"
            )
            raise EmbeddingError(message)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, a_min=1e-12, a_max=None)
        return vectors.tolist()


@lru_cache
def get_embedder() -> E5SmallV2Embedder:
    return E5SmallV2Embedder(get_settings().embedding_model_name)
