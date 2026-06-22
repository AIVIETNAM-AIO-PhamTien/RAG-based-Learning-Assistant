from sklearn.metrics.pairwise import cosine_similarity

from src.config import get_settings
from src.filters import normalize_query
from src.schemas import SourceChunk
from src.store import LocalVectorStore


class Retriever:
    def __init__(self):
        settings = get_settings()
        self.vectorizer, self.matrix, self.chunks = LocalVectorStore(settings.index_dir).load()

    def search(self, query: str, top_k: int | None = None) -> list[SourceChunk]:
        settings = get_settings()
        limit = top_k or settings.top_k
        query_vector = self.vectorizer.transform([normalize_query(query)])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indexes = scores.argsort()[::-1][:limit]
        results: list[SourceChunk] = []
        for index in ranked_indexes:
            chunk = self.chunks[int(index)].model_copy()
            chunk.score = float(scores[index])
            if chunk.score > 0:
                results.append(chunk)
        return results
