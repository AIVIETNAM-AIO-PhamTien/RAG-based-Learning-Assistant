from pathlib import Path

import joblib

from src.schemas import SourceChunk


class LocalVectorStore:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.vectorizer_path = index_dir / "vectorizer.joblib"
        self.matrix_path = index_dir / "matrix.joblib"
        self.chunks_path = index_dir / "chunks.joblib"

    def exists(self) -> bool:
        return self.vectorizer_path.exists() and self.matrix_path.exists() and self.chunks_path.exists()

    def save(self, vectorizer, matrix, chunks: list[SourceChunk]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, self.vectorizer_path)
        joblib.dump(matrix, self.matrix_path)
        joblib.dump([chunk.model_dump() for chunk in chunks], self.chunks_path)

    def load(self):
        if not self.exists():
            raise FileNotFoundError(
                f"Index not found in {self.index_dir}. Run: python -m src.indexing"
            )
        vectorizer = joblib.load(self.vectorizer_path)
        matrix = joblib.load(self.matrix_path)
        chunks = [SourceChunk(**item) for item in joblib.load(self.chunks_path)]
        return vectorizer, matrix, chunks
