from pathlib import Path

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import get_settings
from src.filters import clean_text, is_useful_text
from src.schemas import IndexResponse, SourceChunk
from src.store import LocalVectorStore


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if is_useful_text(chunk):
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def load_pdf_chunks(data_dir: Path, chunk_size: int, overlap: int) -> list[SourceChunk]:
    chunks: list[SourceChunk] = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for page_index, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            for chunk_index, chunk_text in enumerate(split_text(text, chunk_size, overlap), start=1):
                chunks.append(
                    SourceChunk(
                        id=f"{pdf_path.stem}:p{page_index}:c{chunk_index}",
                        source=pdf_path.name,
                        page=page_index,
                        text=chunk_text,
                    )
                )
    return chunks


def build_index() -> IndexResponse:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    chunks = load_pdf_chunks(settings.data_dir, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise RuntimeError(f"No readable PDF chunks found in {settings.data_dir.resolve()}")

    vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1, 2), max_features=50000)
    matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
    LocalVectorStore(settings.index_dir).save(vectorizer, matrix, chunks)
    return IndexResponse(
        chunks=len(chunks),
        documents=len({chunk.source for chunk in chunks}),
        index_dir=str(settings.index_dir),
    )


if __name__ == "__main__":
    result = build_index()
    print(result.model_dump_json(indent=2))
