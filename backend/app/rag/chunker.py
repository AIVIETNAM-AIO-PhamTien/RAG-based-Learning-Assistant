from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    page: int
    text: str
    parent_text: str | None = None


def chunk_pages(pages: list[PageText], chunk_size: int, overlap: int) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[TextChunk] = []
    step = chunk_size - overlap
    for page in pages:
        text = " ".join(page.text.split())
        if not text:
            continue
        for start in range(0, len(text), step):
            chunk_text = text[start : start + chunk_size].strip()
            if chunk_text:
                chunks.append(TextChunk(page=page.page, text=chunk_text, parent_text=chunk_text))
            if start + chunk_size >= len(text):
                break
    return chunks
