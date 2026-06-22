from src.indexing import split_text


def preview_chunks(text: str, chunk_size: int = 900, overlap: int = 160) -> list[str]:
    return split_text(text, chunk_size=chunk_size, overlap=overlap)
