import pytest

from app.rag.chunker import PageText, chunk_pages


def test_chunk_pages_preserves_page_and_overlap() -> None:
    chunks = chunk_pages([PageText(page=3, text="abcdefghij")], chunk_size=5, overlap=2)

    assert [chunk.page for chunk in chunks] == [3, 3, 3]
    assert [chunk.text for chunk in chunks] == ["abcde", "defgh", "ghij"]


def test_chunk_pages_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_pages([PageText(page=1, text="abc")], chunk_size=5, overlap=5)
