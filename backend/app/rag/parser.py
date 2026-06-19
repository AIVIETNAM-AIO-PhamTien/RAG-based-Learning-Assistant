from pathlib import Path

import fitz

from app.rag.chunker import PageText


class PdfParseError(ValueError):
    pass


def parse_pdf_text(path: Path) -> tuple[list[PageText], int]:
    try:
        document = fitz.open(path)
    except Exception as exc:
        raise PdfParseError(f"Could not open PDF: {exc}") from exc

    with document:
        if document.is_encrypted:
            raise PdfParseError("Encrypted PDFs are not supported in Phase 1")

        pages: list[PageText] = []
        for index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(PageText(page=index, text=text))

        if not pages:
            raise PdfParseError("No text layer found in PDF")

        return pages, document.page_count
