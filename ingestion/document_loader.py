from pathlib import Path

import pymupdf

from .models import DocumentPage
from .text_cleaner import clean_text


def load_pdf(
    pdf_path: str,
    document_id: str
) -> list[DocumentPage]:

    """Extract text from a PDF page by page."""

    path = Path(pdf_path)

    if not document_id or not document_id.strip():
        raise ValueError("Document ID cannot be empty.")

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file, received: {pdf_path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, received: {path.suffix}"
        )

    extracted_pages = []

    with pymupdf.open(str(path)) as pdf:
        for page_index, page in enumerate(pdf):

            raw_text = page.get_text("text")
            cleaned_text = clean_text(raw_text)

            if not cleaned_text:
                continue

            page_data = DocumentPage(
                document_id=document_id,
                source=path.name,
                page_number=page_index + 1,
                text=cleaned_text
            )

            extracted_pages.append(page_data)

    return extracted_pages