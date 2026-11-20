
from .models import DocumentPage, DocumentChunk


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list[DocumentChunk]:
    """
    Split extracted PDF pages into overlapping chunks.

    Args:
        pages: Extracted PDF pages.
        chunk_size: Maximum approximate chunk size in characters.
        chunk_overlap: Number of overlapping characters.

    Returns:
        A list of DocumentChunk objects.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    all_chunks = []

    for page in pages:

        text = page.text.strip()

        if not text:
            continue

        start = 0
        chunk_index = 1

        while start < len(text):

            end = min(
                start + chunk_size,
                len(text)
            )

            # Prefer ending at a word boundary
            if end < len(text):

                boundary = text.rfind(
                    " ",
                    start,
                    end
                )

                if boundary > start + chunk_size // 2:
                    end = boundary

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunk = DocumentChunk(
                    document_id=page.document_id,
                    chunk_id=(
                        f"{page.document_id}_"
                        f"p{page.page_number:03d}_"
                        f"c{chunk_index:03d}"
                    ),
                    source=page.source,
                    page_number=page.page_number,
                    text=chunk_text
                )

                all_chunks.append(chunk)

                chunk_index += 1

            if end >= len(text):
                break

            next_start = end - chunk_overlap

            if next_start <= start:
                next_start = start + 1

            start = next_start

    return all_chunks
