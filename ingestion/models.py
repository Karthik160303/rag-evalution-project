
from pydantic import BaseModel


class Document(BaseModel):
    """Represents an original uploaded document."""

    document_id: str
    filename: str
    source: str
    content: str


class DocumentPage(BaseModel):
    """Represents extracted text from one PDF page."""

    document_id: str
    source: str
    page_number: int
    text: str


class DocumentChunk(BaseModel):
    """Represents a smaller piece of a document."""

    document_id: str
    chunk_id: str
    source: str
    page_number: int | None = None
    text: str
    embedding: list[float] | None = None


class RetrievedChunk(BaseModel):
    """Represents a chunk returned by the retriever."""

    chunk_id: str
    document_id: str
    source: str
    page_number: int | None = None
    text: str
    score: float
