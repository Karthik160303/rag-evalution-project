
import pymupdf

from ingestion.document_loader import load_pdf
from ingestion.text_cleaner import clean_text
from ingestion.chunking import chunk_pages
from ingestion.models import DocumentPage


def test_clean_text():

    raw_text = "Docker     is   a\n\n\ncontainer platform."

    cleaned = clean_text(raw_text)

    assert cleaned == "Docker is a\n\ncontainer platform."


def test_chunk_pages():

    page = DocumentPage(
        document_id="doc001",
        source="docker.pdf",
        page_number=1,
        text="Docker is a containerization platform. " * 100
    )

    chunks = chunk_pages(
        pages=[page],
        chunk_size=100,
        chunk_overlap=20
    )

    assert len(chunks) > 1

    assert chunks[0].document_id == "doc001"

    assert chunks[0].page_number == 1

    assert chunks[0].embedding is None

    assert chunks[0].chunk_id == "doc001_p001_c001"


def test_load_pdf(tmp_path):

    pdf_path = tmp_path / "sample.pdf"

    # Create a temporary PDF for testing
    pdf = pymupdf.open()

    page = pdf.new_page()

    page.insert_text(
        (72, 72),
        "Docker is a containerization platform."
    )

    pdf.save(str(pdf_path))

    pdf.close()

    # Test our PDF loader
    pages = load_pdf(
        pdf_path=str(pdf_path),
        document_id="doc001"
    )

    assert len(pages) == 1

    assert pages[0].document_id == "doc001"

    assert pages[0].page_number == 1

    assert "Docker" in pages[0].text
