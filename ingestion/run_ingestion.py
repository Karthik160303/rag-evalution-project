
from ingestion.document_loader import load_pdf
from ingestion.chunking import chunk_pages


def main():

    pdf_path = "data/raw/docker_guide.pdf"

    document_id = "docker001"

    # Step 1: Load PDF
    print("\nLoading PDF...")

    pages = load_pdf(
        pdf_path=pdf_path,
        document_id=document_id
    )

    print(f"Extracted pages: {len(pages)}")

    # Step 2: Generate chunks
    print("\nGenerating chunks...")

    chunks = chunk_pages(
        pages=pages,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"Generated chunks: {len(chunks)}")

    # Step 3: Display sample chunks
    print("\nSample chunks:\n")

    for chunk in chunks[:3]:

        print("-" * 60)

        print(f"Chunk ID: {chunk.chunk_id}")

        print(f"Document ID: {chunk.document_id}")

        print(f"Page Number: {chunk.page_number}")

        print(f"Text: {chunk.text[:200]}")

        print()


if __name__ == "__main__":
    main()
