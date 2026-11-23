
from pathlib import Path

from ingestion.document_loader import load_pdf
from ingestion.chunking import chunk_pages
from ingestion.embedding import EmbeddingService

from vectorstore.qdrant_store import QdrantVectorStore
from vectorstore.indexing import DocumentIndexer

from retrieval.retriever import Retriever


# PDF configuration
PDF_PATH = Path("data/raw/docker_guide.pdf")
DOCUMENT_ID = "docker_manual_001"


def main():
  
    # Step 1: Check PDF
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    print("\nSTEP 1: Loading PDF...")

    pages = load_pdf(
        pdf_path=str(PDF_PATH),
        document_id=DOCUMENT_ID
    )

    print(f"Pages extracted: {len(pages)}")

    if not pages:
        print("No text extracted from the PDF.")
        return

    # Step 2: Chunk the pages
    print("\nSTEP 2: Creating chunks...")

    chunks = chunk_pages(
        pages=pages,
        chunk_size=1000,
        chunk_overlap=200
    )

    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        print("No chunks generated.")
        return

    # Step 3: Initialize embedding service
    print("\nSTEP 3: Initializing embedding service...")

    embedding_service = EmbeddingService()

    # Step 4: Initialize Qdrant
    print("\nSTEP 4: Initializing Qdrant...")

    vector_store = QdrantVectorStore()

    try:

        # Step 5: Index document chunks
        print("\nSTEP 5: Generating embeddings and indexing...")

        indexer = DocumentIndexer(
            embedding_service=embedding_service,
            vector_store=vector_store,
            batch_size=32
        )

        indexed_count = indexer.index_chunks(chunks)

        print(f"Successfully indexed: {indexed_count} chunks")

        # Step 6: Retrieve relevant chunks
        print("\nSTEP 6: Testing semantic retrieval...")

        retriever = Retriever(
            embedding_service=embedding_service,
            vector_store=vector_store
        )

        query = "What is Docker and how does it work?"

        results = retriever.retrieve(
            query=query,
            top_k=5,
            document_id=DOCUMENT_ID
        )

        print(f"\nQuery: {query}")
        print(f"Results found: {len(results)}")

        for index, result in enumerate(results, start=1):

            print(f"\n--- Result {index} ---")
            print(f"Chunk ID: {result.chunk_id}")
            print(f"Source: {result.source}")
            print(f"Page: {result.page_number}")
            print(f"Score: {result.score}")
            print(f"Text: {result.text}")

    finally:
        vector_store.close()


if __name__ == "__main__":
    main()
