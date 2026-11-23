
from ingestion.embedding import EmbeddingService
from ingestion.models import DocumentChunk
from vectorstore.qdrant_store import QdrantVectorStore
from vectorstore.indexing import DocumentIndexer
from retrieval.retriever import Retriever


def main():

    embedding_service = EmbeddingService()

    vector_store = QdrantVectorStore()

    try:
        vector_store.create_collection()

        chunks = [
            DocumentChunk(
                document_id="demo001",
                chunk_id="demo001_p001_c001",
                source="docker.pdf",
                page_number=1,
                text=(
                    "Docker is a containerization platform "
                    "that allows developers to package "
                    "applications with their dependencies."
                )
            ),
            DocumentChunk(
                document_id="demo001",
                chunk_id="demo001_p001_c002",
                source="docker.pdf",
                page_number=1,
                text=(
                    "Docker containers provide consistent "
                    "execution environments across systems."
                )
            ),
            DocumentChunk(
                document_id="demo001",
                chunk_id="demo001_p001_c003",
                source="docker.pdf",
                page_number=2,
                text=(
                    "Python is a programming language used "
                    "for software development and automation."
                )
            )
        ]

        indexer = DocumentIndexer(
            embedding_service=embedding_service,
            vector_store=vector_store
        )

        total = indexer.index_chunks(chunks)

        print("Indexed chunks:", total)

        retriever = Retriever(
            embedding_service=embedding_service,
            vector_store=vector_store
        )

        results = retriever.retrieve(
            query="How does Docker package applications?",
            top_k=2,
            document_id="demo001"
        )

        print("\nRetrieved results:")

        for result in results:
            print("\nChunk ID:", result.chunk_id)
            print("Source:", result.source)
            print("Page:", result.page_number)
            print("Score:", result.score)
            print("Text:", result.text)

    finally:
        vector_store.close()


if __name__ == "__main__":
    main()
