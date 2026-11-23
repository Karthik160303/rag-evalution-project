
from ingestion.embedding import EmbeddingService
from vectorstore.qdrant_store import QdrantVectorStore
from retrieval.retriever import Retriever


DOCUMENT_ID = "docker_manual_001"


def main():

    embedding_service = EmbeddingService()
    vector_store = QdrantVectorStore()

    try:

        retriever = Retriever(
            embedding_service=embedding_service,
            vector_store=vector_store
        )

        query = input("Enter evaluation question: ").strip()

        if not query:
            raise ValueError("Question cannot be empty.")

        results = retriever.retrieve(
            query=query,
            top_k=10,
            document_id=DOCUMENT_ID
        )

        print("\nRETRIEVED CHUNKS")
        print("=" * 60)

        for index, chunk in enumerate(results, start=1):

            print(f"\nResult: {index}")
            print(f"Chunk ID: {chunk.chunk_id}")
            print(f"Page: {chunk.page_number}")
            print(f"Score: {chunk.score}")
            print(f"Text: {chunk.text}")
            print("-" * 60)

    finally:
        vector_store.close()


if __name__ == "__main__":
    main()
