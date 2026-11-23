
from ingestion.embedding import EmbeddingService

from vectorstore.qdrant_store import QdrantVectorStore

from retrieval.retriever import Retriever

from generation.llm_client import OllamaLLM
from generation.rag_chain import RAGChain


DOCUMENT_ID = "docker_manual_001"


def main():

    print("\nInitializing RAG system...")

    # Step 1: Initialize embedding service
    embedding_service = EmbeddingService()

    # Step 2: Connect to Qdrant
    vector_store = QdrantVectorStore()

    try:

        if not vector_store.collection_exists():
            print("Qdrant collection does not exist.")
            print("Run run_pdf_pipeline.py first to index your PDF.")
            return

        # Step 3: Initialize retriever
        retriever = Retriever(
            embedding_service=embedding_service,
            vector_store=vector_store
        )

        # Step 4: Initialize LLM
        llm = OllamaLLM(
            model="llama3.2:3b"
        )

        # Step 5: Create RAG chain
        rag = RAGChain(
            retriever=retriever,
            llm=llm
        )

        print("\nRAG system is ready!")
        print("Type 'exit' to stop.")

        while True:

            query = input("\nAsk a question: ").strip()

            if query.lower() == "exit":
                print("Closing RAG system...")
                break

            if not query:
                print("Please enter a question.")
                continue

            print("\nGenerating answer...")

            result = rag.ask(
                query=query,
                top_k=5,
                document_id=DOCUMENT_ID
            )

            print("\n" + "=" * 60)
            print("ANSWER")
            print("=" * 60)

            print(result.answer)

            print("\n" + "=" * 60)
            print("SOURCES")
            print("=" * 60)

            if not result.retrieved_chunks:
                print("No sources found.")
            else:
                seen_sources = set()

                for chunk in result.retrieved_chunks:

                    source_key = (
                        chunk.source,
                        chunk.page_number
                    )

                    if source_key not in seen_sources:

                        seen_sources.add(source_key)

                        print(
                            f"{chunk.source} - "
                            f"Page {chunk.page_number}"
                        )

            print("\n" + "=" * 60)

    finally:
        vector_store.close()


if __name__ == "__main__":
    main()
