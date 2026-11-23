
from ingestion.embedding import EmbeddingService
from ingestion.models import RetrievedChunk
from vectorstore.qdrant_store import QdrantVectorStore


class Retriever:
    """Retrieves relevant document chunks using semantic search."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None
    ) -> list[RetrievedChunk]:
        """Find the most relevant chunks for a question."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = (
            self.embedding_service.embed_query(query.strip())
        )

        return self.vector_store.search(
            query_embedding=query_embedding,
            limit=top_k,
            document_id=document_id
        )
