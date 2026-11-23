
from vectorstore.qdrant_store import QdrantVectorStore
from ingestion.models import DocumentChunk


def test_store_and_search_chunks(tmp_path):

    store = QdrantVectorStore(
        path=tmp_path / "test_qdrant",
        collection_name="test_chunks",
        vector_size=3
    )

    try:
        store.create_collection()

        chunks = [
            DocumentChunk(
                document_id="doc001",
                chunk_id="doc001_p001_c001",
                source="docker.pdf",
                page_number=1,
                text="Docker is a containerization platform."
            )
        ]

        embeddings = [
            [0.1, 0.2, 0.3]
        ]

        store.upsert_chunks(
            chunks=chunks,
            embeddings=embeddings
        )

        results = store.search(
            query_embedding=[0.1, 0.2, 0.3],
            limit=1
        )

        assert len(results) == 1

        assert results[0].document_id == "doc001"

        assert results[0].chunk_id == "doc001_p001_c001"

        assert results[0].text == (
            "Docker is a containerization platform."
        )

        assert results[0].score > 0.99

    finally:
        store.close()
