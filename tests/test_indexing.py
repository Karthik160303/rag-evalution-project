
from unittest.mock import Mock

from ingestion.models import DocumentChunk
from vectorstore.indexing import DocumentIndexer


def test_index_chunks_in_batches():

    embedding_service = Mock()
    vector_store = Mock()

    embedding_service.embed_documents.side_effect = [
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        [[0.7, 0.8, 0.9]]
    ]

    chunks = [
        DocumentChunk(
            document_id="doc001",
            chunk_id=f"doc001_p001_c00{i}",
            source="docker.pdf",
            page_number=1,
            text=f"Docker content {i}"
        )
        for i in range(1, 4)
    ]

    indexer = DocumentIndexer(
        embedding_service=embedding_service,
        vector_store=vector_store,
        batch_size=2
    )

    total = indexer.index_chunks(chunks)

    assert total == 3

    assert embedding_service.embed_documents.call_count == 2

    assert vector_store.upsert_chunks.call_count == 2


def test_empty_chunks():

    indexer = DocumentIndexer(
        embedding_service=Mock(),
        vector_store=Mock()
    )

    assert indexer.index_chunks([]) == 0
