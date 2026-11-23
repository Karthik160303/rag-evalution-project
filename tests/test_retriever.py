
from unittest.mock import Mock

import pytest

from retrieval.retriever import Retriever
from ingestion.models import RetrievedChunk


def test_retrieve_chunks():

    embedding_service = Mock()

    vector_store = Mock()

    embedding_service.embed_query.return_value = [
        0.1, 0.2, 0.3
    ]

    expected_chunks = [
        RetrievedChunk(
            chunk_id="doc001_p001_c001",
            document_id="doc001",
            source="docker.pdf",
            page_number=1,
            text="Docker is a containerization platform.",
            score=0.95
        )
    ]

    vector_store.search.return_value = expected_chunks

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store
    )

    results = retriever.retrieve(
        query="What is Docker?",
        top_k=5
    )

    assert len(results) == 1

    assert results[0].chunk_id == "doc001_p001_c001"

    assert results[0].score == 0.95

    embedding_service.embed_query.assert_called_once_with(
        "What is Docker?"
    )

    vector_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        limit=5,
        document_id=None
    )


def test_empty_query():

    retriever = Retriever(
        embedding_service=Mock(),
        vector_store=Mock()
    )

    with pytest.raises(ValueError):
        retriever.retrieve("")
