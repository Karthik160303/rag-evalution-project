
from unittest.mock import Mock

import pytest

from generation.rag_chain import RAGChain
from ingestion.models import RetrievedChunk


@pytest.fixture
def sample_chunk():

    return RetrievedChunk(
        chunk_id="test001",
        document_id="doc001",
        source="docker_guide.pdf",
        page_number=1,
        text="Docker is a containerization platform.",
        score=0.95
    )


def test_rag_chain_generates_answer(sample_chunk):

    retriever = Mock()
    llm = Mock()

    retriever.retrieve.return_value = [sample_chunk]
    llm.generate.return_value = "Docker is a containerization platform."

    rag = RAGChain(
        retriever=retriever,
        llm=llm
    )

    result = rag.ask(
        query="What is Docker?",
        top_k=5,
        document_id="doc001"
    )

    assert result.answer == "Docker is a containerization platform."
    assert len(result.retrieved_chunks) == 1

    retriever.retrieve.assert_called_once()
    llm.generate.assert_called_once()


def test_rag_chain_handles_no_results():

    retriever = Mock()
    llm = Mock()

    retriever.retrieve.return_value = []

    rag = RAGChain(
        retriever=retriever,
        llm=llm
    )

    result = rag.ask("What is Docker?")

    assert result.retrieved_chunks == []
    assert "could not find relevant information" in result.answer.lower()

    llm.generate.assert_not_called()


def test_rag_chain_rejects_empty_query():

    rag = RAGChain(
        retriever=Mock(),
        llm=Mock()
    )

    with pytest.raises(ValueError):
        rag.ask("")
