
import pytest

from ingestion.embedding import (
    EmbeddingService,
    EmbeddingError
)


def test_embed_document(monkeypatch):
    service = EmbeddingService()

    captured_payload = {}

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "embeddings": [[0.1, 0.2, 0.3]]
            }

    def mock_post(url, json, timeout):
        captured_payload.update(json)
        return MockResponse()

    monkeypatch.setattr(
        "ingestion.embedding.requests.post",
        mock_post
    )

    embedding = service.embed_document(
        "Docker is a containerization platform."
    )

    assert embedding == [0.1, 0.2, 0.3]

    assert captured_payload["model"] == "nomic-embed-text"

    assert captured_payload["input"] == [
        "search_document: Docker is a containerization platform."
    ]


def test_embed_query(monkeypatch):
    service = EmbeddingService()

    captured_payload = {}

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "embeddings": [[0.4, 0.5, 0.6]]
            }

    def mock_post(url, json, timeout):
        captured_payload.update(json)
        return MockResponse()

    monkeypatch.setattr(
        "ingestion.embedding.requests.post",
        mock_post
    )

    embedding = service.embed_query(
        "What is Docker?"
    )

    assert embedding == [0.4, 0.5, 0.6]

    assert captured_payload["input"] == [
        "search_query: What is Docker?"
    ]


def test_embed_multiple_documents(monkeypatch):
    service = EmbeddingService()

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "embeddings": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6]
                ]
            }

    def mock_post(url, json, timeout):
        return MockResponse()

    monkeypatch.setattr(
        "ingestion.embedding.requests.post",
        mock_post
    )

    embeddings = service.embed_documents([
        "Docker is a containerization platform.",
        "Python is a programming language."
    ])

    assert len(embeddings) == 2

    assert len(embeddings[0]) == 3

    assert len(embeddings[1]) == 3


def test_empty_input():
    service = EmbeddingService()

    with pytest.raises(ValueError):
        service.embed_document("")


def test_invalid_embedding_response(monkeypatch):
    service = EmbeddingService()

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {}

    def mock_post(url, json, timeout):
        return MockResponse()

    monkeypatch.setattr(
        "ingestion.embedding.requests.post",
        mock_post
    )

    with pytest.raises(EmbeddingError):
        service.embed_document("Test document")
