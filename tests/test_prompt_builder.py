
from ingestion.models import RetrievedChunk
from generation.prompt_builder import build_prompt


def test_build_prompt():

    chunk = RetrievedChunk(
        chunk_id="test001",
        document_id="doc001",
        source="docker_guide.pdf",
        page_number=1,
        text="Docker is a containerization platform.",
        score=0.95
    )

    prompt = build_prompt(
        query="What is Docker?",
        chunks=[chunk]
    )

    assert "What is Docker?" in prompt
    assert "Docker is a containerization platform." in prompt
    assert "docker_guide.pdf" in prompt
    assert "Page: 1" in prompt


def test_empty_query():

    try:
        build_prompt("", [])
    except ValueError:
        assert True
    else:
        assert False
