from ingestion.models import DocumentChunk
from evaluation.models import EvaluationSample


def test_document_chunk_creation():

    chunk = DocumentChunk(
        document_id="doc001",
        chunk_id="doc001_chunk001",
        source="docker.pdf",
        page_number=1,
        text="Docker is a containerization platform."
    )

    assert chunk.document_id == "doc001"
    assert chunk.page_number == 1
    assert chunk.embedding is None


def test_evaluation_sample_creation():

    sample = EvaluationSample(
        question_id="q001",
        question="What is Docker?",
        ground_truth="Docker is a containerization platform.",
        relevant_chunk_ids=["doc001_chunk001"]
    )

    assert sample.expected_answerable is True
    assert len(sample.relevant_chunk_ids) == 1