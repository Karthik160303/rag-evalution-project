
import pytest
from unittest.mock import Mock

from evaluation.answer_evaluator import AnswerEvaluator
from ingestion.models import RetrievedChunk


@pytest.fixture
def sample_chunk():

    return RetrievedChunk(
        chunk_id="test001",
        document_id="doc001",
        source="docker_guide.pdf",
        page_number=1,
        text="The docker ps command shows running containers.",
        score=0.95
    )


def test_answer_evaluation(sample_chunk):

    llm = Mock()

    llm.generate.return_value = """
    {
        "faithfulness_score": 1.0,
        "answer_relevancy_score": 0.95,
        "explanation": "The answer is supported by context."
    }
    """

    evaluator = AnswerEvaluator(llm)

    result = evaluator.evaluate(
        question="What does docker ps do?",
        answer="It shows running containers.",
        chunks=[sample_chunk]
    )

    assert result.faithfulness_score == 1.0
    assert result.answer_relevancy_score == 0.95

    llm.generate.assert_called_once()


def test_empty_question(sample_chunk):

    evaluator = AnswerEvaluator(llm=Mock())

    with pytest.raises(ValueError):
        evaluator.evaluate(
            question="",
            answer="Some answer",
            chunks=[sample_chunk]
        )


def test_empty_context():

    evaluator = AnswerEvaluator(llm=Mock())

    with pytest.raises(ValueError):
        evaluator.evaluate(
            question="What is Docker?",
            answer="Docker is a platform.",
            chunks=[]
        )


def test_invalid_llm_response(sample_chunk):

    llm = Mock()
    llm.generate.return_value = "Invalid response"

    evaluator = AnswerEvaluator(llm)

    with pytest.raises(ValueError):
        evaluator.evaluate(
            question="What is Docker?",
            answer="Docker is a platform.",
            chunks=[sample_chunk]
        )


def test_invalid_score(sample_chunk):

    llm = Mock()

    llm.generate.return_value = """
    {
        "faithfulness_score": 2.0,
        "answer_relevancy_score": 0.9,
        "explanation": "Invalid score."
    }
    """

    evaluator = AnswerEvaluator(llm)

    with pytest.raises(ValueError):
        evaluator.evaluate(
            question="What is Docker?",
            answer="Docker is a platform.",
            chunks=[sample_chunk]
        )
