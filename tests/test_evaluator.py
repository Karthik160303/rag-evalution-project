
import pytest

from evaluation.evaluator import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_rate_at_k,
    evaluate_retrieval
)


@pytest.fixture
def sample_data():

    retrieved = [
        "chunk_101",
        "chunk_205",
        "chunk_301",
        "chunk_405",
        "chunk_501"
    ]

    relevant = [
        "chunk_101",
        "chunk_301",
        "chunk_999"
    ]

    return retrieved, relevant


def test_precision_at_k(sample_data):

    retrieved, relevant = sample_data

    result = precision_at_k(retrieved, relevant, 5)

    assert result == pytest.approx(0.4)


def test_recall_at_k(sample_data):

    retrieved, relevant = sample_data

    result = recall_at_k(retrieved, relevant, 5)

    assert result == pytest.approx(2 / 3)


def test_reciprocal_rank(sample_data):

    retrieved, relevant = sample_data

    result = reciprocal_rank(retrieved, relevant)

    assert result == 1.0


def test_hit_rate_at_k(sample_data):

    retrieved, relevant = sample_data

    result = hit_rate_at_k(retrieved, relevant, 5)

    assert result == 1.0


def test_no_relevant_results():

    retrieved = ["chunk_101", "chunk_205"]
    relevant = ["chunk_999"]

    assert reciprocal_rank(retrieved, relevant) == 0.0

    assert hit_rate_at_k(
        retrieved, relevant, 2
    ) == 0.0


def test_invalid_k():

    with pytest.raises(ValueError):
        precision_at_k(["a"], ["a"], 0)


def test_empty_relevant_ids():

    with pytest.raises(ValueError):
        recall_at_k(["a"], [], 5)


def test_evaluate_retrieval(sample_data):

    retrieved, relevant = sample_data

    result = evaluate_retrieval(
        retrieved,
        relevant,
        k=5
    )

    assert result["precision_at_k"] == pytest.approx(0.4)
    assert result["recall_at_k"] == pytest.approx(2 / 3)
    assert result["mrr"] == 1.0
    assert result["hit_rate_at_k"] == 1.0
