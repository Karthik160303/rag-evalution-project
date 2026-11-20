
from pydantic import BaseModel


class EvaluationSample(BaseModel):
    """
    Ground truth data for evaluating RAG.
    """

    question_id: str
    question: str
    ground_truth: str
    relevant_chunk_ids: list[str]
    expected_answerable: bool = True


class RetrievalResult(BaseModel):
    """
    Captures retrieval results.
    """

    question_id: str
    retrieved_chunk_ids: list[str]
    retrieval_latency_ms: float


class GenerationResult(BaseModel):
    """
    Captures LLM generation results.
    """

    question_id: str
    answer: str
    cited_chunk_ids: list[str]
    generation_latency_ms: float


class EvaluationResult(BaseModel):
    """
    Stores evaluation metrics.
    """

    question_id: str
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    answer_correctness: float | None = None
    faithfulness: float | None = None
