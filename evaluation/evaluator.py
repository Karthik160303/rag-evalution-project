
def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer.")


def _validate_relevant_ids(relevant_ids: list[str]) -> None:
    if not relevant_ids:
        raise ValueError(
            "Relevant chunk IDs cannot be empty. "
            "Provide at least one relevant chunk ID."
        )


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> float:
    """
    Calculate Precision@K.

    Precision@K = Relevant retrieved chunks / K
    """

    _validate_k(k)
    _validate_relevant_ids(relevant_ids)

    relevant = set(relevant_ids)
    retrieved = retrieved_ids[:k]

    relevant_count = sum(
        1 for chunk_id in retrieved
        if chunk_id in relevant
    )

    return relevant_count / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> float:
    """
    Calculate Recall@K.

    Recall@K = Relevant retrieved chunks / Total relevant chunks
    """

    _validate_k(k)
    _validate_relevant_ids(relevant_ids)

    relevant = set(relevant_ids)
    retrieved = set(retrieved_ids[:k])

    relevant_count = len(relevant.intersection(retrieved))

    return relevant_count / len(relevant)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str]
) -> float:
    """
    Calculate Reciprocal Rank.

    Reciprocal Rank = 1 / Rank of first relevant result
    """

    _validate_relevant_ids(relevant_ids)

    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def hit_rate_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> float:
    """
    Calculate Hit Rate@K.

    Returns 1 if at least one relevant chunk is retrieved.
    Otherwise returns 0.
    """

    _validate_k(k)
    _validate_relevant_ids(relevant_ids)

    relevant = set(relevant_ids)

    return float(
        any(
            chunk_id in relevant
            for chunk_id in retrieved_ids[:k]
        )
    )


def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5
) -> dict[str, float]:

    return {
        "precision_at_k": precision_at_k(
            retrieved_ids, relevant_ids, k
        ),
        "recall_at_k": recall_at_k(
            retrieved_ids, relevant_ids, k
        ),
        "mrr": reciprocal_rank(
            retrieved_ids, relevant_ids
        ),
        "hit_rate_at_k": hit_rate_at_k(
            retrieved_ids, relevant_ids, k
        )
    }
