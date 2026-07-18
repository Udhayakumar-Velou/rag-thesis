from typing import Sequence


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """
    Compute Recall@K.

    Recall@K = (# relevant retrieved in top-k) / (# relevant documents)
    """

    if not relevant_ids:
        return 0.0

    retrieved = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    return len(retrieved & relevant) / len(relevant)


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """
    Compute Precision@K.

    Precision@K = (# relevant retrieved in top-k) / k
    """

    if k <= 0:
        return 0.0

    retrieved = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    return len(retrieved & relevant) / k


def hit_rate_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    """
    Compute Hit Rate@K.

    Returns:
        1.0 if at least one relevant document is retrieved.
        0.0 otherwise.
    """

    retrieved = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    return 1.0 if retrieved & relevant else 0.0


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) for a single query.

    MRR = 1 / rank of the first relevant document
    """

    relevant = set(relevant_ids)

    for rank, document_id in enumerate(retrieved_ids, start=1):
        if document_id in relevant:
            return 1.0 / rank

    return 0.0