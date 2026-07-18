from dataclasses import dataclass
from time import perf_counter
from typing import List

from evaluation.dataset import EvaluationDataset
from evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    hit_rate_at_k,
    mean_reciprocal_rank,
)
from retrieval.base import BaseRetriever


@dataclass
class EvaluationResult:
    """
    Evaluation result for a single query.
    """

    query: str

    recall: float

    precision: float

    hit_rate: float

    mrr: float

    latency_ms: float


class Evaluator:
    """
    Evaluates a retriever on an evaluation dataset.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.top_k = top_k

    def evaluate(
        self,
        dataset: EvaluationDataset,
    ) -> List[EvaluationResult]:

        results: List[EvaluationResult] = []

        for sample in dataset:

            start = perf_counter()

            retrieved_chunks = self.retriever.retrieve(
                query=sample.query,
                top_k=self.top_k,
            )

            latency_ms = (perf_counter() - start) * 1000

            # Convert retrieved chunks to unique document IDs
            retrieved_document_ids = []

            seen = set()

            for chunk in retrieved_chunks:

                if chunk.document_id not in seen:
                    retrieved_document_ids.append(chunk.document_id)
                    seen.add(chunk.document_id)

            results.append(
                EvaluationResult(
                    query=sample.query,
                    recall=recall_at_k(
                        retrieved_document_ids,
                        sample.relevant_document_ids,
                        self.top_k,
                    ),
                    precision=precision_at_k(
                        retrieved_document_ids,
                        sample.relevant_document_ids,
                        self.top_k,
                    ),
                    hit_rate=hit_rate_at_k(
                        retrieved_document_ids,
                        sample.relevant_document_ids,
                        self.top_k,
                    ),
                    mrr=mean_reciprocal_rank(
                        retrieved_document_ids,
                        sample.relevant_document_ids,
                    ),
                    latency_ms=latency_ms,
                )
            )

        return results