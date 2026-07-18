from typing import Dict, List, Tuple

from models.chunk import Chunk
from retrieval.base import BaseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever using Reciprocal Rank Fusion (RRF).

    Combines Dense Retrieval (FAISS) and BM25 retrieval.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        candidate_multiplier: int = 10,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Chunk]:

        candidate_k = max(top_k * self.candidate_multiplier, 50)

        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=candidate_k,
        )

        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=candidate_k,
        )

        fused_scores: Dict[Tuple[str, int], float] = {}
        documents: Dict[Tuple[str, int], Chunk] = {}

        # Dense contribution
        for rank, chunk in enumerate(dense_results, start=1):

            key = (chunk.document_id, chunk.id)

            documents[key] = chunk

            fused_scores[key] = (
                fused_scores.get(key, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        # BM25 contribution
        for rank, chunk in enumerate(bm25_results, start=1):

            key = (chunk.document_id, chunk.id)

            documents[key] = chunk

            fused_scores[key] = (
                fused_scores.get(key, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        ranked = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            documents[key]
            for key, _ in ranked[:top_k]
        ]