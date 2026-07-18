from typing import Dict, List

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
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Chunk]:

        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        fused_scores: Dict[int, float] = {}
        documents: Dict[int, Chunk] = {}

        # Dense contribution
        for rank, chunk in enumerate(dense_results, start=1):
            chunk_id = chunk.id

            documents[chunk_id] = chunk

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0)
                + 1 / (self.rrf_k + rank)
            )

        # BM25 contribution
        for rank, chunk in enumerate(bm25_results, start=1):
            chunk_id = chunk.id

            documents[chunk_id] = chunk

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0)
                + 1 / (self.rrf_k + rank)
            )

        ranked = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        results = [
            documents[doc_id]
            for doc_id, _ in ranked[:top_k]
        ]

        return results