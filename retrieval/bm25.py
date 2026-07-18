from typing import List

from rank_bm25 import BM25Okapi

from models.chunk import Chunk
from retrieval.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """
    BM25 keyword-based retriever.
    """

    def __init__(self, chunks: List[Chunk]):

        self.chunks = chunks

        self.corpus = [
            chunk.text.lower().split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Chunk]:

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        results: List[Chunk] = []

        for idx in ranked_indices[:top_k]:
            results.append(self.chunks[idx])

        return results