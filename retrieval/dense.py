from typing import List

from models.chunk import Chunk
from retrieval.base import BaseRetriever
from vectorstore.faiss_store import FAISSStore


class DenseRetriever(BaseRetriever):
    """
    Dense retriever using FAISS.
    """

    def __init__(self, vector_store: FAISSStore):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Chunk]:

        return self.vector_store.search(
            query=query,
            top_k=top_k
        )