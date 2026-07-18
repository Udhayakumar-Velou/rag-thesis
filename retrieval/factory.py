from embeddings.e5_embedding import E5Embedding
from models.chunk import Chunk
from retrieval.base import BaseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from vectorstore.faiss_store import FAISSStore


class RetrieverFactory:
    """
    Factory for creating retrieval methods.
    """

    @staticmethod
    def create(
        retriever_type: str,
        chunks: list[Chunk],
        embedding_model: E5Embedding,
    ) -> BaseRetriever:

        retriever_type = retriever_type.lower()

        if retriever_type == "dense":

            vector_store = FAISSStore(embedding_model)

            vector_store.add_documents(chunks)

            return DenseRetriever(vector_store)

        if retriever_type == "bm25":

            return BM25Retriever(chunks)

        if retriever_type == "hybrid":

            vector_store = FAISSStore(embedding_model)

            vector_store.add_documents(chunks)

            dense = DenseRetriever(vector_store)

            bm25 = BM25Retriever(chunks)

            return HybridRetriever(
                dense_retriever=dense,
                bm25_retriever=bm25,
            )

        raise ValueError(
            f"Unknown retriever type: {retriever_type}"
        )