from sentence_transformers import SentenceTransformer
import torch
from typing import List
import numpy as np


class E5Embedding:
    """
    Wrapper for the intfloat/e5-base-v2 embedding model.
    """

    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        # Automatically use Apple GPU (MPS) if available
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        print(f"Loading embedding model on {self.device}...")

        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.
        """
        return self.model.encode(
            f"query: {query}",
            convert_to_numpy=True
        )

    def embed_document(self, document: str) -> np.ndarray:
        """
        Embed a single document.
        """
        return self.model.encode(
            f"passage: {document}",
            convert_to_numpy=True
        )

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Embed multiple queries.
        """
        queries = [f"query: {q}" for q in queries]

        return self.model.encode(
            queries,
            convert_to_numpy=True
        )

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Embed multiple documents.
        """
        documents = [f"passage: {d}" for d in documents]

        return self.model.encode(
            documents,
            convert_to_numpy=True
        )