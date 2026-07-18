from typing import List

import faiss
import numpy as np

from embeddings.e5_embedding import E5Embedding
from models.chunk import Chunk


class FAISSStore:
    """
    FAISS vector store for dense retrieval using cosine similarity.
    """

    def __init__(self, embedding_model: E5Embedding):

        self.embedding_model = embedding_model
        self.index = None
        self.chunks: List[Chunk] = []

    def add_documents(self, chunks: List[Chunk]) -> None:
        """
        Add chunk embeddings to the FAISS index.
        """

        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedding_model.embed_documents(texts)

        # Convert to float32 (required by FAISS)
        embeddings = embeddings.astype(np.float32)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        # Inner Product + Normalized vectors = Cosine Similarity
        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        self.chunks = chunks

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Chunk]:
        """
        Search the FAISS index.
        """

        if self.index is None:
            return []

        query_embedding = self.embedding_model.embed_query(query)

        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)

        # Normalize query embedding
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results: List[Chunk] = []

        for idx in indices[0]:
            if idx != -1:
                results.append(self.chunks[idx])

        return results