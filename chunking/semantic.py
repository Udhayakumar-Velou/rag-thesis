from typing import List

import nltk
from sklearn.metrics.pairwise import cosine_similarity

from models.chunk import Chunk
from chunking.base import BaseChunker
from chunking.utils import create_chunk
from embeddings.e5_embedding import E5Embedding


class SemanticChunker(BaseChunker):
    """
    Semantic chunker using E5 sentence embeddings.
    """

    def __init__(
        self,
        embedding_model: E5Embedding,
        similarity_threshold: float = 0.75,
        max_chunk_sentences: int = 5,
    ):

        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_chunk_sentences = max_chunk_sentences

    def split(
        self,
        text: str,
        document_id: str = "doc_0",
    ) -> List[Chunk]:

        if not text or not text.strip():
            return []

        sentences = nltk.sent_tokenize(text)

        if len(sentences) == 1:
            return [
                create_chunk(
                    chunk_id=0,
                    document_id=document_id,
                    text=sentences[0],
                    start=0,
                    end=len(sentences[0]),
                )
            ]

        embeddings = self.embedding_model.embed_documents(sentences)

        chunks = []

        current_chunk = [sentences[0]]
        chunk_start = text.find(sentences[0])
        chunk_id = 0

        for i in range(1, len(sentences)):

            similarity = cosine_similarity(
                embeddings[i - 1].reshape(1, -1),
                embeddings[i].reshape(1, -1),
            )[0][0]

            if (
                similarity >= self.similarity_threshold
                and len(current_chunk) < self.max_chunk_sentences
            ):
                current_chunk.append(sentences[i])

            else:

                chunk_text = " ".join(current_chunk)

                start = chunk_start
                end = start + len(chunk_text)

                chunks.append(
                    create_chunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        text=chunk_text,
                        start=start,
                        end=end,
                    )
                )

                chunk_id += 1

                current_chunk = [sentences[i]]
                chunk_start = text.find(sentences[i], end)

        if current_chunk:

            chunk_text = " ".join(current_chunk)

            start = chunk_start
            end = start + len(chunk_text)

            chunks.append(
                create_chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=chunk_text,
                    start=start,
                    end=end,
                )
            )

        return chunks