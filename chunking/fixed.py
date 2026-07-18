from typing import List

from models.chunk import Chunk
from chunking.base import BaseChunker
from chunking.utils import create_chunk


class FixedChunker(BaseChunker):
    """
    Simple fixed-size chunker with overlap.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[Chunk]:
        if not text.strip():
            return []

        chunks = []
        step = self.chunk_size - self.chunk_overlap
        chunk_id = 0

        for start in range(0, len(text), step):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    create_chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        start=start,
                        end=end,
                    )
                )
                chunk_id += 1

        return chunks