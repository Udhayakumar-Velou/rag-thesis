from typing import List
import re

from chunking.base import BaseChunker
from chunking.utils import create_chunk
from models.chunk import Chunk


class AdaptiveChunker(BaseChunker):
    """
    Lightweight adaptive chunker.

    Adapts chunk boundaries based on paragraph size:
    - Small paragraphs are merged.
    - Medium paragraphs are kept intact.
    - Large paragraphs are split by sentences.
    """

    def __init__(
        self,
        max_chunk_size: int = 500,
        min_chunk_size: int = 250,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def split(
        self,
        text: str,
        document_id: str = "doc_0",
    ) -> List[Chunk]:

        if not text.strip():
            return []

        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if p.strip()
        ]

        chunks: List[Chunk] = []

        buffer = ""
        chunk_id = 0
        search_start = 0

        for paragraph in paragraphs:

            # Large paragraph
            if len(paragraph) > self.max_chunk_size:

                if buffer:

                    start = text.find(buffer, search_start)
                    end = start + len(buffer)

                    chunks.append(
                        create_chunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            text=buffer,
                            start=start,
                            end=end,
                        )
                    )

                    chunk_id += 1
                    search_start = end
                    buffer = ""

                pieces = self._split_large_paragraph(paragraph)

                for piece in pieces:

                    start = text.find(piece, search_start)

                    if start == -1:
                        start = search_start

                    end = start + len(piece)

                    chunks.append(
                        create_chunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            text=piece,
                            start=start,
                            end=end,
                        )
                    )

                    chunk_id += 1
                    search_start = end

                continue

            # Merge small paragraphs
            if len(buffer) + len(paragraph) + 2 <= self.max_chunk_size:

                if buffer:
                    buffer += "\n\n"

                buffer += paragraph

            else:

                start = text.find(buffer, search_start)
                end = start + len(buffer)

                chunks.append(
                    create_chunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        text=buffer,
                        start=start,
                        end=end,
                    )
                )

                chunk_id += 1
                search_start = end

                buffer = paragraph

        if buffer:

            start = text.find(buffer, search_start)
            end = start + len(buffer)

            chunks.append(
                create_chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=buffer,
                    start=start,
                    end=end,
                )
            )

        return chunks

    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """
        Split large paragraphs into sentence-based chunks.
        """

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        chunks = []

        current = ""

        for sentence in sentences:

            if len(current) + len(sentence) + 1 <= self.max_chunk_size:

                if current:
                    current += " "

                current += sentence

            else:

                if current:
                    chunks.append(current)

                current = sentence

        if current:
            chunks.append(current)

        return chunks