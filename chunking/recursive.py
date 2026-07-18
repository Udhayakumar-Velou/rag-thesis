from typing import List

from models.chunk import Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking.base import BaseChunker
from chunking.utils import create_chunk


class RecursiveChunker(BaseChunker):
    """
    Recursive character-based chunker using LangChain.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, text: str) -> List[Chunk]:
        """
        Split text into recursive chunks.
        """

        if not text or not text.strip():
            return []

        split_chunks = self.splitter.split_text(text)

        chunks = []
        current_position = 0

        for idx, chunk in enumerate(split_chunks):

            start = text.find(chunk, current_position)

            if start == -1:
                start = current_position

            end = start + len(chunk)

            chunks.append(
                create_chunk(
                    chunk_id=idx,
                    text=chunk,
                    start=start,
                    end=end,
                )
            )

            current_position = end

        return chunks