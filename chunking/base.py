from abc import ABC, abstractmethod
from typing import List

from models.chunk import Chunk


class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.
    """

    @abstractmethod
    def split(
        self,
        text: str,
        document_id: str = "doc_0",
    ) -> List[Chunk]:
        """
        Split a document into chunks.

        Args:
            text: Input document.
            document_id: Identifier of the source document.

        Returns:
            List of Chunk objects.
        """
        pass