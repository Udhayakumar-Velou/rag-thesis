from abc import ABC, abstractmethod
from typing import List

from models.chunk import Chunk


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval methods.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Chunk]:
        """
        Retrieve the most relevant chunks.
        """
        pass