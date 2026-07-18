from abc import ABC, abstractmethod
from typing import List, Dict


class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.
    """

    @abstractmethod
    def split(self, text: str) -> List[Dict]:
        """
        Split a document into chunks.

        Args:
            text: Input document.

        Returns:
            List of chunk dictionaries.
        """
        pass