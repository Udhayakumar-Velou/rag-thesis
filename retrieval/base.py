from abc import ABC, abstractmethod
from typing import List, Dict


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval methods.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve the most relevant chunks.
        """
        pass