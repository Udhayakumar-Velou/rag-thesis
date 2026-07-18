from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """
    Abstract base class for all document loaders.
    """

    @abstractmethod
    def load(self, file_path: str) -> str:
        """
        Load a document and return its text.
        """
        pass