from abc import ABC, abstractmethod

from models.document import Document


class BaseLoader(ABC):
    """
    Abstract base class for all document loaders.
    """

    @abstractmethod
    def load(self, file_path: str) -> Document:
        """
        Load a document.

        Args:
            file_path: Path to the input document.

        Returns:
            Document object containing the document ID and text.
        """
        pass