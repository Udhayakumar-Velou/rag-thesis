from pathlib import Path

from loaders.base import BaseLoader
from models.document import Document


class TextLoader(BaseLoader):
    """
    Loader for plain text (.txt) documents.
    """

    def load(self, file_path: str) -> Document:
        """
        Load a text document and return a Document object.
        """

        path = Path(file_path)

        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        return Document(
            id=path.stem,
            text=text,
        )