from pathlib import Path

from pypdf import PdfReader

from loaders.base import BaseLoader
from models.document import Document


class PDFLoader(BaseLoader):
    """
    Loader for PDF documents.
    """

    def load(self, file_path: str) -> Document:
        """
        Load a PDF document and return a Document object.
        """

        path = Path(file_path)

        reader = PdfReader(path)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return Document(
            id=path.stem,
            text=text.strip(),
        )