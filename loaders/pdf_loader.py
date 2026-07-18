from pypdf import PdfReader

from loaders.base import BaseLoader


class PDFLoader(BaseLoader):
    """
    Loader for PDF documents.
    """

    def load(self, file_path: str) -> str:

        reader = PdfReader(file_path)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        return text