from typing import List, Tuple
import re

from chunking.base import BaseChunker
from chunking.utils import create_chunk
from models.chunk import Chunk


class HierarchicalChunker(BaseChunker):
    """
    Lightweight hierarchical chunker.

    Detects document sections and preserves hierarchy while
    ensuring chunks do not exceed the configured size.
    """

    def __init__(
        self,
        max_chunk_size: int = 500,
    ):
        self.max_chunk_size = max_chunk_size

    def split(self, text: str) -> List[Chunk]:
        """
        Split a document into hierarchical chunks.
        """

        if not text or not text.strip():
            return []

        sections = self._split_sections(text)

        chunks: List[Chunk] = []
        chunk_id = 0

        for section_name, level, section_text in sections:

            paragraphs = [
                p.strip()
                for p in section_text.split("\n\n")
                if p.strip()
            ]

            for paragraph in paragraphs:

                paragraph_chunks = self._split_large_paragraph(paragraph)

                search_start = 0

                for piece in paragraph_chunks:

                    start = text.find(piece, search_start)

                    if start == -1:
                        start = search_start

                    end = start + len(piece)

                    chunks.append(
                        create_chunk(
                            chunk_id=chunk_id,
                            text=piece,
                            start=start,
                            end=end,
                            section=section_name,
                            level=level,
                        )
                    )

                    chunk_id += 1
                    search_start = end

        return chunks

    def _split_sections(self, text: str) -> List[Tuple[str, int, str]]:
        """
        Split document into sections.

        Returns:
            (section_name, level, section_text)
        """

        lines = text.splitlines()

        sections = []

        current_title = "Document"
        current_level = 0
        current_content = []

        for line in lines:

            if self._is_heading(line):

                if current_content:

                    sections.append(
                        (
                            current_title,
                            current_level,
                            "\n".join(current_content).strip(),
                        )
                    )

                current_title = self._clean_heading(line)
                current_level = self._heading_level(line)
                current_content = []

            else:

                current_content.append(line)

        if current_content:

            sections.append(
                (
                    current_title,
                    current_level,
                    "\n".join(current_content).strip(),
                )
            )

        return sections

    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """
        Split paragraphs larger than max_chunk_size.
        """

        if len(paragraph) <= self.max_chunk_size:
            return [paragraph]

        words = paragraph.split()

        chunks = []
        current = ""

        for word in words:

            if len(current) + len(word) + 1 <= self.max_chunk_size:

                if current:
                    current += " "

                current += word

            else:

                chunks.append(current)

                current = word

        if current:
            chunks.append(current)

        return chunks

    def _clean_heading(self, heading: str) -> str:
        """
        Remove markdown symbols and numbering.
        """

        heading = heading.strip()

        heading = re.sub(r"^#+\s*", "", heading)

        heading = re.sub(r"^\d+(\.\d+)*\s*", "", heading)

        heading = heading.title()

        return heading

    def _heading_level(self, heading: str) -> int:
        """
        Determine heading hierarchy level.
        """

        heading = heading.strip()

        if heading.startswith("#"):
            return len(heading) - len(heading.lstrip("#"))

        match = re.match(r"^(\d+(?:\.\d+)*)", heading)

        if match:
            return match.group(1).count(".") + 1

        if heading.isupper():
            return 1

        return 1

    def _is_heading(self, line: str) -> bool:
        """
        Detect common heading formats.
        """

        line = line.strip()

        if not line:
            return False

        # Markdown headings
        if line.startswith("#"):
            return True

        # Numbered headings
        if re.match(r"^\d+(\.\d+)*\s+", line):
            return True

        # UPPERCASE headings
        if line.isupper() and len(line.split()) <= 8:
            return True

        return False