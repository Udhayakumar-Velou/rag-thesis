from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    """
    Represents a document chunk used throughout the RAG pipeline.
    """

    id: int
    text: str
    start: int
    end: int
    length: int

    # Optional metadata (used by advanced chunking strategies)
    section: Optional[str] = None
    level: int = 0
    parent: Optional[int] = None