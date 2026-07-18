from dataclasses import dataclass
from typing import Optional


@dataclass
class Chunk:
    """
    Represents a document chunk used throughout the RAG pipeline.
    """

    # Unique chunk identifier
    id: int

    # Original document identifier
    document_id: str

    # Chunk content
    text: str

    # Character positions in the original document
    start: int
    end: int

    # Length of the chunk
    length: int

    # Optional metadata (used by advanced chunking strategies)
    section: Optional[str] = None
    level: int = 0
    parent: Optional[int] = None