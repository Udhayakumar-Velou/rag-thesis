"""
Data model representing a generated RAG answer.
"""

from dataclasses import dataclass
from typing import List

from models.chunk import Chunk


@dataclass
class GeneratedAnswer:
    """
    Output produced by the RAG pipeline.
    """

    query: str
    answer: str
    retrieved_chunks: List[Chunk]
    generation_time: float