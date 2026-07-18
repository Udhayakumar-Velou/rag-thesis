from dataclasses import dataclass


@dataclass
class Document:
    """
    Represents a source document before chunking.
    """

    id: str
    text: str