from models.chunk import Chunk


def create_chunk(
    chunk_id: int,
    text: str,
    start: int,
    end: int,
    section: str = None,
    level: int = 0,
    parent: int = None,
) -> Chunk:
    """
    Create a Chunk object with consistent metadata.
    """

    return Chunk(
        id=chunk_id,
        text=text,
        start=start,
        end=end,
        length=len(text),
        section=section,
        level=level,
        parent=parent,
    )