from dataclasses import dataclass

from chunking.base import BaseChunker
from embeddings.e5_embedding import E5Embedding
from llm.base import BaseLLM


@dataclass
class ExperimentConfig:
    """
    Configuration for a single retrieval experiment.
    """

    # Experiment name
    name: str

    # Dataset
    dataset_name: str
    dataset_path: str

    # Components
    chunker: BaseChunker
    embedding_model: E5Embedding
    retriever_type: str
    llm: BaseLLM           # <-- NEW

    # Retrieval settings
    top_k: int = 5