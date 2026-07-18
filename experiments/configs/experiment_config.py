from dataclasses import dataclass

from chunking.base import BaseChunker
from retrieval.base import BaseRetriever
from embeddings.e5_embedding import E5Embedding


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

    # Retrieval settings
    top_k: int = 5