from pathlib import Path
from typing import Dict, List

from beir.datasets.data_loader import GenericDataLoader

from models.document import Document


class BEIRLoader:
    """
    Loader for BEIR benchmark datasets.

    Loads:
        - Corpus
        - Queries
        - Qrels (ground-truth relevance)
    """

    def load(
        self,
        dataset_path: str,
    ) -> tuple[
        List[Document],
        Dict[str, str],
        Dict[str, Dict[str, int]],
    ]:
        """
        Load a BEIR dataset.

        Args:
            dataset_path: Path to the BEIR dataset directory.

        Returns:
            documents:
                List of Document objects.

            queries:
                Mapping of query_id -> query text.

            qrels:
                Mapping of query_id ->
                    {
                        document_id: relevance_score
                    }
        """

        corpus, queries, qrels = GenericDataLoader(
            Path(dataset_path)
        ).load(split="test")

        documents = []

        for document_id, document in corpus.items():

            title = document.get("title", "")
            text = document.get("text", "")

            full_text = f"{title}\n{text}".strip()

            documents.append(
                Document(
                    id=document_id,
                    text=full_text,
                )
            )

        return documents, queries, qrels