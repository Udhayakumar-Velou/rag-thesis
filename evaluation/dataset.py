from dataclasses import dataclass, field
from typing import Dict, Iterator, List

from models.document import Document


@dataclass
class QuerySample:
    """
    Represents a single evaluation query and its ground truth.
    """

    query_id: str
    query: str
    relevant_document_ids: List[str]


@dataclass
class EvaluationDataset:
    """
    Represents an evaluation dataset consisting of:
        - Documents (corpus)
        - Queries
        - Ground-truth relevance
    """

    name: str
    documents: List[Document] = field(default_factory=list)
    queries: List[QuerySample] = field(default_factory=list)

    def add_document(
        self,
        document: Document,
    ) -> None:
        """
        Add a document to the corpus.
        """

        self.documents.append(document)

    def add_query(
        self,
        query_id: str,
        query: str,
        relevant_document_ids: List[str],
    ) -> None:
        """
        Add a query and its relevant document IDs.
        """

        self.queries.append(
            QuerySample(
                query_id=query_id,
                query=query,
                relevant_document_ids=relevant_document_ids,
            )
        )

    @classmethod
    def from_beir(
        cls,
        name: str,
        documents: List[Document],
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
    ) -> "EvaluationDataset":
        """
        Create an EvaluationDataset from BEIR data.
        """

        dataset = cls(name=name)
        dataset.documents = documents

        for query_id, query_text in queries.items():

            relevant_documents = list(
                qrels.get(query_id, {}).keys()
            )

            dataset.add_query(
                query_id=query_id,
                query=query_text,
                relevant_document_ids=relevant_documents,
            )

        return dataset

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self) -> Iterator[QuerySample]:
        return iter(self.queries)