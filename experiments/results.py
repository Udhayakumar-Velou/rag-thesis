from dataclasses import dataclass
from statistics import mean

from evaluation.evaluator import EvaluationResult


@dataclass
class ExperimentResult:
    """
    Aggregated result of an experiment.
    """

    experiment_name: str

    dataset: str

    chunker: str

    retriever: str

    top_k: int

    recall: float

    precision: float

    hit_rate: float

    mrr: float

    latency_ms: float

    num_queries: int

    @classmethod
    def from_evaluation(
        cls,
        experiment_name: str,
        dataset: str,
        chunker: str,
        retriever: str,
        top_k: int,
        results: list[EvaluationResult],
    ):

        return cls(
            experiment_name=experiment_name,
            dataset=dataset,
            chunker=chunker,
            retriever=retriever,
            top_k=top_k,
            recall=mean(r.recall for r in results),
            precision=mean(r.precision for r in results),
            hit_rate=mean(r.hit_rate for r in results),
            mrr=mean(r.mrr for r in results),
            latency_ms=mean(r.latency_ms for r in results),
            num_queries=len(results),
        )