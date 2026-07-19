from dataclasses import dataclass, asdict
from statistics import mean
from pathlib import Path
import json

from evaluation.evaluator import EvaluationResult
from evaluation.ragas_metrics import RagasResult


@dataclass
class ExperimentResult:
    """
    Aggregated result of a complete RAG experiment.
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

    faithfulness: float

    answer_relevancy: float


    @classmethod
    def from_evaluation(
        cls,
        experiment_name: str,
        dataset: str,
        chunker: str,
        retriever: str,
        top_k: int,
        results: list[EvaluationResult],
        ragas_result: RagasResult,
    ):

        return cls(

            experiment_name=experiment_name,

            dataset=dataset,

            chunker=chunker,

            retriever=retriever,

            top_k=top_k,

            recall=mean(
                r.recall for r in results
            ),

            precision=mean(
                r.precision for r in results
            ),

            hit_rate=mean(
                r.hit_rate for r in results
            ),

            mrr=mean(
                r.mrr for r in results
            ),

            latency_ms=mean(
                r.latency_ms for r in results
            ),

            num_queries=len(results),

            faithfulness=ragas_result.faithfulness,

            answer_relevancy=ragas_result.answer_relevancy,
        )


    def save(
        self,
        path: str = "experiments/results/result.json",
    ) -> None:
        """
        Save experiment result as JSON.
        """

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                asdict(self),
                file,
                indent=4,
            )


        print(
            f"Result saved: {output_path}"
        )