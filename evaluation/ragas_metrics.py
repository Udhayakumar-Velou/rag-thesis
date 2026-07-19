"""
RAGAS evaluation utilities.
"""

from dataclasses import dataclass
from statistics import mean

from ragas import SingleTurnSample
from ragas.metrics import AnswerRelevancy, Faithfulness

from evaluation.dataset import EvaluationDataset
from llm.answer import GeneratedAnswer


@dataclass
class RagasResult:
    """
    Aggregated RAGAS metrics.
    """

    faithfulness: float
    answer_relevancy: float


class RagasEvaluator:
    """
    Computes RAGAS metrics for generated answers.
    """

    def __init__(
        self,
        llm,
        embeddings,
    ):
        """
        Initialize RAGAS metrics.

        llm:
            Local LLM used by RAGAS (Ollama Qwen2.5)

        embeddings:
            Embedding model used for similarity metrics
            (Answer Relevancy)
        """

        self.faithfulness_metric = Faithfulness(
            llm=llm
        )

        self.answer_relevancy_metric = AnswerRelevancy(
            llm=llm,
            embeddings=embeddings,
        )


    def evaluate(
        self,
        generated_answers: list[GeneratedAnswer],
        dataset: EvaluationDataset,
    ) -> RagasResult:

        faithfulness_scores: list[float] = []
        answer_relevancy_scores: list[float] = []


        for generated_answer in generated_answers:

            sample = SingleTurnSample(
                user_input=generated_answer.query,

                response=generated_answer.answer,

                retrieved_contexts=[
                    chunk.text
                    for chunk in generated_answer.retrieved_chunks
                ],
            )


            # -------------------------
            # Faithfulness
            # -------------------------

            faithfulness = (
                self.faithfulness_metric.single_turn_score(
                    sample
                )
            )


            # -------------------------
            # Answer Relevancy
            # -------------------------

            answer_relevancy = (
                self.answer_relevancy_metric.single_turn_score(
                    sample
                )
            )


            faithfulness_scores.append(
                faithfulness
            )

            answer_relevancy_scores.append(
                answer_relevancy
            )


        return RagasResult(
            faithfulness=(
                mean(faithfulness_scores)
                if faithfulness_scores
                else 0.0
            ),

            answer_relevancy=(
                mean(answer_relevancy_scores)
                if answer_relevancy_scores
                else 0.0
            ),
        )