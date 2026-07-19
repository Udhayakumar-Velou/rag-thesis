"""
RAGAS evaluation utilities.
"""

from dataclasses import dataclass
from statistics import mean
import gc
import torch

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
            Local LLM used by RAGAS (Ollama)

        embeddings:
            Embedding model used for Answer Relevancy
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

        total = len(generated_answers)

        for index, generated_answer in enumerate(generated_answers, start=1):

            print("=" * 70)
            print(f"Evaluating Sample {index}/{total}")
            print("=" * 70)

            sample = SingleTurnSample(
                user_input=generated_answer.query,
                response=generated_answer.answer,
                retrieved_contexts=[
                    chunk.text
                    for chunk in generated_answer.retrieved_chunks
                ],
            )

            try:
                # -------------------------
                # Faithfulness
                # -------------------------
                faithfulness = self.faithfulness_metric.single_turn_score(
                    sample
                )

                # -------------------------
                # Answer Relevancy
                # -------------------------
                answer_relevancy = self.answer_relevancy_metric.single_turn_score(
                    sample
                )

                print(f"Faithfulness     : {faithfulness:.4f}")
                print(f"Answer Relevancy : {answer_relevancy:.4f}")

                faithfulness_scores.append(faithfulness)
                answer_relevancy_scores.append(answer_relevancy)

            except Exception as e:

                print(f"\n❌ Sample {index} failed.")
                print(e)

            finally:
                # Free memory after every sample
                gc.collect()

                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        print("\n" + "=" * 70)
        print("RAGAS Evaluation Finished")
        print("=" * 70)
        print(f"Successful Samples : {len(faithfulness_scores)} / {total}")

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