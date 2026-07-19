import csv
import os
from datetime import datetime

from chunking.fixed import FixedChunker
from chunking.recursive import RecursiveChunker
from chunking.semantic import SemanticChunker
from chunking.hierarchical import HierarchicalChunker
from chunking.adaptive import AdaptiveChunker

from embeddings.e5_embedding import E5Embedding

from experiments.configs.experiment_config import ExperimentConfig
from experiments.experiment_runner import ExperimentRunner

from llm.ollama_llm import OllamaLLM


RESULTS_DIR = "results/logs"
SUCCESS_CSV = os.path.join(RESULTS_DIR, "experiment_results.csv")
FAILURE_CSV = os.path.join(RESULTS_DIR, "experiment_failures.csv")


def initialize_csv():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(SUCCESS_CSV):
        with open(SUCCESS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp",
                "Experiment",
                "Dataset",
                "Chunker",
                "Retriever",
                "TopK",
                "Queries",
                "Recall",
                "Precision",
                "HitRate",
                "MRR",
                "Latency(ms)",
                "Faithfulness",
                "AnswerRelevancy",
            ])

    if not os.path.exists(FAILURE_CSV):
        with open(FAILURE_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp",
                "Experiment",
                "Dataset",
                "Chunker",
                "Retriever",
                "Error",
            ])


def save_success(result):

    with open(SUCCESS_CSV, "a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            result.experiment_name,
            result.dataset,
            result.chunker,
            result.retriever,
            result.top_k,
            result.num_queries,
            result.recall,
            result.precision,
            result.hit_rate,
            result.mrr,
            result.latency_ms,
            result.faithfulness,
            result.answer_relevancy,
        ])


def save_failure(dataset, chunker, retriever, experiment_name, error):

    with open(FAILURE_CSV, "a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            experiment_name,
            dataset,
            chunker,
            retriever,
            str(error),
        ])


def main():

    initialize_csv()

    embedding_model = E5Embedding()
    llm = OllamaLLM(model_name="qwen2.5:7b")

    datasets = [
        ("nq", "data/datasets/beir/nq"),
        ("hotpotqa", "data/datasets/beir/hotpotqa"),
        ("scifact", "data/datasets/beir/scifact"),
    ]

    chunkers = [
        ("fixed", FixedChunker()),
        ("recursive", RecursiveChunker()),
        ("semantic", SemanticChunker(embedding_model=embedding_model)),
        ("hierarchical", HierarchicalChunker()),
        ("adaptive", AdaptiveChunker()),
    ]

    retrievers = [
        "dense",
        "bm25",
        "hybrid",
    ]

    total = len(datasets) * len(chunkers) * len(retrievers)
    current = 1
    successful = 0
    failed = 0

    for dataset_name, dataset_path in datasets:

        print("\n" + "=" * 80)
        print(f"Starting Dataset: {dataset_name.upper()}")
        print("=" * 80)

        for chunker_name, chunker in chunkers:

            for retriever in retrievers:

                experiment_name = f"{chunker_name}_{retriever}_{dataset_name}"

                print("\n" + "=" * 70)
                print(f"Experiment {current}/{total}")
                print(f"Running : {experiment_name}")
                print("=" * 70)

                config = ExperimentConfig(
                    name=experiment_name,
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    chunker=chunker,
                    embedding_model=embedding_model,
                    retriever_type=retriever,
                    llm=llm,
                    top_k=5,
                )

                try:

                    runner = ExperimentRunner(config)
                    result = runner.run()

                    save_success(result)

                    successful += 1

                    print("\n===== Experiment Result =====")
                    print(f"Experiment : {result.experiment_name}")
                    print(f"Dataset    : {result.dataset}")
                    print(f"Chunker    : {result.chunker}")
                    print(f"Retriever  : {result.retriever}")

                    print("\n===== Retrieval Metrics =====")
                    print(f"Recall      : {result.recall:.4f}")
                    print(f"Precision   : {result.precision:.4f}")
                    print(f"Hit Rate    : {result.hit_rate:.4f}")
                    print(f"MRR         : {result.mrr:.4f}")
                    print(f"Latency(ms) : {result.latency_ms:.2f}")

                    print("\n===== RAGAS Metrics =====")
                    print(f"Faithfulness     : {result.faithfulness:.4f}")
                    print(f"Answer Relevancy : {result.answer_relevancy:.4f}")

                    print("\n✅ Saved to experiment_results.csv")

                except Exception as e:

                    failed += 1

                    save_failure(
                        dataset_name,
                        chunker_name,
                        retriever,
                        experiment_name,
                        e,
                    )

                    print("\n❌ Experiment Failed")
                    print(f"Experiment : {experiment_name}")
                    print(f"Error      : {e}")

                current += 1

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS FINISHED")
    print("=" * 80)
    print(f"Successful : {successful}")
    print(f"Failed     : {failed}")
    print(f"Results CSV : {SUCCESS_CSV}")
    print(f"Failures CSV: {FAILURE_CSV}")


if __name__ == "__main__":
    main()