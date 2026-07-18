from chunking.adaptive import AdaptiveChunker
from chunking.fixed import FixedChunker
from chunking.hierarchical import HierarchicalChunker
from chunking.recursive import RecursiveChunker
from chunking.semantic import SemanticChunker
from embeddings.e5_embedding import E5Embedding
from experiments.configs.experiment_config import ExperimentConfig
from experiments.experiment_runner import ExperimentRunner
from experiments.utils.csv_writer import CSVWriter


def main():

    embedding_model = E5Embedding()

    # -----------------------------
    # Datasets
    # -----------------------------
    datasets = [
        ("nq", "data/datasets/beir/nq"),
        ("scifact", "data/datasets/beir/scifact"),
        ("hotpotqa", "data/datasets/beir/hotpotqa"),
    ]

    # -----------------------------
    # Chunkers
    # -----------------------------
    chunkers = [
        FixedChunker(),
        RecursiveChunker(),
        SemanticChunker(
            embedding_model=embedding_model,
        ),
        HierarchicalChunker(),
        AdaptiveChunker(),
    ]

    # -----------------------------
    # Retrievers
    # -----------------------------
    retrievers = [
        "dense",
        "bm25",
        "hybrid",
    ]

    results = []

    total = len(datasets) * len(chunkers) * len(retrievers)
    current = 1

    # ======================================================
    # Loop through all datasets
    # ======================================================

    for dataset_name, dataset_path in datasets:

        print()
        print("=" * 80)
        print(f"DATASET : {dataset_name.upper()}")
        print("=" * 80)

        # ==================================================
        # Loop through all chunkers
        # ==================================================

        for chunker in chunkers:

            # ==============================================
            # Loop through all retrievers
            # ==============================================

            for retriever in retrievers:

                experiment_name = (
                    f"{chunker.__class__.__name__}_"
                    f"{retriever}_"
                    f"{dataset_name}"
                )

                print("=" * 70)
                print(f"Running Experiment {current}/{total}")
                print(f"Experiment : {experiment_name}")
                print("=" * 70)

                config = ExperimentConfig(
                    name=experiment_name,
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    chunker=chunker,
                    embedding_model=embedding_model,
                    retriever_type=retriever,
                    top_k=5,
                )

                runner = ExperimentRunner(config)

                result = runner.run()

                results.append(result)

                print()
                print(f"Recall     : {result.recall:.4f}")
                print(f"Precision  : {result.precision:.4f}")
                print(f"Hit Rate   : {result.hit_rate:.4f}")
                print(f"MRR        : {result.mrr:.4f}")
                print(f"Latency(ms): {result.latency_ms:.2f}")
                print()

                current += 1

    print()
    print("=" * 80)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 80)

    print()

    print(
        f"{'Dataset':12}"
        f"{'Chunker':25}"
        f"{'Retriever':12}"
        f"{'Recall':10}"
        f"{'MRR':10}"
    )

    print("-" * 80)

    for result in results:

        print(
            f"{result.dataset:12}"
            f"{result.chunker:25}"
            f"{result.retriever:12}"
            f"{result.recall:.4f}    "
            f"{result.mrr:.4f}"
        )

    print()

    CSVWriter.write(results)

    print("\nResults successfully exported.")


if __name__ == "__main__":
    main()