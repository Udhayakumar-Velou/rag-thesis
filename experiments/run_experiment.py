from chunking.fixed import FixedChunker
from embeddings.e5_embedding import E5Embedding
from experiments.configs.experiment_config import ExperimentConfig
from experiments.experiment_runner import ExperimentRunner


def main():

    config = ExperimentConfig(
        name="fixed_dense_nq",
        dataset_name="nq",
        dataset_path="data/datasets/beir/nq",
        chunker=FixedChunker(),
        embedding_model=E5Embedding(),
        retriever_type="hybrid",
        top_k=5,
    )

    runner = ExperimentRunner(config)

    result = runner.run()

    print("\n===== Experiment Result =====")
    print(f"Experiment : {result.experiment_name}")
    print(f"Dataset    : {result.dataset}")
    print(f"Chunker    : {result.chunker}")
    print(f"Retriever  : {result.retriever}")
    print(f"Top-K      : {result.top_k}")
    print(f"Queries    : {result.num_queries}")

    print("\nMetrics")
    print(f"Recall     : {result.recall:.4f}")
    print(f"Precision  : {result.precision:.4f}")
    print(f"Hit Rate   : {result.hit_rate:.4f}")
    print(f"MRR         : {result.mrr:.4f}")
    print(f"Latency(ms): {result.latency_ms:.2f}")


if __name__ == "__main__":
    main()