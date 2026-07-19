from chunking.fixed import FixedChunker
from embeddings.e5_embedding import E5Embedding
from experiments.configs.experiment_config import ExperimentConfig
from experiments.experiment_runner import ExperimentRunner
from llm.ollama_llm import OllamaLLM


def main():

    config = ExperimentConfig(
        name="test_rag_experiment",

        dataset_name="scifact",

        dataset_path="data/datasets/beir/scifact",

        chunker=FixedChunker(
            chunk_size=500,
            chunk_overlap=50,
        ),

        embedding_model=E5Embedding(),

        retriever_type="dense",

        llm=OllamaLLM(),

        top_k=5,
    )


    runner = ExperimentRunner(config)

    result = runner.run()

    print(result)

    result.save(
        "experiments/results/test_rag_experiment.json"
    )


if __name__ == "__main__":
    main()