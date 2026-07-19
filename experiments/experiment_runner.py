from evaluation.dataset import EvaluationDataset
from evaluation.evaluator import Evaluator
from evaluation.ragas_metrics import RagasEvaluator
from experiments.configs.experiment_config import ExperimentConfig
from experiments.results import ExperimentResult
from loaders.beir_loader import BEIRLoader
from retrieval.factory import RetrieverFactory

from llm.answer import GeneratedAnswer
from llm.generator import RAGGenerator
from llm.ragas_ollama import RagasOllamaLLM


class ExperimentRunner:
    """
    Executes a complete retrieval experiment.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self) -> ExperimentResult:
        """
        Execute the complete retrieval pipeline.
        """

        # --------------------------------------------------
        # Load BEIR dataset
        # --------------------------------------------------

        documents, queries, qrels = BEIRLoader().load(
            self.config.dataset_path
        )

        # --------------------------------------------------
        # DEVELOPMENT MODE
        # --------------------------------------------------

        MAX_DOCUMENTS = 100

        if MAX_DOCUMENTS is not None:

            documents = documents[:MAX_DOCUMENTS]

            available_document_ids = {
                document.id for document in documents
            }

            filtered_queries = {}
            filtered_qrels = {}

            for query_id, relevance in qrels.items():

                valid_relevance = {
                    document_id: score
                    for document_id, score in relevance.items()
                    if document_id in available_document_ids
                }

                if valid_relevance:
                    filtered_qrels[query_id] = valid_relevance
                    filtered_queries[query_id] = queries[query_id]

            queries = filtered_queries
            qrels = filtered_qrels

            print("=" * 60)
            print("DEVELOPMENT MODE")
            print("=" * 60)
            print(f"Documents : {len(documents)}")
            print(f"Queries   : {len(queries)}")
            print("=" * 60)

        # --------------------------------------------------
        # Create evaluation dataset
        # --------------------------------------------------

        dataset = EvaluationDataset.from_beir(
            name=self.config.dataset_name,
            documents=documents,
            queries=queries,
            qrels=qrels,
        )

        # --------------------------------------------------
        # Chunk documents
        # --------------------------------------------------

        print("Creating chunks...")

        chunks = []

        for document in documents:
            chunks.extend(
                self.config.chunker.split(
                    text=document.text,
                    document_id=document.id,
                )
            )

        print(f"Chunks Created : {len(chunks)}")
        print("=" * 60)

        # --------------------------------------------------
        # Create retriever
        # --------------------------------------------------

        print("Building retriever...")

        retriever = RetrieverFactory.create(
            retriever_type=self.config.retriever_type,
            chunks=chunks,
            embedding_model=self.config.embedding_model,
        )

        print("Retriever ready.")
        print("=" * 60)

        # --------------------------------------------------
        # Create RAG Generator
        # --------------------------------------------------

        generator = RAGGenerator(
            retriever=retriever,
            llm=self.config.llm,
        )

        # --------------------------------------------------
        # Generate Answers
        # --------------------------------------------------

        print("Generating answers...")

        generated_answers: list[GeneratedAnswer] = []

        for sample in dataset:

            generated_answer = generator.generate(
                query=sample.query,
                top_k=self.config.top_k,
            )

            generated_answers.append(generated_answer)

        print(f"Generated {len(generated_answers)} answers.")
        print("=" * 60)

        # --------------------------------------------------
        # RAGAS Evaluation
        # --------------------------------------------------

        print("Running RAGAS evaluation...")

        ragas_llm = RagasOllamaLLM(
            model_name="qwen2.5:3b"
        )

        ragas_evaluator = RagasEvaluator(
        llm=ragas_llm,
        embeddings=self.config.embedding_model,
    )

        ragas_result = ragas_evaluator.evaluate(
            generated_answers=generated_answers,
            dataset=dataset,
        )

        print("RAGAS evaluation completed.")
        print("=" * 60)

        # --------------------------------------------------
        # Retrieval Evaluation
        # --------------------------------------------------

        print("Running retrieval evaluation...")

        evaluator = Evaluator(
            retriever=retriever,
            top_k=self.config.top_k,
        )

        evaluation_results = evaluator.evaluate(dataset)

        print("Retrieval evaluation completed.")
        print("=" * 60)

        # --------------------------------------------------
        # Aggregate Results
        # --------------------------------------------------

        return ExperimentResult.from_evaluation(
            experiment_name=self.config.name,
            dataset=self.config.dataset_name,
            chunker=self.config.chunker.__class__.__name__,
            retriever=self.config.retriever_type,
            top_k=self.config.top_k,
            results=evaluation_results,
            ragas_result=ragas_result,
        )