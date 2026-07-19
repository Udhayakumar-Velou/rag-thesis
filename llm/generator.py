"""
RAG generation pipeline.
"""

import time

from retrieval.base import BaseRetriever

from .answer import GeneratedAnswer
from .base import BaseLLM
from .prompt_builder import PromptBuilder


class RAGGenerator:
    """
    End-to-end Retrieval-Augmented Generation pipeline.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
    ):
        self.retriever = retriever
        self.llm = llm

    def generate(
        self,
        query: str,
        top_k: int = 5,
    ) -> GeneratedAnswer:
        """
        Generate an answer using retrieval and Gemini.
        """

        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        # Build prompt
        prompt = PromptBuilder.build_prompt(
            question=query,
            chunks=retrieved_chunks,
        )

        # Measure generation time
        start = time.perf_counter()

        answer = self.llm.generate(prompt)

        generation_time = time.perf_counter() - start

        return GeneratedAnswer(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            generation_time=generation_time,
        )