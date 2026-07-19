"""
Builds prompts for Retrieval-Augmented Generation (RAG).
"""

from typing import List

from models.chunk import Chunk

from .prompts import ANSWER_TEMPLATE, SYSTEM_PROMPT


class PromptBuilder:
    """
    Builds prompts for Gemini using retrieved chunks.
    """

    @staticmethod
    def build_context(chunks: List[Chunk]) -> str:
        """
        Convert retrieved chunks into a formatted context string.
        """

        if not chunks:
            return "No relevant context retrieved."

        context_parts = []

        for i, chunk in enumerate(chunks, start=1):
            context_parts.append(
                f"[Document {i}]\n{chunk.text}"
            )

        return "\n\n".join(context_parts)

    @staticmethod
    def build_prompt(
        question: str,
        chunks: List[Chunk],
    ) -> str:
        """
        Build the final prompt sent to Gemini.
        """

        context = PromptBuilder.build_context(chunks)

        user_prompt = ANSWER_TEMPLATE.format(
            context=context,
            question=question,
        )

        return f"{SYSTEM_PROMPT}\n\n{user_prompt}"