"""
Gemini adapter for RAGAS 0.2.15.
"""

from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig


class RagasGeminiLLM(BaseRagasLLM):
    """
    Adapter that allows RAGAS to use our GeminiLLM.
    """

    def __init__(self, gemini_llm):

        self.gemini_llm = gemini_llm

        self.run_config = RunConfig()


    def generate_text(
        self,
        prompt,
        n=1,
        temperature=0.0,
        stop=None,
        callbacks=None,
        **kwargs,
    ):
        """
        Synchronous generation.
        """

        return self.gemini_llm.generate(
            str(prompt)
        )


    async def agenerate_text(
        self,
        prompt,
        n=1,
        temperature=0.0,
        stop=None,
        callbacks=None,
        **kwargs,
    ):
        """
        Async generation.
        """

        return self.gemini_llm.generate(
            str(prompt)
        )