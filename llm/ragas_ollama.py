"""
Ollama wrapper for RAGAS 0.2.15
"""

from typing import Optional

import requests

from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig

from langchain_core.outputs import LLMResult, Generation


class RagasOllamaLLM(BaseRagasLLM):
    """
    Ollama LLM adapter for RAGAS evaluation.

    Uses local Ollama models instead of external APIs.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
    ):
        self.model_name = model_name
        self.run_config = RunConfig()


    def generate_text(
        self,
        prompt,
        n: int = 1,
        temperature: float = 0.0,
        stop: Optional[list[str]] = None,
        callbacks=None,
        **kwargs,
    ):
        """
        Generate text using Ollama.
        """

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model_name,
                "prompt": str(prompt),
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0,
                },
            },
        )

        response.raise_for_status()

        text = response.json()["response"]

        return LLMResult(
            generations=[
                [
                    Generation(
                        text=text
                    )
                ]
            ]
        )


    async def agenerate_text(
        self,
        prompt,
        n: int = 1,
        temperature: float = 0.0,
        stop: Optional[list[str]] = None,
        callbacks=None,
        **kwargs,
    ):
        """
        Async generation wrapper.
        """

        return self.generate_text(
            prompt=prompt,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )


    def is_finished(
        self,
        result,
    ) -> bool:
        """
        Required by RAGAS BaseRagasLLM.
        """

        return True