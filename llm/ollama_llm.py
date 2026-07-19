"""
Ollama LLM implementation.
"""

import requests

from .base import BaseLLM


class OllamaLLM(BaseLLM):

    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
    ):
        self.model_name = model_name


    def generate(
        self,
        prompt: str,
    ) -> str:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 512,
                    "temperature": 0,
                    "num_ctx": 4096
                },
            },
        )

        response.raise_for_status()

        return response.json()["response"].strip()