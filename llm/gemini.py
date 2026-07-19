"""
Gemini implementation of the BaseLLM interface.
"""

import os

from dotenv import load_dotenv
from google import genai

from .base import BaseLLM

load_dotenv()


class GeminiLLM(BaseLLM):
    """
    Gemini implementation using Google's GenAI SDK.
    """

    def __init__(self, model_name: str = None):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not found.")

        self.client = genai.Client(api_key=api_key)

        self.model_name = model_name or os.getenv(
            "GEMINI_MODEL",
            "gemini-flash-latest",
        )

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        return response.text.strip()