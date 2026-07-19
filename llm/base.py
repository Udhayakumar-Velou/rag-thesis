from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Base interface for all language models.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from a prompt.
        """
        pass