"""LLM provider abstraction."""

from abc import ABC, abstractmethod

from app.models.conversation import ChatMessage


class LLMClient(ABC):
    """Generate a response from a language model provider."""

    @abstractmethod
    async def generate(self, system_prompt: str, user_message: str, history: list[ChatMessage]) -> str:
        """Generate a response for a customer message."""
