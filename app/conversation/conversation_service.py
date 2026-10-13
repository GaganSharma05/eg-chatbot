"""Replaceable in-memory conversation history for V1."""

from collections import defaultdict

from app.models.conversation import ChatMessage


class ConversationService:
    """Keep a bounded history by session ID."""

    def __init__(self, max_messages_per_session: int = 10) -> None:
        self._history: dict[str, list[ChatMessage]] = defaultdict(list)
        self._product_interest: set[str] = set()
        self._max_messages = max_messages_per_session

    async def get_history(self, session_id: str) -> list[ChatMessage]:
        return list(self._history[session_id])

    async def save(self, session_id: str, message: ChatMessage) -> None:
        history = self._history[session_id]
        history.append(message)
        del history[:-self._max_messages]

    async def record_product_interest(self, product: str) -> None:
        """Record customer product interest in memory for future demand analysis."""
        self._product_interest.add(product)
