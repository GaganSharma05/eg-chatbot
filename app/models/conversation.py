"""Conversation domain models."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message retained as conversation context."""

    role: Literal["user", "assistant"]
    content: str
