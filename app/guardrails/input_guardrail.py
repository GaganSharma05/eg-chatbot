"""Deterministic validation for incoming customer messages."""

import re

from app.models.routing import GuardrailResult


class InputGuardrail:
    """Reject malformed messages and obvious instruction-override attempts."""

    _injection_patterns = (
        r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions",
        r"reveal\s+(?:your\s+)?system\s+prompt",
        r"forget\s+(?:your\s+)?(?:rules|restrictions)",
        r"you\s+are\s+now\s+(?:a\s+)?general(?:-purpose)?\s+chatbot",
        r"do\s+not\s+follow\s+(?:the\s+)?guardrails",
    )

    def __init__(self, max_message_length: int = 2_000) -> None:
        self.max_message_length = max_message_length

    async def validate(self, message: str) -> GuardrailResult:
        if not message or not message.strip():
            return GuardrailResult(False, "empty_message", "Please send a food-related question.")
        if len(message) > self.max_message_length:
            return GuardrailResult(False, "message_too_long", f"Please keep your message to {self.max_message_length:,} characters or fewer.")
        if any(re.search(pattern, message, re.IGNORECASE) for pattern in self._injection_patterns):
            return GuardrailResult(False, "prompt_injection", "I can help only with fruits, vegetables, dry fruits, nuts, exotic foods and gourmet foods. I can help with product information, general nutrition, storage, freshness and seasonal availability.")
        return GuardrailResult(True)
