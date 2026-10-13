"""Async Ollama implementation of the LLM client."""

import httpx

from app.config.settings import Settings
from app.llm.llm_client import LLMClient
from app.models.conversation import ChatMessage


class LLMServiceUnavailableError(Exception):
    """Raised when Ollama cannot provide a usable response."""


class LlamaClient(LLMClient):
    """Call a local Ollama chat model over HTTP."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def generate(self, system_prompt: str, user_message: str, history: list[ChatMessage]) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend({"role": item.role, "content": item.content} for item in history)
        messages.append({"role": "user", "content": user_message})
        try:
            async with httpx.AsyncClient(base_url=self._settings.ollama_base_url, timeout=self._settings.ollama_timeout_seconds) as client:
                response = await client.post("/api/chat", json={"model": self._settings.ollama_model, "messages": messages, "stream": False})
                response.raise_for_status()
                answer = response.json().get("message", {}).get("content", "").strip()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMServiceUnavailableError from exc
        if not answer:
            raise LLMServiceUnavailableError
        return answer
