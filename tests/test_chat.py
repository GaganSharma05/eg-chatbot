import pytest

from app.chatbot.chat_orchestrator import ChatOrchestrator
from app.conversation.conversation_service import ConversationService
from app.guardrails.domain_router import DomainRouter
from app.guardrails.input_guardrail import InputGuardrail
from app.guardrails.output_guardrail import OutputGuardrail
from app.llm.llm_client import LLMClient
from app.models.conversation import ChatMessage
from app.schemas.request import ChatRequest
from app.services.catalog_services import CatalogService
from tests.helpers import production_router


class FakeLLMClient(LLMClient):
    async def generate(self, system_prompt: str, user_message: str, history: list[ChatMessage]) -> str:
        return "Store ripe mangoes in the refrigerator to slow further ripening."


@pytest.mark.asyncio
async def test_storage_question_returns_structured_response() -> None:
    chatbot = ChatOrchestrator(InputGuardrail(), production_router(), ConversationService(), FakeLLMClient(), OutputGuardrail(), CatalogService("app/services/default.json"))
    response = await chatbot.chat(ChatRequest(message="How should I store ripe mangoes?", session_id="customer-123"))
    assert response.in_scope
    assert response.intent == "storage"
    assert response.product_domain == "fresh_fruit"
    assert response.identified_products == ["mango"]
