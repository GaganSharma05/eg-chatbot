import pytest

from app.chatbot.chat_orchestrator import (
    UNKNOWN_CATALOG_PRODUCT_RESPONSE,
    ChatOrchestrator,
)
from app.conversation.conversation_service import ConversationService
from app.guardrails.domain_router import DomainRouter
from app.guardrails.input_guardrail import InputGuardrail
from app.guardrails.output_guardrail import OutputGuardrail
from app.llm.llm_client import LLMClient
from app.models.conversation import ChatMessage
from app.schemas.request import ChatRequest
from tests.helpers import production_router


class RecordingLLMClient(LLMClient):
    def __init__(self, answer: str = "Blueberries are a sweet and versatile fruit.") -> None:
        self.calls: list[tuple[str, str, list[ChatMessage]]] = []
        self.answer = answer

    async def generate(
        self, system_prompt: str, user_message: str, history: list[ChatMessage]
    ) -> str:
        self.calls.append((system_prompt, user_message, history))
        return self.answer


class MockCatalogService:
    def __init__(self, product: dict[str, object] | None) -> None:
        self.product = product
        self.calls: list[str] = []

    async def find_product(self, product_query: str) -> dict[str, object] | None:
        self.calls.append(product_query)
        return self.product


class RecordingConversationService(ConversationService):
    def __init__(self) -> None:
        super().__init__()
        self.interest_calls: list[str] = []

    async def record_product_interest(self, product: str) -> None:
        self.interest_calls.append(product)
        await super().record_product_interest(product)


def _orchestrator(
    llm: RecordingLLMClient,
    catalog: MockCatalogService,
    conversations: RecordingConversationService | None = None,
) -> ChatOrchestrator:
    return ChatOrchestrator(
        InputGuardrail(),
        production_router(),
        conversations or RecordingConversationService(),
        llm,
        OutputGuardrail(),
        catalog,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "What are the benefits of mango?",
        "How should I store avocados?",
        "How do I know if an avocado is fresh?",
        "When is watermelon generally in season?",
    ],
)
async def test_general_questions_do_not_call_catalog(message: str) -> None:
    llm = RecordingLLMClient()
    catalog = MockCatalogService(None)

    response = await _orchestrator(llm, catalog).chat(ChatRequest(message=message))

    assert response.in_scope
    assert catalog.calls == []
    assert len(llm.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Is blueberry available?",
        "What is the price of avocados?",
        "Can I order dragon fruit?",
    ],
)
async def test_catalog_requests_call_catalog(message: str) -> None:
    llm = RecordingLLMClient()
    catalog = MockCatalogService(
        {
            "id": "Imported-Blueberries",
            "name": "Imported Blueberries",
            "category": "Exotic",
            "variety": "Canadian",
            "unit": "1 Pack (140 gms)",
            "price": 225,
            "description": "Sweet and juicy imported blueberries.",
        }
    )

    await _orchestrator(llm, catalog).chat(ChatRequest(message=message))

    assert len(catalog.calls) == 1


@pytest.mark.asyncio
async def test_known_product_uses_deterministic_verified_catalog_response() -> None:
    llm = RecordingLLMClient()
    catalog = MockCatalogService(
        {
            "id": "Imported-Blueberries",
            "name": "Imported Blueberries",
            "category": "Exotic",
            "variety": "Canadian",
            "unit": "1 Pack (140 gms)",
            "price": 225,
            "description": "Sweet and juicy imported blueberries.",
        }
    )

    response = await _orchestrator(llm, catalog).chat(
        ChatRequest(message="Is blueberry available?")
    )

    assert response.answer == (
        "Exotic Grove has Imported Blueberries. "
        "Imported Blueberries unit: 1 Pack (140 gms). "
        "Imported Blueberries price: ₹225. Sweet and juicy imported blueberries."
    )
    assert llm.calls == []


@pytest.mark.asyncio
async def test_wrong_product_price_uses_verified_catalog_fallback() -> None:
    llm = RecordingLLMClient("Apples cost Rs 225.")
    catalog = MockCatalogService(
        {
            "id": "Imported-Blueberries",
            "name": "Imported Blueberries",
            "unit": "1 Pack (140 gms)",
            "price": 225,
            "description": "Sweet and juicy imported blueberries.",
        }
    )

    response = await _orchestrator(llm, catalog).chat(
        ChatRequest(message="What is the price of blueberries?")
    )

    assert response.answer == (
        "Exotic Grove has Imported Blueberries. "
        "Imported Blueberries unit: 1 Pack (140 gms). "
        "Imported Blueberries price: ₹225. Sweet and juicy imported blueberries."
    )
    assert llm.calls == []
    assert catalog.calls == ["Imported-Blueberries"]


@pytest.mark.asyncio
async def test_correct_product_price_is_accepted_without_fallback() -> None:
    llm = RecordingLLMClient("Blueberries cost Rs 225 for 1 Pack (140 gms).")
    catalog = MockCatalogService(
        {
            "id": "Imported-Blueberries",
            "name": "Imported Blueberries",
            "unit": "1 Pack (140 gms)",
            "price": 225,
            "description": "Sweet and juicy imported blueberries.",
        }
    )

    response = await _orchestrator(llm, catalog).chat(
        ChatRequest(message="What is the price of blueberries?")
    )

    assert response.answer == (
        "Exotic Grove has Imported Blueberries. "
        "Imported Blueberries unit: 1 Pack (140 gms). "
        "Imported Blueberries price: ₹225. Sweet and juicy imported blueberries."
    )
    assert llm.calls == []


@pytest.mark.asyncio
async def test_unknown_product_returns_deterministic_response_and_records_interest() -> None:
    llm = RecordingLLMClient()
    catalog = MockCatalogService(None)
    conversations = RecordingConversationService()

    response = await _orchestrator(llm, catalog, conversations).chat(
        ChatRequest(message="Do you have mangoes?", session_id="customer-123")
    )

    assert response.answer == UNKNOWN_CATALOG_PRODUCT_RESPONSE.format(product="mango")
    assert response.in_scope
    assert response.identified_products == ["mango"]
    assert catalog.calls == []
    assert llm.calls == []
    assert conversations.interest_calls == ["mango"]
    assert [message.role for message in await conversations.get_history("customer-123")] == [
        "user",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_catalog_request_keeps_existing_conversation_history() -> None:
    llm = RecordingLLMClient()
    catalog = MockCatalogService(
        {
            "id": "Imported-Blueberries",
            "name": "Imported Blueberries",
            "category": "Exotic",
            "unit": "1 Pack",
            "price": 225,
            "description": "Sweet and juicy imported blueberries.",
        }
    )
    conversations = RecordingConversationService()
    await conversations.save("customer-123", ChatMessage(role="user", content="Hello"))

    await _orchestrator(llm, catalog, conversations).chat(
        ChatRequest(message="Is blueberry available?", session_id="customer-123")
    )

    assert llm.calls == []
    assert len(await conversations.get_history("customer-123")) == 3
