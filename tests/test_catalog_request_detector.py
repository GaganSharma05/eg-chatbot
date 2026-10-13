import pytest

from app.guardrails.domain_router import DomainRouter
from app.services.catalog_request_detector import CatalogRequestDetector
from tests.helpers import production_router


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Do you have mangoes?", True),
        ("Is blueberry available?", True),
        ("What is the price of avocados?", True),
        ("How much are your walnuts?", True),
        ("Are almonds in stock?", True),
        ("Can I buy dragon fruit?", True),
        ("Do you sell pomegranates?", True),
        ("Does Exotic Grove sell blueberries?", True),
        ("Does Exotic Grove have imported blueberries?", True),
        ("Does Exotic Grove carry dragon fruit?", True),
        ("Does Exotic Grove offer blueberries?", True),
        ("Can you deliver mangoes?", True),
        ("What fruits do you have?", True),
        ("What are the benefits of mango?", False),
        ("Tell me about dragon fruit.", False),
        ("How should I store avocados?", False),
        ("How do I know if an avocado is fresh?", False),
        ("When is watermelon generally in season?", False),
        ("What nutrients does mango have?", False),
        ("What does Exotic Grove do?", False),
        ("I never said I wanted to buy mango.", False),
        ("I never said I wanted to purchase blueberries.", False),
        ("I didn't say I wanted to order mango.", False),
        ("I want to buy mango.", True),
        ("I want to purchase blueberries.", True),
        ("Can I order mango?", True),
    ],
)
async def test_detects_catalog_lookup_from_customer_context(
    message: str, expected: bool
) -> None:
    route = await production_router().route(message)

    assert CatalogRequestDetector().requires_catalog_lookup(message, route) is expected


@pytest.mark.asyncio
async def test_detects_contextual_availability_without_a_current_product() -> None:
    """"This" can refer to a product named in an earlier conversation turn."""
    message = "Is this available in Mumbai?"
    route = await production_router().route(message)

    assert not route.identified_products
    assert CatalogRequestDetector().requires_catalog_lookup(message, route)
