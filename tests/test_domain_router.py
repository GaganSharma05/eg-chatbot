import pytest

from app.guardrails.domain_router import DomainRouter
from app.models.routing import InformationIntent, ProductDomain
from tests.helpers import production_router


@pytest.mark.asyncio
async def test_routes_mango_storage_question() -> None:
    route = await production_router().route("How should I store ripe mangoes?")
    assert route.in_scope
    assert route.intent is InformationIntent.STORAGE
    assert route.product_domain is ProductDomain.FRESH_FRUIT
    assert route.identified_products == ["mango"]


@pytest.mark.asyncio
async def test_rejects_programming_question() -> None:
    route = await production_router().route("Help me write a Python program.")
    assert not route.in_scope


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Explain recursion using mangoes as an example.",
        "Explain binary search using blueberries.",
        "Write a sorting algorithm using apples as sample data.",
        "Explain data structures using mangoes.",
        "Debug code that processes blueberries.",
    ],
)
async def test_rejects_general_programming_concepts_with_products(
    message: str,
) -> None:
    route = await production_router().route(message)

    assert route.in_scope is False
    assert route.explicitly_out_of_scope is True


@pytest.mark.asyncio
async def test_mixed_catalog_programming_route_keeps_supported_product_only() -> None:
    route = await production_router().route(
        "What is the price of blueberries, and how do I calculate the total in Python?"
    )

    assert route.in_scope is True
    assert route.explicitly_out_of_scope is False
    assert route.identified_products == ["blueberry"]
    assert [match.original_phrase for match in route.product_resolution.matches] == [
        "blueberries"
    ]


@pytest.mark.asyncio
async def test_mixed_brand_programming_route_keeps_brand_portion() -> None:
    route = await production_router().route(
        "Tell me about Exotic Grove and explain recursion."
    )

    assert route.in_scope is True
    assert route.intent is InformationIntent.BRAND_INFORMATION


@pytest.mark.asyncio
async def test_routes_nutrition_multiple_products_and_freshness() -> None:
    router = production_router()
    nutrition = await router.route("Compare spinach and kale nutritionally.")
    freshness = await router.route("How do I know if an avocado is fresh?")
    assert nutrition.intent is InformationIntent.NUTRITION
    assert nutrition.identified_products == ["spinach", "kale"]
    assert freshness.intent is InformationIntent.FRESHNESS


@pytest.mark.asyncio
async def test_routes_watermelon_seasonality() -> None:
    route = await production_router().route("When is watermelon generally in season?")
    assert route.intent is InformationIntent.SEASONALITY


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "What is Exotic Grove?",
        "Tell me about Exotic Grove.",
        "Is Exotic Grove a registered firm?",
        "Tell me about exotic-grove",
    ],
)
async def test_routes_brand_questions_without_a_product(message: str) -> None:
    route = await production_router().route(message)

    assert route.in_scope is True
    assert route.intent is InformationIntent.BRAND_INFORMATION
    assert route.product_domain is None
    assert route.identified_products == []


@pytest.mark.asyncio
async def test_brand_name_does_not_hide_a_real_product() -> None:
    route = await production_router().route("Does Exotic Grove sell mangoes?")

    assert route.in_scope is True
    assert route.intent is InformationIntent.BRAND_INFORMATION
    assert route.product_domain is ProductDomain.FRESH_FRUIT
    assert route.identified_products == ["mango"]
