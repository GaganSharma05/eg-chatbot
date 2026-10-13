import pytest

from app.guardrails.domain_router import (
    DomainRouter,
    InformationIntent,
)
from tests.helpers import production_router


@pytest.mark.asyncio
async def test_exotic_grove_routes_as_brand() -> None:

    result = await production_router().route(
        "What is Exotic Grove?"
    )

    assert result.in_scope is True
    assert result.intent is InformationIntent.BRAND_INFORMATION
    assert result.identified_products == []


@pytest.mark.asyncio
async def test_hyphenated_exotic_grove_routes_as_brand() -> None:

    result = await production_router().route(
        "Tell me about exotic-grove"
    )

    assert result.in_scope is True
    assert result.intent is InformationIntent.BRAND_INFORMATION
    assert result.identified_products == []


@pytest.mark.asyncio
async def test_brand_company_question_is_in_scope() -> None:

    result = await production_router().route(
        "Is Exotic Grove a registered firm?"
    )

    assert result.in_scope is True
    assert result.intent is InformationIntent.BRAND_INFORMATION


@pytest.mark.asyncio
async def test_product_question_still_routes_normally() -> None:

    result = await production_router().route(
        "Do you have blueberries?"
    )

    assert result.in_scope is True
    assert "blueberry" in result.identified_products
