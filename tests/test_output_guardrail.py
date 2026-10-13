import pytest

from app.guardrails.output_guardrail import OutputGuardrail


BLUEBERRIES = {
    "canonical_id": "blueberry",
    "catalog_product_id": "Imported-Blueberries",
    "name": "Imported Blueberries",
    "price": 225,
    "unit": "1 Pack (140 gms)",
}
BLUEBERRY_RECORD = {
    "id": "Imported-Blueberries",
    "name": "Imported Blueberries",
    "price": 225,
    "unit": "1 Pack (140 gms)",
}


@pytest.mark.asyncio
async def test_unsupported_availability_claim_rejected() -> None:
    result = await OutputGuardrail().validate("Imported Blueberries are available.", None)
    assert not result.allowed
    assert "within 30 minutes" in result.answer


@pytest.mark.asyncio
async def test_unsupported_price_claim_rejected() -> None:
    result = await OutputGuardrail().validate("Imported Blueberries cost ₹225.", None)
    assert not result.allowed
    assert "within 30 minutes" in result.answer


@pytest.mark.asyncio
async def test_unsupported_delivery_claim_rejected() -> None:
    result = await OutputGuardrail().validate("Exotic Grove delivers imported blueberries.", None)
    assert not result.allowed
    assert "within 30 minutes" in result.answer


@pytest.mark.asyncio
async def test_unsupported_stock_claim_rejected() -> None:
    result = await OutputGuardrail().validate(
        "Exotic Grove has imported blueberries, but they are currently out of stock.",
        None,
    )
    assert not result.allowed
    assert "within 30 minutes" in result.answer


@pytest.mark.asyncio
async def test_unsupported_negative_stock_claim_rejected() -> None:
    result = await OutputGuardrail().validate(
        "I don't have any imported blueberries on hand.",
        None,
        {"name": "Imported Blueberries", "description": "Sweet and juicy imported blueberries."},
    )
    assert not result.allowed
    assert "within 30 minutes" in result.answer


@pytest.mark.asyncio
async def test_verified_availability_claim_accepted() -> None:
    result = await OutputGuardrail().validate(
        "Imported Blueberries are available.",
        None,
        {"name": "Imported Blueberries", "available": True},
    )
    assert result.allowed
    assert result.answer == "Imported Blueberries are available."


@pytest.mark.asyncio
async def test_verified_price_claim_accepted() -> None:
    result = await OutputGuardrail().validate(
        "Imported Blueberries cost ₹225.",
        None,
        {"name": "Imported Blueberries", "price": 225},
    )
    assert result.allowed
    assert result.answer == "Imported Blueberries cost ₹225."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        "Blueberries cost ₹225.",
        "Blueberries cost Rs 225.",
        "Blueberries cost Rs. 225.",
        "Blueberries cost INR 225.",
    ],
)
async def test_verified_price_formats_are_product_specific(answer: str) -> None:
    result = await OutputGuardrail().validate(
        answer, None, BLUEBERRY_RECORD, [BLUEBERRIES]
    )

    assert result.allowed
    assert result.answer == answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        "Apples cost Rs 225.",
        "Apples cost Rs 280.",
        "Blueberries cost Rs 280.",
        "The price is Rs 225.",
    ],
)
async def test_price_claim_rejects_wrong_or_missing_product_association(
    answer: str,
) -> None:
    result = await OutputGuardrail().validate(
        answer, None, BLUEBERRY_RECORD, [BLUEBERRIES]
    )

    assert not result.allowed
    assert "within 30 minutes" in result.answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "allowed"),
    [
        ("Blueberries cost Rs 225 for 1 Pack (140 gms).", True),
        ("Blueberries cost Rs 225.", True),
        ("Blueberries cost Rs 225 for 1 kg.", False),
        ("Blueberries cost Rs 225 for a 250 g pack.", False),
    ],
)
async def test_price_claim_validates_unit_and_pack_size(
    answer: str, allowed: bool
) -> None:
    result = await OutputGuardrail().validate(
        answer, None, BLUEBERRY_RECORD, [BLUEBERRIES]
    )

    assert result.allowed is allowed


@pytest.mark.asyncio
async def test_price_claim_does_not_invent_unit_when_catalog_has_none() -> None:
    record = {"id": "Loose-Fruit", "name": "Loose Fruit", "price": 50}
    expected = {
        "canonical_id": "loose_fruit",
        "catalog_product_id": "Loose-Fruit",
        "name": "Loose Fruit",
        "price": 50,
        "unit": None,
    }

    without_unit = await OutputGuardrail().validate(
        "Loose Fruit costs Rs 50.", None, record, [expected]
    )
    with_unit = await OutputGuardrail().validate(
        "Loose Fruit costs Rs 50 for 1 kg.", None, record, [expected]
    )

    assert without_unit.allowed
    assert not with_unit.allowed


@pytest.mark.asyncio
async def test_each_multi_product_price_uses_its_own_record() -> None:
    apples = {
        "canonical_id": "apple",
        "catalog_product_id": "Imported-Apples",
        "name": "Imported Apples",
        "price": 280,
        "unit": "1 Kg",
    }
    records = [BLUEBERRY_RECORD, {"id": "Imported-Apples", "name": "Imported Apples", "price": 280, "unit": "1 Kg"}]

    correct = await OutputGuardrail().validate(
        "Blueberries cost Rs 225 and apples cost Rs 280.",
        None,
        records,
        [BLUEBERRIES, apples],
    )
    swapped = await OutputGuardrail().validate(
        "Blueberries cost Rs 280 and apples cost Rs 225.",
        None,
        records,
        [BLUEBERRIES, apples],
    )

    assert correct.allowed
    assert not swapped.allowed


@pytest.mark.asyncio
async def test_verified_quality_description_accepted() -> None:
    result = await OutputGuardrail().validate(
        "Imported Blueberries are sweet and juicy.",
        None,
        {"name": "Imported Blueberries", "description": "Sweet and juicy berries."},
    )
    assert result.allowed
    assert result.answer == "Imported Blueberries are sweet and juicy."


@pytest.mark.asyncio
async def test_unsupported_verified_catalog_quality_claim_rejected() -> None:
    result = await OutputGuardrail().validate(
        "Exotic Grove has Imported Blueberries grown locally.",
        None,
        {"name": "Imported Blueberries", "description": "Sweet and juicy imported blueberries."},
    )
    assert not result.allowed
    assert "within 30 minutes" in result.answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        "Imported Blueberries are organic.",
        "Imported Blueberries are locally sourced.",
    ],
)
async def test_unverified_product_attributes_are_rejected_without_context(
    answer: str,
) -> None:
    result = await OutputGuardrail().validate(answer, None)

    assert not result.allowed
    assert "within 30 minutes" in result.answer


@pytest.mark.asyncio
async def test_medical_cure_claim_rejected() -> None:
    result = await OutputGuardrail().validate("Mango cures diabetes.", None)
    assert not result.allowed
    assert "cannot diagnose" in result.answer


@pytest.mark.asyncio
async def test_medical_prevention_claim_rejected() -> None:
    result = await OutputGuardrail().validate("Dragon fruit protects against cancer.", None)
    assert not result.allowed
    assert "cannot diagnose" in result.answer


@pytest.mark.asyncio
async def test_normal_nutrition_response_accepted() -> None:
    answer = "Mangoes contain vitamin C, fiber, and natural sugars, so they can be a refreshing fruit choice."
    result = await OutputGuardrail().validate(answer, None)
    assert result.allowed
    assert result.answer == answer


@pytest.mark.asyncio
async def test_general_product_description_accepted() -> None:
    answer = "Dragon fruit has a mild, lightly sweet taste with a soft texture and crunchy seeds."
    result = await OutputGuardrail().validate(answer, None)
    assert result.allowed
    assert result.answer == answer


@pytest.mark.asyncio
async def test_general_seasonal_availability_response_accepted() -> None:
    answer = "Watermelons are generally available in summer and taste best when heavy for their size."
    result = await OutputGuardrail().validate(answer, None)
    assert result.allowed
    assert result.answer == answer
