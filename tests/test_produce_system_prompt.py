from app.prompts.produce_system_prompt import PRODUCE_SYSTEM_PROMPT


def test_catalog_context_is_authoritative_and_limited_to_present_fields() -> None:
    assert "VERIFIED CATALOG DATA" in PRODUCE_SYSTEM_PROMPT
    assert "treat it as authoritative" in PRODUCE_SYSTEM_PROMPT
    assert "only when they are actually present" in PRODUCE_SYSTEM_PROMPT
    assert "availability only when it is explicitly provided" in PRODUCE_SYSTEM_PROMPT
    assert "Do not add quality claims" in PRODUCE_SYSTEM_PROMPT


def test_prompt_requires_concise_customer_responses() -> None:
    assert "under 100 words" in PRODUCE_SYSTEM_PROMPT
    assert "40 to 80 words" in PRODUCE_SYSTEM_PROMPT
    assert "Answer directly first" in PRODUCE_SYSTEM_PROMPT
    assert "short paragraphs" in PRODUCE_SYSTEM_PROMPT


def test_prompt_preserves_medical_safety() -> None:
    assert "Do not diagnose or prescribe" in PRODUCE_SYSTEM_PROMPT
    assert "cures or treats disease" in PRODUCE_SYSTEM_PROMPT
    assert "stopping medication" in PRODUCE_SYSTEM_PROMPT
    assert "replaces professional medical care" in PRODUCE_SYSTEM_PROMPT


def test_prompt_rejects_unsupported_catalog_claims() -> None:
    assert "Never invent stock, price, delivery, availability" in PRODUCE_SYSTEM_PROMPT
    assert "does not mean Exotic Grove sells it" in PRODUCE_SYSTEM_PROMPT
    assert "do not make Exotic Grove availability, price, stock, delivery, or sales claims" in PRODUCE_SYSTEM_PROMPT


def test_prompt_handles_exotic_grove_questions_with_official_link() -> None:
    assert "https://exotic-grove.vercel.app/about" in PRODUCE_SYSTEM_PROMPT
    assert "Do not invent other company facts" in PRODUCE_SYSTEM_PROMPT


def test_prompt_allows_concise_general_food_information() -> None:
    assert "taste, nutrition, storage, freshness, seasonality, common uses" in PRODUCE_SYSTEM_PROMPT
    assert "fresh fruits, vegetables, dry fruits, nuts, exotic foods, and gourmet foods" in PRODUCE_SYSTEM_PROMPT
