import json
from dataclasses import fields
from pathlib import Path

import pytest

from app.config.settings import DEFAULT_PRODUCT_VOCABULARY_PATH, Settings
from app.models.product_resolution import (
    ProductMatch,
    ProductMatchSource,
    ProductMatchStatus,
)
from app.services.catalog_services import CatalogService
from app.services.product_resolver import ProductResolver, ProductVocabularyError


def _product(
    product_id: str,
    name: str,
    *,
    tags: list[str] | None = None,
) -> dict[str, object]:
    product: dict[str, object] = {
        "id": product_id,
        "name": name,
        "category": "Test",
        "unit": "1 unit",
        "price": 100,
        "description": f"Test description for {name}.",
    }
    if tags is not None:
        product["tags"] = tags
    return product


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_vocabulary(path: Path, *products: dict[str, object]) -> None:
    _write_json(path, {"products": list(products)})


def _concept(canonical_id: str, *aliases: str) -> dict[str, object]:
    return {"canonical_id": canonical_id, "aliases": list(aliases)}


async def _resolver(
    tmp_path: Path,
    vocabulary: list[dict[str, object]],
    catalog_products: list[dict[str, object]] | None = None,
) -> tuple[ProductResolver, CatalogService, Path]:
    vocabulary_path = tmp_path / "vocabulary.json"
    catalog_path = tmp_path / "catalog.json"
    _write_vocabulary(vocabulary_path, *vocabulary)
    _write_json(catalog_path, {"products": catalog_products or []})
    catalog = CatalogService(catalog_path)
    assert await catalog.refresh_catalog()
    return ProductResolver(catalog, vocabulary_path), catalog, catalog_path


def test_valid_external_vocabulary_loads() -> None:
    catalog = CatalogService("unused.json")

    resolver = ProductResolver(catalog, DEFAULT_PRODUCT_VOCABULARY_PATH)

    assert resolver is not None


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("{invalid", "not valid JSON"),
        (json.dumps({"products": []}), "must not be empty"),
        (json.dumps({}), "products list"),
    ],
)
def test_malformed_or_empty_vocabulary_is_rejected(
    tmp_path: Path, content: str, expected: str
) -> None:
    path = tmp_path / "vocabulary.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ProductVocabularyError, match=expected):
        ProductResolver(CatalogService("unused.json"), path)


def test_missing_vocabulary_file_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ProductVocabularyError, match="missing or unreadable"):
        ProductResolver(CatalogService("unused.json"), tmp_path / "missing.json")


@pytest.mark.parametrize(
    ("products", "expected"),
    [
        (
            [_concept("mango", "mango"), _concept("mango", "mangoes")],
            "Duplicate canonical ID",
        ),
        ([_concept("mango", "mango", "MANGO")], "Duplicate alias"),
        (
            [_concept("mango", "fruit"), _concept("apple", "FRUIT")],
            "Conflicting alias",
        ),
        ([_concept("Mango", "mango")], "lowercase snake_case"),
        ([_concept("mango", "")], "Invalid alias"),
    ],
)
def test_invalid_vocabulary_entries_are_rejected(
    tmp_path: Path,
    products: list[dict[str, object]],
    expected: str,
) -> None:
    path = tmp_path / "vocabulary.json"
    _write_vocabulary(path, *products)

    with pytest.raises(ProductVocabularyError, match=expected):
        ProductResolver(CatalogService("unused.json"), path)


def test_catalog_product_id_cannot_map_to_multiple_concepts(tmp_path: Path) -> None:
    path = tmp_path / "vocabulary.json"
    products = [
        {
            **_concept("blueberry", "blueberry"),
            "domain": "fresh_fruit",
            "catalog_product_ids": ["Catalog-SKU"],
        },
        {
            **_concept("berry", "berry"),
            "domain": "fresh_fruit",
            "catalog_product_ids": ["Catalog-SKU"],
        },
    ]
    _write_vocabulary(path, *products)

    with pytest.raises(ProductVocabularyError, match="multiple concepts"):
        ProductResolver(CatalogService("unused.json"), path)


def test_vocabulary_setting_is_independent_of_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    configured = Settings(_env_file=None)

    assert configured.product_vocabulary_path == DEFAULT_PRODUCT_VOCABULARY_PATH
    assert configured.product_vocabulary_path.is_absolute()
    assert configured.product_vocabulary_path.is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected_id", "expected_text"),
    [
        ("Tell me about mango.", "mango", "mango"),
        ("Tell me about MANGOES.", "mango", "MANGOES"),
        ("Can I buy sweet_potato?", "sweet_potato", "sweet_potato"),
    ],
)
async def test_external_exact_alias_case_and_canonical_matching(
    tmp_path: Path, message: str, expected_id: str, expected_text: str
) -> None:
    resolver, _, _ = await _resolver(
        tmp_path,
        [_concept("mango", "mango", "mangoes"), _concept("sweet_potato", "sweet potato")],
    )

    result = await resolver.resolve(message)

    assert len(result.matches) == 1
    match = result.matches[0]
    assert match.status is ProductMatchStatus.MATCHED
    assert match.canonical_id == expected_id
    assert match.matched_text == expected_text
    assert match.source is ProductMatchSource.EXTERNAL


@pytest.mark.asyncio
async def test_multiple_products_are_ordered_by_textual_occurrence(tmp_path: Path) -> None:
    resolver, _, _ = await _resolver(
        tmp_path,
        [
            _concept("mango", "mango", "mangoes"),
            _concept("apple", "apple", "apples"),
            _concept("walnut", "walnut", "walnuts"),
        ],
    )

    result = await resolver.resolve("Walnuts, mangoes, and apples please.")

    assert [match.canonical_id for match in result.matches] == [
        "walnut",
        "mango",
        "apple",
    ]


@pytest.mark.asyncio
async def test_longest_overlapping_product_name_wins(tmp_path: Path) -> None:
    resolver, _, _ = await _resolver(
        tmp_path,
        [
            _concept("potato", "potato", "potatoes"),
            _concept("sweet_potato", "sweet potato", "sweet potatoes"),
        ],
    )

    result = await resolver.resolve("Do you have sweet potatoes?")

    assert [match.canonical_id for match in result.matches] == ["sweet_potato"]
    assert result.matches[0].original_phrase == "sweet potatoes"


@pytest.mark.asyncio
async def test_duplicate_occurrences_return_one_canonical_product(tmp_path: Path) -> None:
    resolver, _, _ = await _resolver(tmp_path, [_concept("mango", "mango", "mangoes")])

    result = await resolver.resolve("Mango and mangoes are both mango references.")

    assert [match.canonical_id for match in result.matches] == ["mango"]


@pytest.mark.asyncio
async def test_product_is_not_matched_inside_unrelated_word(tmp_path: Path) -> None:
    resolver, _, _ = await _resolver(tmp_path, [_concept("mango", "mango")])

    result = await resolver.resolve("Mangosteen has a thick rind.")

    assert result.matches == []


@pytest.mark.asyncio
async def test_component_match_preserves_full_requested_phrase(tmp_path: Path) -> None:
    resolver, _, _ = await _resolver(
        tmp_path, [_concept("sweet_potato", "sweet potato", "sweet potatoes")]
    )

    result = await resolver.resolve("Do you have sweet potato chips?")

    assert result.matches[0].canonical_id == "sweet_potato"
    assert result.matches[0].matched_text == "sweet potato"
    assert result.matches[0].original_phrase == "sweet potato chips"


@pytest.mark.asyncio
async def test_explicit_unknown_product_phrase_is_unmatched(tmp_path: Path) -> None:
    resolver, _, _ = await _resolver(tmp_path, [_concept("mango", "mango")])

    result = await resolver.resolve("Do you have purple yam?")

    assert result.matches == [
        ProductMatch(
            status=ProductMatchStatus.UNMATCHED,
            original_phrase="purple yam",
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "I don't want to buy mango, but what is the price?",
        "I don't want to buy mango, but can you tell me whether it is available?",
        "I don't want to buy mango; I am only asking about its health benefits.",
    ],
)
async def test_purchase_extraction_excludes_trailing_clause(
    tmp_path: Path, message: str
) -> None:
    resolver, _, _ = await _resolver(tmp_path, [_concept("mango", "mango")])

    result = await resolver.resolve(message)

    assert len(result.matches) == 1
    assert result.matches[0].canonical_id == "mango"
    assert result.matches[0].original_phrase == "mango"


@pytest.mark.asyncio
async def test_known_and_unknown_requested_products_are_distinguished(tmp_path: Path) -> None:
    resolver, _, _ = await _resolver(tmp_path, [_concept("mango", "mango", "mangoes")])

    result = await resolver.resolve("Do you have mangoes and purple yam?")

    assert result.matches == [
        ProductMatch(
            status=ProductMatchStatus.MATCHED,
            original_phrase="mangoes",
            canonical_id="mango",
            matched_text="mangoes",
            source=ProductMatchSource.EXTERNAL,
        ),
        ProductMatch(
            status=ProductMatchStatus.UNMATCHED,
            original_phrase="purple yam",
        ),
    ]


@pytest.mark.asyncio
async def test_no_product_evidence_is_distinct_from_unmatched(tmp_path: Path) -> None:
    resolver, _, _ = await _resolver(tmp_path, [_concept("mango", "mango")])

    result = await resolver.resolve("How do I write Python?")

    assert result.matches == []


@pytest.mark.asyncio
async def test_product_absent_from_catalog_remains_external_match(tmp_path: Path) -> None:
    resolver, _, _ = await _resolver(
        tmp_path,
        [_concept("sweet_potato", "sweet potato", "sweet potatoes")],
        [_product("Blueberry", "Blueberry")],
    )

    result = await resolver.resolve("Do you have sweet potatoes?")

    match = result.matches[0]
    assert match.canonical_id == "sweet_potato"
    assert match.source is ProductMatchSource.EXTERNAL
    assert all(field.name != "available" for field in fields(ProductMatch))


@pytest.mark.asyncio
async def test_external_concept_carries_explicit_authoritative_catalog_mapping() -> None:
    catalog = CatalogService("app/services/default.json")
    resolver = ProductResolver(catalog, DEFAULT_PRODUCT_VOCABULARY_PATH)

    result = await resolver.resolve("Do you have blueberries?")

    match = result.matches[0]
    assert match.canonical_id == "blueberry"
    assert match.catalog_product_ids == ["Imported-Blueberries"]
    assert match.product_domain == "fresh_fruit"


@pytest.mark.asyncio
async def test_catalog_name_id_and_supported_tag_are_dynamic_aliases(tmp_path: Path) -> None:
    resolver, _, _ = await _resolver(
        tmp_path,
        [_concept("mango", "mango")],
        [_product("Golden-Kiwi", "Golden Kiwi", tags=["SunGold kiwi"])],
    )

    by_name = await resolver.resolve("Tell me about Golden Kiwi")
    by_id = await resolver.resolve("Tell me about golden-kiwi")
    by_tag = await resolver.resolve("Tell me about SunGold kiwi")

    assert by_name.matches[0].canonical_id == "golden_kiwi"
    assert by_id.matches[0].canonical_id == "golden_kiwi"
    assert by_tag.matches[0].canonical_id == "golden_kiwi"
    assert all(
        result.matches[0].source is ProductMatchSource.CATALOG
        for result in (by_name, by_id, by_tag)
    )


@pytest.mark.asyncio
async def test_successful_refresh_adds_updates_and_removes_catalog_vocabulary(
    tmp_path: Path,
) -> None:
    resolver, catalog, catalog_path = await _resolver(
        tmp_path,
        [_concept("mango", "mango")],
        [_product("Purple-Yam", "Purple Yam")],
    )
    assert (await resolver.resolve("Purple Yam")).matches[0].canonical_id == "purple_yam"

    _write_json(
        catalog_path,
        {"products": [_product("Japanese-Yam", "Japanese Mountain Yam")]},
    )
    assert await catalog.refresh_catalog()

    assert (await resolver.resolve("Purple Yam")).matches == []
    updated = await resolver.resolve("Japanese Mountain Yam")
    assert updated.matches[0].canonical_id == "japanese_yam"


@pytest.mark.asyncio
async def test_updated_catalog_name_replaces_old_name_for_stable_product_id(
    tmp_path: Path,
) -> None:
    resolver, catalog, catalog_path = await _resolver(
        tmp_path,
        [_concept("mango", "mango")],
        [_product("Yam-SKU", "Purple Mountain Yam")],
    )
    assert (await resolver.resolve("Purple Mountain Yam")).matches[0].canonical_id == "yam_sku"

    _write_json(
        catalog_path,
        {"products": [_product("Yam-SKU", "Japanese Mountain Yam")]},
    )
    assert await catalog.refresh_catalog()

    assert (await resolver.resolve("Purple Mountain Yam")).matches == []
    updated = await resolver.resolve("Japanese Mountain Yam")
    assert updated.matches[0].canonical_id == "yam_sku"


@pytest.mark.asyncio
async def test_failed_refresh_preserves_previous_catalog_vocabulary(tmp_path: Path) -> None:
    resolver, catalog, catalog_path = await _resolver(
        tmp_path,
        [_concept("mango", "mango")],
        [_product("Purple-Yam", "Purple Yam")],
    )
    catalog_path.write_text("{invalid", encoding="utf-8")

    assert await catalog.refresh_catalog() is False
    result = await resolver.resolve("Purple Yam")

    assert result.matches[0].canonical_id == "purple_yam"
    assert result.matches[0].source is ProductMatchSource.CATALOG


@pytest.mark.asyncio
async def test_cross_source_alias_conflict_is_explicitly_ambiguous(tmp_path: Path) -> None:
    resolver, _, _ = await _resolver(
        tmp_path,
        [_concept("apple", "apple", "apples")],
        [_product("Green-Apple", "Apple")],
    )

    result = await resolver.resolve("Do you have apple?")

    assert result.matches == [
        ProductMatch(
            status=ProductMatchStatus.AMBIGUOUS,
            original_phrase="apple",
            matched_text="apple",
            candidate_ids=["apple", "green_apple"],
        )
    ]


@pytest.mark.asyncio
async def test_catalog_alias_conflict_is_deterministically_ambiguous(tmp_path: Path) -> None:
    resolver, _, _ = await _resolver(
        tmp_path,
        [_concept("mango", "mango")],
        [_product("Red-Apple", "Orchard Apple"), _product("Green-Apple", "Orchard Apple")],
    )

    result = await resolver.resolve("Orchard Apple")

    assert result.matches[0].status is ProductMatchStatus.AMBIGUOUS
    assert result.matches[0].candidate_ids == ["green_apple", "red_apple"]
    assert result.matches[0].canonical_id is None
    assert result.matches[0].source is None
