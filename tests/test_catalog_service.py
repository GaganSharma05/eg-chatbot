import json
import asyncio
from pathlib import Path

import pytest

from app.services.catalog_services import CatalogService


def _catalog(*products: dict[str, object]) -> dict[str, object]:
    return {"groupName": "Test Grove", "products": list(products)}


def _product(
    product_id: str = "Imported-Blueberries",
    name: str = "Imported Blueberries",
    category: str = "Exotic",
    price: int = 225,
) -> dict[str, object]:
    return {
        "id": product_id,
        "name": name,
        "category": category,
        "unit": "1 Pack",
        "price": price,
        "description": f"Fresh {name}.",
        "media": ["product.png"],
    }


def _write_catalog(path: Path, catalog: object) -> None:
    path.write_text(json.dumps(catalog), encoding="utf-8")


@pytest.mark.asyncio
async def test_valid_catalog_load_and_cache_creation(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    expected = _catalog(_product())
    _write_catalog(catalog_path, expected)
    service = CatalogService(catalog_path)

    assert await service.refresh_catalog() is True
    cached = await service.get_catalog()
    assert cached == expected
    assert cached is not expected


@pytest.mark.asyncio
async def test_known_product_lookup_by_id_and_name(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    _write_catalog(catalog_path, _catalog(_product()))
    service = CatalogService(catalog_path)
    await service.refresh_catalog()

    assert (await service.find_product("imported-blueberries"))["id"] == "Imported-Blueberries"
    assert (await service.find_product("IMPORTED BLUEBERRIES"))["name"] == "Imported Blueberries"


@pytest.mark.asyncio
async def test_alias_plural_and_joined_product_lookup(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    _write_catalog(
        catalog_path,
        _catalog(
            _product(),
            _product("Avocados", "Avocados"),
            _product("Dragonfruit", "Dragon Fruit"),
        ),
    )
    service = CatalogService(catalog_path)
    await service.refresh_catalog()

    assert (await service.find_product("blueberry"))["id"] == "Imported-Blueberries"
    assert (await service.find_product("avocado"))["id"] == "Avocados"
    assert (await service.find_product("dragonfruit"))["id"] == "Dragonfruit"


@pytest.mark.asyncio
async def test_unknown_product_returns_none(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    _write_catalog(catalog_path, _catalog(_product()))
    service = CatalogService(catalog_path)
    await service.refresh_catalog()

    assert await service.find_product("kumquat") is None


@pytest.mark.asyncio
async def test_refresh_fully_replaces_cache(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    _write_catalog(catalog_path, _catalog(_product("Blueberry", "Blueberry", price=100)))
    service = CatalogService(catalog_path)
    assert await service.refresh_catalog()

    _write_catalog(catalog_path, _catalog(_product("Avocados", "Avocados", price=400)))
    assert await service.refresh_catalog()

    assert await service.find_product("blueberry") is None
    assert (await service.find_product("avocado"))["price"] == 400
    assert [product["id"] for product in (await service.get_catalog())["products"]] == ["Avocados"]


@pytest.mark.asyncio
async def test_invalid_json_preserves_previous_valid_cache(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    _write_catalog(catalog_path, _catalog(_product()))
    service = CatalogService(catalog_path)
    assert await service.refresh_catalog()
    before = await service.get_catalog()

    catalog_path.write_text("{ invalid json", encoding="utf-8")
    assert await service.refresh_catalog() is False
    assert await service.get_catalog() == before


@pytest.mark.asyncio
async def test_invalid_product_structure_preserves_previous_valid_cache(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    _write_catalog(catalog_path, _catalog(_product()))
    service = CatalogService(catalog_path)
    assert await service.refresh_catalog()
    before = await service.get_catalog()

    invalid_product = _product()
    invalid_product.pop("description")
    _write_catalog(catalog_path, _catalog(invalid_product))
    assert await service.refresh_catalog() is False
    assert await service.get_catalog() == before


@pytest.mark.asyncio
async def test_missing_catalog_is_not_ready_and_logs_safe_diagnostic(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    service = CatalogService(tmp_path / "missing.json")

    with caplog.at_level("WARNING"):
        assert await service.ensure_ready() is False

    assert await service.get_catalog() is None
    assert "Catalog refresh failed" in caplog.text


@pytest.mark.asyncio
async def test_concurrent_readers_see_only_complete_catalog_snapshots(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    old_catalog = _catalog(_product("Blueberry", "Blueberry", price=100))
    new_catalog = _catalog(_product("Avocados", "Avocados", price=400))
    _write_catalog(catalog_path, old_catalog)
    service = CatalogService(catalog_path)
    assert await service.refresh_catalog()
    _write_catalog(catalog_path, new_catalog)

    refresh = asyncio.create_task(service.refresh_catalog())
    reads = await asyncio.gather(*(service.get_catalog() for _ in range(50)))
    assert await refresh

    assert all(snapshot in (old_catalog, new_catalog) for snapshot in reads)
    assert await service.get_catalog() == new_catalog


@pytest.mark.parametrize(
    ("first_id", "duplicate_id"),
    [
        ("Duplicate-SKU", "Duplicate-SKU"),
        ("Duplicate-SKU", "duplicate-sku"),
        ("Duplicate-SKU", "  Duplicate-SKU  "),
        ("Berry-Boxes", "berry boxes"),
    ],
)
def test_duplicate_normalized_catalog_ids_have_actionable_validation_error(
    first_id: str, duplicate_id: str
) -> None:
    duplicate_catalog = _catalog(
        _product(first_id, "First Product"),
        _product(duplicate_id, "Second Product"),
    )

    with pytest.raises(
        ValueError,
        match="Duplicate normalized catalog product ID",
    ):
        CatalogService._build_snapshot(duplicate_catalog)


@pytest.mark.asyncio
async def test_duplicate_id_refresh_preserves_snapshot_and_allows_recovery(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    catalog_path = tmp_path / "catalog.json"
    original = _catalog(
        _product("Alpha-SKU", "Alpha Fruit", price=100),
        _product("Beta-SKU", "Beta Fruit", price=200),
    )
    _write_catalog(catalog_path, original)
    service = CatalogService(catalog_path)
    assert await service.refresh_catalog()

    duplicate_refresh = _catalog(
        _product("Duplicate-SKU", "Replacement One", price=300),
        _product(" duplicate sku ", "Replacement Two", price=400),
    )
    _write_catalog(catalog_path, duplicate_refresh)

    with caplog.at_level("WARNING"):
        assert await service.refresh_catalog() is False

    assert "Duplicate normalized catalog product ID" in caplog.text
    assert await service.get_catalog() == original
    assert (await service.find_product("Alpha-SKU"))["name"] == "Alpha Fruit"
    assert (await service.find_product("Beta-SKU"))["price"] == 200
    assert await service.find_product("Replacement One") is None
    assert await service.find_product("Replacement Two") is None

    recovered = _catalog(_product("Gamma-SKU", "Gamma Fruit", price=500))
    _write_catalog(catalog_path, recovered)
    assert await service.refresh_catalog()
    assert await service.get_catalog() == recovered
    assert (await service.find_product("Gamma-SKU"))["price"] == 500
    assert await service.find_product("Alpha-SKU") is None
