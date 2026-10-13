"""Local catalog loading and authoritative in-memory catalog access."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REQUIRED_PRODUCT_FIELDS = ("id", "name", "category", "unit", "price", "description")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CatalogSnapshot:
    """An internally consistent catalog and its derived lookup indexes."""

    catalog: dict[str, Any]
    products: tuple[dict[str, Any], ...]
    product_index: dict[str, dict[str, Any]]
    category_index: dict[str, tuple[dict[str, Any], ...]]


class CatalogService:
    """Load a catalog JSON file and provide read-only local catalog access."""

    def __init__(self, catalog_path: str | Path) -> None:
        self._catalog_path = Path(catalog_path)
        # Retained for compatibility with the original service.
        self._cache: dict[str, Any] | None = None
        self._snapshot: _CatalogSnapshot | None = None
        self._refresh_lock = asyncio.Lock()

    async def refresh_catalog(self) -> bool:
        """Load and atomically replace the current catalog cache.

        The existing cache remains unchanged if the new JSON cannot be
        loaded or validated.
        """
        async with self._refresh_lock:
            try:
                raw_catalog = await asyncio.to_thread(
                    self._catalog_path.read_text, encoding="utf-8"
                )
                catalog_data = json.loads(raw_catalog)
                new_snapshot = self._build_snapshot(catalog_data)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.warning(
                    "Catalog refresh failed; retaining the previous snapshot (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
                return False

            # No await occurs between validation and replacement: readers see a
            # complete old snapshot or a complete new snapshot, never a partial one.
            self._snapshot = new_snapshot
            self._cache = new_snapshot.catalog
            return True

    async def ensure_ready(self) -> bool:
        """Ensure at least one valid catalog snapshot is available."""
        if self._snapshot is not None:
            return True
        return await self.refresh_catalog()

    async def get_catalog(self) -> dict[str, Any] | None:
        """Return a copy of the current cached catalog."""
        snapshot = self._snapshot
        return deepcopy(snapshot.catalog) if snapshot is not None else None

    async def find_product(self, product_query: str) -> dict[str, Any] | None:
        """Find one product by case-insensitive id, name, or safe alias."""
        snapshot = self._snapshot
        if snapshot is None:
            # The service remains usable when constructed during application
            # setup before an event loop is available for an eager refresh.
            await self.refresh_catalog()
            snapshot = self._snapshot
        if snapshot is None or not isinstance(product_query, str):
            return None

        product = snapshot.product_index.get(_normalise_product_term(product_query))
        return deepcopy(product) if product is not None else None

    async def find_category(self, category_query: str) -> list[dict[str, Any]]:
        """Return products in a category, or an empty list when it is unknown."""
        snapshot = self._snapshot
        if snapshot is None:
            await self.refresh_catalog()
            snapshot = self._snapshot
        if snapshot is None or not isinstance(category_query, str):
            return []

        products = snapshot.category_index.get(_normalise_product_term(category_query), ())
        return deepcopy(list(products))

    async def get_compact_catalog_record(
        self, product_query: str
    ) -> dict[str, Any] | None:
        """Return the LLM-suitable compact record for a matching product."""
        product = await self.find_product(product_query)
        return self.compact_catalog_record(product) if product is not None else None

    async def get_compact_catalog(self) -> list[dict[str, Any]]:
        """Return compact records for the complete current catalog."""
        snapshot = self._snapshot
        if snapshot is None:
            return []
        return [self.compact_catalog_record(product) for product in snapshot.products]

    @staticmethod
    def compact_catalog_record(product: dict[str, Any]) -> dict[str, Any]:
        """Select the stable, decision-useful fields for an LLM catalog record."""
        return {field: deepcopy(product[field]) for field in _REQUIRED_PRODUCT_FIELDS}


    async def clear_cache(self) -> None:
        """Remove the current in-memory catalog cache."""
        self._snapshot = None
        self._cache = None

    @staticmethod
    def _build_snapshot(catalog_data: Any) -> _CatalogSnapshot:
        if not isinstance(catalog_data, dict):
            raise ValueError("Catalog root must be an object")

        products_data = catalog_data.get("products")
        if not isinstance(products_data, list):
            raise ValueError("Catalog products must be a list")

        # Copy before indexing so the snapshot owns all data it exposes.
        catalog = deepcopy(catalog_data)
        products = catalog["products"]
        product_index: dict[str, dict[str, Any]] = {}
        category_index: dict[str, list[dict[str, Any]]] = {}
        normalized_product_ids: dict[str, str] = {}

        for product in products:
            if not isinstance(product, dict) or any(
                field not in product for field in _REQUIRED_PRODUCT_FIELDS
            ):
                raise ValueError("Each product must include all required fields")

            raw_product_id = str(product["id"])
            normalized_product_id = _normalise_product_term(raw_product_id)
            previous_product_id = normalized_product_ids.get(normalized_product_id)
            if previous_product_id is not None:
                raise ValueError(
                    "Duplicate normalized catalog product ID: "
                    f"{raw_product_id!r} conflicts with {previous_product_id!r}"
                )
            normalized_product_ids[normalized_product_id] = raw_product_id

            for term in _product_terms(product):
                # Keep the first catalog entry for an ambiguous broad alias.
                product_index.setdefault(term, product)

            category = _normalise_product_term(str(product["category"]))
            category_index.setdefault(category, []).append(product)

        return _CatalogSnapshot(
            catalog=catalog,
            products=tuple(products),
            product_index=product_index,
            category_index={key: tuple(value) for key, value in category_index.items()},
        )


def _normalise_product_term(value: str) -> str:
    """Normalize spacing, punctuation, case, and a conservative final plural."""
    words = re.findall(r"[a-z0-9]+", value.casefold())
    if words and len(words[-1]) > 3:
        if words[-1].endswith("ies"):
            words[-1] = f"{words[-1][:-3]}y"
        elif words[-1].endswith("s") and not words[-1].endswith("ss"):
            words[-1] = words[-1][:-1]
    return " ".join(words)


def _product_terms(product: dict[str, Any]) -> set[str]:
    """Create searchable terms for one catalog product."""

    terms = {
        _normalise_product_term(str(product["id"])),
        _normalise_product_term(str(product["name"])),
    }

    # Individual words from id and name.
    for value in (product["id"], product["name"]):
        words = re.findall(r"[a-z0-9]+", str(value).casefold())
        terms.update(
            _normalise_product_term(word)
            for word in words
            if len(word) > 2
        )

    # Support optional catalog tags.
    tags = product.get("tags", [])

    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str):
                normalized_tag = _normalise_product_term(tag)

                if normalized_tag:
                    terms.add(normalized_tag)

                    # Also index meaningful individual words.
                    words = re.findall(
                        r"[a-z0-9]+",
                        tag.casefold(),
                    )

                    terms.update(
                        _normalise_product_term(word)
                        for word in words
                        if len(word) > 2
                    )

    # Support spacing variations:
    # "Dragon Fruit" → "dragonfruit"
    joined = _normalise_product_term(
        str(product["name"])
    ).replace(" ", "")

    if joined:
        terms.add(joined)

    return terms
