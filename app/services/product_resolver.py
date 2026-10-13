"""Deterministic product recognition from stable and catalog vocabulary."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models.product_resolution import (
    ProductMatch,
    ProductMatchSource,
    ProductMatchStatus,
    ProductResolution,
)
from app.services.catalog_services import CatalogService


class ProductVocabularyError(ValueError):
    """Raised when the configured stable vocabulary cannot be used safely."""


@dataclass(frozen=True, slots=True)
class _VocabularyEntry:
    canonical_id: str
    source: ProductMatchSource
    priority: int
    catalog_product_ids: tuple[str, ...] = ()
    product_domain: str | None = None


@dataclass(frozen=True, slots=True)
class _Token:
    normalized: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Candidate:
    start_token: int
    end_token: int
    entries: tuple[_VocabularyEntry, ...]

    @property
    def token_length(self) -> int:
        return self.end_token - self.start_token


class ProductResolver:
    """Recognize product concepts without making availability decisions.

    Stable vocabulary is loaded and validated at construction. Catalog
    vocabulary is derived from the current CatalogService snapshot for every
    call, so successful catalog refreshes are visible immediately and failed
    refreshes continue to use CatalogService's preserved valid snapshot.
    """

    _token_pattern = re.compile(r"[a-z0-9]+", re.IGNORECASE)
    _canonical_pattern = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    _product_domains = {
        "fresh_fruit",
        "vegetable",
        "dry_fruit",
        "nut",
        "exotic_food",
        "gourmet_food",
    }
    _catalog_category_domains = {
        "fruit": "fresh_fruit",
        "fruits": "fresh_fruit",
        "vegetable": "vegetable",
        "vegetables": "vegetable",
        "dry": "dry_fruit",
        "dry fruit": "dry_fruit",
        "nut": "nut",
        "nuts": "nut",
        "exotic": "exotic_food",
        "gourmet": "gourmet_food",
    }
    _requested_phrase_patterns = (
        re.compile(
            r"\b(?:do|does)\s+(?:you|exotic[\s-]+grove)\s+"
            r"(?:have|sell|carry)\s+(?P<phrase>[^?.!]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:can\s+i\s+)?(?:buy|order)\s+(?P<phrase>[^?.!]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:can|will)\s+you\s+deliver\s+(?P<phrase>[^?.!]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:is|are)\s+(?P<phrase>[^?.!]+?)\s+(?:available|in\s+stock)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:price|cost)\s+of\s+(?P<phrase>[^?.!]+)",
            re.IGNORECASE,
        ),
    )

    def __init__(
        self,
        catalog_service: CatalogService,
        vocabulary_path: str | Path,
    ) -> None:
        self._catalog_service = catalog_service
        self._vocabulary_path = Path(vocabulary_path)
        self._catalog_id_to_external: dict[str, tuple[str, str | None]] = {}
        self._external_index = self._load_external_vocabulary()

    async def resolve(self, message: str) -> ProductResolution:
        """Resolve product concepts in deterministic textual order."""
        if not isinstance(message, str) or not message.strip():
            return ProductResolution()

        tokens = self._tokenize(message)
        if not tokens:
            return ProductResolution()

        combined_index = {
            alias: list(entries) for alias, entries in self._external_index.items()
        }
        catalog = None
        if await self._catalog_service.ensure_ready():
            catalog = await self._catalog_service.get_catalog()
        if catalog is not None:
            for alias, entries in self._catalog_index(catalog).items():
                combined_index.setdefault(alias, []).extend(entries)

        candidates = self._find_candidates(tokens, combined_index)
        selected = self._select_non_overlapping(candidates)
        requested_phrases = self._requested_phrases(message)
        matches = self._build_matches(message, tokens, selected, requested_phrases)

        represented_phrases = {match.original_phrase for match in matches}
        for candidate in selected:
            matched_text = message[
                tokens[candidate.start_token].start : tokens[candidate.end_token - 1].end
            ]
            containing_phrase = self._containing_requested_phrase(
                matched_text, requested_phrases
            )
            if containing_phrase is not None:
                represented_phrases.add(containing_phrase)
        for requested_phrase in requested_phrases:
            if requested_phrase not in represented_phrases:
                matches.append(
                    ProductMatch(
                        status=ProductMatchStatus.UNMATCHED,
                        original_phrase=requested_phrase,
                    )
                )
        matches.sort(
            key=lambda match: message.casefold().find(match.original_phrase.casefold())
        )
        return ProductResolution(matches=matches)

    def _load_external_vocabulary(
        self,
    ) -> dict[tuple[str, ...], tuple[_VocabularyEntry, ...]]:
        try:
            raw = self._vocabulary_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except OSError as exc:
            raise ProductVocabularyError(
                "Product vocabulary file is missing or unreadable"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ProductVocabularyError("Product vocabulary is not valid JSON") from exc

        if not isinstance(data, dict) or not isinstance(data.get("products"), list):
            raise ProductVocabularyError("Product vocabulary must contain a products list")
        if not data["products"]:
            raise ProductVocabularyError("Product vocabulary products list must not be empty")

        canonical_ids: set[str] = set()
        alias_owners: dict[tuple[str, ...], str] = {}
        index: dict[tuple[str, ...], list[_VocabularyEntry]] = {}
        for product in data["products"]:
            if not isinstance(product, dict):
                raise ProductVocabularyError("Each vocabulary product must be an object")
            canonical_id = product.get("canonical_id")
            aliases = product.get("aliases")
            product_domain = product.get("domain")
            catalog_product_ids = product.get("catalog_product_ids", [])
            if (
                not isinstance(canonical_id, str)
                or self._canonical_pattern.fullmatch(canonical_id) is None
            ):
                raise ProductVocabularyError("Canonical IDs must use lowercase snake_case")
            if canonical_id in canonical_ids:
                raise ProductVocabularyError(f"Duplicate canonical ID: {canonical_id}")
            canonical_ids.add(canonical_id)
            if (
                product_domain is not None
                and product_domain not in self._product_domains
            ):
                raise ProductVocabularyError(
                    f"Invalid product domain for {canonical_id}"
                )
            if (
                not isinstance(catalog_product_ids, list)
                or any(
                    not isinstance(product_id, str) or not product_id.strip()
                    for product_id in catalog_product_ids
                )
                or len(set(catalog_product_ids)) != len(catalog_product_ids)
            ):
                raise ProductVocabularyError(
                    f"Invalid catalog product IDs for {canonical_id}"
                )
            for catalog_product_id in catalog_product_ids:
                normalized_catalog_id = catalog_product_id.casefold()
                existing = self._catalog_id_to_external.get(normalized_catalog_id)
                if existing is not None and existing[0] != canonical_id:
                    raise ProductVocabularyError(
                        f"Catalog product ID maps to multiple concepts: {catalog_product_id}"
                    )
                self._catalog_id_to_external[normalized_catalog_id] = (
                    canonical_id,
                    product_domain,
                )
            if not isinstance(aliases, list) or not aliases:
                raise ProductVocabularyError(
                    f"Aliases must be a non-empty list for {canonical_id}"
                )

            seen_product_aliases: set[tuple[str, ...]] = set()
            normalized_aliases: list[tuple[str, ...]] = []
            for alias in aliases:
                normalized = self._normalize_phrase(alias) if isinstance(alias, str) else ()
                if not normalized:
                    raise ProductVocabularyError(f"Invalid alias for {canonical_id}")
                if normalized in seen_product_aliases:
                    raise ProductVocabularyError(f"Duplicate alias for {canonical_id}: {alias}")
                seen_product_aliases.add(normalized)
                owner = alias_owners.get(normalized)
                if owner is not None and owner != canonical_id:
                    raise ProductVocabularyError(
                        f"Conflicting alias maps to {owner} and {canonical_id}: {alias}"
                    )
                alias_owners[normalized] = canonical_id
                normalized_aliases.append(normalized)

            canonical_alias = self._normalize_phrase(canonical_id)
            all_aliases = [canonical_alias, *normalized_aliases]
            for alias in dict.fromkeys(all_aliases):
                priority = 0 if alias == canonical_alias else 1
                index.setdefault(alias, []).append(
                    _VocabularyEntry(
                        canonical_id=canonical_id,
                        source=ProductMatchSource.EXTERNAL,
                        priority=priority,
                        catalog_product_ids=tuple(catalog_product_ids),
                        product_domain=product_domain,
                    )
                )
        return {alias: tuple(entries) for alias, entries in index.items()}

    def _catalog_index(
        self, catalog: dict[str, Any]
    ) -> dict[tuple[str, ...], tuple[_VocabularyEntry, ...]]:
        index: dict[tuple[str, ...], list[_VocabularyEntry]] = {}
        products = catalog.get("products", [])
        if not isinstance(products, list):
            return {}
        for product in products:
            if not isinstance(product, dict):
                continue
            product_id = product.get("id")
            name = product.get("name")
            if not isinstance(product_id, str) or not isinstance(name, str):
                continue
            external_mapping = self._catalog_id_to_external.get(product_id.casefold())
            canonical_id = (
                external_mapping[0]
                if external_mapping is not None
                else self._canonicalize_catalog_id(product_id)
            )
            category = product.get("category")
            product_domain = (
                self._catalog_category_domains.get(category.casefold().strip())
                if isinstance(category, str)
                else None
            )
            if external_mapping is not None and external_mapping[1] is not None:
                product_domain = external_mapping[1]
            aliases: list[tuple[tuple[str, ...], int]] = [
                (self._normalize_phrase(product_id), 2),
                (self._normalize_phrase(name), 3),
            ]
            tags = product.get("tags")
            if isinstance(tags, list):
                aliases.extend(
                    (self._normalize_phrase(tag), 3)
                    for tag in tags
                    if isinstance(tag, str)
                )
            for alias, priority in aliases:
                if alias:
                    entry = _VocabularyEntry(
                        canonical_id=canonical_id,
                        source=ProductMatchSource.CATALOG,
                        priority=priority,
                        catalog_product_ids=(product_id,),
                        product_domain=product_domain,
                    )
                    if entry not in index.setdefault(alias, []):
                        index[alias].append(entry)
        return {alias: tuple(entries) for alias, entries in index.items()}

    def _find_candidates(
        self,
        tokens: list[_Token],
        index: dict[tuple[str, ...], list[_VocabularyEntry]],
    ) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        normalized = [token.normalized for token in tokens]
        for alias, entries in index.items():
            width = len(alias)
            for start in range(len(tokens) - width + 1):
                if tuple(normalized[start : start + width]) == alias:
                    candidates.append(
                        _Candidate(start, start + width, tuple(entries))
                    )
        return candidates

    @staticmethod
    def _select_non_overlapping(candidates: list[_Candidate]) -> list[_Candidate]:
        selected: list[_Candidate] = []
        occupied: set[int] = set()
        for candidate in sorted(
            candidates,
            key=lambda item: (
                item.start_token,
                -item.token_length,
                min(entry.priority for entry in item.entries),
            ),
        ):
            span = set(range(candidate.start_token, candidate.end_token))
            if span & occupied:
                continue
            selected.append(candidate)
            occupied.update(span)
        return sorted(selected, key=lambda item: item.start_token)

    def _build_matches(
        self,
        message: str,
        tokens: list[_Token],
        candidates: list[_Candidate],
        requested_phrases: list[str],
    ) -> list[ProductMatch]:
        matches: list[ProductMatch] = []
        emitted_canonical_ids: set[str] = set()
        emitted_ambiguities: set[tuple[str, ...]] = set()
        for candidate in candidates:
            matched_text = message[
                tokens[candidate.start_token].start : tokens[candidate.end_token - 1].end
            ]
            entries_by_id: dict[str, _VocabularyEntry] = {}
            for entry in sorted(candidate.entries, key=lambda item: item.priority):
                entries_by_id.setdefault(entry.canonical_id, entry)
            candidate_ids = sorted(entries_by_id)
            original_phrase = self._containing_requested_phrase(
                matched_text, requested_phrases
            ) or matched_text
            if len(candidate_ids) > 1:
                ambiguity = tuple(candidate_ids)
                if ambiguity in emitted_ambiguities:
                    continue
                emitted_ambiguities.add(ambiguity)
                matches.append(
                    ProductMatch(
                        status=ProductMatchStatus.AMBIGUOUS,
                        original_phrase=original_phrase,
                        matched_text=matched_text,
                        candidate_ids=candidate_ids,
                    )
                )
                continue

            canonical_id = candidate_ids[0]
            if canonical_id in emitted_canonical_ids:
                continue
            emitted_canonical_ids.add(canonical_id)
            selected_entry = entries_by_id[canonical_id]
            matching_entries = [
                entry
                for entry in candidate.entries
                if entry.canonical_id == canonical_id
            ]
            catalog_product_ids = list(
                dict.fromkeys(
                    product_id
                    for entry in matching_entries
                    for product_id in entry.catalog_product_ids
                )
            )
            product_domain = next(
                (
                    entry.product_domain
                    for entry in matching_entries
                    if entry.product_domain is not None
                ),
                None,
            )
            matches.append(
                ProductMatch(
                    status=ProductMatchStatus.MATCHED,
                    original_phrase=original_phrase,
                    canonical_id=canonical_id,
                    matched_text=matched_text,
                    source=selected_entry.source,
                    catalog_product_ids=catalog_product_ids,
                    product_domain=product_domain,
                )
            )
        return matches

    @classmethod
    def _tokenize(cls, text: str) -> list[_Token]:
        return [
            _Token(match.group(0).casefold(), match.start(), match.end())
            for match in cls._token_pattern.finditer(text)
        ]

    @classmethod
    def _normalize_phrase(cls, value: str) -> tuple[str, ...]:
        return tuple(match.group(0).casefold() for match in cls._token_pattern.finditer(value))

    @classmethod
    def _canonicalize_catalog_id(cls, product_id: str) -> str:
        return "_".join(cls._normalize_phrase(product_id))

    @classmethod
    def _requested_phrases(cls, message: str) -> list[str]:
        for pattern in cls._requested_phrase_patterns:
            match = pattern.search(message)
            if match:
                phrase = match.group("phrase").strip(" \t\r\n,;:")
                phrase = re.split(
                    r"\s*(?:;|,?\s+\bbut\b\s+)\s*",
                    phrase,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0]
                phrase = re.sub(r"^(?:a|an|the|your)\s+", "", phrase, flags=re.IGNORECASE)
                phrase = re.sub(r"\s+please$", "", phrase, flags=re.IGNORECASE)
                return [
                    part.strip()
                    for part in re.split(r"\s*(?:,|\band\b|\bor\b)\s*", phrase, flags=re.IGNORECASE)
                    if part.strip()
                    and part.strip().casefold()
                    not in {"it", "them", "this", "that", "these", "those"}
                ]
        return []

    @classmethod
    def _containing_requested_phrase(
        cls, matched_text: str, requested_phrases: list[str]
    ) -> str | None:
        matched_tokens = cls._normalize_phrase(matched_text)
        for phrase in requested_phrases:
            phrase_tokens = cls._normalize_phrase(phrase)
            width = len(matched_tokens)
            if any(
                phrase_tokens[start : start + width] == matched_tokens
                for start in range(len(phrase_tokens) - width + 1)
            ):
                return phrase
        return None
