"""Stable contracts for deterministic product-vocabulary resolution."""

from dataclasses import dataclass, field
from enum import Enum


class ProductMatchStatus(str, Enum):
    """Outcome for one requested or recognized product phrase."""

    MATCHED = "matched"
    UNMATCHED = "unmatched"
    AMBIGUOUS = "ambiguous"


class ProductMatchSource(str, Enum):
    """Authoritative vocabulary source for a resolved match."""

    EXTERNAL = "external_vocabulary"
    CATALOG = "catalog_vocabulary"


@dataclass(frozen=True, slots=True)
class ProductMatch:
    """One resolution record.

    ``canonical_id`` and ``source`` are populated only for unambiguous matches.
    ``candidate_ids`` is populated only for ambiguous matches and is sorted.
    ``original_phrase`` preserves the containing requested phrase when one can
    be identified; ``matched_text`` preserves the exact text span that matched.
    ``catalog_product_ids`` contains only explicit authoritative mappings; an
    empty list means the concept is recognized but has no safe catalog mapping.
    ``product_domain`` is concept metadata used for routing, not availability.
    """

    status: ProductMatchStatus
    original_phrase: str
    canonical_id: str | None = None
    matched_text: str | None = None
    source: ProductMatchSource | None = None
    candidate_ids: list[str] = field(default_factory=list)
    catalog_product_ids: list[str] = field(default_factory=list)
    product_domain: str | None = None


@dataclass(frozen=True, slots=True)
class ProductResolution:
    """Ordered product matches for one message.

    Records are ordered by their first textual occurrence. Overlapping aliases
    prefer the longest phrase, and a canonical product appears at most once.
    An empty list means the message supplied no confidently extractable product
    evidence; it is distinct from an explicit ``UNMATCHED`` record.
    """

    matches: list[ProductMatch] = field(default_factory=list)
