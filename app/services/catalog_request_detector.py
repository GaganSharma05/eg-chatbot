"""Deterministic detection of requests that need catalog data."""

from __future__ import annotations

import re

from app.models.routing import DomainRoute
from app.models.product_resolution import ProductMatch, ProductMatchStatus


class CatalogRequestDetector:
    """Decide whether a customer message needs a catalog lookup.

    The detector deliberately receives the ``DomainRoute`` created by
    ``DomainRouter`` instead of maintaining a second product registry.  The
    route is not used to decide product availability: a phrase such as "is
    this available" may refer to a product mentioned in prior conversation.
    """

    _commerce_patterns = (
        # Availability and stock. Seller-directed wording keeps a bare "have"
        # in nutritional questions from being interpreted as availability.
        r"\b(?:is|are)\s+this\s+available\b",
        r"\b(?:do|does|can|will)\s+you\s+(?:have|carry|sell)\b",
        r"\b(?:do|does|can|will)\s+exotic[\s-]+grove\s+"
        r"(?:have|carry|sell|offer)\b",
        r"\b(?:have\s+you\s+got|in\s+stock|stock\s+(?:of|for))\b",
        # Price and purchase intent.
        r"\b(?:price|prices|cost|costs|how\s+much|charge|charges)\b",
        # Fulfilment intent.
        r"\b(?:deliver|delivery|ship|shipping)\b",
        # Category/product-list requests directed to the store.
        r"\bwhat\s+(?:fruits?|vegetables?|products?|items?|nuts?|dry\s+fruits?)\s+do\s+you\s+(?:have|sell|carry)\b",
        r"\b(?:show|list)\s+(?:me\s+)?(?:your\s+)?(?:fruits?|vegetables?|products?|items?|nuts?|dry\s+fruits?)\b",
        r"\bwhat\s+(?:fruits?|vegetables?|products?|items?|nuts?|dry\s+fruits?)\s+"
        r"does\s+exotic[\s-]+grove\s+(?:have|sell|carry)\b",
    )
    _purchase_pattern = re.compile(r"\b(?:buy|purchase|order|orders)\b")
    _negated_purchase_pattern = re.compile(
        r"\b(?:"
        r"(?:do\s+not|don't|dont|does\s+not|doesn't|never)\s+"
        r"(?:(?:want|need|plan|intend)\s+)?(?:to\s+)?"
        r"(?:buy|purchase|order)|"
        r"(?:never|did\s+not|didn't|didnt)\s+(?:say|said)\s+"
        r"(?:i\s+)?(?:want|wanted|need|needed|plan|planned|intend|intended)\s+"
        r"to\s+(?:buy|purchase|order)"
        r")\b"
    )
    _clause_separator = re.compile(
        r"(?<=[.!?;])\s+|,?\s+\bbut\b\s+", re.IGNORECASE
    )
    _verified_attribute_pattern = re.compile(
        r"\b(?:organic|locally\s+sourced|local(?:ly)?\s+grown|certified|"
        r"quality|freshness|origin|sourcing)\b",
        re.IGNORECASE,
    )
    _general_inventory_pattern = re.compile(
        r"\b(?:what\s+(?:does\s+exotic[\s-]+grove|do\s+you)\s+"
        r"(?:sell|carry|offer)|which\s+products\s+do\s+you\s+have|"
        r"what\s+products\s+(?:do\s+you\s+have|are\s+available)|"
        r"(?:show|list|describe)\s+(?:me\s+)?(?:the\s+)?"
        r"(?:available\s+)?products)\b",
        re.IGNORECASE,
    )

    def requires_catalog_lookup(self, message: str, route: DomainRoute) -> bool:
        """Return whether this request requires the authoritative catalog.

        ``route`` is intentionally part of the public API so callers reuse the
        already-computed product identification.  Its product/domain metadata
        is not duplicated here and remains available to the next phase when a
        lookup is performed.
        """
        if not isinstance(message, str):
            return False
        if route.explicitly_out_of_scope:
            return False

        normalized = " ".join(message.casefold().split())
        # Generic availability is catalog-related only when the router has
        # already identified the product. "This available" remains supported
        # for conversational follow-ups, where the product is in prior context.
        has_product_evidence = any(
            match.status
            in {
                ProductMatchStatus.MATCHED,
                ProductMatchStatus.UNMATCHED,
                ProductMatchStatus.AMBIGUOUS,
            }
            for match in route.product_resolution.matches
        )
        if has_product_evidence and re.search(r"\bavailable\b", normalized):
            return True
        if (
            len(route.product_resolution.matches) > 1
            and re.search(r"\btell\s+me\s+about\b", normalized)
        ):
            return True
        if self._verified_attribute_pattern.search(normalized) and has_product_evidence:
            return True
        if self._general_inventory_pattern.search(normalized):
            return True
        if any(
            re.search(pattern, normalized) is not None
            for pattern in self._commerce_patterns
        ):
            return True
        return any(
            self._purchase_pattern.search(clause) is not None
            and self._negated_purchase_pattern.search(clause) is None
            for clause in self._clause_separator.split(normalized)
        )

    def catalog_matches(
        self, message: str, route: DomainRoute
    ) -> list[ProductMatch]:
        """Return product matches with affirmative catalog intent in their clause."""
        matches = route.product_resolution.matches
        if not matches:
            return []
        normalized = " ".join(message.casefold().split())
        if (
            len(matches) > 1
            and re.search(r"\btell\s+me\s+about\b", normalized)
        ):
            return list(matches)

        clauses = [
            clause.strip()
            for clause in self._clause_separator.split(message)
            if clause.strip()
        ]
        selected_state: dict[int, bool] = {}
        productless_followup = False
        for clause in clauses:
            clause_has_product = any(
                self._match_occurs_in_clause(match, clause) for match in matches
            )
            if self._clause_has_catalog_intent(clause) and not clause_has_product:
                productless_followup = True

        for index, match in enumerate(matches):
            for clause in clauses:
                if not self._match_occurs_in_clause(match, clause):
                    continue
                if self._negated_purchase_pattern.search(clause) is not None:
                    selected_state[index] = False
                elif self._clause_has_catalog_intent(clause):
                    selected_state[index] = True

        selected = [
            match
            for index, match in enumerate(matches)
            if selected_state.get(index, False)
        ]

        if not selected and productless_followup and len(matches) == 1:
            return list(matches)
        return selected

    def is_general_inventory_query(self, message: str) -> bool:
        """Return whether the message requests a store-wide product list."""
        return bool(self._general_inventory_pattern.search(message))

    def _clause_has_catalog_intent(self, clause: str) -> bool:
        normalized = " ".join(clause.casefold().split())
        if re.search(r"\bavailable\b", normalized):
            return True
        if self._verified_attribute_pattern.search(normalized):
            return True
        if any(re.search(pattern, normalized) for pattern in self._commerce_patterns):
            return True
        return (
            self._purchase_pattern.search(normalized) is not None
            and self._negated_purchase_pattern.search(normalized) is None
        )

    @staticmethod
    def _match_occurs_in_clause(match: ProductMatch, clause: str) -> bool:
        evidence = match.matched_text or match.original_phrase
        return bool(evidence) and evidence.casefold() in clause.casefold()


def requires_catalog_lookup(message: str, route: DomainRoute) -> bool:
    """Convenience function for the deterministic catalog-request decision."""
    return CatalogRequestDetector().requires_catalog_lookup(message, route)
