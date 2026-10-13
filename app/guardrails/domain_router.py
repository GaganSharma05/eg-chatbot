"""Deterministic domain and intent routing using structured product resolution."""

import re

from app.models.routing import DomainRoute, InformationIntent, ProductDomain
from app.models.product_resolution import ProductMatchStatus, ProductResolution
from app.services.product_resolver import ProductResolver


class DomainRouter:
    """Route food questions without owning product vocabulary."""
    _intent_keywords = {
        InformationIntent.STORAGE: ("store", "storage", "refrigerat", "freeze", "keep"),
        InformationIntent.FRESHNESS: ("fresh", "freshness", "ripe", "spoil", "spoiled", "select", "choose"),
        InformationIntent.SEASONALITY: ("season", "seasonal", "harvest", "when is"),
        InformationIntent.NUTRITION: ("nutrition", "nutritious", "calorie", "vitamin", "mineral", "fibre", "fiber", "healthy", "benefit"),
    }
    _domain_terms = {
        ProductDomain.FRESH_FRUIT: ("fruit", "fruits"), ProductDomain.VEGETABLE: ("vegetable", "vegetables"),
        ProductDomain.DRY_FRUIT: ("dry fruit", "dried fruit"), ProductDomain.NUT: ("nut", "nuts"),
        ProductDomain.EXOTIC_FOOD: ("exotic food", "exotic foods", "exotic fruit"), ProductDomain.GOURMET_FOOD: ("gourmet", "gourmet food"),
    }
    _brand_pattern = re.compile(r"(?<!\w)exotic[\s-]+grove(?!\w)", re.IGNORECASE)
    _programming_pattern = re.compile(
        r"\b(?:python|javascript|programming|html|css|recursion|recursive|"
        r"algorithms?|data\s+structures?|binary\s+search|debugging|debug|"
        r"code\s+execution|programming\s+logic)\b|"
        r"\b(?:write|create|build|develop|code|sort)\b.{0,80}"
        r"\b(?:program|function|website|code)\b",
        re.IGNORECASE,
    )
    _mixed_clause_separator = re.compile(
        r"(?<=[.!?;])\s+|,\s*(?:and|but)\s+|"
        r"\s+(?:and|but)\s+(?=(?:how|explain|write|debug|run|execute|"
        r"implement|calculate|tell)\b)",
        re.IGNORECASE,
    )

    def __init__(self, product_resolver: ProductResolver) -> None:
        self._product_resolver = product_resolver

    async def route(self, message: str) -> DomainRoute:
        product_resolution = await self._product_resolver.resolve(message)
        supported_message, has_programming = self.supported_message(message)
        if has_programming and not supported_message:
            return self._out_of_scope_route(product_resolution)
        if has_programming:
            product_resolution = ProductResolution(
                matches=[
                    match
                    for match in product_resolution.matches
                    if self._match_is_in_supported_text(match, supported_message)
                ]
            )

        normalized = supported_message.casefold()
        matched_products = [
            match
            for match in product_resolution.matches
            if match.status is ProductMatchStatus.MATCHED
            and match.canonical_id is not None
        ]
        products = [match.canonical_id for match in matched_products]
        product_domains = [
            ProductDomain(match.product_domain)
            for match in matched_products
            if match.product_domain is not None
        ]
        if self._brand_pattern.search(supported_message):
            return DomainRoute(
                True,
                product_domains[0] if product_domains else None,
                InformationIntent.BRAND_INFORMATION,
                products,
                product_resolution,
            )

        intent = self._find_intent(normalized)
        domain = product_domains[0] if product_domains else self._find_domain(normalized)
        has_ambiguous_product = any(
            match.status is ProductMatchStatus.AMBIGUOUS
            for match in product_resolution.matches
        )
        in_scope = domain is not None or bool(matched_products) or has_ambiguous_product
        return DomainRoute(
            in_scope,
            domain,
            intent or (InformationIntent.INFORMATION if in_scope else None),
            products,
            product_resolution,
        )

    def supported_message(self, message: str) -> tuple[str, bool]:
        """Return the supported clauses and whether programming was removed."""
        if self._programming_pattern.search(message) is None:
            return message, False
        clauses = [
            clause.strip(" \t\r\n,;:")
            for clause in self._mixed_clause_separator.split(message)
            if clause.strip(" \t\r\n,;:")
        ]
        supported = [
            clause
            for clause in clauses
            if self._programming_pattern.search(clause) is None
        ]
        return " ".join(supported), True

    @staticmethod
    def _match_is_in_supported_text(match: object, supported_text: str) -> bool:
        matched_text = getattr(match, "matched_text", None)
        original_phrase = getattr(match, "original_phrase", "")
        evidence = matched_text or original_phrase
        return bool(evidence) and str(evidence).casefold() in supported_text.casefold()

    @staticmethod
    def _out_of_scope_route(product_resolution: ProductResolution) -> DomainRoute:
        matched_products = [
            match
            for match in product_resolution.matches
            if match.status is ProductMatchStatus.MATCHED
            and match.canonical_id is not None
        ]
        product_domains = [
            ProductDomain(match.product_domain)
            for match in matched_products
            if match.product_domain is not None
        ]
        return DomainRoute(
            False,
            product_domains[0] if product_domains else None,
            None,
            [match.canonical_id for match in matched_products],
            product_resolution,
            True,
        )

    def _find_intent(self, message: str) -> InformationIntent | None:
        for intent, keywords in self._intent_keywords.items():
            if any(keyword in message for keyword in keywords):
                return intent
        return None

    def _find_domain(self, message: str) -> ProductDomain | None:
        for domain, terms in self._domain_terms.items():
            if any(term in message for term in terms):
                return domain
        return None
