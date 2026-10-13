"""Deterministic checks for generated answers."""

import re
from typing import Any

from app.models.routing import InformationIntent, OutputGuardrailResult


class OutputGuardrail:
    """Prevent unsafe medical claims and unverified catalog claims."""

    _medical_claim = re.compile(
        r"\b("
        r"cure[sd]?|"
        r"treat(?:s|ment)?|"
        r"diagnos(?:e|is)|"
        r"prescri(?:be|ption)|"
        r"stop (?:your )?(?:medication|medicine)|"
        r"replace (?:medical )?(?:care|treatment)|"
        r"protect(?:s|ing)? against (?:cancer|disease|diseases|diabetes)|"
        r"prevent(?:s|ing)? (?:cancer|disease|diseases|diabetes)|"
        r"reduce(?:s|ing)? (?:the )?risk of (?:cancer|disease|diseases|diabetes)"
        r")\b",
        re.IGNORECASE,
    )

    _company_catalog_claim = re.compile(
        r"\b(?:exotic grove|we|i)\s+(?:sell|sells|have|has|stock|stocks|carry|carries)\b",
        re.IGNORECASE,
    )
    _negative_company_catalog_claim = re.compile(
        r"\b(?:exotic grove|we|i)\s+(?:(?:do|does)\s+not|don't|doesn't)"
        r"\s+(?:sell|have|stock|carry)\b",
        re.IGNORECASE,
    )
    _availability_claim = re.compile(r"\b(?:available|in stock)\b", re.IGNORECASE)
    _stock_claim = re.compile(r"\b(?:in|out of)\s+stock\b|\bstock(?:ed|s)?\b", re.IGNORECASE)
    _delivery_claim = re.compile(r"\b(?:deliver|delivers|delivery|ship|shipping)\b", re.IGNORECASE)
    _retailer_cue = re.compile(r"\b(?:exotic grove|we|our|store|catalog)\b", re.IGNORECASE)
    _general_availability_cue = re.compile(
        r"\b(?:generally|usually|commonly|widely|seasonally|summer|winter|monsoon|"
        r"spring|autumn|fall|season|seasons|region|regions|market|markets|"
        r"january|february|march|april|may|june|july|august|september|october|"
        r"november|december)\b",
        re.IGNORECASE,
    )
    _catalog_fact_terms = re.compile(
        r"\b(?:fresh|premium|sweet|sweeter|juicy|crunchy|creamy|buttery|mild|tart|"
        r"imported|local|locally|native|grown|sourced|source|farm|organic|"
        r"smaller|larger|region|regions)\b",
        re.IGNORECASE,
    )
    _verified_attribute_claim = re.compile(
        r"\b(?:organic|locally\s+sourced|local(?:ly)?\s+grown|farm[- ]picked|"
        r"certified|grade\s+[a-z])\b",
        re.IGNORECASE,
    )
    _price_amount = re.compile(
        r"(?:₹|\brs\.?\s*|\binr\s*|\b(?:price|costs?)\s*(?:is|of)?\s*)"
        r"([0-9][0-9,]*(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )
    _quantity_unit = re.compile(
        r"\b([0-9]+(?:\.[0-9]+)?)\s*"
        r"(kilograms?|kgs?|grams?|g|gms?|packs?|pieces?|pcs?)\b",
        re.IGNORECASE,
    )
    _unverified_catalog_response = (
        "I’ll update you within 30 minutes with the availability and price for this product."
    )

    _max_words = 100

    async def validate(
        self,
        answer: str,
        intent: InformationIntent | None,
        catalog_context: dict[str, Any] | list[dict[str, Any]] | None = None,
        expected_products: list[dict[str, Any]] | None = None,
    ) -> OutputGuardrailResult:
        if self._medical_claim.search(answer):
            return OutputGuardrailResult(
                "I can provide general nutritional information about fruits and vegetables, "
                "but I cannot diagnose, treat, or recommend food as a replacement for medical "
                "care. Please consult a qualified healthcare professional for personalised advice.",
                False,
            )

        if self._has_unsupported_catalog_claim(
            answer, catalog_context, expected_products
        ):
            return OutputGuardrailResult(
                self._unverified_catalog_response,
                False,
            )

        return OutputGuardrailResult(
            self._limit_words(answer),
            True,
        )

    def _has_unsupported_catalog_claim(
        self,
        answer: str,
        catalog_context: dict[str, Any] | list[dict[str, Any]] | None,
        expected_products: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Check catalog assertions only against context supplied by orchestration."""
        records = self._catalog_records(catalog_context)
        has_negative_company_claim = self._negative_company_catalog_claim.search(answer) is not None
        has_company_claim = (
            self._company_catalog_claim.search(answer) is not None
            or has_negative_company_claim
        )
        has_availability_claim = (
            self._has_catalog_availability_claim(answer)
            or has_negative_company_claim
        )
        has_price_claim = bool(self._price_amount.findall(answer))
        has_delivery_claim = self._delivery_claim.search(answer) is not None
        has_verified_attribute_claim = (
            self._verified_attribute_claim.search(answer) is not None
        )

        if not (
            has_company_claim
            or has_availability_claim
            or has_price_claim
            or has_delivery_claim
            or has_verified_attribute_claim
        ):
            return False
        if not records:
            return True
        if (
            (
                has_company_claim
                or has_availability_claim
                or has_price_claim
                or has_delivery_claim
                or has_verified_attribute_claim
            )
            and self._has_unsupported_catalog_fact(answer, records)
        ):
            return True
        if has_company_claim and not any(self._record_matches_answer(record, answer) for record in records):
            return True
        if has_availability_claim and not any(self._supports_availability(record) for record in records):
            return True
        if has_price_claim and not self._supports_price(
            records, answer, expected_products
        ):
            return True
        if has_delivery_claim and not any(self._supports_delivery(record) for record in records):
            return True
        return False

    @staticmethod
    def _catalog_records(
        catalog_context: dict[str, Any] | list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if isinstance(catalog_context, dict):
            return [catalog_context]
        if isinstance(catalog_context, list):
            return [record for record in catalog_context if isinstance(record, dict)]
        return []

    def _has_catalog_availability_claim(self, answer: str) -> bool:
        if self._availability_claim.search(answer) is None:
            if self._stock_claim.search(answer) is not None:
                return True
            return False
        if self._stock_claim.search(answer) is not None:
            return True
        if self._retailer_cue.search(answer):
            return True
        return self._general_availability_cue.search(answer) is None

    @staticmethod
    def _supports_availability(record: dict[str, Any]) -> bool:
        return record.get("availability") is True or record.get("available") is True

    @staticmethod
    def _supports_delivery(record: dict[str, Any]) -> bool:
        return record.get("delivery") is True or record.get("delivery_available") is True

    def _supports_price(
        self,
        records: list[dict[str, Any]],
        answer: str,
        expected_products: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Validate every price against the trusted product named in its clause."""
        identities = expected_products or [
            {
                "catalog_product_id": record.get("id"),
                "name": record.get("name"),
                "price": record.get("price"),
                "unit": record.get("unit"),
            }
            for record in records
        ]
        claims = self._price_claim_segments(answer)
        if not claims:
            return False

        for segment, amounts in claims:
            matching_identities = [
                identity
                for identity in identities
                if self._identity_matches_text(identity, segment)
            ]
            if len(matching_identities) != 1:
                return False
            identity = matching_identities[0]
            verified_price = identity.get("price")
            if not isinstance(verified_price, (int, float)):
                return False
            if any(amount != float(verified_price) for amount in amounts):
                return False
            if not self._supports_claimed_units(identity.get("unit"), segment):
                return False
        return True

    def _price_claim_segments(self, answer: str) -> list[tuple[str, list[float]]]:
        claims: list[tuple[str, list[float]]] = []
        normalized_answer = re.sub(r"\brs\.\s*", "Rs ", answer, flags=re.IGNORECASE)
        for sentence in re.split(r"[.!?;\n]+", normalized_answer):
            segments = re.split(
                r"\b(?:and|but|while)\b", sentence, flags=re.IGNORECASE
            )
            for segment in segments:
                amounts = [
                    float(value.replace(",", ""))
                    for value in self._price_amount.findall(segment)
                ]
                if amounts:
                    claims.append((segment, amounts))
        return claims

    @classmethod
    def _identity_matches_text(
        cls, identity: dict[str, Any], text: str
    ) -> bool:
        text_tokens = cls._identity_tokens(text)
        aliases: list[tuple[str, ...]] = []
        for key in ("canonical_id", "catalog_product_id", "id", "name"):
            value = identity.get(key)
            if isinstance(value, str):
                tokens = cls._identity_tokens(value)
                if tokens:
                    aliases.append(tokens)
        return any(cls._contains_tokens(text_tokens, alias) for alias in aliases)

    @classmethod
    def _identity_tokens(cls, value: str) -> tuple[str, ...]:
        return tuple(
            cls._singular(token)
            for token in re.findall(r"[a-z0-9]+", value.casefold())
        )

    @staticmethod
    def _contains_tokens(
        text_tokens: tuple[str, ...], alias_tokens: tuple[str, ...]
    ) -> bool:
        width = len(alias_tokens)
        return any(
            text_tokens[index : index + width] == alias_tokens
            for index in range(len(text_tokens) - width + 1)
        )

    def _supports_claimed_units(self, verified_unit: object, segment: str) -> bool:
        claimed_units = self._normalised_units(segment)
        if not claimed_units:
            return True
        if not isinstance(verified_unit, str):
            return False
        return claimed_units.issubset(self._normalised_units(verified_unit))

    def _normalised_units(self, value: str) -> set[tuple[float, str]]:
        return {
            (float(amount), self._normalise_unit(unit))
            for amount, unit in self._quantity_unit.findall(value)
        }

    @staticmethod
    def _normalise_unit(unit: str) -> str:
        normalized = unit.casefold()
        if normalized.startswith(("kg", "kilogram")):
            return "kg"
        if normalized.startswith(("g", "gm", "gram")):
            return "g"
        if normalized.startswith("pack"):
            return "pack"
        return "piece"

    def _has_unsupported_catalog_fact(
        self, answer: str, records: list[dict[str, Any]]
    ) -> bool:
        """Reject sensitive catalog facts not present in verified context."""
        supported_text = " ".join(
            str(value).casefold()
            for record in records
            for value in record.values()
            if isinstance(value, (str, int, float, bool))
        )
        for match in self._catalog_fact_terms.finditer(answer):
            term = match.group(0).casefold()
            if term.endswith("er") and term[:-2] in supported_text:
                return True
            if term not in supported_text:
                return True
        return False

    @staticmethod
    def _record_matches_answer(record: dict[str, Any], answer: str) -> bool:
        """Avoid accepting a sales claim for a different catalog product."""
        product_values = (record.get("id"), record.get("name"))
        record_terms = {
            OutputGuardrail._singular(word)
            for value in product_values
            if isinstance(value, str)
            for word in re.findall(r"[a-z]{4,}", value.casefold())
        }
        answer_terms = {
            OutputGuardrail._singular(word)
            for word in re.findall(r"[a-z]{4,}", answer.casefold())
        }
        # Pronouns may refer to an item already established in the conversation.
        return not record_terms or bool(record_terms & answer_terms) or " it " in f" {answer.casefold()} "

    @staticmethod
    def _singular(word: str) -> str:
        if word.endswith("ies"):
            return f"{word[:-3]}y"
        if word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    def _limit_words(self, answer: str) -> str:
        """Keep answers within the maximum customer-facing word limit."""
        words = answer.split()

        if len(words) <= self._max_words:
            return answer

        limited_words = words[: self._max_words]
        limited_text = " ".join(limited_words)

        sentence_end = max(
            limited_text.rfind("."),
            limited_text.rfind("!"),
            limited_text.rfind("?"),
        )

        if sentence_end >= 0:
            return limited_text[: sentence_end + 1]

        return limited_text
