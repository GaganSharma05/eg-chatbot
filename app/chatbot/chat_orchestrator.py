"""Business workflow for a produce chat request."""

import re

from app.conversation.conversation_service import ConversationService
from app.guardrails.domain_router import DomainRouter
from app.guardrails.input_guardrail import InputGuardrail
from app.guardrails.output_guardrail import OutputGuardrail
from app.llm.llm_client import LLMClient
from app.models.conversation import ChatMessage
from app.models.routing import DomainRoute, InformationIntent
from app.models.product_resolution import ProductMatch, ProductMatchStatus
from app.prompts.produce_system_prompt import PRODUCE_SYSTEM_PROMPT
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse
from app.services.catalog_request_detector import CatalogRequestDetector
from app.services.catalog_services import CatalogService


OUT_OF_SCOPE_RESPONSE = "I'm designed to help with fruits, vegetables, dry fruits, nuts, exotic foods and gourmet foods. I can help with product information, general nutrition, storage, freshness and seasonal availability."
UNKNOWN_CATALOG_PRODUCT_RESPONSE = "I’ll update you within 30 minutes with the availability and price for {product}."
EXOTIC_GROVE_ABOUT_RESPONSE = (
    "Exotic Grove helps customers with fresh fruits, vegetables, dry fruits, nuts, "
    "exotic foods and gourmet foods. For official details, visit "
    "https://exotic-grove.vercel.app/about"
)
EXOTIC_GROVE_REGISTRATION_RESPONSE = (
    "Please see Exotic Grove's official About Us page for verified business "
    "information: https://exotic-grove.vercel.app/about"
)
AMBIGUOUS_PRODUCT_RESPONSE = (
    "I found multiple possible products for {product}. Please be more specific."
)


class ChatOrchestrator:
    """Coordinate guardrails, routing, context, generation, and persistence."""
    
    
    def __init__(
        self,
        input_guardrail: InputGuardrail,
        domain_router: DomainRouter,
        conversation_service: ConversationService,
        llm_client: LLMClient,
        output_guardrail: OutputGuardrail,
        catalog_service: CatalogService,
        catalog_request_detector: CatalogRequestDetector | None = None,
    ) -> None:
        self._input_guardrail = input_guardrail
        self._domain_router = domain_router
        self._conversation_service = conversation_service
        self._llm_client = llm_client
        self._output_guardrail = output_guardrail
        self._catalog_service = catalog_service
        self._catalog_request_detector = catalog_request_detector or CatalogRequestDetector()


    async def chat(self, request: ChatRequest) -> ChatResponse:
        input_result = await self._input_guardrail.validate(request.message)
        if not input_result.allowed:
            return ChatResponse(answer=input_result.safe_response or OUT_OF_SCOPE_RESPONSE, in_scope=False)
        route = await self._domain_router.route(request.message)
        supported_message, has_programming = self._domain_router.supported_message(
            request.message
        )
        needs_catalog = self._catalog_request_detector.requires_catalog_lookup(
            supported_message, route
        )
        # ==========================================================
        # BRAND / COMPANY QUERIES
        # ==========================================================
        # Pure brand queries are deterministic. Brand-qualified product and
        # category requests continue into the catalog workflow.

        if (
            route.intent is InformationIntent.BRAND_INFORMATION
            and not needs_catalog
        ):

            response = ChatResponse(
                answer=self._brand_information_response(request.message),
                identified_products=route.identified_products,
                in_scope=True,
                intent=InformationIntent.BRAND_INFORMATION.value,
                product_domain=(
                    route.product_domain.value if route.product_domain else None
                ),
            )

            await self._record_and_persist(
                request=request,
                identified_products=route.identified_products,
                response=response,
            )

            return response
        if not route.in_scope and not needs_catalog:
            return ChatResponse(answer=OUT_OF_SCOPE_RESPONSE, in_scope=False)

        catalog_context = ""
        verified_catalog_context: dict[str, object] | list[dict[str, object]] | None = None
        expected_products: list[dict[str, object]] = []
        if needs_catalog:
            if self._catalog_request_detector.is_general_inventory_query(
                supported_message
            ):
                inventory = await self._catalog_service.get_compact_catalog()
                answer = (
                    await self._validated_catalog_answer(
                        inventory,
                        route.intent,
                        [
                            self._expected_product_context(None, product)
                            for product in inventory
                        ],
                    )
                    if inventory
                    else "I don't have verified catalog items to list right now."
                )
                response = ChatResponse(
                    answer=answer,
                    in_scope=True,
                    intent=route.intent.value if route.intent else None,
                    product_domain=(
                        route.product_domain.value if route.product_domain else None
                    ),
                    identified_products=route.identified_products,
                )
                await self._record_and_persist(
                    request, route.identified_products, response
                )
                return response

            resolution_matches = self._catalog_request_detector.catalog_matches(
                supported_message, route
            )
            catalog_products: list[dict[str, object]] = []
            missing_products: list[str] = []
            ambiguous_products: list[str] = []
            ordered_outcomes: list[tuple[str, dict[str, object] | str]] = []
            seen_catalog_products: set[str] = set()
            seen_missing_products: set[str] = set()
            seen_ambiguous_products: set[str] = set()
            for match in resolution_matches:
                if match.status is ProductMatchStatus.UNMATCHED:
                    if match.original_phrase not in seen_missing_products:
                        seen_missing_products.add(match.original_phrase)
                        missing_products.append(match.original_phrase)
                        ordered_outcomes.append(("missing", match.original_phrase))
                    continue
                if match.status is ProductMatchStatus.AMBIGUOUS:
                    if match.original_phrase not in seen_ambiguous_products:
                        seen_ambiguous_products.add(match.original_phrase)
                        ambiguous_products.append(match.original_phrase)
                        ordered_outcomes.append(("ambiguous", match.original_phrase))
                    continue
                if len(match.catalog_product_ids) != 1:
                    if len(match.catalog_product_ids) > 1:
                        if match.original_phrase not in seen_ambiguous_products:
                            seen_ambiguous_products.add(match.original_phrase)
                            ambiguous_products.append(match.original_phrase)
                            ordered_outcomes.append(
                                ("ambiguous", match.original_phrase)
                            )
                    else:
                        missing_label = self._missing_product_label(match)
                        if missing_label not in seen_missing_products:
                            seen_missing_products.add(missing_label)
                            missing_products.append(missing_label)
                            ordered_outcomes.append(("missing", missing_label))
                    continue
                catalog_product = await self._catalog_service.find_product(
                    match.catalog_product_ids[0]
                )
                if catalog_product is None:
                    missing_label = self._missing_product_label(match)
                    if missing_label not in seen_missing_products:
                        seen_missing_products.add(missing_label)
                        missing_products.append(missing_label)
                        ordered_outcomes.append(("missing", missing_label))
                    continue
                product_key = str(
                    catalog_product.get("id", catalog_product.get("name", ""))
                ).casefold()
                if product_key in seen_catalog_products:
                    continue
                seen_catalog_products.add(product_key)
                catalog_products.append(catalog_product)
                ordered_outcomes.append(("available", catalog_product))
                expected_products.append(
                    self._expected_product_context(match, catalog_product)
                )

            outcome_count = len(ordered_outcomes)
            if outcome_count > 1:
                response = ChatResponse(
                    answer=await self._combined_catalog_answer(
                        ordered_outcomes, route.intent, expected_products
                    ),
                    in_scope=route.in_scope,
                    intent=route.intent.value if route.intent else None,
                    product_domain=(
                        route.product_domain.value
                        if route.product_domain
                        else None
                    ),
                    identified_products=route.identified_products,
                )
                await self._record_and_persist(
                    request, route.identified_products, response
                )
                return response
            if missing_products:
                response = self._missing_product_response(
                    route, missing_products[0]
                )
                await self._record_and_persist(
                    request, route.identified_products, response
                )
                return response
            if ambiguous_products:
                response = ChatResponse(
                    answer=AMBIGUOUS_PRODUCT_RESPONSE.format(
                        product=ambiguous_products[0]
                    ),
                    in_scope=route.in_scope,
                    intent=route.intent.value if route.intent else None,
                    product_domain=(
                        route.product_domain.value
                        if route.product_domain
                        else None
                    ),
                    identified_products=route.identified_products,
                )
                await self._record_and_persist(
                    request, route.identified_products, response
                )
                return response

            if catalog_products:
                catalog_context = "".join(
                    self._verified_catalog_context(product)
                    for product in catalog_products
                )
                verified_catalog_context = (
                    catalog_products[0]
                    if len(catalog_products) == 1
                    else catalog_products
                )

            category_query = self._catalog_category_query(supported_message)
            if not catalog_products and category_query:
                category_products = await self._catalog_service.find_category(category_query)
                if not category_products:
                    response = ChatResponse(
                        answer=UNKNOWN_CATALOG_PRODUCT_RESPONSE.format(product=category_query),
                        in_scope=route.in_scope,
                        intent=route.intent.value if route.intent else None,
                        product_domain=route.product_domain.value if route.product_domain else None,
                        identified_products=route.identified_products,
                    )
                    await self._record_and_persist(request, route.identified_products, response)
                    return response
                catalog_context = "".join(
                    self._verified_catalog_context(product) for product in category_products
                )
                verified_catalog_context = category_products
                expected_products = [
                    self._expected_product_context(None, product)
                    for product in category_products
                ]

            if verified_catalog_context is not None:
                response = ChatResponse(
                    answer=await self._validated_catalog_answer(
                        verified_catalog_context, route.intent, expected_products
                    ),
                    in_scope=True,
                    intent=route.intent.value if route.intent else None,
                    product_domain=(
                        route.product_domain.value if route.product_domain else None
                    ),
                    identified_products=route.identified_products,
                )
                await self._record_and_persist(
                    request, route.identified_products, response
                )
                return response

        if (
            has_programming
            and needs_catalog
            and verified_catalog_context is not None
        ):
            response = ChatResponse(
                answer=self._safe_verified_catalog_answer(
                    verified_catalog_context
                ),
                in_scope=True,
                intent=route.intent.value if route.intent else None,
                product_domain=(
                    route.product_domain.value if route.product_domain else None
                ),
                identified_products=route.identified_products,
            )
            await self._record_and_persist(
                request, route.identified_products, response
            )
            return response

        history = await self._conversation_service.get_history(request.session_id) if request.session_id else []
        answer = await self._llm_client.generate(
            f"{PRODUCE_SYSTEM_PROMPT}{catalog_context}",
            supported_message,
            history,
        )
        guarded_output = await self._output_guardrail.validate(
            answer,
            route.intent,
            verified_catalog_context,
            expected_products or None,
        )
        final_answer = guarded_output.answer
        if needs_catalog and verified_catalog_context is not None and not guarded_output.allowed:
            final_answer = self._safe_verified_catalog_answer(verified_catalog_context)
        response = ChatResponse(answer=final_answer, in_scope=True, intent=route.intent.value if route.intent else None, product_domain=route.product_domain.value if route.product_domain else None, identified_products=route.identified_products)
        await self._record_and_persist(request, route.identified_products, response)
        return response

    @staticmethod
    def _expected_product_context(
        match: ProductMatch | None, product: dict[str, object]
    ) -> dict[str, object]:
        """Build trusted identity and commercial facts for output validation."""
        context = {
            "catalog_product_id": product.get("id"),
            "name": product.get("name"),
            "price": product.get("price"),
            "unit": product.get("unit"),
        }
        if match is not None:
            context["canonical_id"] = match.canonical_id
        return context

    @staticmethod
    def _verified_catalog_context(product: dict[str, object]) -> str:
        """Format only verified catalog fields for the model's request context."""
        fields = (
            "name",
            "category",
            "variety",
            "unit",
            "price",
            "description",
            "availability",
            "available",
            "delivery",
            "delivery_available",
        )
        lines = [
            f"{field.title()}: {product[field]}"
            for field in fields
            if field in product
        ]
        return (
            "\n\nVERIFIED CATALOG DATA (use only these facts for catalog claims):\n"
            + "\n".join(lines)
            + "\nDo not infer availability, delivery terms, quality, or product facts beyond this verified data."
        )

    @staticmethod
    def _missing_product_label(match: ProductMatch) -> str:
        if match.canonical_id:
            return match.canonical_id.replace("_", " ")
        return match.original_phrase

    @staticmethod
    def _missing_product_response(route: DomainRoute, product: str) -> ChatResponse:
        return ChatResponse(
            answer=UNKNOWN_CATALOG_PRODUCT_RESPONSE.format(product=product),
            in_scope=route.in_scope,
            intent=route.intent.value if route.intent else None,
            product_domain=(
                route.product_domain.value if route.product_domain else None
            ),
            identified_products=route.identified_products,
        )

    async def _combined_catalog_answer(
        self,
        ordered_outcomes: list[tuple[str, dict[str, object] | str]],
        intent: InformationIntent | None,
        expected_products: list[dict[str, object]],
    ) -> str:
        parts: list[str] = []
        index = 0
        while index < len(ordered_outcomes):
            status, value = ordered_outcomes[index]
            if status == "available" and isinstance(value, dict):
                product_id = value.get("id")
                expected = [
                    product
                    for product in expected_products
                    if product.get("catalog_product_id") == product_id
                ]
                parts.append(
                    await self._validated_catalog_answer(value, intent, expected)
                )
                index += 1
                continue
            if status == "missing":
                missing_run: list[str] = []
                while (
                    index < len(ordered_outcomes)
                    and ordered_outcomes[index][0] == "missing"
                ):
                    missing_run.append(str(ordered_outcomes[index][1]))
                    index += 1
                parts.append(
                    UNKNOWN_CATALOG_PRODUCT_RESPONSE.format(
                        product=self._join_product_labels(missing_run)
                    )
                )
                continue
            parts.append(AMBIGUOUS_PRODUCT_RESPONSE.format(product=str(value)))
            index += 1
        return " ".join(parts)

    @staticmethod
    def _join_product_labels(products: list[str]) -> str:
        if len(products) == 1:
            return products[0]
        if len(products) == 2:
            return f"{products[0]} and {products[1]}"
        return f"{', '.join(products[:-1])}, and {products[-1]}"

    @staticmethod
    def _catalog_category_query(message: str) -> str | None:
        """Extract only explicit category-list wording, without a product registry."""
        match = re.search(
            r"\b(?:what|show|list)\s+(?:me\s+)?(?:your\s+)?"
            r"(fruits?|vegetables?|products?|items?|nuts?|dry\s+fruits?)\b",
            message.casefold(),
        )
        return match.group(1) if match else None

    @staticmethod
    def _brand_information_response(message: str) -> str:
        """Return a concise controlled response for Exotic Grove questions."""

        if re.search(
            r"\b(?:firm|company|business|registered|registration|"
            r"incorporated|legal)\b",
            message,
            re.IGNORECASE,
        ):
            return EXOTIC_GROVE_REGISTRATION_RESPONSE

        return EXOTIC_GROVE_ABOUT_RESPONSE

    @classmethod
    def _safe_verified_catalog_answer(
        cls,
        catalog_context: dict[str, object] | list[dict[str, object]],
        *,
        include_description: bool = True,
    ) -> str:
        if isinstance(catalog_context, list):
            names = [
                str(product["name"])
                for product in catalog_context[:5]
                if isinstance(product, dict) and "name" in product
            ]
            return "Verified catalog items include: " + ", ".join(names) + "."

        name = str(catalog_context.get("name", "this product"))
        details = [f"Exotic Grove has {name}."]
        if "unit" in catalog_context:
            details.append(f"{name} unit: {catalog_context['unit']}.")
        if "price" in catalog_context:
            details.append(f"{name} price: ₹{catalog_context['price']}.")
        if include_description and "description" in catalog_context:
            description = cls._safe_catalog_description(catalog_context["description"])
            if description:
                details.append(description)
        return " ".join(details)

    @staticmethod
    def _safe_catalog_description(value: object) -> str | None:
        """Keep ordinary product copy while excluding instructions and claims."""
        if not isinstance(value, str):
            return None
        description = " ".join(value.split())
        if not description:
            return None
        unsafe = re.compile(
            r"(?:\bignore\b.{0,40}\b(?:instruction|prompt|rule)s?\b|"
            r"\b(?:reveal|expose|print|show)\b.{0,30}\b(?:secret|password|"
            r"api\s*key|system\s+prompt)\b|(?:₹|\brs\.?\s*|\binr\s*)\d|"
            r"\b(?:price|costs?)\b.{0,15}\d|https?://|"
            r"\b[\w.+-]+@[\w.-]+\b)",
            re.IGNORECASE,
        )
        return None if unsafe.search(description) else description

    async def _validated_catalog_answer(
        self,
        catalog_context: dict[str, object] | list[dict[str, object]],
        intent: InformationIntent | None,
        expected_products: list[dict[str, object]],
    ) -> str:
        """Validate deterministic catalog text and retry without descriptions."""
        answer = self._safe_verified_catalog_answer(catalog_context)
        guarded = await self._output_guardrail.validate(
            answer, intent, catalog_context, expected_products or None
        )
        if guarded.allowed:
            return guarded.answer
        fallback = self._safe_verified_catalog_answer(
            catalog_context, include_description=False
        )
        guarded_fallback = await self._output_guardrail.validate(
            fallback, intent, catalog_context, expected_products or None
        )
        return guarded_fallback.answer

    async def _record_and_persist(
        self,
        request: ChatRequest,
        identified_products: list[str],
        response: ChatResponse,
    ) -> None:
        for product in identified_products:
            await self._conversation_service.record_product_interest(product)
        if request.session_id:
            await self._conversation_service.save(request.session_id, ChatMessage(role="user", content=request.message))
            await self._conversation_service.save(request.session_id, ChatMessage(role="assistant", content=response.answer))
