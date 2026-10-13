"""Routing enums and result models for the food chatbot."""

from dataclasses import dataclass, field
from enum import Enum

from app.models.product_resolution import ProductResolution


class ProductDomain(str, Enum):
    FRESH_FRUIT = "fresh_fruit"
    VEGETABLE = "vegetable"
    DRY_FRUIT = "dry_fruit"
    NUT = "nut"
    EXOTIC_FOOD = "exotic_food"
    GOURMET_FOOD = "gourmet_food"


class InformationIntent(str, Enum):
    INFORMATION = "information"
    BRAND_INFORMATION = "brand_information"
    NUTRITION = "nutrition"
    STORAGE = "storage"
    FRESHNESS = "freshness"
    SEASONALITY = "seasonality"


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    safe_response: str | None = None


@dataclass(frozen=True, slots=True)
class DomainRoute:
    in_scope: bool
    product_domain: ProductDomain | None = None
    intent: InformationIntent | None = None
    identified_products: list[str] = field(default_factory=list)
    product_resolution: ProductResolution = field(default_factory=ProductResolution)
    explicitly_out_of_scope: bool = False


@dataclass(frozen=True, slots=True)
class OutputGuardrailResult:
    answer: str
    allowed: bool
