"""Shared factories for integration tests."""

from app.config.settings import settings
from app.guardrails.domain_router import DomainRouter
from app.services.catalog_services import CatalogService
from app.services.product_resolver import ProductResolver


def production_router(catalog_service: CatalogService | None = None) -> DomainRouter:
    """Build a router with the production vocabulary and a catalog snapshot."""
    catalog = catalog_service or CatalogService(settings.catalog_path)
    return DomainRouter(ProductResolver(catalog, settings.product_vocabulary_path))
