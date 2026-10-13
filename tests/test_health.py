"""Health endpoint tests."""

import pytest
from quart.testing.app import LifespanError

from app import main as main_module
from app.api import chat_routes
from app.main import create_app


@pytest.mark.asyncio
async def test_health_returns_ok() -> None:
    app = create_app()
    test_client = app.test_client()

    response = await test_client.get("/health")

    assert response.status_code == 200
    assert await response.get_json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_cors_allows_only_configured_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module.settings, "cors_allowed_origins", "https://shop.example.com")
    test_client = create_app().test_client()

    allowed = await test_client.options(
        "/api/v1/chat",
        headers={
            "Origin": "https://shop.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    denied = await test_client.get(
        "/health", headers={"Origin": "https://attacker.example"}
    )

    assert allowed.headers["Access-Control-Allow-Origin"] == "https://shop.example.com"
    assert "Access-Control-Allow-Origin" not in denied.headers


class StubCatalogService:
    def __init__(self, ready: bool) -> None:
        self.ready = ready

    async def ensure_ready(self) -> bool:
        return self.ready


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("is_ready", "status_code", "body"),
    [
        (True, 200, {"status": "ready"}),
        (False, 503, {"status": "not_ready"}),
    ],
)
async def test_readiness_reflects_catalog_state_without_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    is_ready: bool,
    status_code: int,
    body: dict[str, str],
) -> None:
    monkeypatch.setattr(chat_routes, "catalog_service", StubCatalogService(is_ready))
    test_client = create_app().test_client()

    response = await test_client.get("/ready")

    assert response.status_code == status_code
    assert await response.get_json() == body
    assert set((await response.get_json()).keys()) == {"status"}


@pytest.mark.asyncio
async def test_startup_fails_when_catalog_cannot_initialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "catalog_service", StubCatalogService(False))
    app = create_app()

    with pytest.raises(LifespanError, match="catalog initialization failed"):
        async with app.test_app():
            pass
