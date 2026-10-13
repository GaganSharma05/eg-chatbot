from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config.settings import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_PRODUCT_VOCABULARY_PATH,
    Settings,
)
from app.services.catalog_services import CatalogService


def test_valid_local_settings_use_absolute_catalog_path() -> None:
    configured = Settings(_env_file=None)

    assert configured.app_env == "development"
    assert configured.catalog_path == DEFAULT_CATALOG_PATH
    assert configured.catalog_path.is_absolute()
    assert configured.product_vocabulary_path == DEFAULT_PRODUCT_VOCABULARY_PATH
    assert configured.product_vocabulary_path.is_absolute()


def test_relative_catalog_path_is_resolved_from_project_root() -> None:
    configured = Settings(
        _env_file=None,
        catalog_path="app/services/default.json",
    )

    assert configured.catalog_path == DEFAULT_CATALOG_PATH


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("app_env", "staging-ish"),
        ("port", 70_000),
        ("ollama_timeout_seconds", 0),
        ("max_message_length", 0),
        ("max_conversation_history", 0),
        ("max_request_body_bytes", 100),
        ("cors_allowed_origins", "https://example.com/not-an-origin"),
        ("log_level", "verbose"),
    ],
)
def test_invalid_configuration_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_configuration_error_hides_input_value() -> None:
    secret_marker = "do-not-expose-this-value"

    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            ollama_base_url=f"http://user:{secret_marker}@localhost:11434",
        )

    assert secret_marker not in str(exc_info.value)


def test_production_requires_explicit_runtime_configuration() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None, app_env="production")

    error = str(exc_info.value)
    assert "CATALOG_PATH" in error
    assert "OLLAMA_BASE_URL" in error
    assert "OLLAMA_MODEL" in error


def test_valid_explicit_production_configuration() -> None:
    configured = Settings(
        _env_file=None,
        app_env="production",
        catalog_path="app/services/default.json",
        ollama_base_url="http://ollama.internal:11434",
        ollama_model="llama3.2:1b",
    )

    assert configured.app_env == "production"


def test_cors_origins_are_normalized() -> None:
    configured = Settings(
        _env_file=None,
        cors_allowed_origins=" https://shop.example.com/,http://localhost:3000 ",
    )

    assert configured.allowed_origins == {
        "https://shop.example.com",
        "http://localhost:3000",
    }


@pytest.mark.asyncio
async def test_default_catalog_loads_after_working_directory_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    configured = Settings(_env_file=None)

    service = CatalogService(configured.catalog_path)

    assert await service.ensure_ready()
    assert (await service.find_product("blueberry"))["id"] == "Imported-Blueberries"
