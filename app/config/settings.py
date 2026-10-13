"""Environment-backed application settings."""

from os import PathLike
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "app" / "services" / "default.json"
DEFAULT_PRODUCT_VOCABULARY_PATH = (
    PROJECT_ROOT / "app" / "services" / "product_vocabulary.json"
)


class Settings(BaseSettings):
    """Configuration loaded from environment variables and an optional .env file."""

    app_env: str = "development"
    host: str = "127.0.0.1"
    port: int = Field(default=5000, ge=1, le=65_535)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    ollama_timeout_seconds: float = Field(default=120.0, gt=0)
    max_message_length: int = Field(default=2_000, gt=0)
    max_conversation_history: int = Field(default=10, gt=0)
    max_request_body_bytes: int = Field(default=16_384, ge=1_024)
    cors_allowed_origins: str = ""
    catalog_path: Path = DEFAULT_CATALOG_PATH
    product_vocabulary_path: Path = DEFAULT_PRODUCT_VOCABULARY_PATH
    log_level: str = "INFO"

    # WhatsApp
    whatsapp_verify_token: str = "exotic_grove_wapp_verify_2026"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
        hide_input_in_errors=True,
    )

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if normalized not in {"development", "test", "production"}:
            raise ValueError("APP_ENV must be development, test, or production")
        return normalized

    @field_validator("host", "ollama_model")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("configuration value must not be empty")
        return normalized

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, value: str) -> str:
        parsed = urlsplit(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("OLLAMA_BASE_URL must be an HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("OLLAMA_BASE_URL must not contain credentials")
        return value.rstrip("/")

    @field_validator("cors_allowed_origins")
    @classmethod
    def validate_cors_allowed_origins(cls, value: str) -> str:
        origins = [origin.strip().rstrip("/") for origin in value.split(",") if origin.strip()]
        for origin in origins:
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("CORS_ALLOWED_ORIGINS must contain HTTP(S) origins")
            if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
                raise ValueError("CORS_ALLOWED_ORIGINS entries must be origins, not URLs with paths")
        return ",".join(origins)

    @property
    def allowed_origins(self) -> frozenset[str]:
        """Return the normalized browser origins allowed to call the API."""
        return frozenset(filter(None, self.cors_allowed_origins.split(",")))

    @field_validator("catalog_path", "product_vocabulary_path", mode="before")
    @classmethod
    def resolve_application_path(cls, value: object) -> Path:
        if not isinstance(value, (str, PathLike)):
            raise ValueError("Path configuration must be a string or path-like value")
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
        return normalized

    @model_validator(mode="after")
    def require_explicit_production_configuration(self) -> "Settings":
        if self.app_env != "production":
            return self
        required = {"catalog_path", "ollama_base_url", "ollama_model"}
        missing = sorted(required - self.model_fields_set)
        if missing:
            names = ", ".join(name.upper() for name in missing)
            raise ValueError(
                f"Production requires explicit configuration for: {names}"
            )
        return self


settings = Settings()
