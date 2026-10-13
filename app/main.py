"""Quart application factory and ASGI entry point."""

import logging

from quart import Quart, ResponseReturnValue, request
from quart.typing import ResponseTypes
from werkzeug.exceptions import RequestEntityTooLarge

from app.api.chat_routes import catalog_service, chat_blueprint
from app.config.settings import settings
from app.web_connect.whatsapp_webhook import whatsapp_webhook_bp


def create_app() -> Quart:
    """Create and configure the Quart application."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app = Quart(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_request_body_bytes
    app.register_blueprint(chat_blueprint)
    app.register_blueprint(whatsapp_webhook_bp)

    @app.after_request
    async def add_cors_headers(response: ResponseTypes) -> ResponseTypes:
        origin = request.headers.get("Origin")
        if origin and origin.rstrip("/") in settings.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Max-Age"] = "600"
            response.headers.add("Vary", "Origin")
        return response

    @app.before_serving
    async def validate_web_chatbot_startup() -> None:
        if not await catalog_service.ensure_ready():
            raise RuntimeError("Web chatbot catalog initialization failed")

    @app.errorhandler(RequestEntityTooLarge)
    async def request_too_large(_: RequestEntityTooLarge) -> ResponseReturnValue:
        return {
            "error": "request_too_large",
            "message": "The request body is too large.",
        }, 413

    return app
    

app = create_app()
