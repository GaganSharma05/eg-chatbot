"""Routes for chatbot-related HTTP endpoints."""

import logging

from pydantic import ValidationError
from quart import Blueprint, jsonify, request

from app.chatbot.chat_orchestrator import ChatOrchestrator
from app.config.settings import settings
from app.conversation.conversation_service import ConversationService
from app.guardrails.domain_router import DomainRouter
from app.guardrails.input_guardrail import InputGuardrail
from app.guardrails.output_guardrail import OutputGuardrail
from app.llm.llama_client import LLMServiceUnavailableError, LlamaClient
from app.schemas.request import ChatRequest
from app.services.catalog_services import CatalogService
from app.services.product_resolver import ProductResolver


chat_blueprint = Blueprint("chat", __name__)
logger = logging.getLogger(__name__)
catalog_service = CatalogService(settings.catalog_path)
product_resolver = ProductResolver(catalog_service, settings.product_vocabulary_path)
orchestrator = ChatOrchestrator(InputGuardrail(settings.max_message_length), DomainRouter(product_resolver), ConversationService(settings.max_conversation_history), LlamaClient(settings), OutputGuardrail(), catalog_service)


@chat_blueprint.get("/health")
async def health() -> tuple[object, int]:
    """Return service health without depending on Ollama availability."""
    return jsonify({"status": "ok"}), 200


@chat_blueprint.get("/ready")
async def ready() -> tuple[object, int]:
    """Return whether required web-chatbot dependencies are initialized."""
    if await catalog_service.ensure_ready():
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "not_ready"}), 503


@chat_blueprint.post("/api/v1/chat")
async def chat() -> tuple[object, int]:
    """Validate a chat request and delegate it to the application workflow."""
    payload = await request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "invalid_request", "message": "A JSON object is required."}), 400
    try:
        chat_request = ChatRequest.model_validate(payload)
    except ValidationError:
        return jsonify({"error": "invalid_request", "message": "Request validation failed."}), 422
    try:
        response = await orchestrator.chat(chat_request)
    except LLMServiceUnavailableError:
        logger.warning("Chat generation service is unavailable")
        return jsonify({"error": "service_unavailable", "message": "The chat service is temporarily unavailable. Please try again shortly."}), 503
    except Exception:
        logger.exception("Unexpected web chatbot failure")
        return jsonify({"error": "internal_error", "message": "An unexpected error occurred. Please try again shortly."}), 500
    return jsonify(response.model_dump()), 200
