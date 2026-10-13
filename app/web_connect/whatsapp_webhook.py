import os

from quart import Blueprint, Response, jsonify, request
from app.config.settings import settings



whatsapp_webhook_bp = Blueprint(
    "whatsapp_webhook",
    __name__,
)


WHATSAPP_VERIFY_TOKEN = settings.whatsapp_verify_token

@whatsapp_webhook_bp.get("/webhook")
async def verify_webhook():
    """
    Meta webhook verification endpoint.

    Meta sends:
    hub.mode
    hub.verify_token
    hub.challenge

    We must return hub.challenge when the verify token matches.
    """

    mode = request.args.get("hub.mode")
    verify_token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if (
        mode == "subscribe"
        and verify_token
        and verify_token == WHATSAPP_VERIFY_TOKEN
    ):
        print("WhatsApp webhook verified successfully.")

        return Response(
            challenge or "",
            status=200,
            content_type="text/plain",
        )

    print("WhatsApp webhook verification failed.")

    return jsonify(
        {
            "error": "Webhook verification failed",
        }
    ), 403


@whatsapp_webhook_bp.post("/webhook")
async def receive_webhook():
    """
    Receive incoming webhook events from Meta.

    For now:
    - Receive the event
    - Print/log the payload
    - Return HTTP 200

    Chatbot processing will be connected after webhook
    connectivity is verified.
    """

    payload = await request.get_json()

    print("WhatsApp webhook received:")
    print(payload)

    return jsonify(
        {
            "status": "received",
        }
    ), 200