"""Request models for the public chatbot API."""

from pydantic import BaseModel, ConfigDict


class ChatRequest(BaseModel):
    """A customer message and optional conversation identifier."""

    model_config = ConfigDict(str_strip_whitespace=False)

    message: str
    session_id: str | None = None
