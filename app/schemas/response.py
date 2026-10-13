"""Response models for the public chatbot API."""

from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    """A structured chatbot result."""

    answer: str
    in_scope: bool
    intent: str | None = None
    product_domain: str | None = None
    identified_products: list[str] = Field(default_factory=list)
