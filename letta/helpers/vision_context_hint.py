"""Request-scoped hints for multi-turn vision conversations."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from letta.schemas.llm_config import LLMConfig
    from letta.schemas.message import Message

VISION_MULTI_TURN_HINT = """<vision_context>
Earlier user messages in this request may include image attachments (image_url / image content parts). They remain visible for this turn—not only the most recent attachment. Describe and compare all images present in the conversation history when asked.
</vision_context>"""


def _message_has_image_content(message: "Message") -> bool:
    content = message.content
    if not content:
        return False
    if not isinstance(content, list):
        return False
    for part in content:
        part_type = getattr(part, "type", None)
        if part_type is not None and str(getattr(part_type, "value", part_type)) == "image":
            return True
        if isinstance(part, dict) and part.get("type") == "image":
            return True
    return False


def conversation_has_user_images(messages: List["Message"]) -> bool:
    """True if any non-system message in the in-context list carries image blocks."""
    for message in messages:
        role = getattr(message.role, "value", message.role)
        if role == "system":
            continue
        if _message_has_image_content(message):
            return True
    return False


def append_vision_context_hint(
    system_prompt: str,
    *,
    llm_config: "LLMConfig",
    messages: Optional[List["Message"]] = None,
) -> str:
    """Append a short vision hint when the model and conversation support images."""
    if not messages or not system_prompt:
        return system_prompt

    from letta.llm_api.model_registry import model_supports_vision

    if not model_supports_vision(llm_config.model, handle=llm_config.handle):
        return system_prompt
    if not conversation_has_user_images(messages):
        return system_prompt
    if VISION_MULTI_TURN_HINT in system_prompt:
        return system_prompt

    return system_prompt.rstrip() + "\n\n" + VISION_MULTI_TURN_HINT
