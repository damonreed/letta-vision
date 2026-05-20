import base64
from unittest.mock import MagicMock, patch

import pytest

from letta.errors import LettaInvalidArgumentError, LettaMessageTooLargeError, LettaVisionCapabilityError
from letta.helpers.message_helper import validate_message_creates_for_vision
from letta.llm_api.model_registry import model_supports_vision
from letta.schemas.letta_message_content import Base64Image, ImageContent, TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import MessageCreate
from letta.schemas.enums import MessageRole


def _llm_config(model: str, handle: str | None = None) -> LLMConfig:
    return LLMConfig(
        model=model,
        model_endpoint_type="openrouter",
        context_window=128000,
        handle=handle or f"openrouter/{model}",
        provider_name="openrouter",
    )


def test_registry_kimi_k26():
    assert model_supports_vision("moonshotai/kimi-k2.6", handle="openrouter/moonshotai/kimi-k2.6")


def test_registry_text_only():
    assert not model_supports_vision("meta-llama/llama-3.1-8b-instruct")


def test_registry_rejects_legacy_and_mini_traps():
    """Glob patterns must not flag known text-only or non-vision API models."""
    traps = [
        ("gpt-4", "openai/gpt-4"),
        ("gpt-4-turbo", "openai/gpt-4-turbo"),
        ("claude-3-5-sonnet-20241022", "anthropic/claude-3-5-sonnet-20241022"),
        ("o3-mini", "openai/o3-mini"),
        ("o3-mini-2025-01-31", "openai/o3-mini-2025-01-31"),
    ]
    for model, handle in traps:
        assert not model_supports_vision(model, handle), f"false positive: {model}"


def test_registry_includes_full_o3_not_mini():
    assert model_supports_vision("o3", handle="openai/o3")
    assert model_supports_vision("o3-pro", handle="openai/o3-pro")
    assert model_supports_vision("o3-2025-04-16", handle="openai/o3-2025-04-16")


def test_validate_rejects_unsupported_media_type():
    tiny = base64.standard_b64encode(b"x").decode()
    mc = MessageCreate(
        role=MessageRole.user,
        content=[ImageContent(source=Base64Image(media_type="image/tiff", data=tiny))],
    )
    with pytest.raises(LettaInvalidArgumentError):
        validate_message_creates_for_vision([mc], _llm_config("moonshotai/kimi-k2.6"))


def test_validate_rejects_oversized_image():
    mock_settings = MagicMock(max_image_bytes=16, max_message_bytes=80 * 1024 * 1024)
    data = base64.standard_b64encode(b"0123456789abcdef" + b"x").decode()
    mc = MessageCreate(
        role=MessageRole.user,
        content=[ImageContent(source=Base64Image(media_type="image/png", data=data))],
    )
    with patch("letta.settings.settings", mock_settings):
        with pytest.raises(LettaMessageTooLargeError):
            validate_message_creates_for_vision([mc], _llm_config("moonshotai/kimi-k2.6"))


def test_validate_rejects_non_vision_model():
    tiny = base64.standard_b64encode(b"x").decode()
    mc = MessageCreate(
        role=MessageRole.user,
        content=[ImageContent(source=Base64Image(media_type="image/png", data=tiny))],
    )
    with pytest.raises(LettaVisionCapabilityError):
        validate_message_creates_for_vision([mc], _llm_config("meta-llama/llama-3.1-8b-instruct"))
