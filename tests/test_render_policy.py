from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import ImageContent, LettaImage, TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message
from letta.services.vision.render_policy import (
    RenderTier,
    compute_image_render_decisions,
    find_image_needing_1mp_now,
)
from letta.settings import settings


def _llm_config() -> LLMConfig:
    return LLMConfig(
        model="openai/gpt-4o",
        model_endpoint_type="openai",
        model_endpoint="https://api.openai.com/v1",
        context_window=128000,
        handle="openai/gpt-4o",
    )


def _user_image_message(image_id: str) -> Message:
    return Message(
        role=MessageRole.user,
        content=[
            TextContent(text="what is this?"),
            ImageContent(source=LettaImage(file_id=image_id, media_type="image/png")),
        ],
    )


def test_find_image_needing_1mp_now_when_full_exceeds_budget():
    cap = settings.vision_context_byte_cap
    messages = [_user_image_message("img-big")]
    meta = {
        "img-big": {
            "file_size_full": cap + 1,
            "file_size_1mp": None,
            "object_url_1mp": None,
        }
    }
    assert find_image_needing_1mp_now(messages, _llm_config(), image_metadata=meta) == "img-big"


def test_find_image_needing_1mp_now_returns_none_when_1mp_baked():
    cap = settings.vision_context_byte_cap
    messages = [_user_image_message("img-big")]
    meta = {
        "img-big": {
            "file_size_full": cap + 1,
            "file_size_1mp": 500_000,
            "object_url_1mp": "sha256/abc_1mp",
        }
    }
    assert find_image_needing_1mp_now(messages, _llm_config(), image_metadata=meta) is None


def test_compute_render_decisions_one_mp_when_full_too_large():
    cap = settings.vision_context_byte_cap
    messages = [_user_image_message("img-big")]
    meta = {
        "img-big": {
            "file_size_full": cap + 1,
            "file_size_1mp": 500_000,
            "object_url_1mp": "sha256/abc_1mp",
        }
    }
    decisions = compute_image_render_decisions(messages, _llm_config(), image_metadata=meta)
    assert decisions["img-big"] == RenderTier.ONE_MP


def test_compute_render_decisions_text_when_1mp_missing():
    cap = settings.vision_context_byte_cap
    messages = [_user_image_message("img-big")]
    meta = {
        "img-big": {
            "file_size_full": cap + 1,
            "file_size_1mp": None,
            "object_url_1mp": None,
        }
    }
    decisions = compute_image_render_decisions(messages, _llm_config(), image_metadata=meta)
    assert decisions["img-big"] == RenderTier.TEXT
