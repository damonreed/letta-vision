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


def test_compute_render_decisions_tool_return_since_last_user_gets_full():
    img = "img-gen"
    messages = [
        Message(role=MessageRole.user, content=[TextContent(text="generate a scene")]),
        _tool_return_generate_image_message(img),
    ]
    meta = {
        img: {
            "file_size_full": 300_000,
            "file_size_1mp": None,
            "object_url_full": "sha256/gen",
        }
    }
    decisions = compute_image_render_decisions(messages, _llm_config(), image_metadata=meta)
    assert decisions[img] == RenderTier.FULL


def _tool_return_generate_image_message(image_id: str) -> Message:
    from letta.schemas.message import ToolReturn

    return Message(
        role=MessageRole.tool,
        tool_returns=[
            ToolReturn(
                tool_call_id="functions.generate_image:1",
                status="success",
                func_response=[
                    TextContent(text='{"status":"ok"}'),
                    ImageContent(source=LettaImage(file_id=image_id, media_type="image/png")),
                ],
            )
        ],
    )


def test_compute_render_decisions_tool_return_image_in_walk_on_later_turn():
    img = "img-gen-persist"
    meta = {
        img: {
            "file_size_full": 300_000,
            "file_size_1mp": 200_000,
            "object_url_1mp": "sha256/gen-persist_1mp",
            "object_url_full": "sha256/gen-persist",
        }
    }
    messages = [
        Message(role=MessageRole.user, content=[TextContent(text="generate scene 1")]),
        _tool_return_generate_image_message(img),
        Message(role=MessageRole.user, content=[TextContent(text="what do you see in that image?")]),
        Message(role=MessageRole.assistant, content=[TextContent(text="working on it")]),
    ]
    decisions = compute_image_render_decisions(messages, _llm_config(), image_metadata=meta)
    assert img in decisions
    assert decisions[img] == RenderTier.ONE_MP
