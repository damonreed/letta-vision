import pytest

from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import ImageContent, LettaImage, TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message, ToolReturn
from letta.services.vision.image_hydration import _hydrate_tool_return_letta_images
from letta.services.vision.render_policy import RenderTier, compute_image_render_decisions
from letta.settings import settings


def _llm_config() -> LLMConfig:
    return LLMConfig(
        model="openai/gpt-4o",
        model_endpoint_type="openai",
        model_endpoint="https://api.openai.com/v1",
        context_window=128000,
        handle="openai/gpt-4o",
    )


def _tool_return_image_message(image_id: str) -> Message:
    return Message(
        role=MessageRole.tool,
        tool_returns=[
            ToolReturn(
                tool_call_id="functions.fetch_image:1",
                status="success",
                func_response=[
                    TextContent(text="summary"),
                    ImageContent(source=LettaImage(file_id=image_id, media_type="image/png")),
                ],
            )
        ],
    )


def test_compute_render_decisions_tool_returns_budget_to_one_mp():
    cap = settings.vision_context_byte_cap
    img = "img-tool"
    messages = [_tool_return_image_message(img)]
    meta = {
        img: {
            "file_size_full": cap + 1,
            "file_size_1mp": 200_000,
            "object_url_1mp": "sha256/abc_1mp",
        }
    }
    decisions = compute_image_render_decisions(messages, _llm_config(), image_metadata=meta)
    assert decisions[img] == RenderTier.ONE_MP


def test_compute_render_decisions_many_tool_returns_demote_oldest():
    cap = settings.vision_context_byte_cap
    onemp = 500_000
    messages = [_tool_return_image_message(f"img-{i}") for i in range(50)]
    meta = {f"img-{i}": {"file_size_full": cap, "file_size_1mp": onemp, "object_url_1mp": f"k{i}"} for i in range(50)}
    decisions = compute_image_render_decisions(messages, _llm_config(), image_metadata=meta)
    assert decisions["img-49"] == RenderTier.ONE_MP
    assert decisions["img-0"] == RenderTier.TEXT


@pytest.mark.asyncio
async def test_hydrate_tool_return_text_tier_replaces_image_with_description():
    class _Store:
        async def get_bytes(self, key):
            raise AssertionError("should not fetch bytes for TEXT tier")

    msg = _tool_return_image_message("img-text")
    metadata = {"img-text": {"description": "Purple hair in bathtub", "object_url_full": "full/key"}}
    await _hydrate_tool_return_letta_images(
        msg,
        metadata,
        _Store(),
        render_decisions={"img-text": RenderTier.TEXT},
    )
    parts = msg.tool_returns[0].func_response
    assert len(parts) == 2
    assert isinstance(parts[1], TextContent)
    assert "Purple hair in bathtub" in parts[1].text
    assert "img-text" in parts[1].text


@pytest.mark.asyncio
async def test_hydrate_tool_return_prepends_caption_and_description():
    class _Store:
        async def get_bytes(self, key):
            return b"\x89PNG"

    msg = _tool_return_image_message("img-meta")
    metadata = {
        "img-meta": {
            "object_url_full": "images/full",
            "media_type": "image/png",
            "caption": "Red barn",
            "description": "Weathered wooden barn at sunset",
        }
    }
    await _hydrate_tool_return_letta_images(msg, metadata, _Store(), render_decisions=None)
    ref = msg.tool_returns[0].func_response[1].text
    assert "Caption: Red barn" in ref
    assert "Description: Weathered wooden barn at sunset" in ref


@pytest.mark.asyncio
async def test_hydrate_tool_return_without_decisions_uses_full_key():
    fetched = []

    class _Store:
        async def get_bytes(self, key):
            fetched.append(key)
            return b"\x89PNG"

    msg = _tool_return_image_message("img-full")
    metadata = {"img-full": {"object_url_full": "images/full", "media_type": "image/png"}}
    await _hydrate_tool_return_letta_images(msg, metadata, _Store(), render_decisions=None)
    parts = msg.tool_returns[0].func_response
    assert len(parts) == 3
    assert isinstance(parts[0], TextContent)
    assert parts[0].text == "summary"
    assert isinstance(parts[1], TextContent)
    assert "Image ID: image-img-full" in parts[1].text
    assert fetched == ["images/full"]
    assert msg.tool_returns[0].func_response[2].source.data
