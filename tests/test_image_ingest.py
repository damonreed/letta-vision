import base64

import pytest

from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import Base64Image, ImageContent, TextContent
from letta.schemas.message import Message, ToolReturn
from letta.schemas.letta_message_content import LettaImage
from letta.services.image_ingest import (
    _extract_assistant_text,
    _parse_caption_json,
    convert_historic_images_in_message,
    ingest_images_in_message,
)


@pytest.mark.asyncio
async def test_ingest_images_in_user_message_replaces_base64(monkeypatch):
    async def fake_ingest(data, media_type, actor, **kwargs):
        assert kwargs.get("provenance") == "uploaded"
        return "image-test-1"

    monkeypatch.setattr("letta.services.image_ingest.ingest_image_sync", fake_ingest)

    msg = Message(
        role=MessageRole.user,
        content=[
            TextContent(text="what is this?"),
            ImageContent(source=Base64Image(media_type="image/png", data=base64.b64encode(b"png").decode())),
        ],
    )
    image_ids = await ingest_images_in_message(msg, actor=None)

    assert image_ids == ["image-test-1"]
    assert msg.content[1].source.type == "letta"
    assert msg.content[1].source.file_id == "image-test-1"
    assert msg.content[1].source.data is None


@pytest.mark.asyncio
async def test_ingest_images_in_tool_return_uses_generated_provenance(monkeypatch):
    async def fake_ingest(data, media_type, actor, **kwargs):
        assert kwargs.get("provenance") == "generated"
        return "image-test-2"

    monkeypatch.setattr("letta.services.image_ingest.ingest_image_sync", fake_ingest)

    msg = Message(
        role=MessageRole.tool,
        content=[TextContent(text="generated image")],
        tool_returns=[
            ToolReturn(
                status="success",
                func_response=[
                    TextContent(text="here is the render"),
                    ImageContent(source=Base64Image(media_type="image/png", data=base64.b64encode(b"png").decode())),
                ],
            )
        ],
    )
    image_ids = await ingest_images_in_message(msg, actor=None)

    assert image_ids == ["image-test-2"]
    func_response = msg.tool_returns[0].func_response
    assert func_response[1].source.type == "letta"
    assert func_response[1].source.file_id == "image-test-2"


@pytest.mark.asyncio
async def test_ingest_images_skips_fetch_image_tool_returns(monkeypatch):
    ingest_called = False

    async def fake_ingest(data, media_type, actor, **kwargs):
        nonlocal ingest_called
        ingest_called = True
        return "image-should-not-create"

    monkeypatch.setattr("letta.services.image_ingest.ingest_image_sync", fake_ingest)

    msg = Message(
        role=MessageRole.tool,
        name="fetch_image",
        content=[TextContent(text="fetch_image result")],
        tool_returns=[
            ToolReturn(
                status="success",
                func_response=[
                    TextContent(text="summary"),
                    ImageContent(
                        source=Base64Image(media_type="image/png", data=base64.b64encode(b"png").decode(), detail="high")
                    ),
                ],
            )
        ],
    )
    image_ids = await ingest_images_in_message(msg, actor=None)

    assert image_ids == []
    assert ingest_called is False
    func_response = msg.tool_returns[0].func_response
    assert func_response[1].source.type == "base64"
    assert func_response[1].source.data is not None


@pytest.mark.asyncio
async def test_convert_historic_images_converts_letta_inline_data(monkeypatch):
    async def fake_ingest(data, media_type, actor, **kwargs):
        return "image-historic-1"

    monkeypatch.setattr("letta.services.image_ingest.ingest_image_sync", fake_ingest)

    msg = Message(
        role=MessageRole.user,
        content=[
            ImageContent(
                source=LettaImage(
                    file_id="image-old",
                    data=base64.b64encode(b"png").decode(),
                    media_type="image/png",
                )
            )
        ],
    )
    image_ids, changed = await convert_historic_images_in_message(msg, actor=None)

    assert changed is True
    assert image_ids == ["image-historic-1"]
    assert msg.content[0].source.file_id == "image-historic-1"
    assert msg.content[0].source.data is None


@pytest.mark.asyncio
async def test_convert_historic_images_converts_fetch_image_tool_return(monkeypatch):
    async def fake_ingest(data, media_type, actor, **kwargs):
        return "image-historic-2"

    monkeypatch.setattr("letta.services.image_ingest.ingest_image_sync", fake_ingest)

    msg = Message(
        role=MessageRole.tool,
        name="fetch_image",
        content=[TextContent(text="fetch_image result")],
        tool_returns=[
            ToolReturn(
                status="success",
                func_response=[
                    ImageContent(source=Base64Image(media_type="image/png", data=base64.b64encode(b"png").decode())),
                ],
            )
        ],
    )
    image_ids, changed = await convert_historic_images_in_message(msg, actor=None)

    assert changed is True
    assert image_ids == ["image-historic-2"]


def test_parse_caption_json_plain_object():
    parsed = _parse_caption_json(
        '{"caption": "Red apple", "description": "A red apple on wood", "details": "Close-up fruit"}'
    )
    assert parsed["caption"] == "Red apple"
    assert parsed["description"] == "A red apple on wood"
    assert parsed["details"] == "Close-up fruit"


def test_parse_caption_json_markdown_fence():
    parsed = _parse_caption_json(
        '```json\n{"caption": "Cat", "description": "Orange tabby cat", "details": "Sitting on sofa"}\n```'
    )
    assert parsed["caption"] == "Cat"
    assert parsed["description"] == "Orange tabby cat"


def test_extract_assistant_text_openai_shape():
    text = _extract_assistant_text({"choices": [{"message": {"content": "OK"}}]})
    assert text == "OK"
