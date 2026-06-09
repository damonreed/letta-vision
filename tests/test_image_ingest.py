import base64

import pytest

from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import Base64Image, ImageContent, TextContent
from letta.schemas.message import Message, ToolReturn
from letta.services.image_ingest import ingest_images_in_message


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
