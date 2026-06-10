import base64

from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import Base64Image, ImageContent, TextContent
from letta.schemas.message import Message, ToolReturn
from letta.services.migration.block_classifier import (
    ImageBlockKind,
    classify_image_block,
    is_unrecoverable_placeholder_text,
    scan_message,
)


def test_unrecoverable_placeholder_detection():
    assert is_unrecoverable_placeholder_text("[Image reference image-abc — use fetch_image to view pixels]")
    assert is_unrecoverable_placeholder_text("[Image omitted]")
    assert is_unrecoverable_placeholder_text("[2 images omitted]")
    assert not is_unrecoverable_placeholder_text("normal text")


def test_classify_base64_block():
    data = base64.b64encode(b"png-bytes").decode()
    block = ImageContent(source=Base64Image(media_type="image/png", data=data))
    result = classify_image_block(block, location="content[0]")
    assert result.kind == ImageBlockKind.convertible
    assert result.content_hash
    assert result.wire_bytes == len(data)


def test_classify_letta_ref_without_data():
    block = {
        "type": "image",
        "source": {"type": "letta", "file_id": "image-123", "data": None},
    }
    result = classify_image_block(block, location="content[0]")
    assert result.kind == ImageBlockKind.already_letta


def test_classify_url_skip():
    block = {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/x.png"},
    }
    result = classify_image_block(block, location="content[0]")
    assert result.kind == ImageBlockKind.url_skip


def test_scan_message_counts_convertible_and_placeholder():
    data = base64.b64encode(b"png").decode()
    msg = Message(
        role=MessageRole.user,
        content=[
            TextContent(text="[Image reference image-x — use fetch_image to view pixels]"),
            ImageContent(source=Base64Image(media_type="image/png", data=data)),
        ],
    )
    stats = scan_message(msg)
    assert stats.convertible_blocks == 1
    assert stats.unrecoverable_placeholders == 1
    assert stats.messages_with_convertible == 1


def test_scan_message_tool_return_base64():
    data = base64.b64encode(b"tool-png").decode()
    msg = Message(
        role=MessageRole.tool,
        content=[TextContent(text="done")],
        tool_returns=[
            ToolReturn(
                status="success",
                func_response=[
                    ImageContent(source=Base64Image(media_type="image/png", data=data)),
                ],
            )
        ],
    )
    stats = scan_message(msg)
    assert stats.convertible_blocks == 1
