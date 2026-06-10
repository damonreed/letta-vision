import base64

from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import ImageContent, LettaImage, TextContent
from letta.schemas.message import Message, ToolReturn
from letta.services.vision.tool_return_storage import (
    message_has_strippable_tool_return_bytes,
    strip_persisted_image_bytes_from_tool_returns,
)


def _fetch_image_tool_message(*, with_data: bool = True) -> Message:
    data = base64.b64encode(b"pixels").decode() if with_data else None
    return Message(
        role=MessageRole.tool,
        name="fetch_image",
        tool_returns=[
            ToolReturn(
                status="success",
                func_response=[
                    TextContent(text="summary"),
                    ImageContent(
                        source=LettaImage(
                            file_id="image-abc",
                            media_type="image/png",
                            data=data,
                            detail="high",
                        )
                    ),
                ],
            )
        ],
    )


def test_strip_persisted_image_bytes_from_tool_returns():
    msg = _fetch_image_tool_message()
    changed, removed = strip_persisted_image_bytes_from_tool_returns(msg)
    assert changed is True
    assert removed > 0
    image = msg.tool_returns[0].func_response[1]
    assert image.source.file_id == "image-abc"
    assert image.source.data is None


def test_strip_is_noop_when_already_ref_only():
    msg = _fetch_image_tool_message(with_data=False)
    changed, removed = strip_persisted_image_bytes_from_tool_returns(msg)
    assert changed is False
    assert removed == 0


def test_message_has_strippable_tool_return_bytes():
    assert message_has_strippable_tool_return_bytes(_fetch_image_tool_message()) is True
    assert message_has_strippable_tool_return_bytes(_fetch_image_tool_message(with_data=False)) is False
