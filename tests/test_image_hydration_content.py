import pytest

from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import ImageContent, LettaImage, TextContent
from letta.schemas.message import Message
from letta.services.vision.image_hydration import _hydrate_content_letta_images
from letta.services.vision.render_policy import RenderTier


@pytest.mark.asyncio
async def test_hydrate_content_text_tier_replaces_image_with_description():
    class _Store:
        async def get_bytes(self, key):
            raise AssertionError("should not fetch bytes for TEXT tier")

    msg = Message(
        role=MessageRole.user,
        content=[
            TextContent(text="what do you see?"),
            ImageContent(source=LettaImage(file_id="img-content", media_type="image/png")),
        ],
    )
    metadata = {"img-content": {"description": "Cerulean lighthouse at dusk", "object_url_full": "full/key"}}
    await _hydrate_content_letta_images(
        msg,
        metadata,
        _Store(),
        decisions={"img-content": RenderTier.TEXT},
    )
    assert len(msg.content) == 2
    assert isinstance(msg.content[0], TextContent)
    assert msg.content[0].text == "what do you see?"
    assert isinstance(msg.content[1], TextContent)
    assert "Cerulean lighthouse at dusk" in msg.content[1].text
    assert "img-content" in msg.content[1].text
