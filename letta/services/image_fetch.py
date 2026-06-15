"""Shared fetch_image tool return builder."""

from __future__ import annotations

from typing import List, Union

from letta.schemas.letta_message_content import ImageContent, LettaImage, TextContent
from letta.schemas.user import User as PydanticUser
from letta.services.image_manager import ImageManager
from letta.services.image_text import format_image_llm_reference_from_metadata, normalize_image_handle
ToolReturn = Union[str, List[Union[TextContent, ImageContent, dict]]]


def multimodal_tool_return(blocks: List[Union[TextContent, ImageContent]]) -> List[dict]:
    """JSON-serializable blocks for tool_returns persistence and client streaming."""
    return [block.model_dump(mode="json") for block in blocks]


async def build_fetch_image_tool_return(handle: str, actor: PydanticUser) -> ToolReturn:
    """Return multimodal tool content so image bytes reach the model and client UI."""
    image_id = normalize_image_handle(handle)
    mgr = ImageManager()
    image = await mgr.get_by_id_async(image_id, actor)
    if not image:
        return f"Image {handle} not found."

    media_type = image.media_type or "image/jpeg"
    # Ref-only at rest; hydration on read supplies pixels for UI/LLM (FR §4.6, §12 r3).
    blocks = [
        TextContent(
            text=format_image_llm_reference_from_metadata(
                image_id,
                {
                    "caption": image.caption,
                    "description": image.description,
                },
            )
        ),
        ImageContent(
            source=LettaImage(
                file_id=image_id,
                media_type=media_type,
                data=None,
                detail="high",
            ),
        ),
    ]
    return multimodal_tool_return(blocks)
