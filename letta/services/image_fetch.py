"""Shared fetch_image tool return builder."""

from __future__ import annotations

import base64
from typing import List, Union

from letta.schemas.letta_message_content import ImageContent, LettaImage, TextContent
from letta.schemas.user import User as PydanticUser
from letta.services.image_manager import ImageManager
from letta.services.object_store.client import get_object_store_client

ToolReturn = Union[str, List[Union[TextContent, ImageContent, dict]]]


def normalize_image_handle(handle: str) -> str:
    """Accept bare uuid or image-<uuid> handles from recall."""
    cleaned = (handle or "").strip()
    if not cleaned:
        return cleaned
    if cleaned.startswith("image-"):
        return cleaned
    return f"image-{cleaned}"


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

    store = get_object_store_client()
    try:
        raw = await store.get_bytes(image.object_url_full)
    except Exception as exc:
        return f"Image {image_id} could not be loaded from object store: {exc}"

    b64 = base64.standard_b64encode(raw).decode("ascii")
    media_type = image.media_type or "image/jpeg"

    summary = image.description or image.caption or f"Image {image_id}"
    # LettaImage (not base64) so message ingest keeps file_id + inline bytes for vision/UI.
    blocks = [
        TextContent(text=f"{summary} ({media_type}, {len(raw)} bytes)"),
        ImageContent(
            source=LettaImage(
                file_id=image_id,
                media_type=media_type,
                data=b64,
                detail="high",
            ),
        ),
    ]
    return multimodal_tool_return(blocks)
