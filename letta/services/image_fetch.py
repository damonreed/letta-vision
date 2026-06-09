"""Shared fetch_image tool return builder."""

from __future__ import annotations

import base64
from typing import List, Union

from letta.schemas.letta_message_content import Base64Image, ImageContent, TextContent
from letta.schemas.user import User as PydanticUser
from letta.services.image_manager import ImageManager
from letta.services.object_store.client import get_object_store_client

ToolReturn = Union[str, List[Union[TextContent, ImageContent]]]


async def build_fetch_image_tool_return(handle: str, actor: PydanticUser) -> ToolReturn:
    """Return multimodal tool content so image bytes bypass string truncation."""
    mgr = ImageManager()
    image = await mgr.get_by_id_async(handle, actor)
    if not image:
        return f"Image {handle} not found."

    store = get_object_store_client()
    raw = await store.get_bytes(image.object_url_full)
    b64 = base64.standard_b64encode(raw).decode("ascii")
    media_type = image.media_type or "application/octet-stream"

    summary = image.description or image.caption or f"Image {handle}"
    return [
        TextContent(text=f"{summary} ({media_type}, {len(raw)} bytes)"),
        ImageContent(source=Base64Image(media_type=media_type, data=b64)),
    ]
