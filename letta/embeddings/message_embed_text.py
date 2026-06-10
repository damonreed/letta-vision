"""Build text payloads for message vector embedding (FR §9 two-embed dance)."""

from __future__ import annotations

import json
from typing import Callable, Iterable, Optional

from letta.schemas.letta_message_content import ImageContent, ImageSourceType, LettaImage
from letta.schemas.message import Message as PydanticMessage
from letta.schemas.user import User as PydanticUser


def collect_letta_image_ids_from_message(message: PydanticMessage) -> list[str]:
    """Return LettaImage file_ids referenced by message content and tool returns (ordered, deduped)."""
    seen: set[str] = set()
    ordered: list[str] = []

    def add(file_id: Optional[str]) -> None:
        if file_id and file_id not in seen:
            seen.add(file_id)
            ordered.append(file_id)

    content = message.content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, ImageContent) and isinstance(block.source, LettaImage):
                add(block.source.file_id)
            elif isinstance(block, dict) and block.get("type") == "image":
                source = block.get("source") or {}
                if source.get("type") == ImageSourceType.letta.value:
                    add(source.get("file_id"))

    for tool_return in message.tool_returns or []:
        func_response = tool_return.func_response
        if not isinstance(func_response, list):
            continue
        for part in func_response:
            if isinstance(part, ImageContent) and isinstance(part.source, LettaImage):
                add(part.source.file_id)
            elif isinstance(part, dict) and part.get("type") == "image":
                source = part.get("source") or {}
                if source.get("type") == ImageSourceType.letta.value:
                    add(source.get("file_id"))

    return ordered


def inject_image_caption_gists(base_text: str, captions: Iterable[str]) -> str:
    """Append short image caption gists to the JSON embed payload (not the 100–200w description)."""
    gist_list = [c.strip() for c in captions if c and str(c).strip()]
    if not gist_list:
        return base_text

    stripped = (base_text or "").strip()
    if not stripped:
        return json.dumps({"image_captions": gist_list})

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            parsed = dict(parsed)
            parsed["image_captions"] = gist_list
            return json.dumps(parsed)
    except (json.JSONDecodeError, ValueError):
        pass

    return json.dumps({"content": stripped, "image_captions": gist_list})


async def load_caption_gists_for_images(
    image_ids: list[str],
    actor: PydanticUser,
) -> list[str]:
    """Resolve caption-tier gists for embedding; skips ids without captions."""
    if not image_ids:
        return []

    from letta.services.image_manager import ImageManager

    manager = ImageManager()
    gists: list[str] = []
    for image_id in image_ids:
        image = await manager.get_by_id_async(image_id, actor)
        caption = (image.caption or "").strip() if image else ""
        if caption:
            gists.append(caption)
    return gists


async def build_message_embed_text(
    message: PydanticMessage,
    actor: PydanticUser,
    *,
    include_image_captions: bool,
    base_extractor: Optional[Callable[[PydanticMessage], str]] = None,
) -> str:
    """Compose message embed text; caption gists are added only when include_image_captions is True (v2+)."""
    if base_extractor is None:
        from letta.services.message_manager import MessageManager

        base_extractor = MessageManager()._extract_message_text

    base = base_extractor(message).strip()
    if not include_image_captions:
        return base

    image_ids = collect_letta_image_ids_from_message(message)
    if not image_ids:
        return base

    gists = await load_caption_gists_for_images(image_ids, actor)
    if not gists:
        return base

    return inject_image_caption_gists(base, gists)
