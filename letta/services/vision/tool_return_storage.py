"""Persist tool-return images as LettaImage refs only (no inline base64 at rest)."""

from __future__ import annotations

from typing import Any, Optional, Union

from letta.schemas.letta_message_content import ImageContent, ImageSourceType, LettaImage, MessageContentType
from letta.schemas.message import Message as PydanticMessage, ToolReturn


def _wire_bytes_for_data_field(data: str) -> int:
    return len(data or "")


def _strip_letta_dict_source(source: dict) -> tuple[dict, int]:
    if source.get("type") != ImageSourceType.letta.value:
        return source, 0
    inline_data = source.get("data")
    if not inline_data or not source.get("file_id"):
        return source, 0
    stripped = dict(source)
    stripped["data"] = None
    return stripped, _wire_bytes_for_data_field(inline_data)


def _strip_letta_image_block(block: Union[ImageContent, dict]) -> tuple[Union[ImageContent, dict], int]:
    if isinstance(block, ImageContent):
        if block.type != MessageContentType.image or not isinstance(block.source, LettaImage):
            return block, 0
        file_id = block.source.file_id
        inline_data = block.source.data
        if not file_id or not inline_data:
            return block, 0
        return (
            ImageContent(
                source=LettaImage(
                    file_id=file_id,
                    data=None,
                    media_type=block.source.media_type,
                    detail=block.source.detail,
                )
            ),
            _wire_bytes_for_data_field(inline_data),
        )

    if not isinstance(block, dict) or block.get("type") != "image":
        return block, 0
    source = block.get("source") or {}
    new_source, removed = _strip_letta_dict_source(source)
    if not removed:
        return block, 0
    return {**block, "source": new_source}, removed


def strip_persisted_image_bytes_from_tool_returns(message: PydanticMessage) -> tuple[bool, int]:
    """Null LettaImage.data in tool_returns when file_id is present (safe for fetch_image refs)."""
    if not message.tool_returns:
        return False, 0

    changed = False
    bytes_removed = 0

    for tool_return in message.tool_returns:
        if not isinstance(tool_return, ToolReturn):
            continue
        func_response = tool_return.func_response
        if not isinstance(func_response, list):
            continue

        updated_parts: list[Any] = []
        part_changed = False
        for part in func_response:
            if (isinstance(part, ImageContent) and part.type == MessageContentType.image) or (
                isinstance(part, dict) and part.get("type") == "image"
            ):
                new_part, removed = _strip_letta_image_block(part)
                if removed:
                    part_changed = True
                    bytes_removed += removed
                updated_parts.append(new_part)
            else:
                updated_parts.append(part)

        if part_changed:
            tool_return.func_response = updated_parts
            changed = True

    return changed, bytes_removed


def message_has_strippable_tool_return_bytes(message: PydanticMessage) -> bool:
    """True when tool_returns contain LettaImage refs with both file_id and inline data."""
    if not message.tool_returns:
        return False
    for tool_return in message.tool_returns:
        func_response = tool_return.func_response if isinstance(tool_return, ToolReturn) else None
        if not isinstance(func_response, list):
            continue
        for part in func_response:
            if isinstance(part, ImageContent) and isinstance(part.source, LettaImage):
                if part.source.file_id and part.source.data:
                    return True
            elif isinstance(part, dict) and part.get("type") == "image":
                source = part.get("source") or {}
                if source.get("type") == ImageSourceType.letta.value and source.get("file_id") and source.get("data"):
                    return True
    return False
