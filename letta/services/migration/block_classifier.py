"""Classify message content blocks for historic base64 image conversion."""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, List, Optional, Union

from letta.schemas.letta_message_content import ImageContent, ImageSourceType, MessageContentType, TextContent
from letta.schemas.message import Message, ToolReturn

def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_UNRECOVERABLE_PLACEHOLDER_RE = re.compile(
    r"^\[(?:Image(?:\s+reference|\s+Here)?|Image omitted|\d+\s+images?\s+omitted)",
    re.IGNORECASE,
)


class ImageBlockKind(str, Enum):
    convertible = "convertible"
    already_letta = "already_letta"
    url_skip = "url_skip"
    other_skip = "other_skip"


@dataclass
class ClassifiedImageBlock:
    kind: ImageBlockKind
    location: str
    wire_bytes: int = 0
    content_hash: Optional[str] = None
    media_type: Optional[str] = None


@dataclass
class MessageScanStats:
    convertible_blocks: int = 0
    already_letta: int = 0
    url_skipped: int = 0
    other_skipped: int = 0
    unrecoverable_placeholders: int = 0
    estimated_bytes_removed: int = 0
    distinct_content_hashes: set[str] = field(default_factory=set)
    messages_scanned: int = 0
    messages_with_convertible: int = 0

    def merge_block(self, block: ClassifiedImageBlock) -> None:
        if block.kind == ImageBlockKind.convertible:
            self.convertible_blocks += 1
            if block.wire_bytes:
                self.estimated_bytes_removed += block.wire_bytes
            if block.content_hash:
                self.distinct_content_hashes.add(block.content_hash)
        elif block.kind == ImageBlockKind.already_letta:
            self.already_letta += 1
        elif block.kind == ImageBlockKind.url_skip:
            self.url_skipped += 1
        else:
            self.other_skipped += 1


def is_unrecoverable_placeholder_text(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return bool(_UNRECOVERABLE_PLACEHOLDER_RE.match(stripped))


def _decode_base64_payload(data: str) -> Optional[bytes]:
    if not data:
        return None
    payload = data
    if payload.startswith("data:"):
        comma = payload.find(",")
        if comma >= 0:
            payload = payload[comma + 1 :]
    try:
        return base64.standard_b64decode(payload)
    except Exception:
        return None


def _wire_bytes_for_base64_field(data: str) -> int:
    """Stored JSON size of the base64 field (conservative estimate for messages table shrink)."""
    return len(data or "")


def classify_image_block(block: Union[ImageContent, dict], *, location: str) -> ClassifiedImageBlock:
    if isinstance(block, ImageContent):
        source = block.source
        source_type = source.type
        if source_type == ImageSourceType.letta:
            inline_data = getattr(source, "data", None)
            if inline_data:
                raw = _decode_base64_payload(inline_data)
                if raw:
                    content_hash = _content_hash(raw)
                    return ClassifiedImageBlock(
                        kind=ImageBlockKind.convertible,
                        location=location,
                        wire_bytes=_wire_bytes_for_base64_field(inline_data),
                        content_hash=content_hash,
                        media_type=getattr(source, "media_type", None) or "image/png",
                    )
            if getattr(source, "file_id", None):
                return ClassifiedImageBlock(kind=ImageBlockKind.already_letta, location=location)
            return ClassifiedImageBlock(kind=ImageBlockKind.other_skip, location=location)

        if source_type == ImageSourceType.base64:
            raw = _decode_base64_payload(source.data)
            if raw:
                return ClassifiedImageBlock(
                    kind=ImageBlockKind.convertible,
                    location=location,
                    wire_bytes=_wire_bytes_for_base64_field(source.data),
                    content_hash=_content_hash(raw),
                    media_type=source.media_type or "image/png",
                )
            return ClassifiedImageBlock(kind=ImageBlockKind.other_skip, location=location)

        if source_type == ImageSourceType.url:
            return ClassifiedImageBlock(kind=ImageBlockKind.url_skip, location=location)

        return ClassifiedImageBlock(kind=ImageBlockKind.other_skip, location=location)

    if not isinstance(block, dict) or block.get("type") != "image":
        return ClassifiedImageBlock(kind=ImageBlockKind.other_skip, location=location)

    source = block.get("source") or {}
    source_type = source.get("type")

    if source_type == "letta":
        inline_data = source.get("data")
        if inline_data:
            raw = _decode_base64_payload(inline_data)
            if raw:
                return ClassifiedImageBlock(
                    kind=ImageBlockKind.convertible,
                    location=location,
                    wire_bytes=_wire_bytes_for_base64_field(inline_data),
                    content_hash=_content_hash(raw),
                    media_type=source.get("media_type") or "image/png",
                )
        if source.get("file_id"):
            return ClassifiedImageBlock(kind=ImageBlockKind.already_letta, location=location)
        return ClassifiedImageBlock(kind=ImageBlockKind.other_skip, location=location)

    if source_type == "base64" and source.get("data"):
        raw = _decode_base64_payload(source["data"])
        if raw:
            return ClassifiedImageBlock(
                kind=ImageBlockKind.convertible,
                location=location,
                wire_bytes=_wire_bytes_for_base64_field(source["data"]),
                content_hash=_content_hash(raw),
                media_type=source.get("media_type") or "image/png",
            )
        return ClassifiedImageBlock(kind=ImageBlockKind.other_skip, location=location)

    if source_type == "url":
        return ClassifiedImageBlock(kind=ImageBlockKind.url_skip, location=location)

    return ClassifiedImageBlock(kind=ImageBlockKind.other_skip, location=location)


def _is_image_block(part: Any) -> bool:
    if isinstance(part, ImageContent):
        return part.type == MessageContentType.image
    return isinstance(part, dict) and part.get("type") == "image"


def iter_message_blocks(message: Message) -> Iterator[tuple[str, Any]]:
    content = message.content
    if isinstance(content, list):
        for idx, block in enumerate(content):
            if isinstance(block, TextContent) or (isinstance(block, dict) and block.get("type") == "text"):
                text = block.text if isinstance(block, TextContent) else (block.get("text") or "")
                yield f"content[{idx}].text", text
            elif _is_image_block(block):
                yield f"content[{idx}].image", block

    tool_returns: Optional[List[ToolReturn]] = message.tool_returns
    if not tool_returns:
        return

    for tr_idx, tool_return in enumerate(tool_returns):
        func_response = tool_return.func_response
        if not isinstance(func_response, list):
            continue
        for part_idx, part in enumerate(func_response):
            if isinstance(part, TextContent) or (isinstance(part, dict) and part.get("type") == "text"):
                text = part.text if isinstance(part, TextContent) else (part.get("text") or "")
                yield f"tool_returns[{tr_idx}].func_response[{part_idx}].text", text
            elif _is_image_block(part):
                yield f"tool_returns[{tr_idx}].func_response[{part_idx}].image", part


def scan_message(message: Message) -> MessageScanStats:
    stats = MessageScanStats()
    had_convertible = False

    for location, block in iter_message_blocks(message):
        if isinstance(block, str):
            if is_unrecoverable_placeholder_text(block):
                stats.unrecoverable_placeholders += 1
            continue

        classified = classify_image_block(block, location=location)
        stats.merge_block(classified)
        if classified.kind == ImageBlockKind.convertible:
            had_convertible = True

    stats.messages_scanned = 1
    if had_convertible:
        stats.messages_with_convertible += 1
    return stats


def merge_scan_stats(target: MessageScanStats, source: MessageScanStats) -> None:
    target.convertible_blocks += source.convertible_blocks
    target.already_letta += source.already_letta
    target.url_skipped += source.url_skipped
    target.other_skipped += source.other_skipped
    target.unrecoverable_placeholders += source.unrecoverable_placeholders
    target.estimated_bytes_removed += source.estimated_bytes_removed
    target.distinct_content_hashes.update(source.distinct_content_hashes)
    target.messages_scanned += source.messages_scanned
    target.messages_with_convertible += source.messages_with_convertible
