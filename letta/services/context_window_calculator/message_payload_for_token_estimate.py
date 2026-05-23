"""
Prepare LLM API message dicts for token *estimates* (compaction thresholds, context UI).

Vision messages embed multi-megabyte base64 data URLs. Serializing them with json.dumps
or str(part) counts bytes as tokens and breaks sliding-window compaction.
"""

from __future__ import annotations

import copy
from typing import Any

# Provider vision billing is tile/dimension-based, not base64-length-based.
ESTIMATED_TOKENS_PER_IMAGE = 1100

_IMAGE_PLACEHOLDER = "[image]"


def _is_heavy_image_payload(value: str) -> bool:
    if not isinstance(value, str) or len(value) < 256:
        return False
    if value.startswith("data:image") and ";base64," in value:
        return True
    sample = value[:2000]
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r")
    return len(value) >= 2000 and all(c in allowed for c in sample)


def _redact_image_url_dict(url_obj: Any) -> dict[str, Any]:
    if isinstance(url_obj, dict):
        detail = url_obj.get("detail", "auto")
        return {"url": _IMAGE_PLACEHOLDER, "detail": detail}
    return {"url": _IMAGE_PLACEHOLDER, "detail": "auto"}


def _redact_content_part(part: Any) -> tuple[Any, int]:
    """Return (redacted part, number of images redacted in this part)."""
    if not isinstance(part, dict):
        if isinstance(part, str) and _is_heavy_image_payload(part):
            return _IMAGE_PLACEHOLDER, 1
        return part, 0

    part_type = part.get("type")
    images = 0

    if part_type == "image_url":
        url_obj = part.get("image_url")
        url = url_obj.get("url") if isinstance(url_obj, dict) else url_obj
        if isinstance(url, str) and _is_heavy_image_payload(url):
            images += 1
            return {"type": "image_url", "image_url": _redact_image_url_dict(url_obj)}, images
        return part, 0

    if part_type in ("input_image", "image"):
        for key in ("image_url", "url", "data"):
            val = part.get(key)
            if isinstance(val, str) and _is_heavy_image_payload(val):
                images += 1
                redacted = copy.copy(part)
                redacted[key] = _IMAGE_PLACEHOLDER
                if part_type == "image" and isinstance(redacted.get("source"), dict):
                    src = dict(redacted["source"])
                    if src.get("type") == "base64" and isinstance(src.get("data"), str):
                        src["data"] = _IMAGE_PLACEHOLDER
                        redacted["source"] = src
                return redacted, images
        source = part.get("source")
        if isinstance(source, dict) and source.get("type") == "base64":
            data = source.get("data")
            if isinstance(data, str) and _is_heavy_image_payload(data):
                images += 1
                return {
                    "type": part_type,
                    "source": {
                        "type": "base64",
                        "media_type": source.get("media_type", "image/png"),
                        "data": _IMAGE_PLACEHOLDER,
                    },
                }, images

    if part_type == "text":
        text = part.get("text", "")
        if isinstance(text, str) and _is_heavy_image_payload(text):
            return {"type": "text", "text": _IMAGE_PLACEHOLDER}, 1
        return part, 0

    # Google-style inline image bytes
    inline = part.get("inlineData") or part.get("inline_data")
    if isinstance(inline, dict):
        data = inline.get("data")
        if isinstance(data, str) and _is_heavy_image_payload(data):
            images += 1
            mime = inline.get("mimeType") or inline.get("mime_type") or "image/png"
            return {"inlineData": {"mimeType": mime, "data": _IMAGE_PLACEHOLDER}}, images

    return part, 0


def _redact_message_content(content: Any) -> tuple[Any, int]:
    images = 0
    if isinstance(content, str):
        if _is_heavy_image_payload(content):
            return _IMAGE_PLACEHOLDER, 1
        return content, 0
    if isinstance(content, list):
        redacted_parts = []
        for part in content:
            redacted, n = _redact_content_part(part)
            redacted_parts.append(redacted)
            images += n
        return redacted_parts, images
    return content, 0


def strip_images_from_api_messages_for_token_estimate(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """
    Deep-copy messages and replace embedded image payloads with placeholders.

    Returns:
        (redacted_messages, image_count) — add image_count * ESTIMATED_TOKENS_PER_IMAGE
        to the text-based token estimate from the redacted payload.
    """
    redacted = copy.deepcopy(messages)
    total_images = 0
    for msg in redacted:
        if not isinstance(msg, dict):
            continue
        if "content" in msg:
            new_content, n = _redact_message_content(msg["content"])
            msg["content"] = new_content
            total_images += n
    return redacted, total_images


def openai_content_block_to_plaintext(block: Any) -> str:
    """Plaintext for summarizer transcripts; never embed base64 image data."""
    if not isinstance(block, dict):
        return str(block)
    block_type = block.get("type")
    if block_type == "text":
        return block.get("text", "") or ""
    if block_type in ("image_url", "input_image", "image"):
        return "[Image omitted]"
    if block_type == "tool_use" or block_type == "tool_result":
        return f"[{block_type}]"
    # Avoid str(block) on unknown dicts that may still contain image bytes
    text = block.get("text")
    if isinstance(text, str):
        return text
    return f"[{block_type or 'content'}]"
