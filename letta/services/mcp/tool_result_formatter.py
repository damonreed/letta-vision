"""
Format MCP CallToolResult content for Letta tool returns.

ZapImage and similar servers return TextContent (JSON with image URLs) plus
ImageContent (large base64). For vision agents we preserve image blocks for the
next LLM call in the same turn; text metadata is kept compact for logs/limits.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from typing import Any, List, Union

from mcp.types import EmbeddedResource as McpEmbeddedResource
from mcp.types import ImageContent as McpImageContent
from mcp.types import TextContent as McpTextContent
from mcp.types import TextResourceContents

from letta.schemas.letta_message_content import Base64Image, ImageContent, TextContent

logger = logging.getLogger(__name__)

# Prepended to image-bearing MCP tool returns. The JSON envelope from servers like
# ZapImage advertises an images[].url, which primes vision models to report the result
# as "URL-only" even though the pixels are attached inline. This note counters that.
_INLINE_IMAGE_VISIBILITY_NOTE = (
    "[The image(s) from this tool are attached inline in this tool result and are "
    "directly visible to you right now. Describe them from what you actually see. "
    "The url in the JSON below is only a storage reference — do NOT call image_fetch "
    "for these; the pixels are already here.]"
)

# Match str(ImageContent) dumps from older parsing paths
_IMAGE_STR_RE = re.compile(
    r"type=['\"]image['\"]\s+data=['\"]([A-Za-z0-9+/=]{100,})",
    re.IGNORECASE,
)
_DATA_URL_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", re.IGNORECASE)
_B64_JSON_FIELD_RE = re.compile(r'"data"\s*:\s*"[A-Za-z0-9+/=]{200,}"')


def _is_image_content(piece: Any) -> bool:
    if isinstance(piece, McpImageContent):
        return True
    if isinstance(piece, dict) and piece.get("type") == "image":
        return True
    return False


def _image_byte_estimate(piece: Any) -> int:
    if isinstance(piece, McpImageContent):
        return len(piece.data or "") * 3 // 4
    if isinstance(piece, dict):
        data = piece.get("data") or ""
        if isinstance(data, str):
            return len(data) * 3 // 4
    return 0


def _mcp_image_to_letta(piece: McpImageContent | dict) -> ImageContent | None:
    """Keep provider pixels intact so ingest can store full resolution.

    LLM context uses the 1MP derivative (enrichment / on-demand render-policy),
    not a pre-ingest downscale. Downscaling here would persist 1k as "full".
    """
    if isinstance(piece, McpImageContent):
        data = piece.data
        media_type = getattr(piece, "mimeType", None) or getattr(piece, "mime_type", None) or "image/png"
    else:
        data = piece.get("data")
        media_type = piece.get("mimeType") or piece.get("mime_type") or "image/png"
    if not data:
        return None
    return ImageContent(source=Base64Image(media_type=media_type, data=data))


def redact_base64_in_text(text: str) -> str:
    """Remove embedded image payloads from text shown in logs/UI (not from vision blocks)."""
    text = _DATA_URL_RE.sub("[image base64 omitted]", text)
    text = _IMAGE_STR_RE.sub("[image base64 omitted]", text)
    text = _B64_JSON_FIELD_RE.sub('"data": "[omitted]"', text)
    return text


def _strip_embedded_base64_from_text(text: str) -> str:
    return redact_base64_in_text(text)


def _resource_field(resource: Any, *names: str) -> Any:
    for name in names:
        if isinstance(resource, dict) and name in resource:
            return resource[name]
        if hasattr(resource, name):
            return getattr(resource, name)
    return None


def _mime_type_is_text(mime_type: str | None) -> bool:
    if not mime_type:
        return False
    base = mime_type.split(";", 1)[0].strip().lower()
    return base.startswith("text/") or base in {"application/json", "application/xml", "application/yaml", "application/x-yaml"}


def _resource_contents_to_text(resource: Any) -> str | None:
    """Extract readable text from MCP TextResourceContents or text-like BlobResourceContents."""
    if resource is None:
        return None

    text = _resource_field(resource, "text")
    if isinstance(text, str) and text:
        return text

    mime_type = _resource_field(resource, "mimeType", "mime_type")
    blob = _resource_field(resource, "blob")
    if isinstance(blob, str) and blob and _mime_type_is_text(mime_type):
        try:
            return base64.b64decode(blob).decode("utf-8")
        except Exception:
            return None

    return None


def _embedded_resource_to_text(piece: Any) -> str | None:
    if isinstance(piece, McpEmbeddedResource):
        resource = piece.resource
    elif isinstance(piece, dict) and piece.get("type") == "resource":
        resource = piece.get("resource")
    else:
        return None

    text = _resource_contents_to_text(resource)
    if text:
        return text

    mime_type = _resource_field(resource, "mimeType", "mime_type")
    uri = _resource_field(resource, "uri")
    blob = _resource_field(resource, "blob")
    if blob:
        size = len(blob) * 3 // 4
        return f"[binary MCP resource omitted: {mime_type or 'application/octet-stream'}, ~{size} bytes, uri={uri}]"
    if uri:
        return f"[MCP resource omitted: {mime_type or 'unknown type'}, uri={uri}]"
    return None


def _append_mcp_text(text_parts: list[str], text: str, *, strip_base64: bool = True) -> None:
    if not text:
        return
    text_parts.append(_strip_embedded_base64_from_text(text) if strip_base64 else text)


def _image_payload_key(img: ImageContent) -> str:
    """Full-content key — batch JPEGs often share identical header prefixes."""
    data = getattr(getattr(img, "source", None), "data", None) or ""
    media_type = getattr(getattr(img, "source", None), "media_type", None) or "image/png"
    digest = hashlib.sha256(data.encode("ascii")).hexdigest()
    return f"{media_type}:{digest}"


def _dedupe_image_parts(images: list[ImageContent]) -> list[ImageContent]:
    seen: set[str] = set()
    unique: list[ImageContent] = []
    for img in images:
        key = _image_payload_key(img)
        if key in seen:
            continue
        seen.add(key)
        unique.append(img)
    return unique


def mcp_content_to_letta_parts(content: list[Any]) -> Union[str, List[Union[TextContent, ImageContent]]]:
    """
    Convert MCP tool result content into Letta text/image blocks for tool_returns.

    Returns a list when any image block is present; otherwise a compact string.
    """
    text_parts: list[str] = []
    image_parts: list[ImageContent] = []

    for piece in content or []:
        if isinstance(piece, McpTextContent):
            _append_mcp_text(text_parts, piece.text or "")
        elif (embedded_text := _embedded_resource_to_text(piece)) is not None:
            _append_mcp_text(text_parts, embedded_text)
        elif hasattr(piece, "text") and getattr(piece, "text", None) and not hasattr(piece, "resource"):
            _append_mcp_text(text_parts, piece.text)
        elif _is_image_content(piece):
            letta_image = _mcp_image_to_letta(piece)
            if letta_image:
                image_parts.append(letta_image)
        else:
            raw = str(piece)
            # Avoid duplicating MCP ImageContent when str(piece) also matches the image regex
            if not image_parts and _IMAGE_STR_RE.search(raw):
                m = _IMAGE_STR_RE.search(raw)
                if m:
                    letta_image = ImageContent(source=Base64Image(media_type="image/png", data=m.group(1)))
                    image_parts.append(letta_image)
            elif len(raw) > 500:
                text_parts.append(f"[omitted non-text MCP content: {len(raw)} chars]")
            elif raw.strip():
                text_parts.append(raw)

    if image_parts:
        parts: List[Union[TextContent, ImageContent]] = []
        body = "\n\n".join(text_parts).strip()
        note = _INLINE_IMAGE_VISIBILITY_NOTE
        body = f"{note}\n\n{body}" if body else note
        parts.append(TextContent(text=body))
        parts.extend(_dedupe_image_parts(image_parts))
        return parts

    body = "\n\n".join(text_parts).strip()
    if not body:
        return "Empty response from tool"
    return body


def format_mcp_result_for_log(result: Any) -> str:
    """Compact, redacted summary for logger.info — never dump raw base64."""
    from letta.schemas.message import tool_return_to_text

    if isinstance(result, list):
        text = tool_return_to_text(result) or "(empty)"
        image_count = sum(
            1
            for part in result
            if isinstance(part, ImageContent) or (isinstance(part, dict) and part.get("type") == "image")
        )
        text = redact_base64_in_text(text)
        if len(text) > 4000:
            text = text[:4000] + f"... [truncated {len(text) - 4000} chars]"
        if image_count:
            return f"{text} (+ {image_count} image block(s); base64 omitted from log)"
        return text

    if isinstance(result, str):
        text = redact_base64_in_text(result)
        if len(text) > 4000:
            text = text[:4000] + f"... [truncated {len(text) - 4000} chars]"
        return text

    return redact_base64_in_text(str(result)[:500])


def format_mcp_tool_content(content: list[Any]) -> str:
    """Build a compact tool return string from MCP result content parts (no image payloads)."""
    text_parts: list[str] = []
    image_count = 0
    image_bytes = 0

    for piece in content or []:
        if isinstance(piece, McpTextContent):
            _append_mcp_text(text_parts, piece.text or "", strip_base64=False)
        elif (embedded_text := _embedded_resource_to_text(piece)) is not None:
            _append_mcp_text(text_parts, embedded_text, strip_base64=False)
        elif hasattr(piece, "text") and getattr(piece, "text", None) and not hasattr(piece, "resource"):
            _append_mcp_text(text_parts, piece.text, strip_base64=False)
        elif _is_image_content(piece):
            image_count += 1
            image_bytes += _image_byte_estimate(piece)
        else:
            raw = str(piece)
            if _IMAGE_STR_RE.search(raw):
                image_count += 1
                m = _IMAGE_STR_RE.search(raw)
                if m:
                    image_bytes += len(m.group(1)) * 3 // 4
            elif len(raw) > 500:
                text_parts.append(f"[omitted non-text MCP content: {len(raw)} chars]")
            elif raw.strip():
                text_parts.append(raw)

    body = "\n\n".join(text_parts).strip()

    if image_count:
        note = (
            f"[{image_count} image(s) from tool omitted from this log"
            f" (~{image_bytes // 1024} KiB base64); use the URL(s) in the JSON above.]"
        )
        body = f"{body}\n\n{note}" if body else note

    if not body:
        return "Empty response from tool"

    body = _strip_embedded_base64_from_text(body)
    return body


def _truncate_text_in_multimodal_list(
    parts: list, return_char_limit: int
) -> list:
    """Truncate only text blocks; preserve image blocks for vision."""
    from letta.schemas.message import tool_return_to_text

    text_budget = return_char_limit
    out = []
    for part in parts:
        if isinstance(part, TextContent):
            text = part.text or ""
            if len(text) > text_budget:
                text = text[:text_budget] + f"... [truncated {len(part.text) - text_budget} chars]"
                text_budget = 0
            else:
                text_budget -= len(text)
            out.append(TextContent(text=text))
        elif isinstance(part, ImageContent):
            out.append(part)
        elif isinstance(part, dict):
            if part.get("type") == "text":
                text = part.get("text", "") or ""
                if len(text) > text_budget:
                    text = text[:text_budget] + f"... [truncated]"
                    text_budget = 0
                else:
                    text_budget -= len(text)
                out.append({**part, "text": text})
            else:
                out.append(part)
        else:
            out.append(part)

    # If still over limit counting placeholders only, drop to text summary
    summary = tool_return_to_text(out) or ""
    if len(summary) > return_char_limit:
        return [TextContent(text=summary[:return_char_limit] + f"... [truncated {len(summary) - return_char_limit} chars]")]
    return out


def compact_tool_return_for_limit(func_return: Any, return_char_limit: int) -> Any:
    """
    Shrink func_return before tool_execution_manager string truncation.

    Multimodal returns with images: truncate text parts only, keep image blocks.
    """
    if return_char_limit is None or return_char_limit <= 0:
        return func_return

    from letta.schemas.message import tool_return_has_images, tool_return_to_text

    if isinstance(func_return, list):
        if tool_return_has_images(func_return):
            text_only = tool_return_to_text(func_return) or ""
            if len(text_only) <= return_char_limit:
                return func_return
            return _truncate_text_in_multimodal_list(func_return, return_char_limit)

        text = tool_return_to_text(func_return) or ""
        if len(text) <= return_char_limit:
            return func_return
        return text[:return_char_limit] + f"... [truncated {len(text) - return_char_limit} chars]"

    if not isinstance(func_return, str):
        text = str(func_return)
    else:
        text = func_return

    if len(text) <= return_char_limit:
        return text

    try:
        payload = json.loads(text.split("\n\n")[0].split("[NOTE:")[0].strip())
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict) and payload.get("images"):
        compact = {k: v for k, v in payload.items() if k != "images"}
        compact["images"] = [
            {key: val for key, val in item.items() if key == "url"} if isinstance(item, dict) else item
            for item in payload.get("images", [])
        ]
        compact["_note"] = "Full tool output truncated; image URLs preserved."
        compact_str = json.dumps(compact, indent=2)
        if len(compact_str) <= return_char_limit:
            return compact_str

    return text[:return_char_limit] + (
        f"... [NOTE: function output was truncated since it exceeded the character limit "
        f"({len(text)} > {return_char_limit})]"
    )
