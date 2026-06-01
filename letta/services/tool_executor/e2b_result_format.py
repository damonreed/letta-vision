"""Convert E2B code-interpreter executions into Letta tool returns (text + inline images)."""

from __future__ import annotations

import hashlib
from typing import Any

from letta.schemas.letta_message_content import Base64Image, ImageContent, TextContent

# E2B Result stores plot/PIL output as base64 in these fields (see e2b_code_interpreter.models.Result).
_E2B_IMAGE_FIELDS = (
    ("png", "image/png"),
    ("jpeg", "image/jpeg"),
)


def _result_image_blocks(result: Any) -> list[ImageContent]:
    """At most one inline image per E2B Result (prefer png over jpeg)."""
    for attr, media_type in _E2B_IMAGE_FIELDS:
        data = getattr(result, attr, None)
        if data:
            return [ImageContent(source=Base64Image(media_type=media_type, data=data))]
    return []


def _image_payload_key(block: ImageContent) -> str:
    data = block.source.data or ""
    digest = hashlib.sha256(data.encode("ascii")).hexdigest()
    return f"{block.source.media_type}:{digest}"


def _dedupe_image_blocks(blocks: list[ImageContent]) -> list[ImageContent]:
    """Drop duplicate PNGs from display + main-result pairs in the same E2B cell."""
    seen: set[str] = set()
    out: list[ImageContent] = []
    for block in blocks:
        key = _image_payload_key(block)
        if key in seen:
            continue
        seen.add(key)
        out.append(block)
    return out


def _collect_image_blocks(execution: Any) -> list[ImageContent]:
    blocks: list[ImageContent] = []
    for result in getattr(execution, "results", None) or []:
        blocks.extend(_result_image_blocks(result))
    return _dedupe_image_blocks(blocks)


def _collect_log_text(execution: Any) -> list[str]:
    parts: list[str] = []
    logs = getattr(execution, "logs", None)
    if not logs:
        return parts
    for stream in ("stdout", "stderr"):
        chunks = getattr(logs, stream, None) or []
        if chunks:
            joined = "".join(chunks) if isinstance(chunks, list) else str(chunks)
            if joined.strip():
                parts.append(joined)
    return parts


def _collect_result_text(execution: Any) -> list[str]:
    parts: list[str] = []
    for result in getattr(execution, "results", None) or []:
        text = getattr(result, "text", None)
        if text:
            parts.append(text)
    return parts


def _format_execution_error(err: Any) -> str:
    if err is None:
        return ""
    name = getattr(err, "name", None) or type(err).__name__
    value = getattr(err, "value", None) or str(err)
    traceback = getattr(err, "traceback", None) or ""
    msg = f"{name}: {value}"
    if traceback:
        msg = f"{msg}\n{traceback}"
    return msg.strip()


def e2b_execution_to_llm_friendly_dict(execution: Any) -> dict:
    """JSON-serializable summary when no inline images are returned."""
    results = getattr(execution, "results", None) or []
    out: dict[str, Any] = {
        "results": [r.text if hasattr(r, "text") else str(r) for r in results],
        "logs": {
            "stdout": getattr(getattr(execution, "logs", None), "stdout", []) or [],
            "stderr": getattr(getattr(execution, "logs", None), "stderr", []) or [],
        },
    }
    err = getattr(execution, "error", None)
    if err is not None:
        out["error"] = {
            "name": getattr(err, "name", None),
            "value": getattr(err, "value", None),
            "traceback": getattr(err, "traceback", None),
        }
    return out


def e2b_execution_to_func_return(execution: Any) -> str | list:
    """
    Build a tool func_return from an E2B Execution.

    When the sandbox produced PNG/JPEG (matplotlib, PIL.Image last expression, etc.),
    returns a multimodal list so the model receives inline image blocks. Otherwise
    returns the legacy JSON-friendly dict (caller typically json.dumps it).
    """
    image_blocks = _collect_image_blocks(execution)

    text_parts = _collect_log_text(execution) + _collect_result_text(execution)
    err_text = _format_execution_error(getattr(execution, "error", None))
    if err_text:
        text_parts.append(err_text)

    if image_blocks:
        blocks: list[TextContent | ImageContent] = []
        if text_parts:
            blocks.append(TextContent(text="\n\n".join(text_parts)))
        blocks.extend(image_blocks)
        return blocks

    return e2b_execution_to_llm_friendly_dict(execution)
