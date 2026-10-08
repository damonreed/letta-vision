"""Resolve image-* handles in Zapimage MCP arguments to fetchable URLs.

The agent passes an image record id. Letta loads the bytes from MinIO, mirrors
them to ``gs://{bucket}/{image_id}`` when needed, and passes Zapimage a
short-lived HTTPS signed URL. That keeps MCP request bodies small.
"""

from __future__ import annotations

from typing import Any

from letta.log import get_logger
from letta.schemas.user import User
from letta.services.image_manager import ImageManager
from letta.services.object_store.gcs_image_mirror import (
    ensure_mirrored_and_sign,
    get_gcs_image_mirror,
)

logger = get_logger(__name__)

# Tool name -> argument fields that may carry an image handle.
IMAGE_HANDLE_TOOLS: dict[str, tuple[str, ...]] = {
    "edit_image": ("image_url",),
    "compose_image": ("image_urls",),
}

IMAGE_HANDLE_NOTE = (
    "An image handle (image-<uuid>) is loaded from storage and sent as a "
    "short-lived HTTPS URL. https:// and data:image/... URLs are also accepted."
)


def _is_image_handle(value: str) -> bool:
    return value.startswith("image-")


def _append_note(text: str) -> str:
    if "image-<uuid>" in text:
        return text
    stripped = text.rstrip()
    if not stripped:
        return IMAGE_HANDLE_NOTE
    return f"{stripped}\n\n{IMAGE_HANDLE_NOTE}"


def annotate_image_tool_schema(tool_json: dict) -> None:
    """Tell the model that edit/compose accept an image handle in place of a URL."""
    fields = IMAGE_HANDLE_TOOLS.get(tool_json.get("name"))
    if not fields:
        return
    tool_json["description"] = _append_note(tool_json.get("description") or "")
    parameters = tool_json.get("parameters")
    if not isinstance(parameters, dict):
        return
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return
    for field in fields:
        prop = properties.get(field)
        if not isinstance(prop, dict):
            continue
        prop["description"] = _append_note(prop.get("description") or "")


async def _to_signed_url(handle: str, actor: User) -> str:
    image = await ImageManager().get_by_id_async(handle, actor)
    if image is None:
        raise ValueError(f"Image not found: {handle}")
    if not image.object_url_full:
        raise ValueError(f"Image {handle} has no stored bytes")
    media_type = (image.media_type or "image/png").split(";", 1)[0].strip() or "image/png"
    if get_gcs_image_mirror() is None:
        raise RuntimeError(
            "LETTA_IMAGE_GCS_BUCKET is not configured; cannot resolve image handles "
            "to signed URLs for Zapimage"
        )
    url = await ensure_mirrored_and_sign(
        handle,
        minio_key=image.object_url_full,
        content_type=media_type,
    )
    logger.info("Resolved %s to a signed GCS URL", handle)
    return url


async def _resolve_value(value: Any, actor: User) -> Any:
    if not isinstance(value, str):
        return value
    handle = value.strip()
    if not _is_image_handle(handle):
        return value
    return await _to_signed_url(handle, actor)


async def resolve_mcp_image_arguments(function_name: str, function_args: dict, actor: User) -> dict:
    """Return a copy of ``function_args`` with image handles replaced by signed URLs.

    ``https://`` and ``data:`` values are left unchanged. The caller's dict is not mutated,
    so the persisted tool call keeps the handle the agent sent.
    """
    fields = IMAGE_HANDLE_TOOLS.get(function_name)
    if not fields:
        return function_args
    resolved = dict(function_args)
    for field in fields:
        if field not in resolved:
            continue
        current = resolved[field]
        if isinstance(current, list):
            resolved[field] = [await _resolve_value(item, actor) for item in current]
        else:
            resolved[field] = await _resolve_value(current, actor)
    return resolved
