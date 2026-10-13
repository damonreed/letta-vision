"""Resolve image-* handles in Zapimage MCP arguments to fetchable URLs.

The agent passes an image record id. Letta ensures the bytes are mirrored to
``gs://{bucket}/{image_id}`` and passes Zapimage either:

- a ``gs://`` URI for Vertex Gemini image models (Vertex reads the object
  directly — no 7 MB inline cap), or
- a short-lived HTTPS signed URL for every other provider (they fetch bytes
  over HTTP).
"""

from __future__ import annotations

from typing import Any

from letta.log import get_logger
from letta.schemas.user import User
from letta.services.image_manager import ImageManager
from letta.services.object_store.gcs_image_mirror import (
    ensure_mirrored,
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
    "short-lived HTTPS URL (or a gs:// URI for google/* Vertex models). "
    "https:// and data:image/... URLs are also accepted."
)


def _is_image_handle(value: str) -> bool:
    return value.startswith("image-")


def _uses_vertex_gs_uri(function_args: dict) -> bool:
    """Vertex Gemini image models accept gs:// fileUri and avoid the 7 MB inline cap."""
    model = function_args.get("model")
    if not isinstance(model, str):
        return False
    return model.strip().startswith("google/")


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


async def _to_provider_url(handle: str, actor: User, *, prefer_gs: bool) -> str:
    image = await ImageManager().get_by_id_async(handle, actor)
    if image is None:
        raise ValueError(f"Image not found: {handle}")
    if not image.object_url_full:
        raise ValueError(f"Image {handle} has no stored bytes")
    media_type = (image.media_type or "image/png").split(";", 1)[0].strip() or "image/png"
    if get_gcs_image_mirror() is None:
        raise RuntimeError(
            "LETTA_IMAGE_GCS_BUCKET is not configured; cannot resolve image handles "
            "for Zapimage"
        )
    if prefer_gs:
        uri = await ensure_mirrored(
            handle,
            minio_key=image.object_url_full,
            content_type=media_type,
        )
        logger.info("Resolved %s to %s for Vertex", handle, uri)
        return uri
    url = await ensure_mirrored_and_sign(
        handle,
        minio_key=image.object_url_full,
        content_type=media_type,
    )
    logger.info("Resolved %s to a signed GCS URL", handle)
    return url


async def _resolve_value(value: Any, actor: User, *, prefer_gs: bool) -> Any:
    if not isinstance(value, str):
        return value
    handle = value.strip()
    if not _is_image_handle(handle):
        return value
    return await _to_provider_url(handle, actor, prefer_gs=prefer_gs)


async def resolve_mcp_image_arguments(function_name: str, function_args: dict, actor: User) -> dict:
    """Return a copy of ``function_args`` with image handles replaced by provider URLs.

    ``https://`` and ``data:`` values are left unchanged. The caller's dict is not mutated,
    so the persisted tool call keeps the handle the agent sent.
    """
    fields = IMAGE_HANDLE_TOOLS.get(function_name)
    if not fields:
        return function_args
    prefer_gs = _uses_vertex_gs_uri(function_args)
    resolved = dict(function_args)
    for field in fields:
        if field not in resolved:
            continue
        current = resolved[field]
        if isinstance(current, list):
            resolved[field] = [
                await _resolve_value(item, actor, prefer_gs=prefer_gs) for item in current
            ]
        else:
            resolved[field] = await _resolve_value(current, actor, prefer_gs=prefer_gs)
    return resolved
