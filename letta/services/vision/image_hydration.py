"""Hydrate LettaImage references with pixels for LLM requests (in-memory only)."""

from __future__ import annotations

import base64
import copy
from typing import Dict, List, Optional, Union

from letta.log import get_logger
from letta.schemas.letta_message_content import ImageContent, LettaImage, TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message
from letta.schemas.user import User as PydanticUser
from letta.services.image_manager import ImageManager
from letta.services.image_text import format_image_llm_reference_from_metadata, normalize_image_handle
from letta.services.vision.render_policy import (
    RenderTier,
    compute_image_render_decisions,
    find_image_needing_1mp_now,
)

logger = get_logger(__name__)

ImagePart = Union[ImageContent, dict]


async def _load_image_metadata(image_ids: set[str], actor: PydanticUser) -> Dict[str, dict]:
    manager = ImageManager()
    meta: Dict[str, dict] = {}
    for image_id in image_ids:
        record = await manager.get_by_id_async(image_id, actor)
        if not record:
            continue
        meta[image_id] = _metadata_entry_from_record(record)
    return meta


def load_image_metadata_for_render_walk_sync(messages: List[Message], actor: PydanticUser) -> Dict[str, dict]:
    """Load persisted image sizes for the byte-budget walk (sync callers)."""
    from letta.utils import run_async_task

    image_ids = _collect_letta_image_ids(messages)
    if not image_ids:
        return {}
    return run_async_task(_load_image_metadata(image_ids, actor))


def _collect_letta_image_ids(messages: List[Message]) -> set[str]:
    from letta.services.vision.render_policy import _content_letta_image_ids, _tool_return_letta_image_ids

    ids: set[str] = set()
    for message in messages:
        ids.update(_content_letta_image_ids(message))
        ids.update(_tool_return_letta_image_ids(message))
    return ids


def _collect_tool_return_letta_image_ids(message: Message) -> set[str]:
    from letta.services.vision.render_policy import _tool_return_letta_image_ids

    return set(_tool_return_letta_image_ids(message))


def _part_text(part) -> str:
    if isinstance(part, TextContent):
        return part.text or ""
    if isinstance(part, dict) and part.get("type") == "text":
        return part.get("text") or ""
    return ""


def _preceding_part_has_image_id(parts: list, file_id: str) -> bool:
    """Skip duplicate reference text when the prior block already names this image."""
    if not parts:
        return False
    text = _part_text(parts[-1])
    if not text:
        return False
    normalized = normalize_image_handle(file_id)
    return normalized in text or file_id in text


def _reference_text_part(file_id: str, info: dict, *, as_dict: bool):
    text = format_image_llm_reference_from_metadata(file_id, info)
    if as_dict:
        return {"type": "text", "text": text}
    return TextContent(text=text)


def _maybe_prepend_reference(parts: list, file_id: str, info: dict, *, as_dict: bool) -> None:
    if _preceding_part_has_image_id(parts, file_id):
        return
    parts.append(_reference_text_part(file_id, info, as_dict=as_dict))


async def _hydrate_letta_image_bytes(
    file_id: str,
    tier: RenderTier,
    info: dict,
    store,
) -> tuple[Optional[str], Optional[str]]:
    if tier == RenderTier.TEXT:
        return None, None
    if tier == RenderTier.FULL:
        key = info.get("object_url_full")
    elif tier == RenderTier.ONE_MP:
        key = info.get("object_url_1mp")
    else:
        return None, None
    if not key:
        return None, None
    raw = await store.get_bytes(key)
    media_type = info.get("media_type") or "image/png"
    return base64.standard_b64encode(raw).decode("ascii"), media_type


async def _apply_letta_image_hydration(
    *,
    file_id: str,
    info: dict,
    tier: RenderTier,
    store,
    updated_parts: list,
    image_part: ImagePart,
    as_dict: bool,
) -> None:
    if tier == RenderTier.TEXT:
        if not _preceding_part_has_image_id(updated_parts, file_id):
            updated_parts.append(_reference_text_part(file_id, info, as_dict=as_dict))
        return

    _maybe_prepend_reference(updated_parts, file_id, info, as_dict=as_dict)

    if isinstance(image_part, ImageContent) and isinstance(image_part.source, LettaImage):
        media_type = image_part.source.media_type or info.get("media_type") or "image/png"
        try:
            data, hydrated_type = await _hydrate_letta_image_bytes(file_id, tier, info, store)
            if data:
                image_part.source.data = data
                image_part.source.media_type = hydrated_type or media_type
        except Exception as exc:
            logger.warning("Failed to hydrate image %s: %s", file_id, exc)
        updated_parts.append(image_part)
        return

    source = dict(image_part.get("source") or {})
    try:
        data, hydrated_type = await _hydrate_letta_image_bytes(file_id, tier, info, store)
        if data:
            source["data"] = data
            source["media_type"] = source.get("media_type") or hydrated_type or info.get("media_type")
            image_part = {**image_part, "source": source}
    except Exception as exc:
        logger.warning("Failed to hydrate image %s: %s", file_id, exc)
    updated_parts.append(image_part)


async def _hydrate_tool_return_letta_images(
    message: Message,
    metadata: Dict[str, dict],
    store,
    *,
    render_decisions: Optional[Dict[str, RenderTier]] = None,
) -> None:
    """Hydrate LettaImage refs in tool returns (fetch_image, MCP, etc.) under render policy."""
    if not message.tool_returns:
        return

    for tool_return in message.tool_returns:
        func_response = tool_return.func_response
        if not isinstance(func_response, list):
            continue

        updated_parts = []
        for part in func_response:
            if isinstance(part, ImageContent) and isinstance(part.source, LettaImage):
                file_id = part.source.file_id
                if not file_id or part.source.data:
                    updated_parts.append(part)
                    continue
                info = metadata.get(file_id, {})
                tier = render_decisions.get(file_id, RenderTier.FULL) if render_decisions else RenderTier.FULL
                await _apply_letta_image_hydration(
                    file_id=file_id,
                    info=info,
                    tier=tier,
                    store=store,
                    updated_parts=updated_parts,
                    image_part=part,
                    as_dict=False,
                )
            elif isinstance(part, dict) and part.get("type") == "image":
                source = part.get("source") or {}
                file_id = source.get("file_id")
                if source.get("type") != "letta" or not file_id or source.get("data"):
                    updated_parts.append(part)
                    continue
                info = metadata.get(file_id, {})
                tier = render_decisions.get(file_id, RenderTier.FULL) if render_decisions else RenderTier.FULL
                await _apply_letta_image_hydration(
                    file_id=file_id,
                    info=info,
                    tier=tier,
                    store=store,
                    updated_parts=updated_parts,
                    image_part=part,
                    as_dict=True,
                )
            else:
                updated_parts.append(part)

        tool_return.func_response = updated_parts


async def hydrate_tool_return_images_in_messages(
    messages: List[Message],
    actor: PydanticUser,
) -> None:
    """Hydrate tool-return LettaImage refs at full resolution (client / DB read paths)."""
    image_ids: set[str] = set()
    for message in messages:
        image_ids.update(_collect_tool_return_letta_image_ids(message))
    if not image_ids:
        return

    metadata = await _load_image_metadata(image_ids, actor)
    from letta.services.object_store.client import get_object_store_client

    store = get_object_store_client()
    for message in messages:
        await _hydrate_tool_return_letta_images(message, metadata, store, render_decisions=None)


def _metadata_entry_from_record(record) -> dict:
    return {
        "file_size_full": record.file_size_full or 0,
        "file_size_1mp": record.file_size_1mp,
        "object_url_full": record.object_url_full,
        "object_url_1mp": record.object_url_1mp,
        "media_type": record.media_type,
        "caption": record.caption,
        "description": record.description,
    }


async def _ensure_1mp_derivatives_for_render(
    messages: List[Message],
    metadata: Dict[str, dict],
    llm_config: LLMConfig,
    actor: PydanticUser,
) -> Dict[str, dict]:
    """Bake and persist any 1MP derivatives required by the render-policy walk."""
    from letta.services.vision.image_derivative import generate_1mp_now

    updated = dict(metadata)
    manager = ImageManager()
    max_passes = len(_collect_letta_image_ids(messages)) + 1

    for _ in range(max_passes):
        image_id = find_image_needing_1mp_now(messages, llm_config, image_metadata=updated)
        if not image_id:
            break
        await generate_1mp_now(image_id, actor)
        record = await manager.get_by_id_async(image_id, actor)
        if record:
            updated[image_id] = _metadata_entry_from_record(record)

    return updated


def _strip_letta_image_bytes(messages: List[Message]) -> None:
    """Remove hydrated base64 so LLM-path tiering is not bypassed by client full-res fills."""
    for message in messages:
        if message.content and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ImageContent) and isinstance(block.source, LettaImage):
                    block.source.data = None
        for tool_return in message.tool_returns or []:
            func_response = tool_return.func_response
            if not isinstance(func_response, list):
                continue
            for part in func_response:
                if isinstance(part, ImageContent) and isinstance(part.source, LettaImage):
                    part.source.data = None
                elif isinstance(part, dict) and part.get("type") == "image":
                    source = part.get("source") or {}
                    if source.get("type") == "letta" and source.get("data"):
                        source = dict(source)
                        source["data"] = None
                        part["source"] = source


async def _hydrate_content_letta_images(
    message: Message,
    metadata: Dict[str, dict],
    store,
    decisions: Dict[str, RenderTier],
) -> None:
    if not message.content or not isinstance(message.content, list):
        return

    updated_parts = []
    for block in message.content:
        if isinstance(block, ImageContent) and isinstance(block.source, LettaImage):
            file_id = block.source.file_id
            if not file_id:
                updated_parts.append(block)
                continue
            tier = decisions.get(file_id, RenderTier.TEXT)
            info = metadata.get(file_id, {})
            await _apply_letta_image_hydration(
                file_id=file_id,
                info=info,
                tier=tier,
                store=store,
                updated_parts=updated_parts,
                image_part=block,
                as_dict=False,
            )
        elif isinstance(block, dict) and block.get("type") == "image":
            source = block.get("source") or {}
            file_id = source.get("file_id")
            if source.get("type") != "letta" or not file_id:
                updated_parts.append(block)
                continue
            tier = decisions.get(file_id, RenderTier.TEXT)
            info = metadata.get(file_id, {})
            await _apply_letta_image_hydration(
                file_id=file_id,
                info=info,
                tier=tier,
                store=store,
                updated_parts=updated_parts,
                image_part=block,
                as_dict=True,
            )
        else:
            updated_parts.append(block)

    message.content = updated_parts


async def prepare_messages_for_vision_llm(
    messages: List[Message],
    llm_config: LLMConfig,
    actor: PydanticUser,
) -> tuple[List[Message], Dict[str, RenderTier]]:
    """Apply render-policy hydration; return messages and render decisions for build_request_data."""
    image_ids = _collect_letta_image_ids(messages)
    if not image_ids:
        return messages, {}

    hydrated = copy.deepcopy(messages)
    _strip_letta_image_bytes(hydrated)
    metadata = await _load_image_metadata(image_ids, actor)
    metadata = await _ensure_1mp_derivatives_for_render(hydrated, metadata, llm_config, actor)
    decisions = compute_image_render_decisions(hydrated, llm_config, image_metadata=metadata)
    from letta.services.object_store.client import get_object_store_client

    store = get_object_store_client()

    for message in hydrated:
        await _hydrate_content_letta_images(message, metadata, store, decisions)
        await _hydrate_tool_return_letta_images(message, metadata, store, render_decisions=decisions)

    return hydrated, decisions
