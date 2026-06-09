"""Hydrate LettaImage references with pixels for LLM requests (in-memory only)."""

from __future__ import annotations

import base64
import copy
from typing import Dict, List, Optional

from letta.log import get_logger
from letta.schemas.letta_message_content import ImageContent, LettaImage
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message
from letta.schemas.user import User as PydanticUser
from letta.services.image_manager import ImageManager
from letta.services.object_store.client import get_object_store_client
from letta.services.vision.image_derivative import generate_1mp_now
from letta.services.vision.render_policy import (
    RenderTier,
    compute_image_render_decisions,
    find_image_needing_1mp_now,
)

logger = get_logger(__name__)


async def _load_image_metadata(image_ids: set[str], actor: PydanticUser) -> Dict[str, dict]:
    manager = ImageManager()
    meta: Dict[str, dict] = {}
    for image_id in image_ids:
        record = await manager.get_by_id_async(image_id, actor)
        if not record:
            continue
        meta[image_id] = _metadata_entry_from_record(record)
    return meta


def _collect_letta_image_ids(messages: List[Message]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        if message.content and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, ImageContent) and isinstance(block.source, LettaImage) and block.source.file_id:
                    ids.add(block.source.file_id)
        ids.update(_collect_tool_return_letta_image_ids(message))
    return ids


def _collect_tool_return_letta_image_ids(message: Message) -> set[str]:
    ids: set[str] = set()
    for tool_return in message.tool_returns or []:
        func_response = tool_return.func_response
        if not isinstance(func_response, list):
            continue
        for part in func_response:
            if isinstance(part, ImageContent) and isinstance(part.source, LettaImage):
                if part.source.file_id and not part.source.data:
                    ids.add(part.source.file_id)
            elif isinstance(part, dict) and part.get("type") == "image":
                source = part.get("source") or {}
                if source.get("type") == "letta" and source.get("file_id") and not source.get("data"):
                    ids.add(source["file_id"])
    return ids


async def _hydrate_tool_return_letta_images(
    message: Message,
    metadata: Dict[str, dict],
    store,
) -> None:
    """Fill missing LettaImage.data on tool returns (fetch_image, MCP, etc.)."""
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
                if file_id and not part.source.data:
                    info = metadata.get(file_id, {})
                    key = info.get("object_url_full")
                    if key:
                        try:
                            raw = await store.get_bytes(key)
                            part.source.data = base64.standard_b64encode(raw).decode("ascii")
                            part.source.media_type = part.source.media_type or info.get("media_type") or "image/png"
                        except Exception as exc:
                            logger.warning("Failed to hydrate tool-return image %s: %s", file_id, exc)
                updated_parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "image":
                source = part.get("source") or {}
                file_id = source.get("file_id")
                if source.get("type") == "letta" and file_id and not source.get("data"):
                    info = metadata.get(file_id, {})
                    key = info.get("object_url_full")
                    if key:
                        try:
                            raw = await store.get_bytes(key)
                            source = dict(source)
                            source["data"] = base64.standard_b64encode(raw).decode("ascii")
                            source["media_type"] = source.get("media_type") or info.get("media_type") or "image/png"
                            part = {**part, "source": source}
                        except Exception as exc:
                            logger.warning("Failed to hydrate tool-return image %s: %s", file_id, exc)
                updated_parts.append(part)
            else:
                updated_parts.append(part)

        tool_return.func_response = updated_parts


async def hydrate_tool_return_images_in_messages(
    messages: List[Message],
    actor: PydanticUser,
) -> None:
    """Hydrate LettaImage refs in tool_returns for client display and downstream LLM calls."""
    image_ids: set[str] = set()
    for message in messages:
        image_ids.update(_collect_tool_return_letta_image_ids(message))
    if not image_ids:
        return

    metadata = await _load_image_metadata(image_ids, actor)
    store = get_object_store_client()
    for message in messages:
        await _hydrate_tool_return_letta_images(message, metadata, store)


def _metadata_entry_from_record(record) -> dict:
    return {
        "file_size_full": record.file_size_full or 0,
        "file_size_1mp": record.file_size_1mp,
        "object_url_full": record.object_url_full,
        "object_url_1mp": record.object_url_1mp,
        "media_type": record.media_type,
        "description": record.description or record.caption,
    }


async def _ensure_1mp_derivatives_for_render(
    messages: List[Message],
    metadata: Dict[str, dict],
    llm_config: LLMConfig,
    actor: PydanticUser,
) -> Dict[str, dict]:
    """Bake and persist any 1MP derivatives required by the render-policy walk."""
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


async def prepare_messages_for_vision_llm(
    messages: List[Message],
    llm_config: LLMConfig,
    actor: PydanticUser,
) -> List[Message]:
    """Apply render-policy hydration to a message list before build_request_data."""
    image_ids = _collect_letta_image_ids(messages)
    if not image_ids:
        return messages

    hydrated = copy.deepcopy(messages)
    metadata = await _load_image_metadata(image_ids, actor)
    metadata = await _ensure_1mp_derivatives_for_render(hydrated, metadata, llm_config, actor)
    decisions = compute_image_render_decisions(hydrated, llm_config, image_metadata=metadata)
    store = get_object_store_client()

    for message in hydrated:
        if not message.content or not isinstance(message.content, list):
            continue
        for block in message.content:
            if not isinstance(block, ImageContent) or not isinstance(block.source, LettaImage):
                continue
            file_id = block.source.file_id
            if not file_id:
                continue
            tier = decisions.get(file_id, RenderTier.TEXT)
            info = metadata.get(file_id, {})
            media_type = block.source.media_type or info.get("media_type") or "image/png"

            try:
                if tier == RenderTier.FULL and info.get("object_url_full"):
                    raw = await store.get_bytes(info["object_url_full"])
                    block.source.data = base64.standard_b64encode(raw).decode("ascii")
                    block.source.media_type = media_type
                elif tier == RenderTier.ONE_MP:
                    key = info.get("object_url_1mp")
                    if not key:
                        logger.warning("ONE_MP tier for %s but object_url_1mp missing; skipping hydration", file_id)
                        continue
                    raw = await store.get_bytes(key)
                    block.source.data = base64.standard_b64encode(raw).decode("ascii")
                    block.source.media_type = media_type
            except Exception as exc:
                logger.warning("Failed to hydrate image %s for LLM request: %s", file_id, exc)

        await _hydrate_tool_return_letta_images(message, metadata, store)

    return hydrated
