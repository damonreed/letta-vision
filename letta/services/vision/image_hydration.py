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
from letta.services.vision.image_derivative import generate_1mp_derivative
from letta.services.vision.render_policy import RenderTier, compute_image_render_decisions

logger = get_logger(__name__)


async def _load_image_metadata(image_ids: set[str], actor: PydanticUser) -> Dict[str, dict]:
    manager = ImageManager()
    meta: Dict[str, dict] = {}
    for image_id in image_ids:
        record = await manager.get_by_id_async(image_id, actor)
        if not record:
            continue
        meta[image_id] = {
            "file_size_full": record.file_size_full or 0,
            "file_size_1mp": record.file_size_1mp,
            "object_url_full": record.object_url_full,
            "object_url_1mp": record.object_url_1mp,
            "media_type": record.media_type,
            "description": record.description or record.caption,
        }
    return meta


def _collect_letta_image_ids(messages: List[Message]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        if not message.content or not isinstance(message.content, list):
            continue
        for block in message.content:
            if isinstance(block, ImageContent) and isinstance(block.source, LettaImage) and block.source.file_id:
                ids.add(block.source.file_id)
    return ids


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
                    if key:
                        raw = await store.get_bytes(key)
                    elif info.get("object_url_full"):
                        raw = await store.get_bytes(info["object_url_full"])
                        raw, media_type, _ = generate_1mp_derivative(raw, media_type)
                    else:
                        continue
                    block.source.data = base64.standard_b64encode(raw).decode("ascii")
                    block.source.media_type = media_type
            except Exception as exc:
                logger.warning("Failed to hydrate image %s for LLM request: %s", file_id, exc)

    return hydrated
