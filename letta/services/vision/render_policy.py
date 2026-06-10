"""Byte-budget render walk for chat image hydration."""

from __future__ import annotations

import base64
from enum import Enum
from typing import Dict, List, Optional, Set

from letta.helpers.vision_context_hint import conversation_has_user_images
from letta.log import get_logger
from letta.schemas.letta_message_content import ImageContent, ImageSourceType, LettaImage, MessageContentType
from letta.schemas.message import Message
from letta.schemas.llm_config import LLMConfig
from letta.settings import settings

logger = get_logger(__name__)

# Pre-seed providers validated at 20MB cap (§17 r2)
_SUPPORTS_IMAGE_HISTORY_MODELS = frozenset(
    {
        "anthropic/claude-sonnet-4",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "google/gemini-2.5-pro-preview",
        "moonshotai/kimi-k2",
    }
)


class RenderTier(str, Enum):
    FULL = "full"
    ONE_MP = "one_mp"
    TEXT = "text"


def supports_image_blocks_in_history(llm_config: LLMConfig) -> bool:
    model = (llm_config.model or llm_config.handle or "").lower()
    if any(m in model for m in _SUPPORTS_IMAGE_HISTORY_MODELS):
        return True
    if "kimi" in model or "claude" in model or "gpt-4" in model or "gemini" in model:
        return True
    return False


def _letta_file_id_from_image_block(block) -> Optional[str]:
    if isinstance(block, ImageContent) and isinstance(block.source, LettaImage):
        return block.source.file_id
    if isinstance(block, dict) and block.get("type") == "image":
        source = block.get("source") or {}
        if source.get("type") == ImageSourceType.letta.value and source.get("file_id"):
            return source["file_id"]
    return None


def _content_letta_image_ids(message: Message) -> List[str]:
    ids: List[str] = []
    if not message.content or not isinstance(message.content, list):
        return ids
    for block in message.content:
        fid = _letta_file_id_from_image_block(block)
        if fid:
            ids.append(fid)
    return ids


def _tool_return_letta_image_ids(message: Message) -> List[str]:
    ids: List[str] = []
    for tool_return in message.tool_returns or []:
        func_response = tool_return.func_response
        if not isinstance(func_response, list):
            continue
        for part in func_response:
            fid = _letta_file_id_from_image_block(part)
            if fid:
                ids.append(fid)
    return ids


def conversation_has_letta_images(messages: List[Message]) -> bool:
    """True if any in-context message carries LettaImage refs (content or tool returns)."""
    for message in messages:
        if _content_letta_image_ids(message) or _tool_return_letta_image_ids(message):
            return True
    return False


def _collect_letta_images(messages: List[Message]) -> List[tuple[str, bool]]:
    """Return (image_id, is_current_turn) newest-first (content + tool returns)."""
    current_turn_ids: Set[str] = set()
    for msg in reversed(messages):
        role = getattr(msg.role, "value", msg.role)
        if role == "user" and msg.content:
            for fid in _content_letta_image_ids(msg):
                current_turn_ids.add(fid)
            break

    found: List[tuple[str, bool]] = []
    seen: Set[str] = set()
    for msg in reversed(messages):
        for fid in _tool_return_letta_image_ids(msg) + _content_letta_image_ids(msg):
            if fid not in seen:
                seen.add(fid)
                found.append((fid, fid in current_turn_ids))
    return found


def find_image_needing_1mp_now(
    messages: List[Message],
    llm_config: LLMConfig,
    *,
    image_metadata: Optional[Dict[str, dict]] = None,
) -> Optional[str]:
    """Return the first current-turn image that needs an on-demand 1MP bake."""
    if not conversation_has_letta_images(messages):
        return None
    if not supports_image_blocks_in_history(llm_config):
        return None

    cap = settings.vision_context_byte_cap
    remaining = cap
    demoted = False
    meta = image_metadata or {}

    for img_id, is_current in _collect_letta_images(messages):
        if demoted:
            return None

        info = meta.get(img_id, {})
        full_size = info.get("file_size_full") or 0
        onemp_size = info.get("file_size_1mp")

        if is_current:
            if full_size <= remaining:
                remaining -= full_size
                continue
            if onemp_size is None and not info.get("object_url_1mp"):
                return img_id
            if onemp_size and onemp_size <= remaining:
                remaining -= onemp_size
            else:
                demoted = True
        else:
            if onemp_size and onemp_size <= remaining:
                remaining -= onemp_size
            else:
                demoted = True

    return None


def compute_image_render_decisions(
    messages: List[Message],
    llm_config: LLMConfig,
    *,
    image_metadata: Optional[Dict[str, dict]] = None,
) -> Dict[str, RenderTier]:
    """Walk message list newest-first spending wire-byte budget."""
    if not conversation_has_letta_images(messages):
        return {}

    if not supports_image_blocks_in_history(llm_config):
        return {img_id: RenderTier.TEXT for img_id, _ in _collect_letta_images(messages)}

    cap = settings.vision_context_byte_cap
    remaining = cap
    demoted = False
    decisions: Dict[str, RenderTier] = {}
    meta = image_metadata or {}

    for img_id, is_current in _collect_letta_images(messages):
        if demoted:
            decisions[img_id] = RenderTier.TEXT
            continue

        info = meta.get(img_id, {})
        full_size = info.get("file_size_full") or 0
        onemp_size = info.get("file_size_1mp")

        if is_current:
            if full_size <= remaining:
                decisions[img_id] = RenderTier.FULL
                remaining -= full_size
            else:
                if onemp_size is None:
                    decisions[img_id] = RenderTier.TEXT
                    demoted = True
                    continue
                if onemp_size <= remaining:
                    decisions[img_id] = RenderTier.ONE_MP
                    remaining -= onemp_size
                else:
                    decisions[img_id] = RenderTier.TEXT
                    demoted = True
        else:
            if onemp_size and onemp_size <= remaining:
                decisions[img_id] = RenderTier.ONE_MP
                remaining -= onemp_size
            else:
                decisions[img_id] = RenderTier.TEXT
                demoted = True

    return decisions
