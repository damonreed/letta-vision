"""Byte-budget render walk for chat image hydration."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Set

from letta.log import get_logger
from letta.schemas.letta_message_content import ImageContent, ImageSourceType, LettaImage
from letta.schemas.message import Message
from letta.schemas.llm_config import LLMConfig
from letta.settings import settings

logger = get_logger(__name__)


class RenderTier(str, Enum):
    FULL = "full"
    ONE_MP = "one_mp"
    TEXT = "text"


def supports_image_blocks_in_history(llm_config: LLMConfig) -> bool:
    """True when the configured model can receive image blocks in message history."""
    from letta.llm_api.model_registry import model_supports_vision

    return model_supports_vision(llm_config.model, handle=llm_config.handle)


def _max_image_parts_for_model(llm_config: LLMConfig) -> Optional[int]:
    from letta.llm_api.model_registry import model_max_image_parts

    return model_max_image_parts(llm_config.model, handle=llm_config.handle)


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


def _last_user_message_index(messages: List[Message]) -> Optional[int]:
    for i in range(len(messages) - 1, -1, -1):
        role = getattr(messages[i].role, "value", messages[i].role)
        if role == "user":
            return i
    return None


def _collect_letta_images(messages: List[Message]) -> List[tuple[str, bool]]:
    """Return (image_id, is_current_turn) newest-first (content + tool returns).

    Current turn = user attachments on the latest user message plus Letta refs
    from tool returns on messages after that user message (generate_image, etc.).
    """
    current_turn_ids: Set[str] = set()
    last_user_idx = _last_user_message_index(messages)
    if last_user_idx is not None:
        current_turn_ids.update(_content_letta_image_ids(messages[last_user_idx]))
        for msg in messages[last_user_idx + 1 :]:
            current_turn_ids.update(_tool_return_letta_image_ids(msg))

    found: List[tuple[str, bool]] = []
    seen: Set[str] = set()
    for msg in reversed(messages):
        for fid in _tool_return_letta_image_ids(msg) + _content_letta_image_ids(msg):
            if fid not in seen:
                seen.add(fid)
                found.append((fid, fid in current_turn_ids))
    return found


def _letta_image_part_counts(messages: List[Message]) -> Dict[str, int]:
    """Total image-part occurrences per image id (one image can appear in several messages)."""
    counts: Dict[str, int] = {}
    for msg in messages:
        for fid in _tool_return_letta_image_ids(msg) + _content_letta_image_ids(msg):
            counts[fid] = counts.get(fid, 0) + 1
    return counts


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
    parts_cap = _max_image_parts_for_model(llm_config)
    part_counts = _letta_image_part_counts(messages) if parts_cap is not None else {}
    parts_remaining = parts_cap

    for img_id, is_current in _collect_letta_images(messages):
        if demoted:
            return None

        if parts_remaining is not None:
            needed_parts = part_counts.get(img_id, 1)
            if needed_parts > parts_remaining:
                # Count-capped: this image and everything older render as text, no bake needed.
                return None
        else:
            needed_parts = 0

        info = meta.get(img_id, {})
        full_size = info.get("file_size_full") or 0
        onemp_size = info.get("file_size_1mp")

        if is_current:
            if full_size <= remaining:
                remaining -= full_size
                if parts_remaining is not None:
                    parts_remaining -= needed_parts
                continue
            if onemp_size is None and not info.get("object_url_1mp"):
                return img_id
            if onemp_size and onemp_size <= remaining:
                remaining -= onemp_size
                if parts_remaining is not None:
                    parts_remaining -= needed_parts
            else:
                demoted = True
        else:
            if onemp_size and onemp_size <= remaining:
                remaining -= onemp_size
                if parts_remaining is not None:
                    parts_remaining -= needed_parts
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
    # Some providers silently drop image parts beyond a per-request cap (keeping the
    # OLDEST parts), which would make the newest images invisible. Enforce the cap
    # ourselves newest-first so older images demote to text instead.
    parts_cap = _max_image_parts_for_model(llm_config)
    part_counts = _letta_image_part_counts(messages) if parts_cap is not None else {}
    parts_remaining = parts_cap

    for img_id, is_current in _collect_letta_images(messages):
        if demoted:
            decisions[img_id] = RenderTier.TEXT
            continue

        if parts_remaining is not None:
            needed_parts = part_counts.get(img_id, 1)
            if needed_parts > parts_remaining:
                decisions[img_id] = RenderTier.TEXT
                demoted = True
                continue
        else:
            needed_parts = 0

        info = meta.get(img_id, {})
        full_size = info.get("file_size_full") or 0
        onemp_size = info.get("file_size_1mp")

        if is_current:
            if full_size <= remaining:
                decisions[img_id] = RenderTier.FULL
                remaining -= full_size
                if parts_remaining is not None:
                    parts_remaining -= needed_parts
            else:
                if onemp_size is None:
                    decisions[img_id] = RenderTier.TEXT
                    demoted = True
                    continue
                if onemp_size <= remaining:
                    decisions[img_id] = RenderTier.ONE_MP
                    remaining -= onemp_size
                    if parts_remaining is not None:
                        parts_remaining -= needed_parts
                else:
                    decisions[img_id] = RenderTier.TEXT
                    demoted = True
        else:
            if onemp_size and onemp_size <= remaining:
                decisions[img_id] = RenderTier.ONE_MP
                remaining -= onemp_size
                if parts_remaining is not None:
                    parts_remaining -= needed_parts
            else:
                decisions[img_id] = RenderTier.TEXT
                demoted = True

    return decisions
