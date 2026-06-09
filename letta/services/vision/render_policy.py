"""Byte-budget render walk for chat image hydration."""

from __future__ import annotations

import base64
from enum import Enum
from typing import Dict, List, Optional, Set

from letta.helpers.message_helper import conversation_has_user_images
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


def _collect_letta_images(messages: List[Message]) -> List[tuple[str, bool]]:
    """Return (image_id, is_current_turn) newest-first."""
    current_turn_ids: Set[str] = set()
    for msg in reversed(messages):
        if msg.role == "user" and msg.content:
            for block in msg.content:
                if isinstance(block, ImageContent) and isinstance(block.source, LettaImage):
                    current_turn_ids.add(block.source.file_id)
            break

    found: List[tuple[str, bool]] = []
    seen: Set[str] = set()
    for msg in reversed(messages):
        if not msg.content:
            continue
        for block in msg.content:
            if isinstance(block, ImageContent) and isinstance(block.source, LettaImage):
                fid = block.source.file_id
                if fid not in seen:
                    seen.add(fid)
                    found.append((fid, fid in current_turn_ids))
    return found


def compute_image_render_decisions(
    messages: List[Message],
    llm_config: LLMConfig,
    *,
    image_metadata: Optional[Dict[str, dict]] = None,
) -> Dict[str, RenderTier]:
    """Walk message list newest-first spending wire-byte budget."""
    if not conversation_has_user_images(messages):
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
                    from letta.services.vision.image_derivative import generate_1mp_derivative
                    from letta.services.object_store.client import get_object_store_client
                    from letta.services.image_manager import ImageManager

                    # on-demand 1MP — caller should persist via enrich path; size estimate only here
                    onemp_size = full_size
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
