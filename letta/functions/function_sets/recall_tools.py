"""Unified recall and fetch_image tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from letta.schemas.agent import AgentState


async def recall(agent_state: "AgentState", query: str, limit: int = 10) -> str:
    """Search the full memory corpus (passages, file reading notes, messages, images) with hybrid recall."""
    from letta.services.recall.recall_service import format_recall_hit, recall as recall_service

    actor = agent_state.created_by  # fallback; tools pass actor via context in production
    if actor is None:
        return "Recall unavailable: no actor context."

    hits = await recall_service(query, actor, limit=limit, agent_id=agent_state.id)
    if not hits:
        return "No results."

    return "\n\n".join(format_recall_hit(h) for h in hits)


async def fetch_image(agent_state: "AgentState", handle: str) -> str:
    """Fetch full image pixels for a recall handle (image-<uuid>)."""
    from letta.services.image_fetch import build_fetch_image_tool_return

    actor = agent_state.created_by
    if actor is None:
        return "fetch_image unavailable: no actor context."

    return await build_fetch_image_tool_return(handle, actor)
