"""Unified recall and fetch_image tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from letta.schemas.agent import AgentState


async def recall(agent_state: "AgentState", query: str, limit: int = 10) -> str:
    """Search the full memory corpus (passages, messages, images) with hybrid recall."""
    from letta.services.recall.recall_service import recall as recall_service

    actor = agent_state.created_by  # fallback; tools pass actor via context in production
    if actor is None:
        return "Recall unavailable: no actor context."

    hits = await recall_service(query, actor, limit=limit, agent_id=agent_state.id)
    if not hits:
        return "No results."

    lines = []
    for h in hits:
        lines.append(f"[{h.layer}] score={h.score:.4f} handle={h.handle}\n{h.snippet}")
    return "\n\n".join(lines)


async def fetch_image(agent_state: "AgentState", handle: str) -> str:
    """Fetch full image pixels for a recall handle (image-<uuid>)."""
    from letta.services.image_manager import ImageManager
    from letta.services.object_store.client import get_object_store_client
    import base64

    actor = agent_state.created_by
    if actor is None:
        return "fetch_image unavailable: no actor context."

    mgr = ImageManager()
    image = await mgr.get_by_id_async(handle, actor)
    if not image:
        return f"Image {handle} not found."

    store = get_object_store_client()
    raw = await store.get_bytes(image.object_url_full)
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{image.media_type};base64,{b64}"
