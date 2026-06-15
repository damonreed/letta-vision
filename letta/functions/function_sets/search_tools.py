"""Cross-layer and image search tool stubs (implemented in core_tool_executor)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Literal

if TYPE_CHECKING:
    from letta.schemas.agent import AgentState


async def search_all(agent_state: "AgentState", query: str, limit: int = 10) -> str:
    """Cross-layer hybrid search over all memory layers."""
    raise NotImplementedError("This should never be invoked directly.")


async def image_fetch(agent_state: "AgentState", handle: str) -> str:
    """Fetch full image pixels for an image handle."""
    raise NotImplementedError("This should never be invoked directly.")


async def image_get_text(
    agent_state: "AgentState",
    handle: str,
    field: Optional[Literal["caption", "description", "details"]] = None,
) -> str:
    """Read caption, description, and/or details for an image handle."""
    raise NotImplementedError("This should never be invoked directly.")


async def image_edit_text(
    agent_state: "AgentState",
    handle: str,
    field: Literal["caption", "description", "details"],
    command: Literal["str_replace", "insert", "set"],
    old_string: Optional[str] = None,
    new_string: Optional[str] = None,
    insert_text: Optional[str] = None,
    insert_line: int = -1,
) -> str:
    """Edit image text metadata and re-embed the image record."""
    raise NotImplementedError("This should never be invoked directly.")


async def image_search(
    agent_state: "AgentState",
    query: str,
    limit: int = 10,
    agent_id: Optional[str] = None,
) -> dict:
    """Hybrid search over image descriptions and captions."""
    raise NotImplementedError("This should never be invoked directly.")
