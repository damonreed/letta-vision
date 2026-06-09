"""Unified embedding config resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from letta.constants import DEFAULT_EMBEDDING_CHUNK_SIZE, DEPLOYMENT_EMBEDDING_DIM
from letta.log import get_logger
from letta.schemas.embedding_config import EmbeddingConfig
from letta.settings import settings

if TYPE_CHECKING:
    from letta.schemas.agent import AgentState
    from letta.schemas.user import User as PydanticUser

logger = get_logger(__name__)

_GEMINI_EMBEDDING_2_HANDLES = frozenset(
    {
        "openrouter/google/gemini-embedding-2-preview",
        "google/gemini-embedding-2-preview",
        "gemini-embedding-2",
    }
)


def _gemini_embedding_2_config(handle: Optional[str] = None) -> EmbeddingConfig:
    return EmbeddingConfig(
        embedding_model="google/gemini-embedding-2-preview",
        embedding_endpoint_type="openrouter",
        embedding_endpoint="https://openrouter.ai/api/v1",
        embedding_dim=DEPLOYMENT_EMBEDDING_DIM,
        output_dimensionality=DEPLOYMENT_EMBEDDING_DIM,
        input_type="search_document",
        normalize=True,
        embedding_chunk_size=DEFAULT_EMBEDDING_CHUNK_SIZE,
        handle=handle or "openrouter/google/gemini-embedding-2-preview",
    )


def _config_from_deployment_handle(handle: str) -> Optional[EmbeddingConfig]:
    """Build a known deployment handle without DB lookup."""
    if handle in _GEMINI_EMBEDDING_2_HANDLES or "gemini-embedding-2" in handle:
        return _gemini_embedding_2_config(handle=handle)
    return None


def _coerce_embedding_config(
    raw: Union[EmbeddingConfig, dict, None],
) -> Optional[EmbeddingConfig]:
    if raw is None:
        return None
    if isinstance(raw, EmbeddingConfig):
        return raw.ensure_space_id()
    return EmbeddingConfig(**raw).ensure_space_id()


def resolve_embedding_config(
    agent_state: Optional["AgentState"] = None,
) -> EmbeddingConfig:
    """Resolve embedding config synchronously (agent override or deployment default)."""
    if agent_state is not None and agent_state.embedding_config is not None:
        config = _coerce_embedding_config(agent_state.embedding_config)
        if config is not None:
            return config

    handle = settings.default_embedding_handle
    if not handle:
        raise ValueError(
            "No embedding configuration resolved: set agent embedding_config or settings.default_embedding_handle"
        )

    config = _config_from_deployment_handle(handle)
    if config is not None:
        return config

    raise ValueError(
        f"Cannot resolve embedding config for handle '{handle}' without async provider lookup; "
        "use resolve_embedding_config_async with an actor, or use a known deployment handle."
    )


async def resolve_embedding_config_async(
    agent_state: Optional["AgentState"] = None,
    actor: Optional["PydanticUser"] = None,
) -> EmbeddingConfig:
    """Resolve embedding config with optional provider-manager lookup."""
    if agent_state is not None and agent_state.embedding_config is not None:
        config = _coerce_embedding_config(agent_state.embedding_config)
        if config is not None:
            return config

    handle = settings.default_embedding_handle
    if not handle:
        raise ValueError(
            "No embedding configuration resolved: set agent embedding_config or settings.default_embedding_handle"
        )

    config = _config_from_deployment_handle(handle)
    if config is not None:
        return config

    if actor is not None:
        from letta.services.provider_manager import ProviderManager

        provider_manager = ProviderManager()
        resolved = await provider_manager.get_embedding_config_from_handle(handle=handle, actor=actor)
        return resolved.ensure_space_id()

    raise ValueError(
        f"No embedding configuration resolved for handle '{handle}': provide actor for provider lookup "
        "or use a known deployment handle."
    )
