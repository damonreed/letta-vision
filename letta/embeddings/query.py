"""Query-time embedding helpers and space guard."""

from __future__ import annotations

from typing import List, Optional, Tuple, Type

from sqlalchemy import Select

from letta.llm_api.llm_client import LLMClient
from letta.log import get_logger
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.user import User as PydanticUser

logger = get_logger(__name__)


async def embed_search_query(
    query_text: str,
    embedding_config: EmbeddingConfig,
    actor: Optional[PydanticUser] = None,
) -> Tuple[List[float], str]:
    """Embed a search query with search_query input_type; return vector and space id."""
    config = embedding_config.ensure_space_id()
    embedding_client = LLMClient.create(
        provider_type=config.embedding_endpoint_type,
        actor=actor,
    )
    embeddings = await embedding_client.request_embeddings(
        [query_text],
        config,
        input_type_override="search_query",
    )
    return embeddings[0], config.embedding_space_id


def apply_embedding_space_guard(query: Select, model_cls: Type, embedding_space_id: str) -> Select:
    """Restrict vector search to rows in the query embedding space with non-null vectors."""
    if hasattr(model_cls, "embedding"):
        query = query.where(model_cls.embedding.isnot(None))
    if hasattr(model_cls, "embedding_space_id"):
        query = query.where(model_cls.embedding_space_id == embedding_space_id)
    return query
