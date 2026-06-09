"""Shared embedding helpers for file reading notes (file_archives)."""

from __future__ import annotations

from typing import Optional

from letta.embeddings.util import prepare_vector_for_write
from letta.schemas.embedding_config import EmbeddingConfig


def prepare_file_archive_embedding_fields(
    data: dict,
    *,
    embedding: Optional[list[float]],
    config: EmbeddingConfig,
) -> dict:
    """Write 768-dim vectors to embedding column with space id stamp."""
    if not embedding:
        return data
    config = config.ensure_space_id()
    data["embedding"] = prepare_vector_for_write(embedding, config)
    data["embedding_config"] = config
    data["embedding_space_id"] = config.embedding_space_id
    return data
