"""Atomic embedding writes with monotonic version guard."""

from __future__ import annotations

from typing import List

from sqlalchemy import or_, update

from letta.log import get_logger
from letta.orm.message import Message as MessageModel
from letta.schemas.embedding_config import EmbeddingConfig
from letta.server.db import db_registry

logger = get_logger(__name__)


async def write_message_embedding_atomic(
    message_id: str,
    organization_id: str,
    embedding: List[float],
    embedding_config: EmbeddingConfig,
    embedding_version: int,
) -> bool:
    """Conditionally update message embedding. Returns True if write applied."""
    config = embedding_config.ensure_space_id()
    async with db_registry.async_session() as session:
        stmt = (
            update(MessageModel)
            .where(MessageModel.id == message_id)
            .where(MessageModel.organization_id == organization_id)
            .where(
                or_(
                    MessageModel.embedding_version.is_(None),
                    MessageModel.embedding_version < embedding_version,
                )
            )
            .values(
                embedding=embedding,
                embedding_config=config.model_dump(),
                embedding_space_id=config.embedding_space_id,
                embedding_version=embedding_version,
            )
        )
        result = await session.execute(stmt)
        await session.commit()
        applied = result.rowcount == 1
        if not applied:
            logger.debug(
                "Skipped message embedding write (monotonic guard): message_id=%s version=%s",
                message_id,
                embedding_version,
            )
        return applied
