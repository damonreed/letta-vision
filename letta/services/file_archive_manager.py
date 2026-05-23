from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
from sqlalchemy import and_, func, or_, select

from letta.constants import MAX_EMBEDDING_DIM
from letta.llm_api.llm_client import LLMClient
from letta.orm.file import FileMetadata as FileMetadataModel
from letta.orm.file_archive import FileArchive as FileArchiveModel
from letta.orm.sources_agents import SourcesAgents
from letta.otel.tracing import trace_method
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.file_archive import FileArchive as PydanticFileArchive
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.services.files.archive_tags import normalize_archive_tags
from letta.settings import DatabaseChoice, settings
from letta.utils import enforce_types


class FileArchiveManager:
    @enforce_types
    @trace_method
    async def _embed_text(self, text: str, embedding_config: EmbeddingConfig, actor: PydanticUser) -> list[float]:
        client = LLMClient.create(provider_type=embedding_config.embedding_endpoint_type, actor=actor)
        embeddings = await client.request_embeddings([text], embedding_config)
        embedded = np.array(embeddings[0])
        if settings.database_engine is DatabaseChoice.POSTGRES:
            if embedded.shape[0] != MAX_EMBEDDING_DIM:
                embedded = np.pad(embedded, (0, MAX_EMBEDDING_DIM - embedded.shape[0]), mode="constant")
        return embedded.tolist()

    @enforce_types
    @trace_method
    async def write_archive(
        self,
        *,
        file_id: str,
        title: str,
        content: str,
        tags: Optional[List[str]],
        author_agent_id: str,
        source_conversation_id: Optional[str],
        embedding_config: EmbeddingConfig,
        actor: PydanticUser,
    ) -> PydanticFileArchive:
        if not (1 <= len(title) <= 200):
            raise ValueError("title must be 1-200 characters")
        if not (1 <= len(content) <= 8000):
            raise ValueError("content must be 1-8000 characters")

        stored_tags = normalize_archive_tags(tags)
        embedding = await self._embed_text(content, embedding_config, actor)

        async with db_registry.async_session() as session:
            file_meta = await session.get(FileMetadataModel, file_id)
            if file_meta is None or file_meta.is_deleted or file_meta.organization_id != actor.organization_id:
                raise ValueError(f"File not found: {file_id}")

            row = FileArchiveModel(
                file_id=file_id,
                title=title,
                content=content,
                tags=stored_tags,
                author_agent_id=author_agent_id,
                source_conversation_id=source_conversation_id,
                embedding=embedding,
                embedding_config=embedding_config.model_dump(),
                organization_id=actor.organization_id,
            )
            await row.create_async(session, actor=actor)
            result = row.to_pydantic()
            result.file_name = file_meta.file_name
            result.tags = stored_tags
            return result

    @enforce_types
    @trace_method
    async def search_archives(
        self,
        *,
        query: str,
        agent_id: str,
        embedding_config: EmbeddingConfig,
        actor: PydanticUser,
        file_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[PydanticFileArchive]:
        embedded_text = await self._embed_text(query, embedding_config, actor)

        async with db_registry.async_session() as session:
            base = (
                select(FileArchiveModel, FileMetadataModel)
                .join(FileMetadataModel, FileArchiveModel.file_id == FileMetadataModel.id)
                .join(SourcesAgents, SourcesAgents.source_id == FileMetadataModel.source_id)
                .where(
                    FileArchiveModel.organization_id == actor.organization_id,
                    FileArchiveModel.is_deleted == False,
                    SourcesAgents.agent_id == agent_id,
                )
            )
            if file_id:
                base = base.where(FileArchiveModel.file_id == file_id)
            if tags:
                normalized = normalize_archive_tags(tags)
                if normalized:
                    tag_filters = [FileArchiveModel.tags.contains([t]) for t in normalized]
                    base = base.where(or_(*tag_filters))

            if settings.database_engine is DatabaseChoice.POSTGRES:
                base = base.order_by(FileArchiveModel.embedding.cosine_distance(embedded_text).asc())
            else:
                from letta.orm.sqlite_functions import adapt_array

                base = base.order_by(
                    func.cosine_distance(FileArchiveModel.embedding, adapt_array(embedded_text)).asc(),
                )

            rows = (await session.execute(base.limit(limit))).all()
            results: List[PydanticFileArchive] = []
            for archive_row, file_meta in rows:
                item = archive_row.to_pydantic()
                item.file_name = file_meta.file_name
                results.append(item)
            return results

    @enforce_types
    @trace_method
    async def list_for_file(
        self, *, file_id: str, actor: PydanticUser, limit: int = 50
    ) -> List[PydanticFileArchive]:
        async with db_registry.async_session() as session:
            query = (
                select(FileArchiveModel)
                .where(
                    FileArchiveModel.file_id == file_id,
                    FileArchiveModel.organization_id == actor.organization_id,
                    FileArchiveModel.is_deleted == False,
                )
                .order_by(FileArchiveModel.created_at.desc())
                .limit(limit)
            )
            rows = (await session.execute(query)).scalars().all()
            return [r.to_pydantic() for r in rows]
