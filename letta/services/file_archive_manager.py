from typing import List, Optional

from sqlalchemy import select

from letta.embeddings.resolver import resolve_embedding_config_async
from letta.llm_api.llm_client import LLMClient
from letta.orm.file import FileMetadata as FileMetadataModel
from letta.orm.file_archive import FileArchive as FileArchiveModel
from letta.otel.tracing import trace_method
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.file_archive import FileArchive as PydanticFileArchive
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.services.file_archive_embedding import prepare_file_archive_embedding_fields
from letta.services.files.archive_tags import normalize_archive_tags
from letta.utils import enforce_types


class FileArchiveManager:
    @enforce_types
    @trace_method
    async def _embed_document_text(self, text: str, actor: PydanticUser) -> tuple[list[float], EmbeddingConfig]:
        config = (await resolve_embedding_config_async(actor=actor)).ensure_space_id()
        client = LLMClient.create(config.embedding_endpoint_type, actor=actor)
        embeddings = await client.request_embeddings([text], config)
        return embeddings[0], config

    @enforce_types
    @trace_method
    async def write_file_archive(
        self,
        *,
        file_id: str,
        title: str,
        content: str,
        tags: Optional[List[str]],
        author_agent_id: str,
        source_conversation_id: Optional[str],
        embedding_config: Optional[EmbeddingConfig] = None,
        actor: PydanticUser,
    ) -> PydanticFileArchive:
        if not (1 <= len(title) <= 200):
            raise ValueError("title must be 1-200 characters")
        if not (1 <= len(content) <= 8000):
            raise ValueError("content must be 1-8000 characters")

        stored_tags = normalize_archive_tags(tags)
        embed_text = f"{title}\n\n{content}".strip()
        embedding, config = await self._embed_document_text(embed_text, actor)

        async with db_registry.async_session() as session:
            file_meta = await session.get(FileMetadataModel, file_id)
            if file_meta is None or file_meta.is_deleted or file_meta.organization_id != actor.organization_id:
                raise ValueError(f"File not found: {file_id}")

            row_data = prepare_file_archive_embedding_fields(
                {
                    "file_id": file_id,
                    "title": title,
                    "content": content,
                    "tags": stored_tags,
                    "author_agent_id": author_agent_id,
                    "source_conversation_id": source_conversation_id,
                    "organization_id": actor.organization_id,
                },
                embedding=embedding,
                config=config,
            )
            row = FileArchiveModel(**row_data)
            await row.create_async(session, actor=actor)
            result = row.to_pydantic()
            result.file_name = file_meta.file_name
            result.tags = stored_tags
            return result

    @enforce_types
    @trace_method
    async def search_file_archives(
        self,
        *,
        query: str,
        agent_id: str,
        embedding_config: Optional[EmbeddingConfig] = None,
        actor: PydanticUser,
        file_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[PydanticFileArchive]:
        from letta.services.recall.hybrid_search import search_file_archives_hybrid

        hits = await search_file_archives_hybrid(
            query,
            actor,
            limit=limit,
            agent_id=agent_id,
            file_id=file_id,
            tags=tags,
        )
        if not hits:
            return []

        async with db_registry.async_session() as session:
            archive_ids = [h.handle for h in hits]
            rows = (
                await session.execute(
                    select(FileArchiveModel, FileMetadataModel)
                    .join(FileMetadataModel, FileArchiveModel.file_id == FileMetadataModel.id)
                    .where(FileArchiveModel.id.in_(archive_ids))
                )
            ).all()
            by_id = {r[0].id: (r[0], r[1]) for r in rows}

        results: List[PydanticFileArchive] = []
        for hit in hits:
            tup = by_id.get(hit.handle)
            if not tup:
                continue
            archive_row, file_meta = tup
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
