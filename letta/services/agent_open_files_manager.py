from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import and_, delete, select

from letta.orm.agent_open_file import AgentOpenFile as AgentOpenFileModel
from letta.orm.file import FileContent as FileContentModel
from letta.orm.file import FileMetadata as FileMetadataModel
from letta.orm.file_core_block import FileCoreBlock as FileCoreBlockModel
from letta.otel.tracing import trace_method
from letta.schemas.agent_open_file import AgentOpenFile as PydanticAgentOpenFile
from letta.schemas.file_core_block import OpenFileCoreView
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.utils import enforce_types


class AgentOpenFilesManager:
    async def _get_open_row(self, session, agent_id: str, file_id: str, org_id: str) -> Optional[AgentOpenFileModel]:
        query = select(AgentOpenFileModel).where(
            AgentOpenFileModel.agent_id == agent_id,
            AgentOpenFileModel.file_id == file_id,
            AgentOpenFileModel.organization_id == org_id,
            AgentOpenFileModel.is_deleted == False,
        )
        return await session.scalar(query)

    @enforce_types
    @trace_method
    async def open_file(
        self,
        *,
        agent_id: str,
        file_id: str,
        actor: PydanticUser,
        max_files_open: int,
    ) -> tuple[PydanticAgentOpenFile, List[str]]:
        """Open a file with LRU eviction. Returns (open_file_row, evicted_file_ids)."""
        evicted: List[str] = []
        async with db_registry.async_session() as session:
            file_meta = await session.get(FileMetadataModel, file_id)
            if file_meta is None or file_meta.is_deleted or file_meta.organization_id != actor.organization_id:
                raise ValueError(f"File not found: {file_id}")

            existing = await self._get_open_row(session, agent_id, file_id, actor.organization_id)
            now = datetime.now(timezone.utc)

            open_query = (
                select(AgentOpenFileModel)
                .where(
                    AgentOpenFileModel.agent_id == agent_id,
                    AgentOpenFileModel.organization_id == actor.organization_id,
                    AgentOpenFileModel.is_deleted == False,
                )
                .order_by(AgentOpenFileModel.last_accessed_at.asc())
            )
            open_rows = list((await session.execute(open_query)).scalars().all())
            other_open = [r for r in open_rows if r.file_id != file_id]

            if not existing and len(open_rows) >= max_files_open:
                to_evict = other_open[: max(0, len(open_rows) - max_files_open + 1)]
                for row in to_evict:
                    evicted.append(row.file_id)
                    await session.delete(row)

            if existing:
                existing.last_accessed_at = now
                await existing.update_async(session, actor=actor, no_refresh=True)
                result = existing
            else:
                result = AgentOpenFileModel(
                    agent_id=agent_id,
                    file_id=file_id,
                    cursor_char=0,
                    opened_at=now,
                    last_accessed_at=now,
                    organization_id=actor.organization_id,
                )
                await result.create_async(session, actor=actor, no_refresh=True)

            pydantic = result.to_pydantic()
            pydantic.file_name = file_meta.file_name
            return pydantic, evicted

    @enforce_types
    @trace_method
    async def close_file(self, *, agent_id: str, file_id: str, actor: PydanticUser) -> bool:
        async with db_registry.async_session() as session:
            row = await self._get_open_row(session, agent_id, file_id, actor.organization_id)
            if row is None:
                return False
            await session.delete(row)
            await session.commit()
            return True

    @enforce_types
    @trace_method
    async def get_open_file(self, *, agent_id: str, file_id: str, actor: PydanticUser) -> Optional[PydanticAgentOpenFile]:
        async with db_registry.async_session() as session:
            row = await self._get_open_row(session, agent_id, file_id, actor.organization_id)
            if row is None:
                return None
            return row.to_pydantic()

    @enforce_types
    @trace_method
    async def clamp_cursors_for_file(self, *, file_id: str, max_char: int, actor: PydanticUser) -> int:
        """Clamp cursor_char for every agent with this file open. Returns rows touched."""
        max_char = max(0, max_char)
        async with db_registry.async_session() as session:
            query = select(AgentOpenFileModel).where(
                AgentOpenFileModel.file_id == file_id,
                AgentOpenFileModel.organization_id == actor.organization_id,
                AgentOpenFileModel.is_deleted == False,
            )
            rows = list((await session.execute(query)).scalars().all())
            for row in rows:
                row.cursor_char = min(row.cursor_char, max_char)
            if rows:
                await session.commit()
            return len(rows)

    @enforce_types
    @trace_method
    async def update_cursor(self, *, agent_id: str, file_id: str, cursor_char: int, actor: PydanticUser) -> None:
        async with db_registry.async_session() as session:
            row = await self._get_open_row(session, agent_id, file_id, actor.organization_id)
            if row is None:
                raise ValueError(f"File {file_id} is not open for agent {agent_id}")
            row.cursor_char = max(0, cursor_char)
            row.last_accessed_at = datetime.now(timezone.utc)
            await row.update_async(session, actor=actor)

    @enforce_types
    @trace_method
    async def list_open_files_with_cores(
        self, *, agent_id: str, actor: PydanticUser
    ) -> List[OpenFileCoreView]:
        async with db_registry.async_session() as session:
            query = (
                select(AgentOpenFileModel, FileMetadataModel, FileCoreBlockModel, FileContentModel)
                .join(FileMetadataModel, AgentOpenFileModel.file_id == FileMetadataModel.id)
                .outerjoin(FileCoreBlockModel, AgentOpenFileModel.file_id == FileCoreBlockModel.id)
                .outerjoin(FileContentModel, AgentOpenFileModel.file_id == FileContentModel.file_id)
                .where(
                    AgentOpenFileModel.agent_id == agent_id,
                    AgentOpenFileModel.organization_id == actor.organization_id,
                    AgentOpenFileModel.is_deleted == False,
                )
                .order_by(AgentOpenFileModel.last_accessed_at.desc())
            )
            rows = (await session.execute(query)).all()
            views: List[OpenFileCoreView] = []
            for open_row, file_meta, core, content in rows:
                summary = core.summary if core and not core.is_deleted else "No headline yet."
                char_limit = core.char_limit if core and not core.is_deleted else 2000
                total_chars = len(content.text) if content and content.text else 0
                views.append(
                    OpenFileCoreView(
                        file_id=open_row.file_id,
                        file_name=file_meta.file_name or open_row.file_id,
                        source_id=file_meta.source_id,
                        summary=summary,
                        char_limit=char_limit,
                        cursor_char=open_row.cursor_char,
                        total_chars=total_chars,
                        is_open=True,
                    )
                )
            return views

    @enforce_types
    @trace_method
    async def close_all(self, *, agent_id: str, actor: PydanticUser) -> int:
        async with db_registry.async_session() as session:
            stmt = delete(AgentOpenFileModel).where(
                AgentOpenFileModel.agent_id == agent_id,
                AgentOpenFileModel.organization_id == actor.organization_id,
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount or 0
