from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select

from letta.orm.file_core_block import DEFAULT_FILE_CORE_CHAR_LIMIT, FileCoreBlock as FileCoreBlockModel
from letta.orm.file_core_block import FileCoreBlockHistory as FileCoreBlockHistoryModel
from letta.otel.tracing import trace_method
from letta.schemas.file_core_block import FileCoreBlock as PydanticFileCoreBlock
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.utils import enforce_types


class FileCoreBlockManager:
    @enforce_types
    @trace_method
    async def get_or_create(
        self,
        *,
        file_id: str,
        organization_id: str,
        actor: PydanticUser,
        default_summary: str = "",
    ) -> PydanticFileCoreBlock:
        async with db_registry.async_session() as session:
            row = await session.get(FileCoreBlockModel, file_id)
            if row and not row.is_deleted:
                return row.to_pydantic()

            summary = default_summary or "No headline yet."
            row = FileCoreBlockModel(
                id=file_id,
                summary=summary[:DEFAULT_FILE_CORE_CHAR_LIMIT],
                char_limit=DEFAULT_FILE_CORE_CHAR_LIMIT,
                organization_id=organization_id,
            )
            await row.create_async(session, actor=actor)
            return row.to_pydantic()

    @enforce_types
    @trace_method
    async def get(self, *, file_id: str, actor: PydanticUser) -> Optional[PydanticFileCoreBlock]:
        async with db_registry.async_session() as session:
            row = await session.get(FileCoreBlockModel, file_id)
            if row is None or row.is_deleted or row.organization_id != actor.organization_id:
                return None
            return row.to_pydantic()

    @enforce_types
    @trace_method
    async def update(
        self,
        *,
        file_id: str,
        new_summary: str,
        agent_id: str,
        actor: PydanticUser,
    ) -> PydanticFileCoreBlock:
        async with db_registry.async_session() as session:
            row = await session.get(FileCoreBlockModel, file_id)
            if row is None or row.is_deleted or row.organization_id != actor.organization_id:
                raise ValueError(f"File core block not found for file_id={file_id}")

            if len(new_summary) > row.char_limit:
                raise ValueError(f"Summary exceeds char_limit of {row.char_limit}")

            previous = row.summary
            row.summary = new_summary
            row.last_updated_by_agent_id = agent_id
            row.last_updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            row.version = (row.version or 1) + 1
            await row.update_async(session, actor=actor)

            history = FileCoreBlockHistoryModel(
                file_id=file_id,
                summary=previous,
                updated_by_agent_id=agent_id,
                organization_id=row.organization_id,
            )
            session.add(history)
            await session.commit()

            return row.to_pydantic()

    @enforce_types
    @trace_method
    async def get_many(self, *, file_ids: list[str], actor: PydanticUser) -> dict[str, PydanticFileCoreBlock]:
        if not file_ids:
            return {}
        async with db_registry.async_session() as session:
            query = select(FileCoreBlockModel).where(
                FileCoreBlockModel.id.in_(file_ids),
                FileCoreBlockModel.organization_id == actor.organization_id,
                FileCoreBlockModel.is_deleted == False,
            )
            rows = (await session.execute(query)).scalars().all()
            return {r.id: r.to_pydantic() for r in rows}
