"""Image record CRUD."""

from __future__ import annotations

import uuid
from typing import List, Optional

from sqlalchemy import select

from letta.log import get_logger
from letta.orm.image import ImageRecord
from letta.otel.tracing import trace_method
from letta.schemas.image import PydanticImage
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.utils import enforce_types

logger = get_logger(__name__)


class ImageManager:
    @enforce_types
    @trace_method
    async def get_by_id_async(self, image_id: str, actor: PydanticUser) -> Optional[PydanticImage]:
        async with db_registry.async_session() as session:
            row = await ImageRecord.read_async(db_session=session, identifier=image_id, actor=actor)
            return row.to_pydantic() if row else None

    @enforce_types
    @trace_method
    async def get_by_hash_async(self, content_hash: str, actor: PydanticUser) -> Optional[PydanticImage]:
        async with db_registry.async_session() as session:
            q = select(ImageRecord).where(
                ImageRecord.organization_id == actor.organization_id,
                ImageRecord.content_hash == content_hash,
                ImageRecord.is_deleted == False,  # noqa: E712
            )
            result = await session.execute(q)
            row = result.scalar_one_or_none()
            return row.to_pydantic() if row else None

    @enforce_types
    @trace_method
    async def list_async(
        self,
        actor: PydanticUser,
        *,
        limit: int = 50,
        enrichment_status: Optional[str] = None,
    ) -> List[PydanticImage]:
        async with db_registry.async_session() as session:
            q = (
                select(ImageRecord)
                .where(ImageRecord.organization_id == actor.organization_id, ImageRecord.is_deleted == False)  # noqa: E712
                .order_by(ImageRecord.created_at.desc())
                .limit(limit)
            )
            if enrichment_status:
                q = q.where(ImageRecord.enrichment_status == enrichment_status)
            result = await session.execute(q)
            return [r.to_pydantic() for r in result.scalars().all()]

    @enforce_types
    @trace_method
    async def create_record_async(self, record: ImageRecord, actor: PydanticUser) -> PydanticImage:
        async with db_registry.async_session() as session:
            created = await record.create_async(session, actor=actor)
            return created.to_pydantic()

    @enforce_types
    @trace_method
    async def delete_async(self, image_id: str, actor: PydanticUser) -> bool:
        async with db_registry.async_session() as session:
            row = await ImageRecord.read_async(db_session=session, identifier=image_id, actor=actor)
            row.is_deleted = True
            await row.update_async(session, actor=actor)
            return True

    @staticmethod
    def new_image_id() -> str:
        return f"image-{uuid.uuid4()}"
