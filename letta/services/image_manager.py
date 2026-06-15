"""Image record CRUD."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional, Tuple

from sqlalchemy import and_, or_, select

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
        limit: Optional[int] = None,
        enrichment_status: Optional[str] = None,
        after_created_at: Optional[datetime] = None,
        after_id: Optional[str] = None,
    ) -> Tuple[List[PydanticImage], bool]:
        async with db_registry.async_session() as session:
            q = (
                select(ImageRecord)
                .where(ImageRecord.organization_id == actor.organization_id, ImageRecord.is_deleted == False)  # noqa: E712
                .order_by(ImageRecord.created_at.desc(), ImageRecord.id.desc())
            )
            if enrichment_status:
                q = q.where(ImageRecord.enrichment_status == enrichment_status)
            if after_created_at is not None and after_id:
                q = q.where(
                    or_(
                        ImageRecord.created_at < after_created_at,
                        and_(ImageRecord.created_at == after_created_at, ImageRecord.id < after_id),
                    )
                )
            if limit is not None:
                q = q.limit(limit + 1)
            result = await session.execute(q)
            rows = list(result.scalars().all())
            has_more = False
            if limit is not None:
                has_more = len(rows) > limit
                rows = rows[:limit]
            return [r.to_pydantic() for r in rows], has_more

    @enforce_types
    @trace_method
    async def create_record_async(self, record: ImageRecord, actor: PydanticUser) -> PydanticImage:
        async with db_registry.async_session() as session:
            created = await record.create_async(session, actor=actor)
            return created.to_pydantic()

    @enforce_types
    @trace_method
    async def update_text_field_async(
        self,
        image_id: str,
        actor: PydanticUser,
        *,
        field: str,
        value: Optional[str],
    ) -> Optional[PydanticImage]:
        async with db_registry.async_session() as session:
            row = await ImageRecord.read_async(db_session=session, identifier=image_id, actor=actor)
            if field == "caption":
                row.caption = value
            elif field == "description":
                row.description = value
            elif field == "details":
                row.details = value
            else:
                raise ValueError(f"Invalid image text field: {field}")
            updated = await row.update_async(session, actor=actor)
            return updated.to_pydantic()

    @enforce_types
    @trace_method
    async def update_metadata_async(
        self,
        image_id: str,
        actor: PydanticUser,
        *,
        caption: Optional[str] = None,
        description: Optional[str] = None,
        details: Optional[str] = None,
    ) -> Optional[PydanticImage]:
        async with db_registry.async_session() as session:
            row = await ImageRecord.read_async(db_session=session, identifier=image_id, actor=actor)
            row.caption = caption
            row.description = description
            row.details = details
            updated = await row.update_async(session, actor=actor)
            return updated.to_pydantic()

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
