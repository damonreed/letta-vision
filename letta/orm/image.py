from typing import TYPE_CHECKING, Optional

from sqlalchemy import Index, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.constants import DEPLOYMENT_EMBEDDING_DIM
from letta.orm.custom_columns import CommonVector, EmbeddingConfigColumn
from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.settings import DatabaseChoice, settings

if TYPE_CHECKING:
    from letta.orm.organization import Organization

if settings.database_engine is DatabaseChoice.POSTGRES:
    from pgvector.sqlalchemy import Vector


class ImageRecord(SqlalchemyBase, OrganizationMixin):
    __tablename__ = "images"
    __pydantic_model__ = None  # set after schema import
    __table_args__ = (
        UniqueConstraint("organization_id", "content_hash", name="uq_images_org_content_hash"),
        Index("ix_images_org_id", "organization_id"),
        Index("ix_images_enrichment_status", "enrichment_status"),
        Index("ix_images_created_at", "created_at"),
        {"extend_existing": True},
    )

    id: Mapped[str] = mapped_column(primary_key=True)
    content_hash: Mapped[str] = mapped_column(nullable=False)
    object_url_full: Mapped[str] = mapped_column(nullable=False)
    object_url_1mp: Mapped[Optional[str]] = mapped_column(nullable=True)
    media_type: Mapped[str] = mapped_column(nullable=False)
    width: Mapped[Optional[int]] = mapped_column(nullable=True)
    height: Mapped[Optional[int]] = mapped_column(nullable=True)
    file_size_full: Mapped[Optional[int]] = mapped_column(nullable=True)
    file_size_1mp: Mapped[Optional[int]] = mapped_column(nullable=True)
    provenance: Mapped[str] = mapped_column(nullable=False)
    generation_prompt: Mapped[Optional[str]] = mapped_column(nullable=True)
    caption: Mapped[Optional[str]] = mapped_column(nullable=True)
    description: Mapped[Optional[str]] = mapped_column(nullable=True)
    details: Mapped[Optional[str]] = mapped_column(nullable=True)
    if settings.database_engine is DatabaseChoice.POSTGRES:
        embedding: Mapped[Optional[list]] = mapped_column(Vector(DEPLOYMENT_EMBEDDING_DIM), nullable=True)
    else:
        embedding = mapped_column(CommonVector, nullable=True)
    embedding_config: Mapped[Optional[dict]] = mapped_column(EmbeddingConfigColumn, nullable=True)
    embedding_space_id: Mapped[Optional[str]] = mapped_column(nullable=True, index=True)
    enrichment_status: Mapped[str] = mapped_column(nullable=False, default="pending")
    enrichment_attempts: Mapped[int] = mapped_column(nullable=False, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(nullable=True)

    organization: Mapped["Organization"] = relationship("Organization", lazy="raise")


from letta.schemas.image import PydanticImage  # noqa: E402

ImageRecord.__pydantic_model__ = PydanticImage
