import uuid
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import JSON, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.constants import MAX_EMBEDDING_DIM
from letta.orm.custom_columns import CommonVector, EmbeddingConfigColumn
from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.file_archive import FileArchive as PydanticFileArchive
from letta.settings import DatabaseChoice, settings

if TYPE_CHECKING:
    from letta.orm.file import FileMetadata

if settings.database_engine is DatabaseChoice.POSTGRES:
    from pgvector.sqlalchemy import Vector


class FileArchive(SqlalchemyBase, OrganizationMixin):
    """Topical notes about a file, written during agent interactions."""

    __tablename__ = "file_archives"
    __pydantic_model__ = PydanticFileArchive

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: f"file_archive-{uuid.uuid4()}")
    file_id: Mapped[str] = mapped_column(String, ForeignKey("files.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    author_agent_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("agents.id", ondelete="SET NULL"), nullable=True)
    source_conversation_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    embedding_config: Mapped[Optional[dict]] = mapped_column(EmbeddingConfigColumn, nullable=True)

    if settings.database_engine is DatabaseChoice.POSTGRES:
        embedding: Mapped[Optional[list]] = mapped_column(Vector(MAX_EMBEDDING_DIM), nullable=True)
    else:
        embedding: Mapped[Optional[list]] = mapped_column(CommonVector, nullable=True)

    file: Mapped["FileMetadata"] = relationship("FileMetadata", lazy="selectin")

    __table_args__ = (
        Index("file_archives_file_id_idx", "file_id"),
        Index("file_archives_created_at_idx", "created_at"),
        Index("ix_file_archives_organization_id", "organization_id"),
    )
