from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.base import Base, CommonSqlalchemyMetaMixins
from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.file_core_block import FileCoreBlock as PydanticFileCoreBlock

if TYPE_CHECKING:
    from letta.orm.file import FileMetadata


DEFAULT_FILE_CORE_CHAR_LIMIT = 2000


class FileCoreBlock(SqlalchemyBase, OrganizationMixin):
    """Stable per-file headline shared across agents. id equals file_id."""

    __tablename__ = "file_core_blocks"
    __pydantic_model__ = PydanticFileCoreBlock
    __table_args__ = (Index("ix_file_core_blocks_org", "organization_id"),)

    id: Mapped[str] = mapped_column(
        String,
        ForeignKey("files.id", ondelete="CASCADE"),
        primary_key=True,
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False, doc="Stable headline describing what the file is")
    char_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=DEFAULT_FILE_CORE_CHAR_LIMIT)
    last_updated_by_agent_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("agents.id", ondelete="SET NULL"), nullable=True)
    last_updated_at: Mapped[datetime] = mapped_column(
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    file: Mapped["FileMetadata"] = relationship("FileMetadata", lazy="selectin")

    @property
    def file_id(self) -> str:
        return self.id


class FileCoreBlockHistory(CommonSqlalchemyMetaMixins, OrganizationMixin, Base):
    """Append-only history of file core headline edits."""

    __tablename__ = "file_core_block_history"
    __table_args__ = (
        Index("ix_file_core_block_history_file_id", "file_id"),
        Index("ix_file_core_block_history_updated_at", "updated_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    file_id: Mapped[str] = mapped_column(String, ForeignKey("files.id", ondelete="CASCADE"), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    updated_by_agent_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("agents.id", ondelete="SET NULL"), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
