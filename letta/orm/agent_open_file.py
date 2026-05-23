import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from letta.orm.mixins import OrganizationMixin
from letta.orm.sqlalchemy_base import SqlalchemyBase
from letta.schemas.agent_open_file import AgentOpenFile as PydanticAgentOpenFile

if TYPE_CHECKING:
    from letta.orm.agent import Agent
    from letta.orm.file import FileMetadata


class AgentOpenFile(SqlalchemyBase, OrganizationMixin):
    """Per-agent open-file state and read cursor."""

    __tablename__ = "agent_open_files"
    __pydantic_model__ = PydanticAgentOpenFile
    __table_args__ = (
        UniqueConstraint("agent_id", "file_id", name="uq_agent_open_file"),
        Index("ix_agent_open_files_agent", "agent_id"),
        Index("ix_agent_open_files_last_accessed", "agent_id", "last_accessed_at"),
    )

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: f"agent_open_file-{uuid.uuid4()}",
    )
    agent_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
    )
    file_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("files.id", ondelete="CASCADE"),
        nullable=False,
    )
    cursor_char: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    agent: Mapped["Agent"] = relationship("Agent", lazy="selectin")
    file: Mapped["FileMetadata"] = relationship("FileMetadata", lazy="selectin")
