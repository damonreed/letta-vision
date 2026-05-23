"""Add three-tier filesystem memory tables

Revision ID: f1a2b3c4d5e6
Revises: 1c28e167b74f
Create Date: 2026-05-23

"""

from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op
from letta.constants import MAX_EMBEDDING_DIM
from letta.orm.custom_columns import EmbeddingConfigColumn
from letta.settings import settings

revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, None] = "1c28e167b74f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if not settings.letta_pg_uri_no_default:
        return

    op.create_table(
        "file_core_blocks",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("char_limit", sa.Integer(), nullable=False, server_default="2000"),
        sa.Column("last_updated_by_agent_id", sa.String(), nullable=True),
        sa.Column("last_updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("_created_by_id", sa.String(), nullable=True),
        sa.Column("_last_updated_by_id", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["id"], ["files.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["last_updated_by_agent_id"], ["agents.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_file_core_blocks_org", "file_core_blocks", ["organization_id"])

    op.create_table(
        "file_core_block_history",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("file_id", sa.String(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("updated_by_agent_id", sa.String(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("_created_by_id", sa.String(), nullable=True),
        sa.Column("_last_updated_by_id", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["file_id"], ["files.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["updated_by_agent_id"], ["agents.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_file_core_block_history_file_id", "file_core_block_history", ["file_id"])
    op.create_index("ix_file_core_block_history_updated_at", "file_core_block_history", ["updated_at"])

    op.create_table(
        "agent_open_files",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("file_id", sa.String(), nullable=False),
        sa.Column("cursor_char", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("opened_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("_created_by_id", sa.String(), nullable=True),
        sa.Column("_last_updated_by_id", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["file_id"], ["files.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("agent_id", "file_id", name="uq_agent_open_file"),
    )
    op.create_index("ix_agent_open_files_agent", "agent_open_files", ["agent_id"])
    op.create_index("ix_agent_open_files_last_accessed", "agent_open_files", ["agent_id", "last_accessed_at"])

    op.create_table(
        "file_archives",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("file_id", sa.String(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tags", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="[]"),
        sa.Column("author_agent_id", sa.String(), nullable=True),
        sa.Column("source_conversation_id", sa.String(), nullable=True),
        sa.Column("embedding_config", EmbeddingConfigColumn(), nullable=True),
        sa.Column("embedding", Vector(dim=MAX_EMBEDDING_DIM), nullable=True),
        sa.Column("organization_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), server_default=sa.text("FALSE"), nullable=False),
        sa.Column("_created_by_id", sa.String(), nullable=True),
        sa.Column("_last_updated_by_id", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["file_id"], ["files.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["author_agent_id"], ["agents.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["source_conversation_id"], ["conversations.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("file_archives_file_id_idx", "file_archives", ["file_id"])
    op.create_index("file_archives_created_at_idx", "file_archives", ["created_at"])
    op.create_index("ix_file_archives_organization_id", "file_archives", ["organization_id"])
    # No HNSW index: MAX_EMBEDDING_DIM is 4096 but pgvector HNSW caps at 2000 dims (same as source/archival passages).
    op.execute("CREATE INDEX file_archives_tags_gin ON file_archives USING gin (tags jsonb_path_ops)")


def downgrade() -> None:
    if not settings.letta_pg_uri_no_default:
        return

    op.drop_index("file_archives_tags_gin", table_name="file_archives")
    op.drop_index("ix_file_archives_organization_id", table_name="file_archives")
    op.drop_index("file_archives_created_at_idx", table_name="file_archives")
    op.drop_index("file_archives_file_id_idx", table_name="file_archives")
    op.drop_table("file_archives")

    op.drop_index("ix_agent_open_files_last_accessed", table_name="agent_open_files")
    op.drop_index("ix_agent_open_files_agent", table_name="agent_open_files")
    op.drop_table("agent_open_files")

    op.drop_index("ix_file_core_block_history_updated_at", table_name="file_core_block_history")
    op.drop_index("ix_file_core_block_history_file_id", table_name="file_core_block_history")
    op.drop_table("file_core_block_history")

    op.drop_index("ix_file_core_blocks_org", table_name="file_core_blocks")
    op.drop_table("file_core_blocks")
