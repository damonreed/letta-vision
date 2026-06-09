"""v0.6.0 unified embedding: dual-column passages, message vectors, images table, HNSW, pg_trgm

Revision ID: v060_unified_emb
Revises: f1a2b3c4d5e6
Create Date: 2026-06-09

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "v060_unified_emb"
down_revision: Union[str, None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DEPLOYMENT_DIM = 768
LEGACY_DIM = 4096


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    for table in ("archival_passages", "source_passages"):
        op.add_column(table, sa.Column("embedding_space_id", sa.String(), nullable=True))
        op.create_index(f"ix_{table}_embedding_space_id", table, ["embedding_space_id"])
        op.alter_column(table, "embedding", new_column_name="embedding_legacy_4096")
        op.add_column(table, sa.Column("embedding", Vector(DEPLOYMENT_DIM), nullable=True))
        op.execute(
            f"""
            UPDATE {table}
               SET embedding_space_id = COALESCE(
                   (embedding_config->>'embedding_space_id'),
                   'legacy-unknown'
               )
             WHERE embedding_space_id IS NULL
            """
        )

    op.add_column("messages", sa.Column("embedding", Vector(DEPLOYMENT_DIM), nullable=True))
    op.add_column("messages", sa.Column("embedding_config", sa.JSON(), nullable=True))
    op.add_column("messages", sa.Column("embedding_space_id", sa.String(), nullable=True))
    op.add_column("messages", sa.Column("embedding_version", sa.Integer(), nullable=True))
    op.create_index("ix_messages_embedding_space_id", "messages", ["embedding_space_id"])

    op.create_table(
        "images",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("organization_id", sa.String(), sa.ForeignKey("organizations.id"), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("object_url_full", sa.String(), nullable=False),
        sa.Column("object_url_1mp", sa.String(), nullable=True),
        sa.Column("media_type", sa.String(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("file_size_full", sa.Integer(), nullable=True),
        sa.Column("file_size_1mp", sa.Integer(), nullable=True),
        sa.Column("provenance", sa.String(), nullable=False),
        sa.Column("generation_prompt", sa.Text(), nullable=True),
        sa.Column("caption", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("details", sa.Text(), nullable=True),
        sa.Column("embedding", Vector(DEPLOYMENT_DIM), nullable=True),
        sa.Column("embedding_config", sa.JSON(), nullable=True),
        sa.Column("embedding_space_id", sa.String(), nullable=True),
        sa.Column("enrichment_status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("enrichment_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("_created_by_id", sa.String(), nullable=True),
        sa.Column("_last_updated_by_id", sa.String(), nullable=True),
        sa.UniqueConstraint("organization_id", "content_hash", name="uq_images_org_content_hash"),
    )
    op.create_index("ix_images_embedding_space_id", "images", ["embedding_space_id"])
    op.create_index("ix_images_enrichment_status", "images", ["enrichment_status"])
    op.create_index("ix_images_created_at", "images", ["created_at"])
    op.create_index("ix_images_org_id", "images", ["organization_id"])

    for table, cols in (
        ("archival_passages", ["text"]),
        ("source_passages", ["text"]),
        ("messages", ["text"]),
        ("images", ["caption", "description", "details"]),
    ):
        for col in cols:
            op.execute(
                f"""
                CREATE INDEX IF NOT EXISTS ix_{table}_{col}_trgm
                ON {table} USING gin ({col} gin_trgm_ops)
                """
            )

    for table in ("archival_passages", "source_passages", "messages", "images"):
        op.execute(
            f"""
            CREATE INDEX IF NOT EXISTS ix_{table}_embedding_hnsw
            ON {table} USING hnsw (embedding vector_cosine_ops)
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    for table in ("archival_passages", "source_passages", "messages", "images"):
        op.execute(f"DROP INDEX IF EXISTS ix_{table}_embedding_hnsw")

    op.drop_table("images")
    op.drop_index("ix_messages_embedding_space_id", table_name="messages")
    op.drop_column("messages", "embedding_version")
    op.drop_column("messages", "embedding_space_id")
    op.drop_column("messages", "embedding_config")
    op.drop_column("messages", "embedding")

    for table in ("archival_passages", "source_passages"):
        op.drop_column(table, "embedding")
        op.alter_column(table, "embedding_legacy_4096", new_column_name="embedding")
        op.drop_index(f"ix_{table}_embedding_space_id", table_name=table)
        op.drop_column(table, "embedding_space_id")
