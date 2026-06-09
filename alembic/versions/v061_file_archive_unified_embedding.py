"""file_archives: dual-column embedding + space id + trigram indexes for recall

Revision ID: v061_file_archive_emb
Revises: v060_unified_emb
Create Date: 2026-06-09

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "v061_file_archive_emb"
down_revision: Union[str, None] = "v060_unified_emb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DEPLOYMENT_DIM = 768


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.add_column("file_archives", sa.Column("embedding_space_id", sa.String(), nullable=True))
    op.create_index("ix_file_archives_embedding_space_id", "file_archives", ["embedding_space_id"])
    op.alter_column("file_archives", "embedding", new_column_name="embedding_legacy_4096")
    op.add_column("file_archives", sa.Column("embedding", Vector(DEPLOYMENT_DIM), nullable=True))
    op.execute(
        """
        UPDATE file_archives
           SET embedding_space_id = COALESCE(
               (embedding_config->>'embedding_space_id'),
               'legacy-unknown'
           )
         WHERE embedding_space_id IS NULL
        """
    )

    for col in ("title", "content"):
        op.execute(
            f"""
            CREATE INDEX IF NOT EXISTS ix_file_archives_{col}_trgm
            ON file_archives USING gin ({col} gin_trgm_ops)
            """
        )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_file_archives_embedding_hnsw
        ON file_archives USING hnsw (embedding vector_cosine_ops)
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("DROP INDEX IF EXISTS ix_file_archives_embedding_hnsw")
    for col in ("title", "content"):
        op.execute(f"DROP INDEX IF EXISTS ix_file_archives_{col}_trgm")
    op.drop_column("file_archives", "embedding")
    op.alter_column("file_archives", "embedding_legacy_4096", new_column_name="embedding")
    op.drop_index("ix_file_archives_embedding_space_id", table_name="file_archives")
    op.drop_column("file_archives", "embedding_space_id")
