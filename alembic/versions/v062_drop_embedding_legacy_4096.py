"""drop embedding_legacy_4096 from passage and archive tables

Revision ID: v062_drop_legacy_emb
Revises: v061_file_archive_emb
Create Date: 2026-06-12

Ship together with removal of embedding_legacy_4096 ORM mappings — SQLAlchemy
includes mapped columns in SELECTs, so dropping the DB column alone breaks reads.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

from letta.constants import MAX_EMBEDDING_DIM

revision: str = "v062_drop_legacy_emb"
down_revision: Union[str, None] = "v061_file_archive_emb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_LEGACY_TABLES = ("archival_passages", "source_passages", "file_archives")


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    for table in _LEGACY_TABLES:
        op.drop_column(table, "embedding_legacy_4096")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    for table in _LEGACY_TABLES:
        op.add_column(table, sa.Column("embedding_legacy_4096", Vector(MAX_EMBEDDING_DIM), nullable=True))
