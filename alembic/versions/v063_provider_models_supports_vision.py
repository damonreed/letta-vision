"""add supports_vision to provider_models

Revision ID: v063_provider_models_vision
Revises: v062_drop_legacy_emb
Create Date: 2026-06-12

Persist OpenRouter architecture.input_modalities at provider sync so vision
resolution survives restarts without a fresh /v1/models fetch.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "v063_provider_models_vision"
down_revision: Union[str, None] = "v062_drop_legacy_emb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "provider_models",
        sa.Column("supports_vision", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("provider_models", "supports_vision")
