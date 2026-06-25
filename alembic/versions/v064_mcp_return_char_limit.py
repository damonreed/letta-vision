"""raise return_char_limit for external_mcp tools

Revision ID: v064_mcp_return_char_limit
Revises: v063_provider_models_vision
Create Date: 2026-06-25

MCP tools (e.g. Scenecraft) often return large JSON payloads that exceed the
default 50k function return limit. Preserve any per-tool limit above the MCP
default.
"""

from typing import Sequence, Union

from alembic import op
from letta.constants import MCP_FUNCTION_RETURN_CHAR_LIMIT
from letta.settings import settings

revision: str = "v064_mcp_return_char_limit"
down_revision: Union[str, None] = "v063_provider_models_vision"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    if not settings.letta_pg_uri_no_default:
        return

    op.execute(
        f"""
        UPDATE tools
        SET return_char_limit = {MCP_FUNCTION_RETURN_CHAR_LIMIT}
        WHERE tool_type = 'external_mcp'
          AND (return_char_limit IS NULL OR return_char_limit < {MCP_FUNCTION_RETURN_CHAR_LIMIT})
        """
    )


def downgrade() -> None:
    if not settings.letta_pg_uri_no_default:
        return

    op.execute(
        f"""
        UPDATE tools
        SET return_char_limit = 50000
        WHERE tool_type = 'external_mcp'
          AND return_char_limit = {MCP_FUNCTION_RETURN_CHAR_LIMIT}
        """
    )
