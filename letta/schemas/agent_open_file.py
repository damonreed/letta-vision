from datetime import datetime
from typing import Optional

from pydantic import Field

from letta.schemas.letta_base import LettaBase


class AgentOpenFile(LettaBase):
    agent_id: str = Field(..., description="Agent ID")
    file_id: str = Field(..., description="File ID")
    cursor_char: int = Field(default=0, description="Current read cursor in characters")
    opened_at: Optional[datetime] = Field(None, description="When the file was opened")
    last_accessed_at: Optional[datetime] = Field(None, description="When the file was last accessed")
    organization_id: Optional[str] = Field(None, description="Organization ID")
    file_name: Optional[str] = Field(None, description="Denormalized file name for display")
    headline: Optional[str] = Field(None, description="File core headline if available")
