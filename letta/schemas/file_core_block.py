from datetime import datetime
from typing import Optional

from pydantic import Field, model_validator

from letta.schemas.letta_base import LettaBase


class FileCoreBlockBase(LettaBase):
    summary: str = Field(..., description="Stable headline describing what the file is")
    char_limit: int = Field(default=2000, description="Maximum character length for the headline")


class FileCoreBlock(FileCoreBlockBase):
    file_id: str = Field(..., description="ID of the file")
    organization_id: Optional[str] = Field(None, description="Organization ID")
    last_updated_by_agent_id: Optional[str] = Field(None, description="Agent that last updated the headline")
    last_updated_at: Optional[datetime] = Field(None, description="When the headline was last updated")
    version: int = Field(default=1, description="Monotonic version number")

    @model_validator(mode="before")
    @classmethod
    def _map_file_id_from_orm(cls, data):
        if isinstance(data, dict) and "file_id" not in data and data.get("id"):
            data = {**data, "file_id": data["id"]}
        return data


class FileCoreBlockUpdate(LettaBase):
    summary: str = Field(..., min_length=1, description="New headline text")


class OpenFileCoreView(LettaBase):
    """File core headline joined with open-file metadata for prompt compilation."""

    file_id: str
    file_name: str
    source_id: str
    summary: str
    char_limit: int = 2000
    cursor_char: int = 0
    total_chars: Optional[int] = None
    is_open: bool = True
