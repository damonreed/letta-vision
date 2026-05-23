from datetime import datetime
from typing import List, Optional

from pydantic import Field

from letta.schemas.enums import PrimitiveType
from letta.schemas.letta_base import LettaBase


class FileArchiveBase(LettaBase):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, max_length=8000)
    tags: List[str] = Field(default_factory=list, description="Normalized tags for scoped search")


class FileArchiveCreate(FileArchiveBase):
    file_id: str = Field(..., description="File this archive is about")


class FileArchive(FileArchiveBase):
    __id_prefix__ = "file_archive"

    id: str = Field(..., description="Archive ID")
    file_id: str = Field(..., description="File ID")
    author_agent_id: Optional[str] = Field(None, description="Agent that wrote the archive")
    source_conversation_id: Optional[str] = Field(None, description="Conversation where the archive was written")
    organization_id: Optional[str] = Field(None, description="Organization ID")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    file_name: Optional[str] = Field(None, description="Joined file name for display")
    similarity: Optional[float] = Field(None, description="Search similarity score when returned from search")


class FileArchiveSearchResult(LettaBase):
    results: List[FileArchive] = Field(default_factory=list)
