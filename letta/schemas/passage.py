from datetime import datetime
from typing import Dict, List, Optional

from pydantic import Field

from letta.helpers.datetime_helpers import get_utc_time
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import PrimitiveType
from letta.schemas.letta_base import OrmMetadataBase


class PassageBase(OrmMetadataBase):
    __id_prefix__ = PrimitiveType.PASSAGE.value

    is_deleted: bool = Field(False, description="Whether this passage is deleted or not.")

    # associated user/agent
    organization_id: Optional[str] = Field(None, description="The unique identifier of the user associated with the passage.")
    archive_id: Optional[str] = Field(None, description="The unique identifier of the archive containing this passage.")

    # origin data source
    source_id: Optional[str] = Field(
        None, description="Deprecated: Use `folder_id` field instead. The data source of the passage.", deprecated=True
    )

    # file association
    file_id: Optional[str] = Field(None, description="The unique identifier of the file associated with the passage.")
    file_name: Optional[str] = Field(None, description="The name of the file (only for source passages).")
    metadata: Optional[Dict] = Field({}, validation_alias="metadata_", description="The metadata of the passage.")
    tags: Optional[List[str]] = Field(None, description="Tags associated with this passage.")


class Passage(PassageBase):
    """Representation of a passage, which is stored in archival memory."""

    id: str = PassageBase.generate_id_field()

    # passage text
    text: str = Field(..., description="The text of the passage.")

    # embeddings
    embedding: Optional[List[float]] = Field(..., description="The embedding of the passage.")
    embedding_config: Optional[EmbeddingConfig] = Field(..., description="The embedding configuration used by the passage.")

    created_at: Optional[datetime] = Field(default_factory=get_utc_time, description="The creation date of the passage.")


class PassageCreate(PassageBase):
    text: str = Field(..., description="The text of the passage.")

    # optionally provide embeddings
    embedding: Optional[List[float]] = Field(None, description="The embedding of the passage.")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the passage.")
    created_at: Optional[datetime] = Field(None, description="Optional creation datetime for the passage.")


class PassageUpdate(PassageCreate):
    id: str = Field(..., description="The unique identifier of the passage.")
    text: Optional[str] = Field(None, description="The text of the passage.")

    # optionally provide embeddings
    embedding: Optional[List[float]] = Field(None, description="The embedding of the passage.")
    embedding_config: Optional[EmbeddingConfig] = Field(None, description="The embedding configuration used by the passage.")
