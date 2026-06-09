from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import PrimitiveType
from letta.schemas.letta_base import OrmMetadataBase

EnrichmentStatus = Literal["pending", "complete", "failed"]


class PydanticImage(OrmMetadataBase):
    __id_prefix__ = PrimitiveType.IMAGE.value

    id: str = Field(..., description="Image record id (image-<uuid>)")
    organization_id: str
    content_hash: str
    object_url_full: str
    object_url_1mp: Optional[str] = None
    media_type: str
    width: Optional[int] = None
    height: Optional[int] = None
    file_size_full: Optional[int] = None
    file_size_1mp: Optional[int] = None
    provenance: Literal["uploaded", "generated"]
    generation_prompt: Optional[str] = None
    caption: Optional[str] = None
    description: Optional[str] = None
    details: Optional[str] = None
    embedding: Optional[list[float]] = None
    embedding_config: Optional[EmbeddingConfig] = None
    embedding_space_id: Optional[str] = None
    enrichment_status: EnrichmentStatus = "pending"
    enrichment_attempts: int = 0
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_deleted: bool = False


class ImageCreate(BaseModel):
    """Payload for synchronous image ingest."""

    data: bytes
    media_type: str
    provenance: Literal["uploaded", "generated"] = "uploaded"
    generation_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
