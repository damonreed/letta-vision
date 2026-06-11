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
    caption: Optional[str] = Field(default=None, description="Short label, 20-50 words.")
    description: Optional[str] = Field(default=None, description="Search-oriented summary, 100-200 words.")
    details: Optional[str] = Field(default=None, description="Thorough literal description, 1000 words.")
    embedding: Optional[list[float]] = None
    embedding_config: Optional[EmbeddingConfig] = None
    embedding_space_id: Optional[str] = None
    enrichment_status: EnrichmentStatus = "pending"
    enrichment_attempts: int = 0
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_deleted: bool = False


class ImageMetadataUpdate(BaseModel):
    """User-editable image text tiers: caption (20-50 words), description (100-200 words), details (1000 words)."""

    caption: Optional[str] = Field(default=None, description="Short label, 20-50 words.")
    description: Optional[str] = Field(default=None, description="Search-oriented summary, 100-200 words.")
    details: Optional[str] = Field(default=None, description="Thorough literal description, 1000 words.")


class ImageListResponse(BaseModel):
    images: list[PydanticImage]
    has_more: bool = False


class ImageSearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1)


class ImageSearchHit(BaseModel):
    handle: str
    description: Optional[str] = None
    score: float


class ImageSearchResponse(BaseModel):
    results: list[ImageSearchHit]


class ImageCreate(BaseModel):
    """Payload for synchronous image ingest."""

    data: bytes
    media_type: str
    provenance: Literal["uploaded", "generated"] = "uploaded"
    generation_prompt: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
