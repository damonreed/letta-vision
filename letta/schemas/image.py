from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

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
    details: Optional[str] = Field(
        default=None,
        description="Prompt-ready literal description, 1500-2000 words with structured section headings.",
    )
    embedding: Optional[list[float]] = None
    embedding_config: Optional[EmbeddingConfig] = None
    embedding_space_id: Optional[str] = None
    enrichment_status: EnrichmentStatus = "pending"
    enrichment_attempts: int = 0
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_deleted: bool = False


class ImageTextEdit(BaseModel):
    """One edit to a caption, description, or details tier."""

    model_config = ConfigDict(extra="forbid")

    field: Literal["caption", "description", "details"] = Field(
        ...,
        description="Text tier to edit: caption, description, or details.",
    )
    command: Literal["str_replace", "insert", "set"] = Field(
        ...,
        description="str_replace needs old_string and new_string. insert needs insert_text (insert_line=-1 appends); do not send that text as new_string. set replaces the whole field with new_string.",
    )
    old_string: Optional[str] = Field(
        default=None,
        description="Exact text to replace. Required for str_replace. Ignored by insert and set.",
    )
    new_string: Optional[str] = Field(
        default=None,
        description="Replacement text. Required for str_replace and set. Not the insert payload.",
    )
    insert_text: Optional[str] = Field(
        default=None,
        description="Text to insert. Required for insert. Do not send this as new_string.",
    )
    insert_line: int = Field(
        default=-1,
        description="Line index for insert. -1 appends. Ignored by str_replace and set.",
    )


class ImageMetadataUpdate(BaseModel):
    """User-editable image text tiers: caption (20-50 words), description (100-200 words), details (1500-2000 words)."""

    caption: Optional[str] = Field(default=None, description="Short label, 20-50 words.")
    description: Optional[str] = Field(default=None, description="Search-oriented summary, 100-200 words.")
    details: Optional[str] = Field(
        default=None,
        description="Prompt-ready literal description, 1500-2000 words with structured section headings.",
    )


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
