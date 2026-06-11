from datetime import datetime
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BeforeValidator
from fastapi.responses import Response

from letta.schemas.image import (
    ImageListResponse,
    ImageMetadataUpdate,
    ImageSearchHit,
    ImageSearchRequest,
    ImageSearchResponse,
    PydanticImage,
)
from letta.schemas.user import User as PydanticUser
from letta.server.rest_api.dependencies import HeaderParams, get_headers, get_letta_server
from letta.server.server import SyncServer
from letta.services.image_ingest import schedule_image_re_enrichment
from letta.services.image_manager import ImageManager
from letta.services.object_store.client import get_object_store_client
from letta.services.recall.hybrid_search import search_images_hybrid
from letta.utils.datetime_cursor import parse_cursor_datetime

router = APIRouter(prefix="/images", tags=["images"])

CursorDatetime = Annotated[Optional[datetime], BeforeValidator(parse_cursor_datetime)]


@router.get("", response_model=ImageListResponse)
async def list_images(
    limit: Optional[int] = Query(None, description="Page size; omit for full org corpus"),
    enrichment_status: Optional[str] = None,
    after_created_at: CursorDatetime = Query(None, description="Cursor: created_at of last row from prior page"),
    after_id: Optional[str] = Query(None, description="Cursor: id tie-breaker when created_at matches"),
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    mgr = ImageManager()
    images, has_more = await mgr.list_async(
        actor,
        limit=limit,
        enrichment_status=enrichment_status,
        after_created_at=after_created_at,
        after_id=after_id,
    )
    return ImageListResponse(images=images, has_more=has_more)


@router.post("/search", response_model=ImageSearchResponse)
async def search_images(
    body: ImageSearchRequest,
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    hits = await search_images_hybrid(body.query, actor, limit=body.limit)
    return ImageSearchResponse(
        results=[
            ImageSearchHit(
                handle=h.handle,
                description=h.description or h.snippet,
                score=h.score,
            )
            for h in hits
        ]
    )


@router.get("/{image_id}", response_model=PydanticImage)
async def get_image(
    image_id: str,
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    mgr = ImageManager()
    image = await mgr.get_by_id_async(image_id, actor)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return image


@router.patch("/{image_id}", response_model=PydanticImage)
async def update_image_metadata(
    image_id: str,
    body: ImageMetadataUpdate,
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    mgr = ImageManager()

    def _norm(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    updated = await mgr.update_metadata_async(
        image_id,
        actor,
        caption=_norm(body.caption),
        description=_norm(body.description),
        details=_norm(body.details),
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Image not found")
    return updated


@router.delete("/{image_id}")
async def delete_image(
    image_id: str,
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    mgr = ImageManager()
    await mgr.delete_async(image_id, actor)
    return {"ok": True}


@router.post("/{image_id}/re-enrich")
async def re_enrich_image(
    image_id: str,
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    mgr = ImageManager()
    image = await mgr.get_by_id_async(image_id, actor)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    await schedule_image_re_enrichment(image_id, actor)
    return {"ok": True, "enrichment_status": "pending"}


@router.get("/{image_id}/content")
async def get_image_content(
    image_id: str,
    variant: str = Query("full", pattern="^(full|1mp)$"),
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    mgr = ImageManager()
    image = await mgr.get_by_id_async(image_id, actor)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    store = get_object_store_client()
    if variant == "1mp" and image.object_url_1mp:
        data = await store.get_bytes(image.object_url_1mp)
        media_type = "image/jpeg"
    else:
        data = await store.get_bytes(image.object_url_full)
        media_type = image.media_type or "application/octet-stream"
    return Response(content=data, media_type=media_type)


@router.get("/{image_id}/url")
async def get_image_url(
    image_id: str,
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    mgr = ImageManager()
    image = await mgr.get_by_id_async(image_id, actor)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    store = get_object_store_client()
    url = await store.presigned_get_url(image.object_url_full)
    return {"url": url}
