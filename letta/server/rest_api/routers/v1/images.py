from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from letta.schemas.image import PydanticImage
from letta.schemas.user import User as PydanticUser
from letta.server.rest_api.dependencies import HeaderParams, get_headers, get_letta_server
from letta.server.server import SyncServer
from letta.services.image_ingest import enrich_image_background
from letta.services.image_manager import ImageManager
from letta.services.object_store.client import get_object_store_client

router = APIRouter(prefix="/images", tags=["images"])


@router.get("", response_model=List[PydanticImage])
async def list_images(
    limit: int = 50,
    enrichment_status: Optional[str] = None,
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    mgr = ImageManager()
    return await mgr.list_async(actor, limit=limit, enrichment_status=enrichment_status)


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
    await enrich_image_background(image_id, actor)
    return {"ok": True}


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
