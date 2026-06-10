from typing import Literal

from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel, ConfigDict, Field

from letta.errors import LettaInvalidArgumentError
from letta.server.rest_api.dependencies import HeaderParams, get_headers, get_letta_server
from letta.server.server import SyncServer

router = APIRouter(prefix="/_internal_search", tags=["_internal_search"])


class MessageSearchCacheWarmScope(BaseModel):
    """Messages currently infer scope from the authenticated actor."""

    model_config = ConfigDict(extra="forbid")


class SearchCacheWarmRequest(BaseModel):
    """Request for warming an internal search cache."""

    model_config = ConfigDict(extra="forbid")

    collection: Literal["messages"] = Field(description="Embedded collection whose cache should be warmed.")
    scope: MessageSearchCacheWarmScope = Field(
        description="Collection-specific scope. Messages currently infer organization from the authenticated actor.",
    )


class SearchCacheWarmResponse(BaseModel):
    """Response for internal search cache warming."""

    collection: Literal["messages"]
    status: str
    warmed: bool


@router.post("/cache-warm", response_model=SearchCacheWarmResponse, status_code=202, operation_id="warm_internal_search_cache")
async def warm_search_cache(
    request: SearchCacheWarmRequest = Body(...),
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """Warm the cache for a supported internal search collection."""
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    _ = actor  # reserved for future collection-specific cache warming
    raise LettaInvalidArgumentError(
        "Internal search cache warming is not supported.",
        argument_name="collection",
    )
