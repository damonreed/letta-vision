from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from letta.schemas.file_archive import FileArchive as PydanticFileArchive
from letta.schemas.file_core_block import FileCoreBlock as PydanticFileCoreBlock, FileCoreBlockUpdate
from letta.schemas.user import User
from letta.server.rest_api.dependencies import HeaderParams, get_headers, get_letta_server
from letta.server.server import SyncServer
from letta.services.agent_open_files_manager import AgentOpenFilesManager
from letta.services.file_archive_manager import FileArchiveManager
from letta.services.file_core_block_manager import FileCoreBlockManager
from letta.validators import AgentId, FileId

router = APIRouter(prefix="/file-memory", tags=["file-memory"])


class FileArchiveSearchRequest(BaseModel):
    query: str
    agent_id: str
    file_id: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = Field(default=10, ge=1, le=50)


@router.get("/files/{file_id}/core", response_model=PydanticFileCoreBlock)
async def get_file_core(
    file_id: FileId,
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    core = await FileCoreBlockManager().get(file_id=file_id, actor=actor)
    if core is None:
        core = await FileCoreBlockManager().get_or_create(
            file_id=file_id, organization_id=actor.organization_id, actor=actor
        )
    return core


@router.patch("/files/{file_id}/core", response_model=PydanticFileCoreBlock)
async def patch_file_core(
    file_id: FileId,
    body: FileCoreBlockUpdate,
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    try:
        return await FileCoreBlockManager().update(
            file_id=file_id, new_summary=body.summary, agent_id="operator", actor=actor
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/agents/{agent_id}/open-files")
async def list_open_files(
    agent_id: AgentId,
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    return await AgentOpenFilesManager().list_open_files_with_cores(agent_id=agent_id, actor=actor)


@router.get("/files/{file_id}/archives", response_model=List[PydanticFileArchive])
async def list_file_archives(
    file_id: FileId,
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    return await FileArchiveManager().list_for_file(file_id=file_id, actor=actor)


@router.post("/archives/search")
async def search_file_archives(
    body: FileArchiveSearchRequest,
    server: SyncServer = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    agent = await server.agent_manager.get_agent_by_id_async(agent_id=body.agent_id, actor=actor)
    results = await FileArchiveManager().search_archives(
        query=body.query,
        agent_id=body.agent_id,
        embedding_config=agent.embedding_config,
        actor=actor,
        file_id=body.file_id,
        tags=body.tags,
        limit=body.limit,
    )
    return {"results": results}
