"""Per-layer vector + lexical hybrid search with RRF fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence, Set

from sqlalchemy import or_, select, text
from sqlalchemy.orm import noload

from letta.embeddings.query import apply_embedding_space_guard, embed_search_query
from letta.embeddings.resolver import resolve_embedding_config_async
from letta.orm.archives_agents import ArchivesAgents
from letta.orm.file import FileMetadata as FileMetadataModel
from letta.orm.file_archive import FileArchive as FileArchiveModel
from letta.orm.image import ImageRecord
from letta.orm.message import Message as MessageModel
from letta.orm.passage import ArchivalPassage, SourcePassage
from letta.orm.sources_agents import SourcesAgents
from letta.schemas.enums import TagMatchMode
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.settings import DatabaseChoice, settings
from letta.services.recall.recall_service import (
    RecallHit,
    _image_recall_snippet,
    _message_linked_image_ids,
    _message_source_group,
    _recall_snippet,
    _truncate_snippet,
    finalize_recall_hits,
    format_recall_hit,
)

RRF_K = 60
FILE_CONTENTS_SEARCH_MAX_CHARS = 1000


@dataclass
class HybridHit:
    layer: str
    handle: str
    score: float
    reasons: List[str] = field(default_factory=list)
    snippet: str = ""
    filename: Optional[str] = None
    file_id: Optional[str] = None
    source_group: Optional[str] = None
    linked_image_ids: List[str] = field(default_factory=list)
    full_text: Optional[str] = None
    tags: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    title: Optional[str] = None
    description: Optional[str] = None

    def to_recall_hit(self, *, truncate: bool = True) -> RecallHit:
        text_out = self.snippet or self.full_text or ""
        if truncate:
            text_out = _truncate_snippet(text_out)
        return RecallHit(
            layer=self.layer,
            snippet=text_out,
            handle=self.handle,
            score=self.score,
            reasons=list(self.reasons),
            filename=self.filename,
            source_group=self.source_group,
            linked_image_ids=list(self.linked_image_ids),
        )


def truncate_at_word_boundary(text: str, max_chars: int) -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > max_chars // 2:
        return cut[:last_space].rstrip() + "…"
    return cut.rstrip() + "…"


def _merge_hybrid_hit(existing: HybridHit, incoming: HybridHit, rrf: float) -> None:
    existing.score += rrf
    existing.reasons.extend(incoming.reasons)
    if incoming.source_group and not existing.source_group:
        existing.source_group = incoming.source_group
    if incoming.filename and not existing.filename:
        existing.filename = incoming.filename
    if incoming.file_id and not existing.file_id:
        existing.file_id = incoming.file_id
    if incoming.full_text and not existing.full_text:
        existing.full_text = incoming.full_text
    if incoming.linked_image_ids and not existing.linked_image_ids:
        existing.linked_image_ids = list(incoming.linked_image_ids)


def fuse_rrf(
    vector_hits: Sequence[HybridHit],
    lexical_hits: Sequence[HybridHit],
    *,
    limit: int,
    key_fn=None,
) -> List[HybridHit]:
    if key_fn is None:
        key_fn = lambda h: f"{h.layer}:{h.handle}"

    fused: dict[str, HybridHit] = {}
    for rank, hit in enumerate(sorted(vector_hits, key=lambda h: -h.score)):
        key = key_fn(hit)
        rrf = 1.0 / (RRF_K + rank + 1)
        if key in fused:
            _merge_hybrid_hit(fused[key], hit, rrf)
        else:
            hit.score = rrf
            fused[key] = hit
    for rank, hit in enumerate(sorted(lexical_hits, key=lambda h: -h.score)):
        key = key_fn(hit)
        rrf = 1.0 / (RRF_K + rank + 1)
        if key in fused:
            _merge_hybrid_hit(fused[key], hit, rrf)
        else:
            hit.score = rrf
            fused[key] = hit

    return sorted(fused.values(), key=lambda h: -h.score)[:limit]


def _filter_archival_by_tags(
    hits: List[HybridHit],
    tags: Optional[List[str]],
    tag_match_mode: TagMatchMode,
) -> List[HybridHit]:
    if not tags:
        return hits
    query_tags = set(tags)
    filtered: List[HybridHit] = []
    for hit in hits:
        passage_tags = set(hit.tags or [])
        if tag_match_mode == TagMatchMode.ALL:
            if query_tags.issubset(passage_tags):
                filtered.append(hit)
        elif query_tags.intersection(passage_tags):
            filtered.append(hit)
    return filtered


def _archival_lexical_sql(*, with_agent_filter: bool) -> str:
    agent_clause = "AND aa.agent_id = :agent_id" if with_agent_filter else ""
    join_clause = "JOIN archives_agents aa ON ap.archive_id = aa.archive_id" if with_agent_filter else ""
    return f"""
        SELECT ap.id, ap.text, similarity(ap.text, :q) AS sim
          FROM archival_passages ap
          {join_clause}
         WHERE ap.organization_id = :org
           {agent_clause}
           AND ap.text IS NOT NULL
           AND similarity(ap.text, :q) > 0.1
         ORDER BY sim DESC
         LIMIT :lim
    """


def _file_archive_lexical_sql(*, with_agent_filter: bool) -> str:
    agent_clause = "AND sa.agent_id = :agent_id" if with_agent_filter else ""
    return f"""
        SELECT fa.id,
               ('[' || COALESCE(fa.title, '') || '] ' || COALESCE(fa.content, '')) AS snippet_text,
               fm.file_name,
               fm.id AS file_id,
               GREATEST(
                   COALESCE(similarity(fa.title, :q), 0),
                   COALESCE(similarity(fa.content, :q), 0)
               ) AS sim
          FROM file_archives fa
          JOIN files fm ON fa.file_id = fm.id
          JOIN sources_agents sa ON sa.source_id = fm.source_id
         WHERE fa.organization_id = :org
           AND fa.is_deleted = FALSE
           {agent_clause}
           AND (
             (fa.title IS NOT NULL AND similarity(fa.title, :q) > 0.1)
             OR (fa.content IS NOT NULL AND similarity(fa.content, :q) > 0.1)
           )
         ORDER BY sim DESC
         LIMIT :lim
    """


def _source_passage_lexical_sql(*, with_agent_filter: bool) -> str:
    if with_agent_filter:
        return """
            SELECT sp.id, sp.text, sp.file_name, sp.file_id, similarity(sp.text, :q) AS sim
              FROM source_passages sp
              JOIN sources_agents sa ON sa.source_id = sp.source_id
             WHERE sp.organization_id = :org
               AND sa.agent_id = :agent_id
               AND sp.text IS NOT NULL
               AND similarity(sp.text, :q) > 0.1
             ORDER BY sim DESC
             LIMIT :lim
        """
    return """
        SELECT id, text, file_name, file_id, similarity(text, :q) AS sim
          FROM source_passages
         WHERE organization_id = :org
           AND text IS NOT NULL
           AND similarity(text, :q) > 0.1
         ORDER BY sim DESC
         LIMIT :lim
    """


async def search_archival_hybrid(
    query: str,
    actor: PydanticUser,
    *,
    limit: int = 10,
    agent_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    tag_match_mode: TagMatchMode = TagMatchMode.ANY,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[HybridHit]:
    if settings.database_engine != DatabaseChoice.POSTGRES or not query or not query.strip():
        return []

    config = await resolve_embedding_config_async(actor=actor)
    query_vec, space_id = await embed_search_query(query, config, actor=actor)

    vector_hits: List[HybridHit] = []
    lexical_hits: List[HybridHit] = []

    async with db_registry.async_session() as session:
        q = (
            select(ArchivalPassage)
            .options(noload(ArchivalPassage.organization), noload(ArchivalPassage.passage_tags))
            .where(ArchivalPassage.organization_id == actor.organization_id)
        )
        if agent_id:
            q = q.join(ArchivesAgents, ArchivalPassage.archive_id == ArchivesAgents.archive_id).where(
                ArchivesAgents.agent_id == agent_id
            )
        if start_date:
            q = q.where(ArchivalPassage.created_at >= start_date)
        if end_date:
            q = q.where(ArchivalPassage.created_at <= end_date)
        q = apply_embedding_space_guard(q, ArchivalPassage, space_id)
        q = q.order_by(ArchivalPassage.embedding.cosine_distance(query_vec).asc()).limit(limit)
        rows = (await session.execute(q)).scalars().all()
        for idx, row in enumerate(rows):
            vector_hits.append(
                HybridHit(
                    layer="archival",
                    handle=row.id,
                    score=1.0 / (idx + 1),
                    reasons=["vector"],
                    snippet=row.text,
                    full_text=row.text,
                    tags=row.tags,
                    created_at=row.created_at,
                )
            )

        stmt = text(_archival_lexical_sql(with_agent_filter=bool(agent_id)))
        params: dict = {"q": query, "org": actor.organization_id, "lim": limit}
        if agent_id:
            params["agent_id"] = agent_id
        result = await session.execute(stmt, params)
        for row in result:
            lexical_hits.append(
                HybridHit(
                    layer="archival",
                    handle=row[0],
                    score=float(row[2]),
                    reasons=["lexical"],
                    snippet=str(row[1] or "").strip(),
                    full_text=str(row[1] or "").strip(),
                )
            )

        if tags or start_date or end_date:
            ids = {h.handle for h in vector_hits + lexical_hits}
            if ids:
                detail_q = select(ArchivalPassage).where(
                    ArchivalPassage.id.in_(ids),
                    ArchivalPassage.organization_id == actor.organization_id,
                )
                detail_rows = (await session.execute(detail_q)).scalars().all()
                by_id = {r.id: r for r in detail_rows}
                for hit in vector_hits + lexical_hits:
                    row = by_id.get(hit.handle)
                    if row:
                        hit.tags = row.tags
                        hit.created_at = row.created_at
                        hit.full_text = row.text
                        hit.snippet = row.text

    fused = fuse_rrf(vector_hits, lexical_hits, limit=limit)
    fused = _filter_archival_by_tags(fused, tags, tag_match_mode)
    if start_date or end_date:
        filtered: List[HybridHit] = []
        for hit in fused:
            ts = hit.created_at
            if start_date and ts and ts < start_date:
                continue
            if end_date and ts and ts > end_date:
                continue
            filtered.append(hit)
        fused = filtered[:limit]
    return fused[:limit]


async def search_file_archives_hybrid(
    query: str,
    actor: PydanticUser,
    *,
    limit: int = 10,
    agent_id: Optional[str] = None,
    file_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> List[HybridHit]:
    if settings.database_engine != DatabaseChoice.POSTGRES or not query or not query.strip():
        return []

    from letta.services.files.archive_tags import normalize_archive_tags

    config = await resolve_embedding_config_async(actor=actor)
    query_vec, space_id = await embed_search_query(query, config, actor=actor)

    vector_hits: List[HybridHit] = []
    lexical_hits: List[HybridHit] = []

    async with db_registry.async_session() as session:
        base = (
            select(FileArchiveModel, FileMetadataModel.file_name, FileMetadataModel.id)
            .join(FileMetadataModel, FileArchiveModel.file_id == FileMetadataModel.id)
            .join(SourcesAgents, SourcesAgents.source_id == FileMetadataModel.source_id)
            .where(
                FileArchiveModel.organization_id == actor.organization_id,
                FileArchiveModel.is_deleted == False,
            )
        )
        if agent_id:
            base = base.where(SourcesAgents.agent_id == agent_id)
        if file_id:
            base = base.where(FileArchiveModel.file_id == file_id)
        if tags:
            normalized = normalize_archive_tags(tags)
            if normalized:
                tag_filters = [FileArchiveModel.tags.contains([t]) for t in normalized]
                base = base.where(or_(*tag_filters))

        base = apply_embedding_space_guard(base, FileArchiveModel, space_id)
        base = base.order_by(FileArchiveModel.embedding.cosine_distance(query_vec).asc()).limit(limit)
        rows = (await session.execute(base)).all()
        for idx, (row, file_name, fid) in enumerate(rows):
            snippet = _recall_snippet("file_archive", row)
            vector_hits.append(
                HybridHit(
                    layer="file_archive",
                    handle=row.id,
                    score=1.0 / (idx + 1),
                    reasons=["vector"],
                    snippet=snippet,
                    full_text=row.content,
                    title=row.title,
                    filename=file_name,
                    file_id=fid,
                    tags=row.tags,
                    created_at=row.created_at,
                    source_group=file_name,
                )
            )

        stmt = text(_file_archive_lexical_sql(with_agent_filter=bool(agent_id)))
        params: dict = {"q": query, "org": actor.organization_id, "lim": limit}
        if agent_id:
            params["agent_id"] = agent_id
        result = await session.execute(stmt, params)
        for row in result:
            lexical_hits.append(
                HybridHit(
                    layer="file_archive",
                    handle=row[0],
                    score=float(row[4]),
                    reasons=["lexical"],
                    snippet=str(row[1] or "").strip(),
                    filename=row[2],
                    file_id=row[3],
                    source_group=row[2],
                )
            )

        if lexical_hits:
            ids = {h.handle for h in lexical_hits}
            detail_q = (
                select(FileArchiveModel, FileMetadataModel.file_name, FileMetadataModel.id)
                .join(FileMetadataModel, FileArchiveModel.file_id == FileMetadataModel.id)
                .where(FileArchiveModel.id.in_(ids))
            )
            detail_rows = (await session.execute(detail_q)).all()
            by_id = {r[0].id: (r[0], r[1], r[2]) for r in detail_rows}
            for hit in lexical_hits:
                tup = by_id.get(hit.handle)
                if tup:
                    archive_row, fname, fid = tup
                    hit.full_text = archive_row.content
                    hit.title = archive_row.title
                    hit.tags = archive_row.tags
                    hit.created_at = archive_row.created_at
                    hit.filename = fname
                    hit.file_id = fid

    return fuse_rrf(vector_hits, lexical_hits, limit=limit)


async def search_source_passages_hybrid(
    query: str,
    actor: PydanticUser,
    *,
    limit: int = 10,
    agent_id: Optional[str] = None,
) -> List[HybridHit]:
    if settings.database_engine != DatabaseChoice.POSTGRES or not query or not query.strip():
        return []

    config = await resolve_embedding_config_async(actor=actor)
    query_vec, space_id = await embed_search_query(query, config, actor=actor)

    vector_hits: List[HybridHit] = []
    lexical_hits: List[HybridHit] = []

    async with db_registry.async_session() as session:
        q = select(SourcePassage).where(SourcePassage.organization_id == actor.organization_id)
        if agent_id:
            q = q.join(SourcesAgents, SourcesAgents.source_id == SourcePassage.source_id).where(
                SourcesAgents.agent_id == agent_id
            )
        q = apply_embedding_space_guard(q, SourcePassage, space_id)
        q = q.order_by(SourcePassage.embedding.cosine_distance(query_vec).asc()).limit(limit)
        rows = (await session.execute(q)).scalars().all()
        for idx, row in enumerate(rows):
            vector_hits.append(
                HybridHit(
                    layer="file",
                    handle=row.id,
                    score=1.0 / (idx + 1),
                    reasons=["vector"],
                    snippet=row.text,
                    full_text=row.text,
                    filename=row.file_name,
                    file_id=row.file_id,
                    source_group=row.file_name,
                )
            )

        stmt = text(_source_passage_lexical_sql(with_agent_filter=bool(agent_id)))
        params: dict = {"q": query, "org": actor.organization_id, "lim": limit}
        if agent_id:
            params["agent_id"] = agent_id
        result = await session.execute(stmt, params)
        for row in result:
            lexical_hits.append(
                HybridHit(
                    layer="file",
                    handle=row[0],
                    score=float(row[4]),
                    reasons=["lexical"],
                    snippet=str(row[1] or "").strip(),
                    full_text=str(row[1] or "").strip(),
                    filename=row[2],
                    file_id=row[3],
                    source_group=row[2],
                )
            )

    return fuse_rrf(vector_hits, lexical_hits, limit=limit)


async def _agent_image_ids(session, agent_id: str, org_id: str) -> Set[str]:
    from letta.embeddings.message_embed_text import collect_letta_image_ids_from_message

    q = select(MessageModel).where(
        MessageModel.organization_id == org_id,
        MessageModel.agent_id == agent_id,
    )
    rows = (await session.execute(q)).scalars().all()
    ids: Set[str] = set()
    for row in rows:
        ids.update(collect_letta_image_ids_from_message(row.to_pydantic()))
    return ids


async def search_images_hybrid(
    query: str,
    actor: PydanticUser,
    *,
    limit: int = 10,
    agent_id: Optional[str] = None,
) -> List[HybridHit]:
    if settings.database_engine != DatabaseChoice.POSTGRES or not query or not query.strip():
        return []

    config = await resolve_embedding_config_async(actor=actor)
    query_vec, space_id = await embed_search_query(query, config, actor=actor)

    vector_hits: List[HybridHit] = []
    lexical_hits: List[HybridHit] = []
    allowed_ids: Optional[Set[str]] = None
    fetch_lim = limit

    async with db_registry.async_session() as session:
        if agent_id:
            allowed_ids = await _agent_image_ids(session, agent_id, actor.organization_id)
            fetch_lim = limit * 5

        q = select(ImageRecord).where(ImageRecord.organization_id == actor.organization_id)
        q = apply_embedding_space_guard(q, ImageRecord, space_id)
        q = q.order_by(ImageRecord.embedding.cosine_distance(query_vec).asc()).limit(fetch_lim)
        rows = (await session.execute(q)).scalars().all()
        idx = 0
        for row in rows:
            if allowed_ids is not None and row.id not in allowed_ids:
                continue
            desc = _image_recall_snippet(row)
            vector_hits.append(
                HybridHit(
                    layer="image",
                    handle=row.id,
                    score=1.0 / (idx + 1),
                    reasons=["vector"],
                    snippet=desc,
                    description=desc,
                )
            )
            idx += 1
            if len(vector_hits) >= limit:
                break

        snippet_expr = (
            "COALESCE(NULLIF(description, ''), NULLIF(caption, ''), NULLIF(details, ''), generation_prompt, '')"
        )
        sim_expr = (
            "GREATEST("
            "COALESCE(similarity(description, :q), 0), "
            "COALESCE(similarity(caption, :q), 0), "
            "COALESCE(similarity(details, :q), 0)"
            ")"
        )
        stmt = text(
            f"""
            SELECT id, {snippet_expr} AS snippet_text, {sim_expr} AS sim
              FROM images
             WHERE organization_id = :org
               AND (
                 (description IS NOT NULL AND similarity(description, :q) > 0.1)
                 OR (caption IS NOT NULL AND similarity(caption, :q) > 0.1)
                 OR (details IS NOT NULL AND similarity(details, :q) > 0.1)
               )
             ORDER BY sim DESC
             LIMIT :lim
            """
        )
        result = await session.execute(stmt, {"q": query, "org": actor.organization_id, "lim": fetch_lim})
        for row in result:
            if allowed_ids is not None and row[0] not in allowed_ids:
                continue
            snippet_text = str(row[1] or "").strip()
            if not snippet_text:
                snippet_text = f"Image {row[0]}. Use image_fetch({row[0]}) to view pixels."
            lexical_hits.append(
                HybridHit(
                    layer="image",
                    handle=row[0],
                    score=float(row[2]),
                    reasons=["lexical"],
                    snippet=snippet_text,
                    description=snippet_text,
                )
            )
            if len(lexical_hits) >= limit:
                break

    return fuse_rrf(vector_hits, lexical_hits, limit=limit)


async def search_messages_hybrid(
    query: str,
    actor: PydanticUser,
    *,
    limit: int = 10,
    agent_id: Optional[str] = None,
) -> List[HybridHit]:
    if settings.database_engine != DatabaseChoice.POSTGRES or not query or not query.strip():
        return []

    config = await resolve_embedding_config_async(actor=actor)
    query_vec, space_id = await embed_search_query(query, config, actor=actor)

    vector_hits: List[HybridHit] = []
    lexical_hits: List[HybridHit] = []

    async with db_registry.async_session() as session:
        q = select(MessageModel).where(MessageModel.organization_id == actor.organization_id)
        if agent_id:
            q = q.where(MessageModel.agent_id == agent_id)
        q = apply_embedding_space_guard(q, MessageModel, space_id)
        q = q.order_by(MessageModel.embedding.cosine_distance(query_vec).asc()).limit(limit)
        rows = (await session.execute(q)).scalars().all()
        for idx, row in enumerate(rows):
            snippet = _recall_snippet("message", row)
            vector_hits.append(
                HybridHit(
                    layer="message",
                    handle=row.id,
                    score=1.0 / (idx + 1),
                    reasons=["vector"],
                    snippet=snippet,
                    full_text=snippet,
                    source_group=_message_source_group(row),
                    linked_image_ids=_message_linked_image_ids(row),
                )
            )

        agent_clause = "AND agent_id = :agent_id" if agent_id else ""
        message_lexical = text(
            f"""
            SELECT id, text, conversation_id, agent_id, similarity(text, :q) AS sim
              FROM messages
             WHERE organization_id = :org
               {agent_clause}
               AND text IS NOT NULL
               AND similarity(text, :q) > 0.1
             ORDER BY sim DESC
             LIMIT :lim
            """
        )
        params: dict = {"q": query, "org": actor.organization_id, "lim": limit}
        if agent_id:
            params["agent_id"] = agent_id
        message_rows = await session.execute(message_lexical, params)
        for row in message_rows:
            conversation_id = row[2]
            agent_id_val = row[3]
            if conversation_id:
                source_group = f"conversation:{conversation_id}"
            elif agent_id_val:
                source_group = f"agent:{agent_id_val}"
            else:
                source_group = None
            lexical_hits.append(
                HybridHit(
                    layer="message",
                    handle=row[0],
                    score=float(row[4]),
                    reasons=["lexical"],
                    snippet=str(row[1] or "").strip(),
                    full_text=str(row[1] or "").strip(),
                    source_group=source_group,
                )
            )

    return fuse_rrf(vector_hits, lexical_hits, limit=limit)


async def search_all_hybrid(
    query: str,
    actor: PydanticUser,
    *,
    limit: int = 10,
    agent_id: Optional[str] = None,
) -> List[RecallHit]:
    """Cross-layer fused search for search_all tool."""
    if settings.database_engine != DatabaseChoice.POSTGRES:
        return []

    archival = await search_archival_hybrid(query, actor, limit=limit, agent_id=agent_id)
    files = await search_source_passages_hybrid(query, actor, limit=limit, agent_id=agent_id)
    archives = await search_file_archives_hybrid(query, actor, limit=limit, agent_id=agent_id)
    images = await search_images_hybrid(query, actor, limit=limit, agent_id=agent_id)
    messages = await search_messages_hybrid(query, actor, limit=limit, agent_id=agent_id)

    all_hybrid = archival + files + archives + images + messages
    merged: dict[str, HybridHit] = {}
    for hit in sorted(all_hybrid, key=lambda h: -h.score):
        key = f"{hit.layer}:{hit.handle}"
        if key in merged:
            merged[key].score += hit.score
            merged[key].reasons.extend(hit.reasons)
        else:
            merged[key] = hit

    recall_hits = [h.to_recall_hit(truncate=True) for h in merged.values()]
    return finalize_recall_hits(recall_hits, limit)


def format_search_all_hit(hit: RecallHit) -> str:
    return format_recall_hit(hit)
