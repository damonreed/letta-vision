"""Unified hybrid recall over passages, messages, images, and file reading notes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import select, text

from letta.embeddings.query import apply_embedding_space_guard, embed_search_query
from letta.embeddings.resolver import resolve_embedding_config_async
from letta.orm.file import FileMetadata as FileMetadataModel
from letta.orm.file_archive import FileArchive as FileArchiveModel
from letta.orm.image import ImageRecord
from letta.orm.message import Message as MessageModel
from letta.orm.passage import ArchivalPassage, SourcePassage
from letta.orm.sources_agents import SourcesAgents
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.settings import DatabaseChoice, settings
from letta.services.message_manager import MessageManager

_message_manager = MessageManager()


@dataclass
class RecallHit:
    layer: str
    snippet: str
    handle: str
    score: float
    reasons: List[str]


def _snippet_for_display(text: str) -> str:
    """Flatten MessageManager JSON extracts into human-readable recall snippets."""
    if not text:
        return ""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return stripped
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if isinstance(parsed, dict):
        for key in ("content", "text", "thinking"):
            value = parsed.get(key)
            if value:
                return str(value)
    return stripped


def _image_recall_snippet(row) -> str:
    for field in ("description", "caption", "details", "generation_prompt"):
        value = getattr(row, field, None)
        if value and str(value).strip():
            return str(value).strip()

    image_id = getattr(row, "id", "image")
    provenance = getattr(row, "provenance", "image") or "image"
    media_type = getattr(row, "media_type", "image") or "image"
    status = getattr(row, "enrichment_status", "unknown") or "unknown"
    return (
        f"{provenance.capitalize()} {media_type} (status: {status}). "
        f"Use fetch_image({image_id}) to view pixels."
    )


def _file_archive_recall_snippet(row) -> str:
    title = (getattr(row, "title", None) or "").strip()
    content = (getattr(row, "content", None) or "").strip()
    file_name = getattr(row, "file_name", None)
    prefix = f"{file_name}: " if file_name else ""
    if title and content:
        return f"{prefix}[{title}] {content}"
    if title:
        return f"{prefix}{title}"
    return f"{prefix}{content}"


def _message_image_fallback_snippet(row) -> str:
    msg = row.to_pydantic()
    content = msg.content
    if not isinstance(content, list):
        return ""

    handles: List[str] = []
    for item in content:
        item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
        if item_type != "image":
            continue
        source = getattr(item, "source", None) or (item.get("source") if isinstance(item, dict) else None)
        if not source:
            continue
        file_id = getattr(source, "file_id", None) or (source.get("file_id") if isinstance(source, dict) else None)
        if file_id:
            handles.append(file_id)

    if not handles:
        return ""
    joined = ", ".join(handles)
    if len(handles) == 1:
        return f"Message with image {joined}. Use fetch_image({handles[0]}) to view pixels."
    return f"Message with images: {joined}. Use fetch_image on a handle to view pixels."


def _recall_snippet(layer: str, row) -> str:
    if layer == "message":
        extracted = _message_manager._extract_message_text(row.to_pydantic())
        snippet = _snippet_for_display(extracted)
        if snippet.strip():
            return snippet
        return _message_image_fallback_snippet(row)

    if layer == "image":
        return _image_recall_snippet(row)

    if layer == "file_archive":
        return _file_archive_recall_snippet(row)

    snippet = getattr(row, "description", None) or getattr(row, "text", None) or getattr(row, "caption", "") or ""
    return str(snippet)


async def _vector_hits_for_file_archives(
    session,
    *,
    actor: PydanticUser,
    agent_id: Optional[str],
    query_vec,
    space_id: str,
    limit: int,
) -> List[RecallHit]:
    q = (
        select(FileArchiveModel)
        .join(FileMetadataModel, FileArchiveModel.file_id == FileMetadataModel.id)
        .join(SourcesAgents, SourcesAgents.source_id == FileMetadataModel.source_id)
        .where(
            FileArchiveModel.organization_id == actor.organization_id,
            FileArchiveModel.is_deleted == False,
        )
    )
    if agent_id:
        q = q.where(SourcesAgents.agent_id == agent_id)
    q = apply_embedding_space_guard(q, FileArchiveModel, space_id)
    q = q.order_by(FileArchiveModel.embedding.cosine_distance(query_vec).asc()).limit(limit)
    rows = (await session.execute(q)).scalars().all()
    hits: List[RecallHit] = []
    for idx, row in enumerate(rows):
        hits.append(
            RecallHit(
                layer="file_archive",
                snippet=_recall_snippet("file_archive", row)[:500],
                handle=row.id,
                score=1.0 / (idx + 1),
                reasons=["vector"],
            )
        )
    return hits


async def recall(
    query: str,
    actor: PydanticUser,
    *,
    limit: int = 10,
    agent_id: Optional[str] = None,
) -> List[RecallHit]:
    if settings.database_engine != DatabaseChoice.POSTGRES:
        return []

    config = await resolve_embedding_config_async(actor=actor)
    query_vec, space_id = await embed_search_query(query, config, actor=actor)

    vector_hits: List[RecallHit] = []
    async with db_registry.async_session() as session:
        for layer, model in (
            ("archival", ArchivalPassage),
            ("source", SourcePassage),
            ("message", MessageModel),
            ("image", ImageRecord),
        ):
            q = select(model).where(model.organization_id == actor.organization_id)
            if agent_id and hasattr(model, "agent_id"):
                q = q.where(model.agent_id == agent_id)
            q = apply_embedding_space_guard(q, model, space_id)
            q = q.order_by(model.embedding.cosine_distance(query_vec).asc()).limit(limit)
            rows = (await session.execute(q)).scalars().all()
            for idx, row in enumerate(rows):
                snippet = _recall_snippet(layer, row)
                vector_hits.append(
                    RecallHit(
                        layer=layer,
                        snippet=snippet[:500],
                        handle=row.id,
                        score=1.0 / (idx + 1),
                        reasons=["vector"],
                    )
                )

        vector_hits.extend(
            await _vector_hits_for_file_archives(
                session,
                actor=actor,
                agent_id=agent_id,
                query_vec=query_vec,
                space_id=space_id,
                limit=limit,
            )
        )

        lexical_hits: List[RecallHit] = []
        if query:
            lexical_specs = (
                ("archival", "archival_passages", "text", "text"),
                ("source", "source_passages", "text", "text"),
                ("message", "messages", "text", "text"),
                (
                    "image",
                    "images",
                    "COALESCE(NULLIF(description, ''), NULLIF(caption, ''), NULLIF(details, ''), generation_prompt, '')",
                    "GREATEST("
                    "COALESCE(similarity(description, :q), 0), "
                    "COALESCE(similarity(caption, :q), 0), "
                    "COALESCE(similarity(details, :q), 0)"
                    ")",
                ),
            )
            for layer, table, snippet_expr, sim_expr in lexical_specs:
                if layer == "image":
                    stmt = text(
                        f"""
                        SELECT id, {snippet_expr} AS snippet_text, {sim_expr} AS sim
                          FROM {table}
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
                else:
                    col = snippet_expr
                    stmt = text(
                        f"""
                        SELECT id, {col}, similarity({col}, :q) AS sim
                          FROM {table}
                         WHERE organization_id = :org
                           AND {col} IS NOT NULL
                           AND similarity({col}, :q) > 0.1
                         ORDER BY sim DESC
                         LIMIT :lim
                        """
                    )
                result = await session.execute(stmt, {"q": query, "org": actor.organization_id, "lim": limit})
                for row in result:
                    snippet_text = str(row[1] or "").strip()
                    if layer == "image" and not snippet_text:
                        snippet_text = f"Image {row[0]}. Use fetch_image({row[0]}) to view pixels."
                    lexical_hits.append(
                        RecallHit(
                            layer=layer,
                            snippet=snippet_text[:500],
                            handle=row[0],
                            score=float(row[2]),
                            reasons=["lexical"],
                        )
                    )

            archive_lexical = text(
                """
                SELECT fa.id,
                       (COALESCE(fm.file_name || ': ', '') || '[' || fa.title || '] ' || fa.content) AS snippet_text,
                       GREATEST(
                           COALESCE(similarity(fa.title, :q), 0),
                           COALESCE(similarity(fa.content, :q), 0)
                       ) AS sim
                  FROM file_archives fa
                  JOIN files fm ON fa.file_id = fm.id
                  JOIN sources_agents sa ON sa.source_id = fm.source_id
                 WHERE fa.organization_id = :org
                   AND fa.is_deleted = FALSE
                   AND (:agent_id IS NULL OR sa.agent_id = :agent_id)
                   AND (
                     (fa.title IS NOT NULL AND similarity(fa.title, :q) > 0.1)
                     OR (fa.content IS NOT NULL AND similarity(fa.content, :q) > 0.1)
                   )
                 ORDER BY sim DESC
                 LIMIT :lim
                """
            )
            archive_rows = await session.execute(
                archive_lexical,
                {"q": query, "org": actor.organization_id, "agent_id": agent_id, "lim": limit},
            )
            for row in archive_rows:
                lexical_hits.append(
                    RecallHit(
                        layer="file_archive",
                        snippet=str(row[1] or "").strip()[:500],
                        handle=row[0],
                        score=float(row[2]),
                        reasons=["lexical"],
                    )
                )

    fused: dict[str, RecallHit] = {}
    for rank, hit in enumerate(sorted(vector_hits, key=lambda h: -h.score)):
        key = f"{hit.layer}:{hit.handle}"
        rrf = 1.0 / (60 + rank + 1)
        if key in fused:
            fused[key].score += rrf
            fused[key].reasons.extend(hit.reasons)
        else:
            hit.score = rrf
            fused[key] = hit
    for rank, hit in enumerate(sorted(lexical_hits, key=lambda h: -h.score)):
        key = f"{hit.layer}:{hit.handle}"
        rrf = 1.0 / (60 + rank + 1)
        if key in fused:
            fused[key].score += rrf
            fused[key].reasons.extend(hit.reasons)
        else:
            hit.score = rrf
            fused[key] = hit

    return sorted(fused.values(), key=lambda h: -h.score)[:limit]
