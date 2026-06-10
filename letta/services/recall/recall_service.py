"""Unified hybrid recall over passages, messages, images, and file reading notes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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

RECALL_PER_SOURCE_DIVERSITY_CAP = 3
RECALL_SNIPPET_MAX_CHARS = 500


@dataclass
class RecallHit:
    layer: str
    snippet: str
    handle: str
    score: float
    reasons: List[str]
    filename: Optional[str] = None
    source_group: Optional[str] = None
    linked_image_ids: List[str] = field(default_factory=list)


def _truncate_snippet(text: str) -> str:
    return (text or "")[:RECALL_SNIPPET_MAX_CHARS]


def _message_source_group(row) -> Optional[str]:
    conversation_id = getattr(row, "conversation_id", None)
    if conversation_id:
        return f"conversation:{conversation_id}"
    agent_id = getattr(row, "agent_id", None)
    if agent_id:
        return f"agent:{agent_id}"
    return None


def _message_linked_image_ids(row) -> List[str]:
    from letta.embeddings.message_embed_text import collect_letta_image_ids_from_message

    return collect_letta_image_ids_from_message(row.to_pydantic())


def _dedup_image_message_hits(hits: List[RecallHit]) -> List[RecallHit]:
    """Drop message hits when the same image is already in the result set (FR §6)."""
    image_handles = {hit.handle for hit in hits if hit.layer == "image"}
    if not image_handles:
        return hits

    dropped_message_handles: set[str] = set()
    image_by_handle = {hit.handle: hit for hit in hits if hit.layer == "image"}

    for hit in hits:
        if hit.layer != "message":
            continue
        for image_id in hit.linked_image_ids:
            if image_id not in image_handles:
                continue
            image_hit = image_by_handle.get(image_id)
            if image_hit is None:
                continue
            if "message" not in image_hit.reasons:
                image_hit.reasons.append("message")
            dropped_message_handles.add(hit.handle)
            break

    if not dropped_message_handles:
        return hits
    return [hit for hit in hits if not (hit.layer == "message" and hit.handle in dropped_message_handles)]


def _apply_diversity_cap(
    hits: List[RecallHit],
    limit: int,
    *,
    per_source_cap: int = RECALL_PER_SOURCE_DIVERSITY_CAP,
) -> List[RecallHit]:
    """Limit hits per file/conversation before applying top-K (FR §6)."""
    selected: List[RecallHit] = []
    source_counts: dict[str, int] = {}

    for hit in sorted(hits, key=lambda h: -h.score):
        if len(selected) >= limit:
            break
        group = hit.source_group
        if group:
            count = source_counts.get(group, 0)
            if count >= per_source_cap:
                continue
            source_counts[group] = count + 1
        selected.append(hit)

    return selected


def _merge_recall_hit(existing: RecallHit, incoming: RecallHit, rrf: float) -> None:
    existing.score += rrf
    existing.reasons.extend(incoming.reasons)
    if incoming.source_group and not existing.source_group:
        existing.source_group = incoming.source_group
    if incoming.filename and not existing.filename:
        existing.filename = incoming.filename
    if incoming.linked_image_ids and not existing.linked_image_ids:
        existing.linked_image_ids = list(incoming.linked_image_ids)


def finalize_recall_hits(hits: List[RecallHit], limit: int) -> List[RecallHit]:
    """Post-fusion dedup and diversity cap."""
    return _apply_diversity_cap(_dedup_image_message_hits(hits), limit)


def format_recall_hit(hit: RecallHit) -> str:
    """Single recall hit line for tool output."""
    header = f"[{hit.layer}] handle={hit.handle}"
    if hit.filename:
        header += f" filename={hit.filename}"
    header += f" score={hit.score:.4f}"
    return f"{header}\n{hit.snippet}"


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
    if title and content:
        return f"[{title}] {content}"
    if title:
        return title
    return content


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


def _file_archive_lexical_sql(*, with_agent_filter: bool) -> str:
    agent_clause = "AND sa.agent_id = :agent_id" if with_agent_filter else ""
    return f"""
                SELECT fa.id,
                       ('[' || COALESCE(fa.title, '') || '] ' || COALESCE(fa.content, '')) AS snippet_text,
                       fm.file_name,
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
                        SELECT sp.id, sp.text, sp.file_name, similarity(sp.text, :q) AS sim
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
                        SELECT id, text, file_name, similarity(text, :q) AS sim
                          FROM source_passages
                         WHERE organization_id = :org
                           AND text IS NOT NULL
                           AND similarity(text, :q) > 0.1
                         ORDER BY sim DESC
                         LIMIT :lim
                        """


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
        select(FileArchiveModel, FileMetadataModel.file_name)
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
    rows = (await session.execute(q)).all()
    hits: List[RecallHit] = []
    for idx, (row, file_name) in enumerate(rows):
        hits.append(
            RecallHit(
                layer="file_archive",
                snippet=_truncate_snippet(_recall_snippet("file_archive", row)),
                handle=row.id,
                score=1.0 / (idx + 1),
                reasons=["vector"],
                filename=file_name or None,
                source_group=file_name or None,
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
            ("file", SourcePassage),
            ("message", MessageModel),
            ("image", ImageRecord),
        ):
            q = select(model).where(model.organization_id == actor.organization_id)
            if layer == "file" and agent_id:
                q = q.join(SourcesAgents, SourcesAgents.source_id == model.source_id).where(
                    SourcesAgents.agent_id == agent_id
                )
            elif agent_id and hasattr(model, "agent_id"):
                q = q.where(model.agent_id == agent_id)
            q = apply_embedding_space_guard(q, model, space_id)
            q = q.order_by(model.embedding.cosine_distance(query_vec).asc()).limit(limit)
            rows = (await session.execute(q)).scalars().all()
            for idx, row in enumerate(rows):
                snippet = _recall_snippet(layer, row)
                filename = getattr(row, "file_name", None) if layer == "file" else None
                source_group = None
                linked_image_ids: List[str] = []
                if layer == "file":
                    source_group = filename
                elif layer == "message":
                    source_group = _message_source_group(row)
                    linked_image_ids = _message_linked_image_ids(row)

                vector_hits.append(
                    RecallHit(
                        layer=layer,
                        snippet=_truncate_snippet(snippet),
                        handle=row.id,
                        score=1.0 / (idx + 1),
                        reasons=["vector"],
                        filename=filename,
                        source_group=source_group,
                        linked_image_ids=linked_image_ids,
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
            message_lexical = text(
                """
                SELECT id, text, conversation_id, agent_id, similarity(text, :q) AS sim
                  FROM messages
                 WHERE organization_id = :org
                   AND text IS NOT NULL
                   AND similarity(text, :q) > 0.1
                 ORDER BY sim DESC
                 LIMIT :lim
                """
            )
            message_rows = await session.execute(
                message_lexical, {"q": query, "org": actor.organization_id, "lim": limit}
            )
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
                    RecallHit(
                        layer="message",
                        snippet=_truncate_snippet(str(row[1] or "").strip()),
                        handle=row[0],
                        score=float(row[4]),
                        reasons=["lexical"],
                        source_group=source_group,
                    )
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
                            snippet=_truncate_snippet(snippet_text),
                            handle=row[0],
                            score=float(row[2]),
                            reasons=["lexical"],
                        )
                    )

            source_lexical = text(_source_passage_lexical_sql(with_agent_filter=bool(agent_id)))
            source_params = {"q": query, "org": actor.organization_id, "lim": limit}
            if agent_id:
                source_params["agent_id"] = agent_id
            source_rows = await session.execute(source_lexical, source_params)
            for row in source_rows:
                file_name = row[2] or None
                lexical_hits.append(
                    RecallHit(
                        layer="file",
                        snippet=_truncate_snippet(str(row[1] or "").strip()),
                        handle=row[0],
                        score=float(row[3]),
                        reasons=["lexical"],
                        filename=file_name,
                        source_group=file_name,
                    )
                )

            archive_lexical = text(_file_archive_lexical_sql(with_agent_filter=bool(agent_id)))
            archive_params = {"q": query, "org": actor.organization_id, "lim": limit}
            if agent_id:
                archive_params["agent_id"] = agent_id
            archive_rows = await session.execute(archive_lexical, archive_params)
            for row in archive_rows:
                archive_name = row[2] or None
                lexical_hits.append(
                    RecallHit(
                        layer="file_archive",
                        snippet=_truncate_snippet(str(row[1] or "").strip()),
                        handle=row[0],
                        score=float(row[3]),
                        reasons=["lexical"],
                        filename=archive_name,
                        source_group=archive_name,
                    )
                )

    fused: dict[str, RecallHit] = {}
    for rank, hit in enumerate(sorted(vector_hits, key=lambda h: -h.score)):
        key = f"{hit.layer}:{hit.handle}"
        rrf = 1.0 / (60 + rank + 1)
        if key in fused:
            _merge_recall_hit(fused[key], hit, rrf)
        else:
            hit.score = rrf
            fused[key] = hit
    for rank, hit in enumerate(sorted(lexical_hits, key=lambda h: -h.score)):
        key = f"{hit.layer}:{hit.handle}"
        rrf = 1.0 / (60 + rank + 1)
        if key in fused:
            _merge_recall_hit(fused[key], hit, rrf)
        else:
            hit.score = rrf
            fused[key] = hit

    return finalize_recall_hits(list(fused.values()), limit)
