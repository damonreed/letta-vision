"""Unified hybrid recall over passages, messages, and images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import select, text

from letta.embeddings.query import apply_embedding_space_guard, embed_search_query
from letta.embeddings.resolver import resolve_embedding_config_async
from letta.orm.image import ImageRecord
from letta.orm.message import Message as MessageModel
from letta.orm.passage import ArchivalPassage, SourcePassage
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.settings import DatabaseChoice, settings


@dataclass
class RecallHit:
    layer: str
    snippet: str
    handle: str
    score: float
    reasons: List[str]


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
            if hasattr(model, "text"):
                pass
            q = q.order_by(model.embedding.cosine_distance(query_vec).asc()).limit(limit)
            rows = (await session.execute(q)).scalars().all()
            for idx, row in enumerate(rows):
                snippet = getattr(row, "description", None) or getattr(row, "text", None) or getattr(row, "caption", "") or ""
                vector_hits.append(
                    RecallHit(
                        layer=layer,
                        snippet=str(snippet)[:500],
                        handle=row.id,
                        score=1.0 / (idx + 1),
                        reasons=["vector"],
                    )
                )

        lexical_hits: List[RecallHit] = []
        if query:
            for layer, table, col in (
                ("archival", "archival_passages", "text"),
                ("source", "source_passages", "text"),
                ("message", "messages", "text"),
                ("image", "images", "description"),
            ):
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
                    lexical_hits.append(
                        RecallHit(
                            layer=layer,
                            snippet=str(row[1])[:500],
                            handle=row[0],
                            score=float(row[2]),
                            reasons=["lexical"],
                        )
                    )

    # RRF fusion (simplified)
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
