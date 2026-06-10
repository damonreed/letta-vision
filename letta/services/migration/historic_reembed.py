"""Part 2 historic re-embed: passages, file archives, and messages (FR v0.6.0 GA §5.1)."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from sqlalchemy import func, or_, select, tuple_, update

from letta.embeddings.resolver import resolve_deployment_embedding_config_async
from letta.llm_api.llm_client import LLMClient
from letta.embeddings.write import write_message_embedding_atomic
from letta.embeddings.util import prepare_vector_for_write
from letta.orm.file_archive import FileArchive
from letta.orm.message import Message as MessageModel
from letta.orm.passage import ArchivalPassage, SourcePassage
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.services.file_archive_embedding import prepare_file_archive_embedding_fields
from letta.services.passage_manager import _prepare_passage_embedding_fields

logger = logging.getLogger(__name__)

DEFAULT_PART2_CHECKPOINT_PATH = Path.home() / ".letta" / "uplift_part2_checkpoint.json"

MESSAGE_EMBED_VERSION = 2

REEMBED_TABLES = ("archival_passages", "source_passages", "file_archives", "messages")
ALL_REEMBED_TABLES = REEMBED_TABLES


def _needs_uplift_clause(model: type, target_space_id: str):
    return or_(
        model.embedding.is_(None),
        model.embedding_space_id.is_distinct_from(target_space_id),
    )


def _needs_message_uplift_clause(model: type, target_space_id: str):
    return or_(
        _needs_uplift_clause(model, target_space_id),
        model.embedding_version.is_(None),
        model.embedding_version < MESSAGE_EMBED_VERSION,
    )


def _passage_embed_text(row) -> str:
    return (row.text or "").strip()


def _file_archive_embed_text(row) -> str:
    title = (row.title or "").strip()
    content = (row.content or "").strip()
    if title and content:
        return f"{title}\n\n{content}"
    return title or content


@dataclass(frozen=True)
class TableSpec:
    name: str
    model: type
    embed_text: Callable[[Any], str]


TABLE_SPECS: dict[str, TableSpec] = {
    "archival_passages": TableSpec("archival_passages", ArchivalPassage, _passage_embed_text),
    "source_passages": TableSpec("source_passages", SourcePassage, _passage_embed_text),
    "file_archives": TableSpec("file_archives", FileArchive, _file_archive_embed_text),
    "messages": TableSpec("messages", MessageModel, lambda _row: ""),
}


@dataclass
class TableCursor:
    last_created_at: Optional[str] = None
    last_id: Optional[str] = None
    processed: int = 0
    succeeded: int = 0
    failed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TableCursor":
        return cls(
            last_created_at=data.get("last_created_at"),
            last_id=data.get("last_id"),
            processed=int(data.get("processed", 0)),
            succeeded=int(data.get("succeeded", 0)),
            failed=int(data.get("failed", 0)),
        )


@dataclass
class Part2Checkpoint:
    organization_id: str
    target_space_id: str
    tables: dict[str, TableCursor] = field(default_factory=dict)

    def cursor_for(self, table: str) -> TableCursor:
        return self.tables.get(table, TableCursor())

    def to_dict(self) -> dict[str, Any]:
        return {
            "organization_id": self.organization_id,
            "target_space_id": self.target_space_id,
            "tables": {name: c.to_dict() for name, c in self.tables.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Part2Checkpoint":
        tables = {name: TableCursor.from_dict(cur) for name, cur in (data.get("tables") or {}).items()}
        return cls(
            organization_id=data.get("organization_id", ""),
            target_space_id=data.get("target_space_id", ""),
            tables=tables,
        )


@dataclass
class TableReembedStats:
    table: str
    pending_count: int = 0
    processed: int = 0
    succeeded: int = 0
    failed: int = 0


@dataclass
class ReembedDryRunReport:
    generated_at: str
    organization_id: str
    target_space_id: str
    deployment_handle: str
    tables: list[TableReembedStats]

    def summary_lines(self) -> list[str]:
        lines = [
            "=== Part 2 Re-embed Dry Run (passages, file archives, messages) ===",
            f"Generated: {self.generated_at}",
            f"Organization: {self.organization_id}",
            f"Deployment handle: {self.deployment_handle}",
            f"Target embedding_space_id: {self.target_space_id}",
            "",
        ]
        total = 0
        for stat in self.tables:
            lines.append(f"{stat.table}: {stat.pending_count} rows need uplift")
            total += stat.pending_count
        lines.extend(["", f"Total embed calls (if live): {total}", "No API calls performed (dry run)."])
        return lines


@dataclass
class ReembedLiveReport:
    generated_at: str
    organization_id: str
    target_space_id: str
    deployment_handle: str
    tables: list[TableReembedStats]
    checkpoint_path: str
    completed: bool = True

    def summary_lines(self) -> list[str]:
        status = "complete" if self.completed else "interrupted (checkpoint saved)"
        lines = [
            "=== Part 2 Re-embed Live Run (passages, file archives, messages) ===",
            f"Generated: {self.generated_at}",
            f"Organization: {self.organization_id}",
            f"Status: {status}",
            f"Target embedding_space_id: {self.target_space_id}",
            f"Checkpoint: {self.checkpoint_path}",
            "",
        ]
        for stat in self.tables:
            lines.append(f"{stat.table}: processed={stat.processed} succeeded={stat.succeeded} failed={stat.failed}")
        return lines


def load_part2_checkpoint(path: Path) -> Optional[Part2Checkpoint]:
    if not path.exists():
        return None
    return Part2Checkpoint.from_dict(json.loads(path.read_text()))


def save_part2_checkpoint(path: Path, checkpoint: Part2Checkpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint.to_dict(), indent=2))


def resolve_tables(table_arg: str) -> list[str]:
    if table_arg == "all":
        return list(ALL_REEMBED_TABLES)
    if table_arg not in TABLE_SPECS:
        raise ValueError(f"Unknown table {table_arg!r}; choose from {', '.join(REEMBED_TABLES)} or all")
    return [table_arg]


def _uplift_clause_for_table(spec: TableSpec, target_space_id: str):
    if spec.name == "messages":
        return _needs_message_uplift_clause(spec.model, target_space_id)
    return _needs_uplift_clause(spec.model, target_space_id)


def _embeddable_text_clause(spec: TableSpec):
    if spec.name == "messages":
        return True
    if spec.name == "file_archives":
        return or_(FileArchive.title != "", FileArchive.content != "")
    return spec.model.text != ""


async def count_pending_rows(
    session,
    spec: TableSpec,
    org_id: str,
    target_space_id: str,
) -> int:
    q = (
        select(func.count())
        .select_from(spec.model)
        .where(spec.model.organization_id == org_id)
        .where(_uplift_clause_for_table(spec, target_space_id))
        .where(_embeddable_text_clause(spec))
    )
    return int((await session.execute(q)).scalar_one() or 0)


async def run_reembed_dry_run(
    actor: PydanticUser,
    *,
    tables: Sequence[str],
) -> ReembedDryRunReport:
    config = await resolve_deployment_embedding_config_async(actor)
    target_space_id = config.compute_space_id()
    stats: list[TableReembedStats] = []

    async with db_registry.async_session() as session:
        for table_name in tables:
            spec = TABLE_SPECS[table_name]
            pending = await count_pending_rows(session, spec, actor.organization_id, target_space_id)
            stats.append(TableReembedStats(table=table_name, pending_count=pending))

    return ReembedDryRunReport(
        generated_at=datetime.utcnow().isoformat() + "Z",
        organization_id=actor.organization_id,
        target_space_id=target_space_id,
        deployment_handle=config.handle or config.embedding_model,
        tables=stats,
    )


async def _request_embeddings_with_retry(
    client: LLMClient,
    texts: list[str],
    config: EmbeddingConfig,
    *,
    max_attempts: int = 5,
) -> list[list[float]]:
    delay = 1.0
    last_error: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            return await client.request_embeddings(texts, config)
        except Exception as e:
            last_error = e
            if attempt == max_attempts - 1:
                break
            logger.warning("Embed batch failed (attempt %s/%s): %s", attempt + 1, max_attempts, e)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
    assert last_error is not None
    raise last_error


async def _write_passage_embedding(
    session,
    *,
    row_id: str,
    org_id: str,
    embedding: list[float],
    config: EmbeddingConfig,
    model: type,
) -> None:
    data = _prepare_passage_embedding_fields({"embedding": embedding, "embedding_config": config})
    cfg = data["embedding_config"]
    await session.execute(
        update(model)
        .where(model.id == row_id, model.organization_id == org_id)
        .values(
            embedding=data["embedding"],
            embedding_config=cfg.model_dump() if isinstance(cfg, EmbeddingConfig) else cfg,
            embedding_space_id=data["embedding_space_id"],
        )
    )


async def _build_message_embed_texts(rows, actor: PydanticUser) -> list[tuple[Any, str]]:
    from letta.embeddings.message_embed_text import build_message_embed_text
    from letta.services.message_manager import MessageManager

    extractor = MessageManager()._extract_message_text
    pairs: list[tuple[Any, str]] = []
    for row in rows:
        text = (
            await build_message_embed_text(
                row.to_pydantic(),
                actor,
                include_image_captions=True,
                base_extractor=extractor,
            )
        ).strip()
        if text:
            pairs.append((row, text))
    return pairs


async def _write_message_embedding(
    *,
    row_id: str,
    org_id: str,
    embedding: list[float],
    config: EmbeddingConfig,
) -> bool:
    prepared = prepare_vector_for_write(embedding, config)
    return await write_message_embedding_atomic(
        message_id=row_id,
        organization_id=org_id,
        embedding=prepared,
        embedding_config=config,
        embedding_version=MESSAGE_EMBED_VERSION,
    )


async def _write_file_archive_embedding(
    session,
    *,
    row_id: str,
    org_id: str,
    embedding: list[float],
    config: EmbeddingConfig,
) -> None:
    data = prepare_file_archive_embedding_fields({}, embedding=embedding, config=config)
    cfg = data["embedding_config"]
    await session.execute(
        update(FileArchive)
        .where(FileArchive.id == row_id, FileArchive.organization_id == org_id)
        .values(
            embedding=data["embedding"],
            embedding_config=cfg.model_dump() if isinstance(cfg, EmbeddingConfig) else cfg,
            embedding_space_id=data["embedding_space_id"],
        )
    )


async def _reembed_table_live(
    actor: PydanticUser,
    table_name: str,
    *,
    target_space_id: str,
    embedding_config: EmbeddingConfig,
    batch_size: int,
    limit: Optional[int],
    throttle_seconds: float,
    checkpoint: Part2Checkpoint,
    checkpoint_path: Path,
) -> TableReembedStats:
    spec = TABLE_SPECS[table_name]
    model = spec.model
    org_id = actor.organization_id
    cursor = checkpoint.cursor_for(table_name)

    cursor_created_at = None
    cursor_id = cursor.last_id
    if cursor.last_created_at:
        cursor_created_at = datetime.fromisoformat(cursor.last_created_at.replace("Z", "+00:00"))

    processed = cursor.processed
    succeeded = cursor.succeeded
    failed = cursor.failed

    client = LLMClient.create(embedding_config.embedding_endpoint_type, actor=actor)
    doc_config = embedding_config.model_copy(update={"input_type": "search_document"})

    logger.info("Re-embedding %s from cursor %s %s", table_name, cursor_created_at, cursor_id)

    while True:
        if limit is not None and processed >= limit:
            break

        fetch_size = batch_size
        if limit is not None:
            fetch_size = min(batch_size, limit - processed)

        async with db_registry.async_session() as session:
            query = (
                select(model)
                .where(model.organization_id == org_id)
                .where(_uplift_clause_for_table(spec, target_space_id))
                .where(_embeddable_text_clause(spec))
                .order_by(model.created_at, model.id)
            )
            if cursor_created_at is not None and cursor_id is not None:
                query = query.where(tuple_(model.created_at, model.id) > tuple_(cursor_created_at, cursor_id))
            query = query.limit(fetch_size)
            rows = (await session.execute(query)).scalars().all()

        if not rows:
            break

        if table_name == "messages":
            embed_pairs = await _build_message_embed_texts(rows, actor)
            embed_rows = [pair[0] for pair in embed_pairs]
            texts = [pair[1] for pair in embed_pairs]
            embed_ids = {row.id for row in embed_rows}
            skipped_rows = [row for row in rows if row.id not in embed_ids]
        else:
            embed_rows = rows
            texts = [spec.embed_text(row) for row in rows]
            skipped_rows = []

        try:
            vectors = await _request_embeddings_with_retry(client, texts, doc_config) if texts else []
        except Exception as e:
            logger.error("Batch embed failed for %s (%s rows): %s", table_name, len(texts), e)
            failed += len(rows)
            processed += len(rows)
            last = rows[-1]
            checkpoint.tables[table_name] = TableCursor(
                last_created_at=last.created_at.isoformat() if last.created_at else None,
                last_id=last.id,
                processed=processed,
                succeeded=succeeded,
                failed=failed,
            )
            save_part2_checkpoint(checkpoint_path, checkpoint)
            raise

        async with db_registry.async_session() as session:
            for row in skipped_rows:
                processed += 1
                succeeded += 1
                logger.debug("Skipped empty embed text for %s %s", table_name, row.id)

            for row, vec in zip(embed_rows, vectors):
                if vec is None:
                    failed += 1
                else:
                    try:
                        if table_name == "file_archives":
                            await _write_file_archive_embedding(
                                session, row_id=row.id, org_id=org_id, embedding=vec, config=doc_config
                            )
                        elif table_name == "messages":
                            applied = await _write_message_embedding(
                                row_id=row.id, org_id=org_id, embedding=vec, config=doc_config
                            )
                            if not applied:
                                logger.debug("Monotonic guard skipped %s %s", table_name, row.id)
                        else:
                            await _write_passage_embedding(
                                session, row_id=row.id, org_id=org_id, embedding=vec, config=doc_config, model=model
                            )
                        succeeded += 1
                        logger.info("Re-embedded %s %s", table_name, row.id)
                    except Exception as e:
                        failed += 1
                        logger.error("Write failed for %s %s: %s", table_name, row.id, e)
                processed += 1
            if table_name != "messages":
                await session.commit()

        last = rows[-1]
        cursor_created_at = last.created_at
        cursor_id = last.id
        checkpoint.tables[table_name] = TableCursor(
            last_created_at=cursor_created_at.isoformat() if cursor_created_at else None,
            last_id=cursor_id,
            processed=processed,
            succeeded=succeeded,
            failed=failed,
        )
        save_part2_checkpoint(checkpoint_path, checkpoint)

        if throttle_seconds > 0:
            await asyncio.sleep(throttle_seconds)

    return TableReembedStats(
        table=table_name,
        processed=processed,
        succeeded=succeeded,
        failed=failed,
    )


async def run_reembed_live(
    actor: PydanticUser,
    *,
    tables: Sequence[str],
    batch_size: int = 32,
    limit: Optional[int] = None,
    throttle_seconds: float = 0.25,
    checkpoint_path: Path = DEFAULT_PART2_CHECKPOINT_PATH,
    resume: bool = True,
) -> ReembedLiveReport:
    config = await resolve_deployment_embedding_config_async(actor)
    target_space_id = config.compute_space_id()
    org_id = actor.organization_id

    checkpoint = load_part2_checkpoint(checkpoint_path) if resume else None
    if checkpoint:
        if checkpoint.organization_id and checkpoint.organization_id != org_id:
            raise ValueError(
                f"Checkpoint organization {checkpoint.organization_id} does not match actor {org_id}; "
                "use --no-resume or a different --checkpoint path"
            )
        if checkpoint.target_space_id and checkpoint.target_space_id != target_space_id:
            raise ValueError(
                f"Checkpoint target space {checkpoint.target_space_id} does not match deployment {target_space_id}; "
                "use --no-resume or a different --checkpoint path"
            )
    else:
        checkpoint = Part2Checkpoint(organization_id=org_id, target_space_id=target_space_id)

    completed = True
    table_stats: list[TableReembedStats] = []

    try:
        for table_name in tables:
            stat = await _reembed_table_live(
                actor,
                table_name,
                target_space_id=target_space_id,
                embedding_config=config,
                batch_size=batch_size,
                limit=limit,
                throttle_seconds=throttle_seconds,
                checkpoint=checkpoint,
                checkpoint_path=checkpoint_path,
            )
            table_stats.append(stat)
    except Exception:
        completed = False
        for table_name in tables:
            if any(s.table == table_name for s in table_stats):
                continue
            cur = checkpoint.cursor_for(table_name)
            if cur.processed:
                table_stats.append(
                    TableReembedStats(
                        table=table_name,
                        processed=cur.processed,
                        succeeded=cur.succeeded,
                        failed=cur.failed,
                    )
                )

    return ReembedLiveReport(
        generated_at=datetime.utcnow().isoformat() + "Z",
        organization_id=org_id,
        target_space_id=target_space_id,
        deployment_handle=config.handle or config.embedding_model,
        tables=table_stats,
        checkpoint_path=str(checkpoint_path),
        completed=completed,
    )
