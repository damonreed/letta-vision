"""Historic inline base64 image → LettaImage reference conversion (Part 1)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

from sqlalchemy import select, tuple_

from letta.orm.message import Message as MessageModel
from letta.schemas.message import Message as PydanticMessage
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.services.image_ingest import convert_historic_images_in_message
from letta.services.migration.block_classifier import MessageScanStats, merge_scan_stats, scan_message

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = Path.home() / ".letta" / "uplift_part1_checkpoint.json"


@dataclass
class ConversionDryRunReport:
    generated_at: str
    organization_id: str
    messages_scanned: int
    stats: MessageScanStats

    def summary_lines(self) -> list[str]:
        s = self.stats
        return [
            "=== Part 1 Conversion Dry Run ===",
            f"Generated: {self.generated_at}",
            f"Organization: {self.organization_id}",
            f"Messages scanned: {self.messages_scanned}",
            f"Messages with convertible blocks: {s.messages_with_convertible}",
            f"Convertible image blocks: {s.convertible_blocks}",
            f"Already LettaImage refs: {s.already_letta}",
            f"URL image blocks (skipped): {s.url_skipped}",
            f"Other skipped image blocks: {s.other_skipped}",
            f"Unrecoverable placeholders: {s.unrecoverable_placeholders}",
            f"Distinct content hashes: {len(s.distinct_content_hashes)}",
            f"Estimated messages content bytes removed: {s.estimated_bytes_removed:,}",
            "",
            "No writes performed (dry run).",
        ]


@dataclass
class ConversionLiveReport:
    generated_at: str
    organization_id: str
    messages_scanned: int
    messages_converted: int
    image_refs_written: int
    stats: MessageScanStats
    checkpoint_path: str
    completed: bool = True

    def summary_lines(self) -> list[str]:
        s = self.stats
        status = "complete" if self.completed else "interrupted (checkpoint saved)"
        return [
            "=== Part 1 Conversion Live Run ===",
            f"Generated: {self.generated_at}",
            f"Organization: {self.organization_id}",
            f"Status: {status}",
            f"Messages scanned: {self.messages_scanned}",
            f"Messages converted: {self.messages_converted}",
            f"Image refs written: {self.image_refs_written}",
            f"Convertible blocks processed: {s.convertible_blocks}",
            f"Already LettaImage refs (skipped): {s.already_letta}",
            f"Checkpoint: {self.checkpoint_path}",
        ]


@dataclass
class ConversionCheckpoint:
    organization_id: str
    last_created_at: Optional[str] = None
    last_id: Optional[str] = None
    messages_scanned: int = 0
    messages_converted: int = 0
    image_refs_written: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversionCheckpoint":
        return cls(
            organization_id=data.get("organization_id", ""),
            last_created_at=data.get("last_created_at"),
            last_id=data.get("last_id"),
            messages_scanned=int(data.get("messages_scanned", 0)),
            messages_converted=int(data.get("messages_converted", 0)),
            image_refs_written=int(data.get("image_refs_written", 0)),
        )


def load_checkpoint(path: Path) -> Optional[ConversionCheckpoint]:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return ConversionCheckpoint.from_dict(data)


def save_checkpoint(path: Path, checkpoint: ConversionCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint.to_dict(), indent=2))


async def persist_message_for_migration(message: PydanticMessage, actor: PydanticUser) -> None:
    """Rewrite message content without triggering background re-embed (Part 2 owns that)."""
    if not message.id:
        raise ValueError("message.id is required for migration persist")

    async with db_registry.async_session() as session:
        row = await MessageModel.read_async(db_session=session, identifier=message.id, actor=actor)
        row.content = message.content
        row.tool_returns = message.tool_returns
        await row.update_async(db_session=session, actor=actor)


async def scan_messages_for_conversion(
    actor: PydanticUser,
    *,
    batch_size: int = 200,
    limit: Optional[int] = None,
    cursor_created_at: Optional[datetime] = None,
    cursor_id: Optional[str] = None,
) -> Tuple[MessageScanStats, int]:
    """Scan messages and classify image blocks without mutating storage."""
    org_id = actor.organization_id
    aggregate = MessageScanStats()
    messages_scanned = 0
    last_created_at = cursor_created_at
    last_id = cursor_id

    while True:
        if limit is not None and messages_scanned >= limit:
            break

        fetch_size = batch_size
        if limit is not None:
            fetch_size = min(batch_size, limit - messages_scanned)

        async with db_registry.async_session() as session:
            query = (
                select(MessageModel)
                .where(MessageModel.organization_id == org_id)
                .order_by(MessageModel.created_at, MessageModel.id)
            )
            if last_created_at is not None and last_id is not None:
                query = query.where(tuple_(MessageModel.created_at, MessageModel.id) > tuple_(last_created_at, last_id))
            query = query.limit(fetch_size)
            rows = (await session.execute(query)).scalars().all()

        if not rows:
            break

        for row in rows:
            message = row.to_pydantic()
            block_stats = scan_message(message)
            merge_scan_stats(aggregate, block_stats)
            messages_scanned += 1
            last_created_at = row.created_at
            last_id = row.id

            if limit is not None and messages_scanned >= limit:
                break

        if len(rows) < fetch_size:
            break

    return aggregate, messages_scanned


async def run_conversion_dry_run(
    actor: PydanticUser,
    *,
    batch_size: int = 200,
    limit: Optional[int] = None,
) -> ConversionDryRunReport:
    stats, messages_scanned = await scan_messages_for_conversion(actor, batch_size=batch_size, limit=limit)
    return ConversionDryRunReport(
        generated_at=datetime.utcnow().isoformat() + "Z",
        organization_id=actor.organization_id,
        messages_scanned=messages_scanned,
        stats=stats,
    )


async def run_conversion_live(
    actor: PydanticUser,
    *,
    batch_size: int = 200,
    limit: Optional[int] = None,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    resume: bool = True,
) -> ConversionLiveReport:
    org_id = actor.organization_id
    checkpoint = load_checkpoint(checkpoint_path) if resume else None
    if checkpoint and checkpoint.organization_id and checkpoint.organization_id != org_id:
        raise ValueError(
            f"Checkpoint organization {checkpoint.organization_id} does not match actor {org_id}; "
            "use --no-resume or a different --checkpoint path"
        )

    cursor_created_at = None
    cursor_id = None
    messages_scanned = 0
    messages_converted = 0
    image_refs_written = 0
    aggregate = MessageScanStats()

    if checkpoint:
        if checkpoint.last_created_at:
            cursor_created_at = datetime.fromisoformat(checkpoint.last_created_at.replace("Z", "+00:00"))
        cursor_id = checkpoint.last_id
        messages_scanned = checkpoint.messages_scanned
        messages_converted = checkpoint.messages_converted
        image_refs_written = checkpoint.image_refs_written
        logger.info(
            "Resuming from checkpoint: scanned=%s converted=%s cursor=%s %s",
            messages_scanned,
            messages_converted,
            cursor_created_at,
            cursor_id,
        )

    completed = True

    try:
        while True:
            if limit is not None and messages_scanned >= limit:
                break

            fetch_size = batch_size
            if limit is not None:
                fetch_size = min(batch_size, limit - messages_scanned)

            async with db_registry.async_session() as session:
                query = (
                    select(MessageModel)
                    .where(MessageModel.organization_id == org_id)
                    .order_by(MessageModel.created_at, MessageModel.id)
                )
                if cursor_created_at is not None and cursor_id is not None:
                    query = query.where(tuple_(MessageModel.created_at, MessageModel.id) > tuple_(cursor_created_at, cursor_id))
                query = query.limit(fetch_size)
                rows = (await session.execute(query)).scalars().all()

            if not rows:
                break

            for row in rows:
                if limit is not None and messages_scanned >= limit:
                    break

                message = row.to_pydantic()
                pre_stats = scan_message(message)
                merge_scan_stats(aggregate, pre_stats)

                image_ids, changed = await convert_historic_images_in_message(message, actor)
                if changed:
                    await persist_message_for_migration(message, actor)
                    messages_converted += 1
                    image_refs_written += len(image_ids)
                    logger.info("Converted message %s (%s image refs)", message.id, len(image_ids))

                messages_scanned += 1
                cursor_created_at = row.created_at
                cursor_id = row.id

                save_checkpoint(
                    checkpoint_path,
                    ConversionCheckpoint(
                        organization_id=org_id,
                        last_created_at=cursor_created_at.isoformat() if cursor_created_at else None,
                        last_id=cursor_id,
                        messages_scanned=messages_scanned,
                        messages_converted=messages_converted,
                        image_refs_written=image_refs_written,
                    ),
                )

            if len(rows) < fetch_size:
                break

    except Exception:
        completed = False
        raise

    if completed and checkpoint_path.exists():
        checkpoint_path.unlink(missing_ok=True)

    return ConversionLiveReport(
        generated_at=datetime.utcnow().isoformat() + "Z",
        organization_id=org_id,
        messages_scanned=messages_scanned,
        messages_converted=messages_converted,
        image_refs_written=image_refs_written,
        stats=aggregate,
        checkpoint_path=str(checkpoint_path),
        completed=completed,
    )
