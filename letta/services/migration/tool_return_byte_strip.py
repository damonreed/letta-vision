"""Historic cleanup: strip persisted base64 from tool_returns (fetch_image, etc.)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select, tuple_

from letta.orm.message import Message as MessageModel
from letta.schemas.message import Message as PydanticMessage
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.services.migration.image_base64_conversion import persist_message_for_migration
from letta.services.vision.tool_return_storage import (
    message_has_strippable_tool_return_bytes,
    strip_persisted_image_bytes_from_tool_returns,
)

logger = logging.getLogger(__name__)

DEFAULT_TOOL_RETURN_STRIP_CHECKPOINT = Path.home() / ".letta" / "uplift_tool_return_strip_checkpoint.json"


@dataclass
class ToolReturnStripStats:
    messages_scanned: int = 0
    messages_stripped: int = 0
    bytes_removed: int = 0


@dataclass
class ToolReturnStripDryRunReport:
    generated_at: str
    organization_id: str
    stats: ToolReturnStripStats

    def summary_lines(self) -> list[str]:
        s = self.stats
        return [
            "=== Tool Return Byte Strip Dry Run ===",
            f"Generated: {self.generated_at}",
            f"Organization: {self.organization_id}",
            f"Messages scanned: {s.messages_scanned}",
            f"Messages with strippable tool_return bytes: {s.messages_stripped}",
            f"Estimated bytes removable: {s.bytes_removed:,}",
            "",
            "No writes performed (dry run).",
        ]


@dataclass
class ToolReturnStripLiveReport:
    generated_at: str
    organization_id: str
    stats: ToolReturnStripStats
    checkpoint_path: str
    completed: bool = True

    def summary_lines(self) -> list[str]:
        s = self.stats
        status = "complete" if self.completed else "interrupted (checkpoint saved)"
        return [
            "=== Tool Return Byte Strip Live Run ===",
            f"Generated: {self.generated_at}",
            f"Organization: {self.organization_id}",
            f"Status: {status}",
            f"Messages scanned: {s.messages_scanned}",
            f"Messages stripped: {s.messages_stripped}",
            f"Bytes removed (JSON field size): {s.bytes_removed:,}",
            f"Checkpoint: {self.checkpoint_path}",
        ]


@dataclass
class ToolReturnStripCheckpoint:
    organization_id: str
    last_created_at: Optional[str] = None
    last_id: Optional[str] = None
    messages_scanned: int = 0
    messages_stripped: int = 0
    bytes_removed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolReturnStripCheckpoint":
        return cls(
            organization_id=data.get("organization_id", ""),
            last_created_at=data.get("last_created_at"),
            last_id=data.get("last_id"),
            messages_scanned=int(data.get("messages_scanned", 0)),
            messages_stripped=int(data.get("messages_stripped", 0)),
            bytes_removed=int(data.get("bytes_removed", 0)),
        )


def load_checkpoint(path: Path) -> Optional[ToolReturnStripCheckpoint]:
    if not path.exists():
        return None
    return ToolReturnStripCheckpoint.from_dict(json.loads(path.read_text()))


def save_checkpoint(path: Path, checkpoint: ToolReturnStripCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint.to_dict(), indent=2))


async def _scan_batch(
    actor: PydanticUser,
    *,
    batch_size: int,
    limit: Optional[int],
    cursor_created_at: Optional[datetime],
    cursor_id: Optional[str],
    dry_run: bool,
    checkpoint: Optional[ToolReturnStripCheckpoint],
    checkpoint_path: Optional[Path],
) -> tuple[ToolReturnStripStats, bool, Optional[datetime], Optional[str]]:
    org_id = actor.organization_id
    stats = ToolReturnStripStats()
    last_created_at = cursor_created_at
    last_id = cursor_id
    completed = True

    if checkpoint:
        stats.messages_scanned = checkpoint.messages_scanned
        stats.messages_stripped = checkpoint.messages_stripped
        stats.bytes_removed = checkpoint.bytes_removed

    try:
        while True:
            if limit is not None and stats.messages_scanned >= limit:
                break

            fetch_size = batch_size
            if limit is not None:
                fetch_size = min(batch_size, limit - stats.messages_scanned)

            async with db_registry.async_session() as session:
                query = (
                    select(MessageModel)
                    .where(MessageModel.organization_id == org_id)
                    .where(MessageModel.tool_returns.isnot(None))
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
                stats.messages_scanned += 1

                if message_has_strippable_tool_return_bytes(message):
                    changed, removed = strip_persisted_image_bytes_from_tool_returns(message)
                    if changed:
                        stats.messages_stripped += 1
                        stats.bytes_removed += removed
                        if not dry_run:
                            await persist_message_for_migration(message, actor)
                            logger.info(
                                "Stripped tool_return bytes from %s (~%s bytes)",
                                message.id,
                                removed,
                            )

                last_created_at = row.created_at
                last_id = row.id

                if checkpoint is not None and checkpoint_path is not None:
                    checkpoint.last_created_at = last_created_at.isoformat() if last_created_at else None
                    checkpoint.last_id = last_id
                    checkpoint.messages_scanned = stats.messages_scanned
                    checkpoint.messages_stripped = stats.messages_stripped
                    checkpoint.bytes_removed = stats.bytes_removed
                    save_checkpoint(checkpoint_path, checkpoint)

                if limit is not None and stats.messages_scanned >= limit:
                    break

            if len(rows) < fetch_size:
                break
    except Exception:
        completed = False
        raise

    return stats, completed, last_created_at, last_id


async def run_tool_return_strip_dry_run(
    actor: PydanticUser,
    *,
    batch_size: int = 200,
    limit: Optional[int] = None,
) -> ToolReturnStripDryRunReport:
    stats, _, _, _ = await _scan_batch(
        actor,
        batch_size=batch_size,
        limit=limit,
        cursor_created_at=None,
        cursor_id=None,
        dry_run=True,
        checkpoint=None,
        checkpoint_path=None,
    )
    return ToolReturnStripDryRunReport(
        generated_at=datetime.utcnow().isoformat() + "Z",
        organization_id=actor.organization_id,
        stats=stats,
    )


async def run_tool_return_strip_live(
    actor: PydanticUser,
    *,
    batch_size: int = 200,
    limit: Optional[int] = None,
    checkpoint_path: Path = DEFAULT_TOOL_RETURN_STRIP_CHECKPOINT,
    resume: bool = True,
) -> ToolReturnStripLiveReport:
    org_id = actor.organization_id
    checkpoint = load_checkpoint(checkpoint_path) if resume else None
    if checkpoint and checkpoint.organization_id and checkpoint.organization_id != org_id:
        raise ValueError(
            f"Checkpoint organization {checkpoint.organization_id} does not match actor {org_id}; "
            "use --no-resume or a different --checkpoint path"
        )

    cursor_created_at = None
    cursor_id = None
    if checkpoint:
        if checkpoint.last_created_at:
            cursor_created_at = datetime.fromisoformat(checkpoint.last_created_at.replace("Z", "+00:00"))
        cursor_id = checkpoint.last_id
    else:
        checkpoint = ToolReturnStripCheckpoint(organization_id=org_id)

    completed = True
    try:
        stats, completed, _, _ = await _scan_batch(
            actor,
            batch_size=batch_size,
            limit=limit,
            cursor_created_at=cursor_created_at,
            cursor_id=cursor_id,
            dry_run=False,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
        )
    except Exception:
        stats = ToolReturnStripStats(
            messages_scanned=checkpoint.messages_scanned,
            messages_stripped=checkpoint.messages_stripped,
            bytes_removed=checkpoint.bytes_removed,
        )
        completed = False

    return ToolReturnStripLiveReport(
        generated_at=datetime.utcnow().isoformat() + "Z",
        organization_id=org_id,
        stats=stats,
        checkpoint_path=str(checkpoint_path),
        completed=completed,
    )
