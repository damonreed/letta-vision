"""Historic inline base64 image → LettaImage reference conversion (Part 1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from sqlalchemy import select, tuple_

from letta.orm.message import Message as MessageModel
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.services.migration.block_classifier import MessageScanStats, merge_scan_stats, scan_message


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
