"""Batch enrichment for historic Part 1 image records (1MP + captions + pixel embed)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import func, select

from letta.orm.image import ImageRecord
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.services.image_ingest import enrich_image_background

logger = logging.getLogger(__name__)

_PENDING_STATUSES = ("pending", "failed")


@dataclass
class EnrichPendingReport:
    generated_at: str
    organization_id: str
    pending_count: int
    processed: int
    succeeded: int
    failed: int
    dry_run: bool

    def summary_lines(self) -> list[str]:
        title = "Enrich Pending Dry Run" if self.dry_run else "Enrich Pending Live Run"
        lines = [
            f"=== Part 1 {title} ===",
            f"Generated: {self.generated_at}",
            f"Organization: {self.organization_id}",
            f"Images needing enrichment: {self.pending_count}",
        ]
        if not self.dry_run:
            lines.extend(
                [
                    f"Processed: {self.processed}",
                    f"Succeeded (now complete): {self.succeeded}",
                    f"Failed this run: {self.failed}",
                ]
            )
        else:
            lines.append("No API calls performed (dry run).")
        return lines


async def count_pending_images(actor: PydanticUser) -> int:
    async with db_registry.async_session() as session:
        q = (
            select(func.count())
            .select_from(ImageRecord)
            .where(
                ImageRecord.organization_id == actor.organization_id,
                ImageRecord.is_deleted == False,  # noqa: E712
                ImageRecord.enrichment_status.in_(_PENDING_STATUSES),
            )
        )
        return int((await session.execute(q)).scalar_one() or 0)


async def _list_pending_images(
    actor: PydanticUser,
    *,
    limit: Optional[int] = None,
) -> list[tuple[str, str]]:
    async with db_registry.async_session() as session:
        q = (
            select(ImageRecord.id, ImageRecord.enrichment_status)
            .where(
                ImageRecord.organization_id == actor.organization_id,
                ImageRecord.is_deleted == False,  # noqa: E712
                ImageRecord.enrichment_status.in_(_PENDING_STATUSES),
            )
            .order_by(ImageRecord.created_at, ImageRecord.id)
        )
        if limit is not None:
            q = q.limit(limit)
        rows = (await session.execute(q)).all()
        return [(row[0], row[1]) for row in rows]


async def run_enrich_pending_dry_run(actor: PydanticUser) -> EnrichPendingReport:
    pending_count = await count_pending_images(actor)
    return EnrichPendingReport(
        generated_at=datetime.utcnow().isoformat() + "Z",
        organization_id=actor.organization_id,
        pending_count=pending_count,
        processed=0,
        succeeded=0,
        failed=0,
        dry_run=True,
    )


async def _enrich_one_image(
    image_id: str,
    status: str,
    actor: PydanticUser,
) -> bool:
    """Returns True if image reached complete status."""
    force = status == "failed"
    try:
        await enrich_image_background(image_id, actor, message_id=None, force=force)

        from letta.services.image_manager import ImageManager

        image = await ImageManager().get_by_id_async(image_id, actor)
        if image and image.enrichment_status == "complete":
            logger.info("Enriched image %s", image_id)
            return True
        logger.warning("Enrichment incomplete for %s (status=%s)", image_id, getattr(image, "enrichment_status", None))
        return False
    except Exception as e:
        logger.error("Enrichment error for %s: %s", image_id, e)
        return False


async def run_enrich_pending_live(
    actor: PydanticUser,
    *,
    limit: Optional[int] = None,
    throttle_seconds: float = 0.5,
    concurrency: int = 1,
) -> EnrichPendingReport:
    pending = await _list_pending_images(actor, limit=limit)
    pending_count = await count_pending_images(actor)
    processed = 0
    succeeded = 0
    failed = 0

    workers = max(1, concurrency)
    sem = asyncio.Semaphore(workers)

    async def _run_one(image_id: str, status: str) -> bool:
        async with sem:
            ok = await _enrich_one_image(image_id, status, actor)
            if throttle_seconds > 0:
                await asyncio.sleep(throttle_seconds)
            return ok

    if workers == 1:
        for image_id, status in pending:
            processed += 1
            if await _run_one(image_id, status):
                succeeded += 1
            else:
                failed += 1
    else:
        results = await asyncio.gather(*[_run_one(image_id, status) for image_id, status in pending])
        processed = len(results)
        succeeded = sum(1 for ok in results if ok)
        failed = processed - succeeded

    return EnrichPendingReport(
        generated_at=datetime.utcnow().isoformat() + "Z",
        organization_id=actor.organization_id,
        pending_count=pending_count,
        processed=processed,
        succeeded=succeeded,
        failed=failed,
        dry_run=False,
    )
