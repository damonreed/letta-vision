"""Read-only inventory and cost estimates for historic embedding uplift."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from letta.embeddings.resolver import resolve_deployment_embedding_config_async
from letta.orm.file_archive import FileArchive
from letta.orm.image import ImageRecord
from letta.orm.message import Message as MessageModel
from letta.orm.passage import ArchivalPassage, SourcePassage
from letta.schemas.user import User as PydanticUser
from letta.server.db import db_registry
from letta.services.migration.block_classifier import MessageScanStats
from letta.services.migration.image_base64_conversion import scan_messages_for_conversion
from letta.services.migration.historic_reembed import MESSAGE_EMBED_VERSION
from letta.services.migration.uplift_cost import (
    CostEstimate,
    Part2Inventory,
    TableUpliftCounts,
    estimate_uplift_cost,
)


@dataclass
class UpliftInventoryReport:
    generated_at: str
    organization_id: str
    part1: MessageScanStats
    part1_dedup_hits: int = 0
    part1_distinct_new_images: int = 0
    messages_table_bytes: int = 0
    part2: Part2Inventory = field(default_factory=Part2Inventory)
    cost: CostEstimate = field(default_factory=CostEstimate)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["part1"]["distinct_content_hashes"] = len(self.part1.distinct_content_hashes)
        return data


async def get_messages_table_size_bytes(session: AsyncSession) -> int:
    result = await session.execute(text("SELECT pg_total_relation_size('messages')"))
    return int(result.scalar_one() or 0)


async def _count_table(
    session: AsyncSession,
    model: type,
    org_id: str,
    target_space_id: str,
    *,
    has_legacy_column: bool = False,
    extra_needs_uplift: Optional[Any] = None,
) -> TableUpliftCounts:
    needs_uplift = or_(
        model.embedding.is_(None),
        model.embedding_space_id.is_distinct_from(target_space_id),
    )
    if extra_needs_uplift is not None:
        needs_uplift = or_(needs_uplift, extra_needs_uplift)

    total_q = select(func.count()).select_from(model).where(model.organization_id == org_id)
    needs_q = select(func.count()).select_from(model).where(model.organization_id == org_id).where(needs_uplift)
    legacy_unknown_q = (
        select(func.count())
        .select_from(model)
        .where(model.organization_id == org_id)
        .where(model.embedding_space_id == "legacy-unknown")
    )

    total_rows = int((await session.execute(total_q)).scalar_one() or 0)
    needs_count = int((await session.execute(needs_q)).scalar_one() or 0)
    legacy_unknown = int((await session.execute(legacy_unknown_q)).scalar_one() or 0)

    legacy_4096_only = 0
    if has_legacy_column:
        legacy_q = (
            select(func.count())
            .select_from(model)
            .where(model.organization_id == org_id)
            .where(model.embedding_legacy_4096.isnot(None))
            .where(model.embedding.is_(None))
        )
        legacy_4096_only = int((await session.execute(legacy_q)).scalar_one() or 0)

    return TableUpliftCounts(
        needs_uplift=needs_count,
        legacy_unknown_space=legacy_unknown,
        legacy_4096_only=legacy_4096_only,
        total_rows=total_rows,
    )


async def collect_part2_inventory(actor: PydanticUser, target_space_id: str, deployment_handle: str) -> Part2Inventory:
    org_id = actor.organization_id
    image_extra = or_(ImageRecord.enrichment_status == "failed", ImageRecord.enrichment_status == "pending")

    async with db_registry.async_session() as session:
        archival = await _count_table(session, ArchivalPassage, org_id, target_space_id, has_legacy_column=True)
        source = await _count_table(session, SourcePassage, org_id, target_space_id, has_legacy_column=True)
        archives = await _count_table(session, FileArchive, org_id, target_space_id, has_legacy_column=True)
        messages = await _count_table(
            session,
            MessageModel,
            org_id,
            target_space_id,
            extra_needs_uplift=or_(
                MessageModel.embedding_version.is_(None),
                MessageModel.embedding_version < MESSAGE_EMBED_VERSION,
            ),
        )
        images = await _count_table(
            session,
            ImageRecord,
            org_id,
            target_space_id,
            extra_needs_uplift=image_extra,
        )

        pending_failed_q = (
            select(func.count())
            .select_from(ImageRecord)
            .where(ImageRecord.organization_id == org_id)
            .where(ImageRecord.enrichment_status.in_(("pending", "failed")))
        )
        images_pending_or_failed = int((await session.execute(pending_failed_q)).scalar_one() or 0)

    return Part2Inventory(
        target_space_id=target_space_id,
        deployment_handle=deployment_handle,
        archival_passages=archival,
        source_passages=source,
        file_archives=archives,
        messages=messages,
        images=images,
        images_pending_or_failed=images_pending_or_failed,
    )


async def load_existing_image_hashes(actor: PydanticUser) -> set[str]:
    async with db_registry.async_session() as session:
        result = await session.execute(
            select(ImageRecord.content_hash).where(ImageRecord.organization_id == actor.organization_id)
        )
        return {row[0] for row in result.all() if row[0]}


async def build_inventory_report(
    actor: PydanticUser,
    *,
    batch_size: int = 200,
    limit: Optional[int] = None,
    include_caption_cost: bool = True,
) -> UpliftInventoryReport:
    config = await resolve_deployment_embedding_config_async(actor)
    target_space_id = config.compute_space_id()
    deployment_handle = config.handle or config.embedding_model

    part1_stats, _ = await scan_messages_for_conversion(actor, batch_size=batch_size, limit=limit)
    existing_hashes = await load_existing_image_hashes(actor)
    dedup_hits = len(part1_stats.distinct_content_hashes & existing_hashes)
    distinct_new = len(part1_stats.distinct_content_hashes - existing_hashes)

    part2 = await collect_part2_inventory(actor, target_space_id, deployment_handle)
    cost = estimate_uplift_cost(part2, part1_stats, include_caption_cost=include_caption_cost)

    async with db_registry.async_session() as session:
        messages_table_bytes = await get_messages_table_size_bytes(session)

    return UpliftInventoryReport(
        generated_at=datetime.utcnow().isoformat() + "Z",
        organization_id=actor.organization_id,
        part1=part1_stats,
        part1_dedup_hits=dedup_hits,
        part1_distinct_new_images=distinct_new,
        messages_table_bytes=messages_table_bytes,
        part2=part2,
        cost=cost,
    )


def format_inventory_report(report: UpliftInventoryReport) -> str:
    p1 = report.part1
    p2 = report.part2
    cost = report.cost
    lines = [
        "=== Historic Uplift Inventory ===",
        f"Generated: {report.generated_at}",
        f"Organization: {report.organization_id}",
        f"Deployment handle: {p2.deployment_handle}",
        f"Target embedding_space_id: {p2.target_space_id}",
        "",
        "--- Part 1: inline base64 conversion ---",
        f"Messages scanned: {p1.messages_scanned}",
        f"Messages with convertible blocks: {p1.messages_with_convertible}",
        f"Convertible image blocks: {p1.convertible_blocks}",
        f"Already LettaImage refs: {p1.already_letta}",
        f"URL image blocks (skipped): {p1.url_skipped}",
        f"Other skipped image blocks: {p1.other_skipped}",
        f"Unrecoverable placeholders: {p1.unrecoverable_placeholders}",
        f"Distinct content hashes: {len(p1.distinct_content_hashes)}",
        f"Dedup hits (existing images rows): {report.part1_dedup_hits}",
        f"Distinct new images to create: {report.part1_distinct_new_images}",
        f"Estimated messages content bytes removed: {p1.estimated_bytes_removed:,}",
        f"messages table total size (pg_total_relation_size): {report.messages_table_bytes:,} bytes",
        "",
        "--- Part 2: re-embed backlog ---",
        f"archival_passages needs uplift: {p2.archival_passages.needs_uplift} / {p2.archival_passages.total_rows}"
        f" (legacy-unknown: {p2.archival_passages.legacy_unknown_space}, legacy-4096-only: {p2.archival_passages.legacy_4096_only})",
        f"source_passages needs uplift: {p2.source_passages.needs_uplift} / {p2.source_passages.total_rows}"
        f" (legacy-unknown: {p2.source_passages.legacy_unknown_space}, legacy-4096-only: {p2.source_passages.legacy_4096_only})",
        f"file_archives needs uplift: {p2.file_archives.needs_uplift} / {p2.file_archives.total_rows}"
        f" (legacy-unknown: {p2.file_archives.legacy_unknown_space}, legacy-4096-only: {p2.file_archives.legacy_4096_only})",
        f"messages needs uplift: {p2.messages.needs_uplift} / {p2.messages.total_rows}",
        f"images needs uplift: {p2.images.needs_uplift} / {p2.images.total_rows}",
        f"images pending/failed enrichment: {p2.images_pending_or_failed}",
        "",
        "--- Part 2: projected API cost (heuristic) ---",
        f"Passage embed calls: {cost.passage_embed_calls}",
        f"File archive embed calls: {cost.file_archive_embed_calls}",
        f"Message embed calls: {cost.message_embed_calls}",
        f"Image pixel embed calls: {cost.image_pixel_embed_calls}",
        f"Image VLM caption calls: {cost.image_vlm_caption_calls}",
        f"Estimated embed tokens: {cost.estimated_tokens:,}",
        f"Estimated USD: ${cost.estimated_usd:.4f}",
    ]
    return "\n".join(lines)
