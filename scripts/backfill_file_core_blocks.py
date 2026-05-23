#!/usr/bin/env python3
"""Backfill file_core_blocks from legacy files_agents visible_content."""

import argparse
import asyncio
import logging

from sqlalchemy import select

from letta.orm.files_agents import FileAgent as FileAgentModel
from letta.services.file_core_block_manager import FileCoreBlockManager
from letta.services.files.char_page_reader import CharPageReader
from letta.server.db import db_registry
from letta.server.server import SyncServer

logger = logging.getLogger(__name__)


def _seed_summary_from_visible(visible: str, limit: int = 2000) -> str:
    text = (visible or "").strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    return (first_line or text)[:limit]


async def backfill(dry_run: bool = False) -> None:
    server = SyncServer()
    actor = await server.user_manager.get_default_actor_async()
    manager = FileCoreBlockManager()

    async with db_registry.async_session() as session:
        rows = (await session.execute(select(FileAgentModel).where(FileAgentModel.is_deleted == False))).scalars().all()

    seeded = 0
    skipped = 0
    for row in rows:
        existing = await manager.get(file_id=row.file_id, actor=actor)
        if existing and existing.summary and existing.summary != "No headline yet.":
            continue

        summary = _seed_summary_from_visible(row.visible_content or "")
        if not summary:
            skipped += 1
            logger.info("skip file_id=%s (no visible_content headline seed)", row.file_id)
            continue

        if dry_run:
            logger.info("would seed file_id=%s summary=%r", row.file_id, summary[:80])
            seeded += 1
            continue

        await manager.get_or_create(
            file_id=row.file_id,
            organization_id=row.organization_id,
            actor=actor,
            default_summary=summary,
        )
        seeded += 1

    logger.info("backfill complete: seeded=%s skipped=%s", seeded, skipped)


def main():
    parser = argparse.ArgumentParser(description="Backfill file_core_blocks from files_agents")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(backfill(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
