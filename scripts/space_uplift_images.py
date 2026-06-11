#!/usr/bin/env python3
"""Re-embed images whose embedding_space_id differs from the deployment target (preview→GA uplift)."""

from __future__ import annotations

import argparse
import asyncio
import logging

from sqlalchemy import select

from letta.embeddings.resolver import resolve_deployment_embedding_config_async
from letta.orm.image import ImageRecord
from letta.server.db import db_registry
from letta.server.server import SyncServer
from letta.services.image_ingest import enrich_image_background

logger = logging.getLogger(__name__)


async def _list_wrong_space_image_ids(actor, target_space_id: str, limit: int | None) -> list[str]:
    async with db_registry.async_session() as session:
        q = (
            select(ImageRecord.id)
            .where(
                ImageRecord.organization_id == actor.organization_id,
                ImageRecord.is_deleted == False,  # noqa: E712
                ImageRecord.embedding_space_id.is_distinct_from(target_space_id),
            )
            .order_by(ImageRecord.created_at, ImageRecord.id)
        )
        if limit is not None:
            q = q.limit(limit)
        return [row[0] for row in (await session.execute(q)).all()]


async def run_space_uplift(
    *,
    limit: int | None = None,
    concurrency: int = 4,
    throttle_seconds: float = 0.25,
) -> int:
    server = SyncServer()
    actor = await server.user_manager.get_default_actor_async()
    config = await resolve_deployment_embedding_config_async(actor)
    target = config.compute_space_id()
    image_ids = await _list_wrong_space_image_ids(actor, target, limit)
    print(f"space uplift: {len(image_ids)} images (target={target})")
    if not image_ids:
        return 0

    ok = 0
    fail = 0
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(image_id: str) -> None:
        nonlocal ok, fail
        async with sem:
            try:
                await enrich_image_background(image_id, actor, force=True)
                async with db_registry.async_session() as session:
                    row = await ImageRecord.read_async(db_session=session, identifier=image_id, actor=actor)
                if row and row.enrichment_status == "complete" and row.embedding_space_id == target:
                    ok += 1
                    logger.info("OK %s", image_id)
                else:
                    fail += 1
                    logger.warning(
                        "INCOMPLETE %s status=%s space=%s",
                        image_id,
                        getattr(row, "enrichment_status", None),
                        getattr(row, "embedding_space_id", None),
                    )
            except Exception as e:
                fail += 1
                logger.error("ERR %s: %s", image_id, e)
            if throttle_seconds > 0:
                await asyncio.sleep(throttle_seconds)

    await asyncio.gather(*[one(image_id) for image_id in image_ids])
    print(f"DONE ok={ok} fail={fail} total={len(image_ids)}")
    return 1 if fail else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--throttle", type=float, default=0.25)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    return asyncio.run(
        run_space_uplift(
            limit=args.limit,
            concurrency=args.concurrency,
            throttle_seconds=args.throttle,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
