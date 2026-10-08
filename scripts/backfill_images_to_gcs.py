#!/usr/bin/env python3
"""Copy existing MinIO image objects into the GCS image mirror.

Objects are named by image id: ``gs://{LETTA_IMAGE_GCS_BUCKET}/{image_id}``.

Usage (inside the letta-vision container or a venv with the same env):

    python scripts/backfill_images_to_gcs.py
    python scripts/backfill_images_to_gcs.py --limit 50
    python scripts/backfill_images_to_gcs.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import sys


async def _run(*, limit: int | None, dry_run: bool) -> int:
    from sqlalchemy import select

    from letta.orm.image import ImageRecord
    from letta.server.db import db_registry
    from letta.services.object_store.client import get_object_store_client
    from letta.services.object_store.gcs_image_mirror import get_gcs_image_mirror

    mirror = get_gcs_image_mirror()
    if mirror is None:
        print("LETTA_IMAGE_GCS_BUCKET is not set", file=sys.stderr)
        return 2

    store = get_object_store_client()
    copied = 0
    skipped = 0
    failed = 0

    async with db_registry.async_session() as session:
        stmt = (
            select(ImageRecord)
            .where(ImageRecord.is_deleted == False)  # noqa: E712
            .order_by(ImageRecord.created_at.asc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        rows = list(result.scalars().all())

    print(f"Scanning {len(rows)} image(s) → gs://{mirror.bucket_name}/{{image_id}}")
    for row in rows:
        image_id = row.id
        key = row.object_url_full
        if not key:
            print(f"SKIP {image_id}: no object_url_full")
            skipped += 1
            continue
        try:
            if await mirror.exists(image_id):
                skipped += 1
                continue
            if dry_run:
                print(f"DRY-RUN would copy {image_id} from {key}")
                copied += 1
                continue
            raw = await store.get_bytes(key)
            media_type = (row.media_type or "image/png").split(";", 1)[0].strip()
            await mirror.put_image(image_id, raw, media_type or "image/png")
            copied += 1
            if copied % 25 == 0:
                print(f"... copied {copied}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {image_id}: {exc}", file=sys.stderr)

    print(f"done copied={copied} skipped={skipped} failed={failed}")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(limit=args.limit, dry_run=args.dry_run)))


if __name__ == "__main__":
    main()
