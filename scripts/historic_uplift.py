#!/usr/bin/env python3
"""Historic embedding uplift management CLI (FR v0.6.0 GA)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from letta.server.server import SyncServer
from letta.services.migration.image_base64_conversion import run_conversion_dry_run
from letta.services.migration.uplift_inventory import build_inventory_report, format_inventory_report

logger = logging.getLogger(__name__)


async def _get_actor():
    server = SyncServer()
    return await server.user_manager.get_default_actor_async()


async def cmd_inventory(args: argparse.Namespace) -> int:
    actor = await _get_actor()
    report = await build_inventory_report(
        actor,
        batch_size=args.batch_size,
        limit=args.limit,
        include_caption_cost=not args.skip_caption_cost,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_inventory_report(report))
    return 0


async def cmd_convert_dry_run(args: argparse.Namespace) -> int:
    actor = await _get_actor()
    report = await run_conversion_dry_run(actor, batch_size=args.batch_size, limit=args.limit)
    if args.json:
        payload = {
            "generated_at": report.generated_at,
            "organization_id": report.organization_id,
            "messages_scanned": report.messages_scanned,
            "stats": {
                **{k: v for k, v in report.stats.__dict__.items() if k != "distinct_content_hashes"},
                "distinct_content_hashes": len(report.stats.distinct_content_hashes),
            },
        }
        print(json.dumps(payload, indent=2))
    else:
        print("\n".join(report.summary_lines()))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Historic embedding uplift (v0.6.0 GA)")
    parser.add_argument("--batch-size", type=int, default=200, help="Messages per scan batch")
    parser.add_argument("--limit", type=int, default=None, help="Max messages to scan (smoke tests)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    sub = parser.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("inventory", help="Part 1 + Part 2 inventory and cost estimate")
    inv.add_argument(
        "--skip-caption-cost",
        action="store_true",
        help="Exclude VLM caption calls from Part 2 cost estimate",
    )
    inv.set_defaults(func=cmd_inventory)

    convert = sub.add_parser("convert", help="Part 1 base64 → object conversion")
    convert.add_argument("--dry-run", action="store_true", help="Scan and report only; no writes")
    convert.set_defaults(func=cmd_convert_dry_run)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.command == "convert" and not args.dry_run:
        print("Live conversion is not implemented yet. Use: convert --dry-run", file=sys.stderr)
        return 2

    return asyncio.run(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
