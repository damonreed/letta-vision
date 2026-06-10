#!/usr/bin/env python3
"""Historic embedding uplift management CLI (FR v0.6.0 GA)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from letta.server.server import SyncServer
from letta.services.migration.enrich_pending_images import run_enrich_pending_dry_run, run_enrich_pending_live
from letta.services.migration.historic_reembed import (
    DEFAULT_PART2_CHECKPOINT_PATH,
    resolve_tables,
    run_reembed_dry_run,
    run_reembed_live,
)
from letta.services.migration.image_base64_conversion import (
    DEFAULT_CHECKPOINT_PATH,
    run_conversion_dry_run,
    run_conversion_live,
)
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


async def cmd_enrich_pending(args: argparse.Namespace) -> int:
    actor = await _get_actor()
    if args.dry_run:
        report = await run_enrich_pending_dry_run(actor)
    else:
        report = await run_enrich_pending_live(
            actor,
            limit=args.limit,
            throttle_seconds=args.throttle,
            concurrency=args.concurrency,
        )
    if args.json:
        print(json.dumps(report.__dict__, indent=2))
    else:
        print("\n".join(report.summary_lines()))
    return 0


async def cmd_reembed(args: argparse.Namespace) -> int:
    actor = await _get_actor()
    tables = resolve_tables(args.table)
    if args.dry_run:
        report = await run_reembed_dry_run(actor, tables=tables)
    else:
        report = await run_reembed_live(
            actor,
            tables=tables,
            batch_size=args.batch_size,
            limit=args.limit,
            throttle_seconds=args.throttle,
            checkpoint_path=Path(args.checkpoint),
            resume=not args.no_resume,
        )
    if args.json:
        print(json.dumps(report.__dict__, indent=2, default=str))
    else:
        print("\n".join(report.summary_lines()))
    return 0


async def cmd_convert_live(args: argparse.Namespace) -> int:
    actor = await _get_actor()
    checkpoint = Path(args.checkpoint)
    report = await run_conversion_live(
        actor,
        batch_size=args.batch_size,
        limit=args.limit,
        checkpoint_path=checkpoint,
        resume=not args.no_resume,
    )
    if args.json:
        payload = {
            "generated_at": report.generated_at,
            "organization_id": report.organization_id,
            "messages_scanned": report.messages_scanned,
            "messages_converted": report.messages_converted,
            "image_refs_written": report.image_refs_written,
            "completed": report.completed,
            "checkpoint_path": report.checkpoint_path,
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
    convert.add_argument(
        "--i-have-a-snapshot",
        action="store_true",
        help="Required for live conversion: confirms a Postgres snapshot was taken",
    )
    convert.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Checkpoint file for resume (default: ~/.letta/uplift_part1_checkpoint.json)",
    )
    convert.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing checkpoint and start from the beginning",
    )
    convert.set_defaults(func=None)

    enrich = sub.add_parser("enrich-pending", help="Part 1: 1MP + captions + pixel embed for pending images")
    enrich.add_argument("--dry-run", action="store_true", help="Count pending images only; no API calls")
    enrich.add_argument("--throttle", type=float, default=0.5, help="Seconds after each image completes (0 to disable)")
    enrich.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Parallel enrichment workers (VLM + embed are I/O bound; try 4-8)",
    )
    enrich.set_defaults(func=cmd_enrich_pending)

    reembed = sub.add_parser("reembed", help="Part 2: re-embed passages and file archives")
    reembed.add_argument("--dry-run", action="store_true", help="Count rows needing uplift only")
    reembed.add_argument(
        "--table",
        default="all",
        choices=["all", "archival_passages", "source_passages", "file_archives"],
        help="Table(s) to re-embed (default: all passage/archive tables)",
    )
    reembed.add_argument("--throttle", type=float, default=0.25, help="Seconds between embed batches (0 to disable)")
    reembed.add_argument(
        "--checkpoint",
        default=str(DEFAULT_PART2_CHECKPOINT_PATH),
        help="Checkpoint file for resume (default: ~/.letta/uplift_part2_checkpoint.json)",
    )
    reembed.add_argument("--no-resume", action="store_true", help="Ignore checkpoint and start from the beginning")
    reembed.set_defaults(func=cmd_reembed)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.command == "convert":
        if args.dry_run:
            return asyncio.run(cmd_convert_dry_run(args))
        if not args.i_have_a_snapshot:
            print(
                "Live conversion mutates messages.content. Take a Postgres snapshot first, then re-run with "
                "--i-have-a-snapshot",
                file=sys.stderr,
            )
            return 2
        return asyncio.run(cmd_convert_live(args))

    return asyncio.run(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
