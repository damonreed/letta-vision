"""Heuristic cost estimates for historic embedding uplift (no DB)."""

from __future__ import annotations

from dataclasses import dataclass, field

from letta.services.migration.block_classifier import MessageScanStats

_DEFAULT_EMBED_COST_PER_1M_TOKENS = 0.10
_DEFAULT_IMAGE_EMBED_COST_PER_CALL = 0.0001
_DEFAULT_VLM_CAPTION_COST_PER_CALL = 0.002


@dataclass
class TableUpliftCounts:
    needs_uplift: int = 0
    legacy_unknown_space: int = 0
    legacy_4096_only: int = 0
    total_rows: int = 0


@dataclass
class Part2Inventory:
    target_space_id: str = ""
    deployment_handle: str = ""
    archival_passages: TableUpliftCounts = field(default_factory=TableUpliftCounts)
    source_passages: TableUpliftCounts = field(default_factory=TableUpliftCounts)
    file_archives: TableUpliftCounts = field(default_factory=TableUpliftCounts)
    messages: TableUpliftCounts = field(default_factory=TableUpliftCounts)
    images: TableUpliftCounts = field(default_factory=TableUpliftCounts)
    images_pending_or_failed: int = 0


@dataclass
class CostEstimate:
    passage_embed_calls: int = 0
    file_archive_embed_calls: int = 0
    message_embed_calls: int = 0
    image_pixel_embed_calls: int = 0
    image_vlm_caption_calls: int = 0
    estimated_tokens: int = 0
    estimated_usd: float = 0.0


def estimate_uplift_cost(part2: Part2Inventory, part1: MessageScanStats, *, include_caption_cost: bool = True) -> CostEstimate:
    passage_rows = part2.archival_passages.needs_uplift + part2.source_passages.needs_uplift
    archive_rows = part2.file_archives.needs_uplift
    message_rows = part2.messages.needs_uplift
    image_rows = part2.images.needs_uplift

    estimated_tokens = passage_rows * 500 + archive_rows * 600 + message_rows * 300

    passage_embed_calls = passage_rows
    file_archive_embed_calls = archive_rows
    message_embed_calls = message_rows
    image_pixel_embed_calls = image_rows
    image_vlm_caption_calls = part2.images_pending_or_failed if include_caption_cost else 0

    embed_token_cost = (estimated_tokens / 1_000_000) * _DEFAULT_EMBED_COST_PER_1M_TOKENS
    image_embed_cost = image_pixel_embed_calls * _DEFAULT_IMAGE_EMBED_COST_PER_CALL
    vlm_cost = image_vlm_caption_calls * _DEFAULT_VLM_CAPTION_COST_PER_CALL

    return CostEstimate(
        passage_embed_calls=passage_embed_calls,
        file_archive_embed_calls=file_archive_embed_calls,
        message_embed_calls=message_embed_calls,
        image_pixel_embed_calls=image_pixel_embed_calls,
        image_vlm_caption_calls=image_vlm_caption_calls,
        estimated_tokens=estimated_tokens,
        estimated_usd=round(embed_token_cost + image_embed_cost + vlm_cost, 4),
    )
