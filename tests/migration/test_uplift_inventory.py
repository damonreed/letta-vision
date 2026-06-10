from letta.services.migration.block_classifier import MessageScanStats
from letta.services.migration.uplift_cost import Part2Inventory, TableUpliftCounts, estimate_uplift_cost


def test_estimate_uplift_cost_sums_rows():
    part2 = Part2Inventory(
        target_space_id="abc123",
        deployment_handle="openrouter/google/gemini-embedding-2-preview",
        archival_passages=TableUpliftCounts(needs_uplift=10, total_rows=10),
        source_passages=TableUpliftCounts(needs_uplift=5, total_rows=5),
        file_archives=TableUpliftCounts(needs_uplift=3, total_rows=3),
        messages=TableUpliftCounts(needs_uplift=100, total_rows=200),
        images=TableUpliftCounts(needs_uplift=4, total_rows=4),
        images_pending_or_failed=7,
    )
    part1 = MessageScanStats()
    cost = estimate_uplift_cost(part2, part1, include_caption_cost=True)

    assert cost.passage_embed_calls == 15
    assert cost.file_archive_embed_calls == 3
    assert cost.message_embed_calls == 100
    assert cost.image_pixel_embed_calls == 4
    assert cost.image_vlm_caption_calls == 7
    assert cost.estimated_tokens > 0
    assert cost.estimated_usd > 0


def test_estimate_uplift_cost_skip_caption():
    part2 = Part2Inventory(
        target_space_id="abc123",
        deployment_handle="handle",
        images_pending_or_failed=50,
    )
    cost = estimate_uplift_cost(part2, MessageScanStats(), include_caption_cost=False)
    assert cost.image_vlm_caption_calls == 0
