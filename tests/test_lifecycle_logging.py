from letta.observability.lifecycle_logging import extract_provider_completion_metadata, merge_step_usage_with_provider
from letta.schemas.usage import LettaUsageStatistics


def test_extract_openrouter_usage_metadata():
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cost": 0.0025,
        "is_byok": False,
        "prompt_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 5},
        "cost_details": {
            "upstream_inference_prompt_cost": 0.001,
            "upstream_inference_completions_cost": 0.0015,
        },
    }
    meta = extract_provider_completion_metadata(
        usage,
        response_model="moonshotai/kimi-k2.6",
        generation_id="gen-abc123",
        finish_reason="tool_calls",
    )
    assert meta["response_model"] == "moonshotai/kimi-k2.6"
    assert meta["generation_id"] == "gen-abc123"
    assert meta["finish_reason"] == "tool_calls"
    assert meta["cost"] == 0.0025
    assert meta["prompt_tokens_details.cached_tokens"] == 10
    assert meta["completion_tokens_details.reasoning_tokens"] == 5
    assert meta["cost_details.upstream_inference_prompt_cost"] == 0.001


def test_merge_step_usage_with_provider_when_step_empty():
    step_usage = LettaUsageStatistics()
    provider = {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160, "cost": 0.01}
    merged = merge_step_usage_with_provider(step_usage, provider)
    assert merged.prompt_tokens == 120
    assert merged.completion_tokens == 40
    assert merged.total_tokens == 160


def test_extract_provider_completion_metadata_ignores_nested_blobs():
    usage = {"cost": 1.0, "prompt_tokens_details": {"ignored": {"nested": True}}}
    meta = extract_provider_completion_metadata(usage)
    assert meta["cost"] == 1.0
    assert "prompt_tokens_details.ignored" not in meta
