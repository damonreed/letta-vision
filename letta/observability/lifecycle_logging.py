"""Structured INFO logs for run/step lifecycle (metadata only, no prompts or tool payloads)."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from letta.log import get_logger
from letta.schemas.openai.chat_completion_response import (
    UsageStatistics,
    UsageStatisticsCompletionTokenDetails,
    UsageStatisticsPromptTokenDetails,
)
from letta.schemas.step import Step
from letta.schemas.step_metrics import StepMetrics
from letta.schemas.usage import LettaUsageStatistics

logger = get_logger(__name__)

# Scalar keys from OpenRouter / OpenAI-compatible usage objects (provider trace response.usage).
_USAGE_SCALAR_KEYS = (
    "cost",
    "is_byok",
    "total_tokens",
    "prompt_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
)

_USAGE_DETAIL_OBJECTS = (
    "prompt_tokens_details",
    "completion_tokens_details",
    "cost_details",
    "input_tokens_details",
    "output_tokens_details",
    "server_tool_use",
)


def extract_provider_completion_metadata(
    usage: Optional[Mapping[str, Any]],
    *,
    response_model: Optional[str] = None,
    generation_id: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> dict[str, Any]:
    """Pull safe completion fields from provider trace usage (e.g. OpenRouter raw_usage)."""
    out: dict[str, Any] = {}
    if response_model:
        out["response_model"] = response_model
    if generation_id:
        out["generation_id"] = generation_id
    if finish_reason:
        out["finish_reason"] = finish_reason

    if not usage:
        return out

    for key in _USAGE_SCALAR_KEYS:
        if key in usage and usage[key] is not None:
            out[key] = usage[key]

    for detail_key in _USAGE_DETAIL_OBJECTS:
        details = usage.get(detail_key)
        if not isinstance(details, dict):
            continue
        for sub_key, sub_val in details.items():
            if sub_val is None or isinstance(sub_val, (dict, list)):
                continue
            out[f"{detail_key}.{sub_key}"] = sub_val

    return out


_PROVIDER_LOG_ALWAYS = frozenset({"generation_id", "response_model", "finish_reason", "cost", "is_byok"})


def _provider_field_is_meaningful(key: str, value: Any) -> bool:
    if key in _PROVIDER_LOG_ALWAYS:
        return value is not None
    if isinstance(value, (int, float)):
        return value != 0
    return bool(value)


def merge_step_usage_with_provider(
    step_usage: LettaUsageStatistics,
    provider_completion: Optional[Mapping[str, Any]],
) -> LettaUsageStatistics:
    """Prefer provider-reported tokens when step usage is empty but OpenRouter usage is present."""
    if not provider_completion:
        return step_usage

    prompt = provider_completion.get("prompt_tokens")
    if prompt is None:
        prompt = provider_completion.get("input_tokens")
    completion = provider_completion.get("completion_tokens")
    if completion is None:
        completion = provider_completion.get("output_tokens")
    total = provider_completion.get("total_tokens")

    has_provider_tokens = any(v for v in (prompt, completion, total) if v)
    step_empty = (step_usage.prompt_tokens or 0) == 0 and (step_usage.completion_tokens or 0) == 0
    if not has_provider_tokens or not step_empty:
        return step_usage

    prompt_tokens = int(prompt or 0)
    completion_tokens = int(completion or 0)
    total_tokens = int(total) if total is not None else prompt_tokens + completion_tokens

    cached = provider_completion.get("prompt_tokens_details.cached_tokens")
    cache_write = provider_completion.get("prompt_tokens_details.cache_write_tokens")
    reasoning = provider_completion.get("completion_tokens_details.reasoning_tokens")

    return LettaUsageStatistics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=int(cached) if cached is not None else step_usage.cached_input_tokens,
        cache_write_tokens=int(cache_write) if cache_write is not None else step_usage.cache_write_tokens,
        reasoning_tokens=int(reasoning) if reasoning is not None else step_usage.reasoning_tokens,
        step_count=step_usage.step_count,
    )


def provider_completion_to_usage_statistics(
    provider_completion: Optional[Mapping[str, Any]],
) -> Optional[UsageStatistics]:
    """Build UsageStatistics for step persistence from provider completion metadata."""
    if not provider_completion:
        return None

    prompt = provider_completion.get("prompt_tokens")
    if prompt is None:
        prompt = provider_completion.get("input_tokens")
    completion = provider_completion.get("completion_tokens")
    if completion is None:
        completion = provider_completion.get("output_tokens")
    if prompt is None and completion is None and provider_completion.get("total_tokens") is None:
        return None

    prompt_tokens = int(prompt or 0)
    completion_tokens = int(completion or 0)
    total_tokens = int(provider_completion["total_tokens"]) if provider_completion.get("total_tokens") is not None else (
        prompt_tokens + completion_tokens
    )

    prompt_details = None
    cached = provider_completion.get("prompt_tokens_details.cached_tokens")
    cache_write = provider_completion.get("prompt_tokens_details.cache_write_tokens")
    if cached is not None or cache_write is not None:
        prompt_details = UsageStatisticsPromptTokenDetails(
            cached_tokens=int(cached) if cached is not None else None,
            cache_creation_tokens=int(cache_write) if cache_write is not None else None,
        )

    completion_details = None
    reasoning = provider_completion.get("completion_tokens_details.reasoning_tokens")
    if reasoning is not None:
        completion_details = UsageStatisticsCompletionTokenDetails(reasoning_tokens=int(reasoning))

    return UsageStatistics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=prompt_details,
        completion_tokens_details=completion_details,
    )


def _ns_to_ms(ns: Optional[int]) -> Optional[float]:
    if ns is None:
        return None
    return round(ns / 1_000_000, 2)


def _format_log_fields(fields: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(fields.keys()):
        value = fields[key]
        if value is None:
            continue
        if isinstance(value, bool):
            parts.append(f"{key}={value}")
        elif isinstance(value, (int, float)):
            parts.append(f"{key}={value}")
        elif isinstance(value, dict):
            for sub_key, sub_val in sorted(value.items()):
                if sub_val is None or isinstance(sub_val, (dict, list)):
                    continue
                parts.append(f"{key}.{sub_key}={sub_val}")
        elif isinstance(value, list):
            parts.append(f"{key}={','.join(str(v) for v in value)}")
        else:
            text = str(value)
            if len(text) > 240:
                text = text[:240] + "..."
            parts.append(f"{key}={text}")
    return " ".join(parts)


def log_run_lifecycle_started(
    *,
    run_id: str,
    agent_id: str,
    organization_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    background: Optional[bool] = None,
    trace_id: Optional[str] = None,
) -> None:
    logger.info(
        "[RUN_LIFECYCLE] event=started %s",
        _format_log_fields(
            {
                "run_id": run_id,
                "agent_id": agent_id,
                "organization_id": organization_id,
                "conversation_id": conversation_id,
                "background": background,
                "trace_id": trace_id,
            }
        ),
    )


def log_run_lifecycle_completed(
    *,
    run_id: str,
    agent_id: str,
    status: str,
    stop_reason: Optional[str] = None,
    organization_id: Optional[str] = None,
    ttft_ns: Optional[int] = None,
    total_duration_ns: Optional[int] = None,
    run_duration_ms: Optional[float] = None,
    num_steps: Optional[int] = None,
    trace_id: Optional[str] = None,
) -> None:
    logger.info(
        "[RUN_LIFECYCLE] event=completed %s",
        _format_log_fields(
            {
                "run_id": run_id,
                "agent_id": agent_id,
                "organization_id": organization_id,
                "status": status,
                "stop_reason": stop_reason,
                "ttft_ms": _ns_to_ms(ttft_ns),
                "total_duration_ms": _ns_to_ms(total_duration_ns),
                "run_duration_ms": run_duration_ms,
                "num_steps": num_steps,
                "trace_id": trace_id,
            }
        ),
    )


def log_step_lifecycle_started(
    *,
    step: Step,
) -> None:
    logger.info(
        "[STEP_LIFECYCLE] event=started %s",
        _format_log_fields(
            {
                "step_id": step.id,
                "run_id": step.run_id,
                "agent_id": step.agent_id,
                "trace_id": step.trace_id,
                "provider_name": step.provider_name,
                "model": step.model,
                "model_handle": step.model_handle,
                "status": step.status,
            }
        ),
    )


def log_step_lifecycle_completed(
    *,
    step: Step,
    step_metrics: Optional[StepMetrics] = None,
    provider_completion: Optional[Mapping[str, Any]] = None,
    tool_names: Optional[list[str]] = None,
    terminal_status: str = "success",
    stream_events_received: Optional[int] = None,
) -> None:
    """Single consolidated log after step row + step_metrics are persisted."""
    fields: dict[str, Any] = {
        "step_id": step.id,
        "run_id": step.run_id,
        "agent_id": step.agent_id,
        "trace_id": step.trace_id,
        "terminal_status": terminal_status,
        "step_status": step.status,
        "provider_name": step.provider_name,
        "model": step.model,
        "model_handle": step.model_handle,
        "stop_reason": step.stop_reason,
        "prompt_tokens": step.prompt_tokens,
        "completion_tokens": step.completion_tokens,
        "total_tokens": step.total_tokens,
        "cached_input_tokens": step.cached_input_tokens,
        "cache_write_tokens": step.cache_write_tokens,
        "reasoning_tokens": step.reasoning_tokens,
        "error_type": step.error_type,
    }

    if step_metrics is not None:
        fields.update(
            {
                "llm_request_ms": _ns_to_ms(step_metrics.llm_request_ns),
                "tool_execution_ms": _ns_to_ms(step_metrics.tool_execution_ns),
                "step_ms": _ns_to_ms(step_metrics.step_ns),
            }
        )

    if tool_names:
        fields["tool_names"] = tool_names
        fields["tool_count"] = len(tool_names)

    if stream_events_received is not None:
        fields["stream_events_received"] = stream_events_received

    if provider_completion:
        provider_token_fields = (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
        )
        provider_tokens_all_zero = all((provider_completion.get(k) or 0) == 0 for k in provider_token_fields)
        if provider_tokens_all_zero and provider_completion.get("generation_id"):
            fields["provider_tokens_pending"] = True

        for key, value in provider_completion.items():
            if value is None:
                continue
            if _provider_field_is_meaningful(key, value):
                fields[f"provider.{key}"] = value

    logger.info("[STEP_LIFECYCLE] event=completed %s", _format_log_fields(fields))
