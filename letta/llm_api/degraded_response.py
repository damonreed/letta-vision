"""Detect OpenRouter / streaming meta-failures (HTTP 200 with no usable output)."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from letta.errors import (
    ErrorCode,
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMEmptyResponseError,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
)
from letta.llm_api.error_utils import is_openrouter_image_payload_limit_message, openrouter_image_payload_limit_user_message
from letta.schemas.enums import ProviderType
from letta.schemas.letta_message_content import (
    ReasoningContent,
    RedactedReasoningContent,
    SummarizedReasoningContent,
    TextContent,
)
from letta.schemas.llm_config import LLMConfig
from letta.schemas.usage import LettaUsageStatistics

_TOKEN_COUNT_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
)


def is_openrouter_llm_config(llm_config: LLMConfig) -> bool:
    if llm_config.provider_name == "openrouter":
        return True
    if llm_config.model_endpoint_type == ProviderType.openrouter:
        return True
    endpoint = llm_config.model_endpoint or ""
    return "openrouter.ai" in endpoint


def _generation_id(raw_usage: Optional[Mapping[str, Any]], response_id: Optional[str]) -> Optional[str]:
    if raw_usage:
        rid = raw_usage.get("id")
        if isinstance(rid, str) and rid:
            return rid
    if response_id:
        return str(response_id)
    return None


def provider_reported_tokens_all_zero(
    raw_usage: Optional[Mapping[str, Any]],
    usage: Optional[LettaUsageStatistics] = None,
) -> bool:
    """True when the provider usage object reports no billable tokens."""
    if raw_usage:
        values = [raw_usage.get(k) for k in _TOKEN_COUNT_KEYS]
        if any(v for v in values if v):
            return False
        # Explicit zeros or missing keys — treat as zero for OpenRouter pending detection.
        return True
    if usage is None:
        return True
    return (usage.prompt_tokens or 0) == 0 and (usage.completion_tokens or 0) == 0


def is_openrouter_meta_failure(
    *,
    llm_config: LLMConfig,
    raw_usage: Optional[Mapping[str, Any]],
    response_id: Optional[str],
    usage: Optional[LettaUsageStatistics],
    has_tool_calls: bool,
) -> bool:
    """HTTP 200 with a generation id but zero provider-reported tokens (Cloudflare edge case)."""
    if not is_openrouter_llm_config(llm_config) or has_tool_calls:
        return False
    gen_id = _generation_id(raw_usage, response_id)
    if not gen_id:
        return False
    return provider_reported_tokens_all_zero(raw_usage, usage)


def _has_meaningful_content(
    content_parts: Optional[Sequence[Any]],
    reasoning_content: Optional[Sequence[Any]],
) -> bool:
    for parts in (content_parts, reasoning_content):
        if not parts:
            continue
        for part in parts:
            if isinstance(part, TextContent):
                if part.text and part.text.strip():
                    return True
            elif isinstance(part, ReasoningContent):
                if part.reasoning and str(part.reasoning).strip():
                    return True
            elif isinstance(part, SummarizedReasoningContent):
                if part.text and part.text.strip():
                    return True
            elif isinstance(part, RedactedReasoningContent):
                return True
            elif hasattr(part, "text"):
                text = getattr(part, "text", None)
                if text and str(text).strip():
                    return True
    return False


def classify_degraded_streaming_completion(
    *,
    llm_config: LLMConfig,
    tool_calls: Sequence[Any],
    tool_call: Any | None,
    content_parts: Optional[Sequence[Any]] = None,
    reasoning_content: Optional[Sequence[Any]] = None,
    usage: Optional[LettaUsageStatistics] = None,
    raw_usage: Optional[Mapping[str, Any]] = None,
    response_id: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> tuple[bool, str]:
    has_tool_calls = bool(tool_calls) or tool_call is not None
    if has_tool_calls:
        return False, ""

    if _has_meaningful_content(content_parts, reasoning_content):
        return False, ""

    if is_openrouter_meta_failure(
        llm_config=llm_config,
        raw_usage=raw_usage,
        response_id=response_id,
        usage=usage,
        has_tool_calls=False,
    ):
        return True, "openrouter_zero_tokens_with_generation_id"

    return True, "empty_content_no_tool_calls"


def validate_streaming_completion_or_raise(
    *,
    llm_config: LLMConfig,
    tool_calls: Sequence[Any],
    tool_call: Any | None,
    content_parts: Optional[Sequence[Any]] = None,
    reasoning_content: Optional[Sequence[Any]] = None,
    usage: Optional[LettaUsageStatistics] = None,
    raw_usage: Optional[Mapping[str, Any]] = None,
    response_id: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> None:
    """Raise LLMEmptyResponseError when the stream completed without usable output."""
    degraded, reason = classify_degraded_streaming_completion(
        llm_config=llm_config,
        tool_calls=tool_calls,
        tool_call=tool_call,
        content_parts=content_parts,
        reasoning_content=reasoning_content,
        usage=usage,
        raw_usage=raw_usage,
        response_id=response_id,
        finish_reason=finish_reason,
    )
    if not degraded:
        return

    gen_id = _generation_id(raw_usage, response_id)
    model = None
    if raw_usage:
        model = raw_usage.get("model")
    message = (
        f"LLM stream completed without usable output ({reason}) "
        f"(model={model or llm_config.model}, generation_id={gen_id}, finish_reason={finish_reason})"
    )
    raise LLMEmptyResponseError(
        message=message,
        code=ErrorCode.INTERNAL_SERVER_ERROR,
        details={
            "degraded_reason": reason,
            "generation_id": gen_id,
            "model": model or llm_config.model,
            "finish_reason": finish_reason,
            "provider_name": llm_config.provider_name,
            "handle": llm_config.handle,
        },
    )


def llm_failure_error_type(error: LLMError) -> str:
    if isinstance(error, LLMEmptyResponseError):
        return "llm_empty_response"
    if isinstance(error, LLMAuthenticationError):
        return "llm_authentication"
    if isinstance(error, LLMRateLimitError):
        return "llm_rate_limit"
    if isinstance(error, LLMServerError):
        return "llm_server_error"
    if isinstance(error, LLMBadRequestError):
        return "llm_bad_request"
    return "llm_api_error"


def friendly_llm_error_message(error: LLMError) -> str:
    """Short operator-facing text for chat/SSE surfaces."""
    message = str(error.message) if getattr(error, "message", None) else str(error)
    detail = None
    if isinstance(error.details, dict):
        detail = error.details.get("detail") or error.details.get("body")
        if isinstance(detail, dict):
            detail = str(detail)
    haystack = f"{message} {detail or ''}"
    if "JSON error injected into SSE stream" in haystack:
        return (
            "The model provider sent a broken streaming response (OpenRouter/upstream). "
            "This is usually temporary — try sending again."
        )
    if "Upstream idle timeout exceeded" in haystack:
        return (
            "The model stream timed out during a long response. "
            "Try again, or use a shorter prompt / smaller image."
        )
    if isinstance(error, LLMBadRequestError) and isinstance(error.details, dict):
        if error.details.get("error_kind") == "openrouter_image_payload_limit":
            return openrouter_image_payload_limit_user_message()
    if is_openrouter_image_payload_limit_message(haystack):
        return openrouter_image_payload_limit_user_message()
    return message


def human_message_for_llm_failure(error: LLMError, *, attempts: Optional[int] = None) -> str:
    if isinstance(error, LLMEmptyResponseError):
        n = attempts or 1
        return (
            f"Note: The model provider failed to return a usable response after {n} attempt(s). "
            "This often indicates a temporary upstream outage (for example at OpenRouter or its CDN edge) "
            "rather than a problem with your request. No assistant reply was produced for this step."
        )
    friendly = friendly_llm_error_message(error)
    return (
        "Note: The model provider returned an error and this step could not complete. "
        f"{friendly}"
    )
