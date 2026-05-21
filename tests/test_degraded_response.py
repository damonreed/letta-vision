import pytest

from letta.errors import LLMEmptyResponseError
from letta.llm_api.degraded_response import (
    classify_degraded_streaming_completion,
    is_openrouter_meta_failure,
    validate_streaming_completion_or_raise,
)
from letta.schemas.enums import ProviderType
from letta.schemas.letta_message_content import TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.openai.chat_completion_response import FunctionCall, ToolCall
from letta.schemas.usage import LettaUsageStatistics


def _openrouter_config() -> LLMConfig:
    return LLMConfig(
        model="moonshotai/kimi-k2.6",
        model_endpoint_type=ProviderType.openrouter,
        model_endpoint="https://openrouter.ai/api/v1",
        context_window=128000,
        provider_name="openrouter",
    )


def test_openrouter_zero_tokens_with_generation_id_is_degraded():
    cfg = _openrouter_config()
    assert is_openrouter_meta_failure(
        llm_config=cfg,
        raw_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        response_id="gen-1779367060-abc",
        usage=LettaUsageStatistics(),
        has_tool_calls=False,
    )


def test_openrouter_with_tool_call_is_not_meta_failure():
    cfg = _openrouter_config()
    assert not is_openrouter_meta_failure(
        llm_config=cfg,
        raw_usage={"prompt_tokens": 0, "completion_tokens": 0},
        response_id="gen-abc",
        usage=LettaUsageStatistics(),
        has_tool_calls=True,
    )


def test_meaningful_content_is_not_degraded():
    degraded, reason = classify_degraded_streaming_completion(
        llm_config=_openrouter_config(),
        tool_calls=[],
        tool_call=None,
        content_parts=[TextContent(text="hello")],
        raw_usage={"prompt_tokens": 0, "completion_tokens": 0, "id": "gen-x"},
        usage=LettaUsageStatistics(),
        response_id="gen-x",
    )
    assert not degraded
    assert reason == ""


def test_empty_stream_raises_llm_empty_response_error():
    with pytest.raises(LLMEmptyResponseError) as exc_info:
        validate_streaming_completion_or_raise(
            llm_config=_openrouter_config(),
            tool_calls=[],
            tool_call=None,
            content_parts=[],
            raw_usage={"prompt_tokens": 0, "completion_tokens": 0, "id": "gen-1779367060-test"},
            usage=LettaUsageStatistics(),
            response_id="gen-1779367060-test",
            finish_reason="stop",
        )
    assert exc_info.value.details["degraded_reason"] == "openrouter_zero_tokens_with_generation_id"


def test_package_llm_failure_notice_message_includes_stats():
    from letta.system import package_llm_failure_notice_message

    packed = package_llm_failure_notice_message(
        timezone="UTC",
        failure_kind="llm_api_error",
        human_message="Provider stream failed.",
        attempts=1,
        model="moonshotai/kimi-k2.6",
        handle="openrouter/moonshotai/kimi-k2.6",
        error_type="llm_bad_request",
        error_class="LLMBadRequestError",
        detail="JSON error injected into SSE stream",
    )
    import json

    data = json.loads(packed)
    assert data["type"] == "system_alert"
    assert data["llm_failure_stats"]["failure_kind"] == "llm_api_error"
    assert "Provider stream failed" in data["message"]


def test_package_llm_degraded_failure_message_includes_stats():
    from letta.system import package_llm_degraded_failure_message

    packed = package_llm_degraded_failure_message(
        timezone="UTC",
        attempts=3,
        model="moonshotai/kimi-k2.6",
        handle="openrouter/moonshotai/kimi-k2.6",
        degraded_reason="openrouter_zero_tokens_with_generation_id",
        generation_id="gen-test",
    )
    import json

    data = json.loads(packed)
    assert data["type"] == "system_alert"
    assert data["degraded_failure_stats"]["attempts"] == 3
    assert "3 attempt" in data["message"]


def test_tool_call_stream_is_valid():
    tool = ToolCall(id="call_1", function=FunctionCall(name="memory", arguments="{}"))
    validate_streaming_completion_or_raise(
        llm_config=_openrouter_config(),
        tool_calls=[tool],
        tool_call=tool,
        content_parts=[],
        raw_usage={"prompt_tokens": 0, "completion_tokens": 0, "id": "gen-x"},
        usage=LettaUsageStatistics(),
        response_id="gen-x",
    )
