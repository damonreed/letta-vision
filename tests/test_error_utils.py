"""Unit tests for provider error classification helpers."""

import httpx
import openai
import pytest

from letta.errors import ContextWindowExceededError, LLMBadRequestError
from letta.llm_api.error_utils import (
    extract_openrouter_provider_name,
    is_ambiguous_provider_rejection_message,
    is_context_window_overflow_message,
)
from letta.llm_api.openai_client import OpenAIClient
from letta.schemas.enums import ProviderCategory
from letta.schemas.llm_config import LLMConfig


SAIL_AMBIGUOUS_OVERFLOW_OR_INVALID = (
    "Error code: 400 - {'error': {'message': 'Provider returned error', 'code': 400, "
    "'metadata': {'raw': '{\"error\":{\"code\":\"invalid_request_error\",\"message\":"
    "\"The request was rejected. Possible causes: input exceeds the model's maximum "
    "context length, or the request contains invalid parameters.\"}}', "
    "'provider_name': 'Sail Research', 'is_byok': False, "
    "'provider_error_code': 'invalid_request_error'}}, "
    "'user_id': 'user_3DXjxzcy2KQlDJkQBsJURK6Kyog'}"
)


@pytest.mark.parametrize(
    "message",
    [
        "Your input exceeds the context window of this model. Please adjust your input and try again.",
        "This model's maximum context length is 8192 tokens. However, your messages resulted in 8198 tokens.",
        "Error code: context_length_exceeded",
        "Input tokens exceed the configured limit of 128000.",
        "The input exceeds the maximum allowed input length.",
    ],
)
def test_is_context_window_overflow_message_true_for_definitive_overflow(message):
    assert is_context_window_overflow_message(message) is True
    assert is_ambiguous_provider_rejection_message(message) is False


def test_sail_ambiguous_rejection_is_not_context_overflow():
    assert is_ambiguous_provider_rejection_message(SAIL_AMBIGUOUS_OVERFLOW_OR_INVALID) is True
    assert is_context_window_overflow_message(SAIL_AMBIGUOUS_OVERFLOW_OR_INVALID) is False
    # Bare substring alone used to false-positive; keep that regression covered.
    assert is_context_window_overflow_message("maximum context length") is False


def test_extract_openrouter_provider_name_from_message_and_body():
    assert extract_openrouter_provider_name(SAIL_AMBIGUOUS_OVERFLOW_OR_INVALID) == "Sail Research"
    assert (
        extract_openrouter_provider_name(
            {
                "error": {
                    "message": "Provider returned error",
                    "metadata": {"provider_name": "Sail Research"},
                }
            }
        )
        == "Sail Research"
    )


def test_openai_client_maps_sail_ambiguous_error_to_bad_request_not_overflow():
    client = OpenAIClient()
    llm_config = LLMConfig(
        model="z-ai/glm-5.3-flash",
        model_endpoint_type="openai",
        model_endpoint="https://openrouter.ai/api/v1",
        provider_name="openrouter",
        handle="openrouter/z-ai/glm-5.3-flash",
        context_window=1048576,
        provider_category=ProviderCategory.base,
    )

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    response = httpx.Response(status_code=400, request=request)
    body = {
        "error": {
            "message": "Provider returned error",
            "code": 400,
            "metadata": {
                "raw": (
                    '{"error":{"code":"invalid_request_error","message":'
                    '"The request was rejected. Possible causes: input exceeds the model\'s '
                    'maximum context length, or the request contains invalid parameters."}}'
                ),
                "provider_name": "Sail Research",
                "is_byok": False,
                "provider_error_code": "invalid_request_error",
            },
        },
        "user_id": "user_test",
    }
    error = openai.BadRequestError(SAIL_AMBIGUOUS_OVERFLOW_OR_INVALID, response=response, body=body)
    result = client.handle_llm_error(error, llm_config=llm_config)

    assert isinstance(result, LLMBadRequestError)
    assert not isinstance(result, ContextWindowExceededError)
    assert result.details.get("error_kind") == "ambiguous_provider_rejection"
    assert result.details.get("upstream_provider") == "Sail Research"
