"""MiniMax OpenAI-compatible API helpers."""

from letta.constants import LLM_MAX_CONTEXT_WINDOW
from letta.llm_api.minimax_openai import (
    apply_minimax_openai_request_extras,
    extract_reasoning_from_message_data,
    is_minimax_openai_compatible,
    minimax_image_detail_for_request,
    normalize_minimax_openai_request_images,
)
from letta.llm_api.model_registry import model_supports_vision
from letta.model_specs.litellm_model_specs import normalize_model_basename
from letta.schemas.enums import ProviderCategory
from letta.schemas.llm_config import LLMConfig
from letta.schemas.providers.openai import OpenAIProvider


def _llm_config(**kwargs) -> LLMConfig:
    defaults = dict(
        model="MiniMax-M3",
        model_endpoint_type="openai",
        model_endpoint="https://api.minimax.io/v1",
        context_window=30000,
        handle="openai-proxy/minimax/MiniMax-M3",
        provider_name="Minimax",
    )
    defaults.update(kwargs)
    return LLMConfig(**defaults)


def test_minimax_openai_detected_from_endpoint():
    assert is_minimax_openai_compatible(_llm_config())


def test_minimax_openai_not_detected_for_unrelated_provider():
    assert not is_minimax_openai_compatible(
        _llm_config(
            model="gpt-4o",
            model_endpoint="https://api.openai.com/v1",
            handle="openai/gpt-4o",
            provider_name="openai",
        )
    )


def test_reasoning_split_extra_body():
    request_data = {"model": "MiniMax-M3", "messages": []}
    apply_minimax_openai_request_extras(request_data, _llm_config())
    assert request_data["extra_body"]["reasoning_split"] is True


def test_minimax_image_detail_maps_auto_to_default():
    assert minimax_image_detail_for_request("auto") == "default"
    assert minimax_image_detail_for_request(None) == "default"
    assert minimax_image_detail_for_request("high") == "high"


def test_normalize_minimax_openai_request_images():
    request_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc", "detail": "auto"},
                    },
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,xyz"},
                    },
                ],
            },
        ],
    }
    normalize_minimax_openai_request_images(request_data)
    user_image = request_data["messages"][0]["content"][1]["image_url"]
    assert user_image["detail"] == "default"
    tool_image = request_data["messages"][1]["content"][0]["image_url"]
    assert tool_image["detail"] == "default"


def test_apply_minimax_extras_normalizes_images():
    request_data = {
        "model": "MiniMax-M3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc", "detail": "auto"},
                    },
                ],
            },
        ],
    }
    apply_minimax_openai_request_extras(request_data, _llm_config())
    assert request_data["messages"][0]["content"][0]["image_url"]["detail"] == "default"
    assert request_data["extra_body"]["reasoning_split"] is True


def test_extract_reasoning_details():
    text = extract_reasoning_from_message_data(
        {
            "reasoning_details": [
                {"type": "reasoning.text", "text": "Step one."},
                {"type": "reasoning.text", "text": "Step two."},
            ]
        }
    )
    assert text == "Step one.\nStep two."


def test_context_window_basename_for_m3():
    assert normalize_model_basename("MiniMax-M3") == "minimax-m3"
    assert LLM_MAX_CONTEXT_WINDOW["minimax-m3"] == 1_000_000


def test_vision_registry_m3_byok_handle():
    assert model_supports_vision(
        "MiniMax-M3",
        handle="openai-proxy/minimax/MiniMax-M3",
    )


def test_byok_sync_context_window_for_m3():
    provider = OpenAIProvider(
        name="Minimax",
        provider_category=ProviderCategory.byok,
        base_url="https://api.minimax.io/v1",
        api_key="test-key",
    )
    _, context = provider._do_model_checks_for_name_and_context_size({"id": "MiniMax-M3"})
    assert context == 1_000_000
