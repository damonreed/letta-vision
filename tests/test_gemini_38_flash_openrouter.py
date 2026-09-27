import pytest

from letta.llm_api.openai_client import (
    GEMINI_38_FLASH_MAX_OUTPUT_TOKENS,
    OpenAIClient,
    gemini_38_flash_reasoning_effort,
)
from letta.schemas.enums import AgentType, MessageRole, ProviderCategory
from letta.schemas.letta_message_content import TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message
from letta.schemas.providers.openrouter import OpenRouterProvider


def _flash_config(**kwargs) -> LLMConfig:
    defaults = dict(
        model="google/gemini-3.8-flash",
        model_endpoint_type="openrouter",
        model_endpoint="https://openrouter.ai/api/v1",
        handle="openrouter/google/gemini-3.8-flash",
        provider_name="openrouter",
        context_window=1_048_576,
        max_tokens=16384,
        temperature=0.7,
        max_reasoning_tokens=1024,
    )
    defaults.update(kwargs)
    return LLMConfig(**defaults)


def _request(llm_config: LLMConfig) -> dict:
    client = OpenAIClient()
    messages = [
        Message(
            role=MessageRole.user,
            content=[TextContent(text="hello")],
            agent_id="agent-abc",
        )
    ]
    return client.build_request_data(
        agent_type=AgentType.letta_v1_agent,
        messages=messages,
        llm_config=llm_config,
        tools=[],
    )


def test_gemini_38_flash_effort_is_always_high():
    for effort in (None, "medium", "low", "high", "xhigh", "max", "minimal", "none"):
        assert gemini_38_flash_reasoning_effort(effort, enabled=True) == "high"
        assert gemini_38_flash_reasoning_effort(effort, enabled=False) == "high"


def test_gemini_38_flash_request_uses_thinking_level_not_budget():
    request_data = _request(_flash_config(enable_reasoner=True, reasoning_effort=None))

    reasoning = request_data["extra_body"]["reasoning"]
    assert reasoning == {"effort": "high"}
    assert "temperature" not in request_data
    assert "top_p" not in request_data
    assert request_data["max_completion_tokens"] == GEMINI_38_FLASH_MAX_OUTPUT_TOKENS
    assert request_data["model"] == "google/gemini-3.8-flash"


def test_gemini_38_flash_request_keeps_explicit_output_cap_and_forces_high_effort():
    request_data = _request(
        _flash_config(enable_reasoner=False, reasoning_effort="low", max_tokens=8192, temperature=1.0)
    )

    assert request_data["extra_body"]["reasoning"] == {"effort": "high"}
    assert request_data["max_completion_tokens"] == 8192
    assert "temperature" not in request_data


def test_gemini_38_flash_disabled_reasoning_drops_to_low():
    config = _flash_config(enable_reasoner=True, reasoning_effort="high")
    updated = LLMConfig.apply_reasoning_setting_to_config(
        config.model_copy(), reasoning=False, agent_type=AgentType.letta_v1_agent
    )
    assert updated.enable_reasoner is True
    assert updated.put_inner_thoughts_in_kwargs is False
    assert updated.reasoning_effort == "high"

    request_data = _request(updated)
    assert request_data["extra_body"]["reasoning"] == {"effort": "high"}


def test_gemini_38_flash_reasoning_setting_is_always_high():
    config = _flash_config(reasoning_effort="minimal")
    for agent_type in (AgentType.letta_v1_agent, AgentType.memgpt_v2_agent):
        for reasoning in (True, False):
            updated = LLMConfig.apply_reasoning_setting_to_config(
                config.model_copy(), reasoning=reasoning, agent_type=agent_type
            )
            assert updated.enable_reasoner is True
            assert updated.reasoning_effort == "high"


@pytest.mark.asyncio
async def test_openrouter_list_stamps_gemini_38_flash(monkeypatch):
    provider = OpenRouterProvider(
        name="openrouter",
        provider_category=ProviderCategory.base,
        base_url="https://openrouter.ai/api/v1",
    )

    async def mock_list(*_args, **_kwargs):
        return {
            "data": [
                {"id": "google/gemini-3.8-flash", "context_length": 1048576},
                {"id": "google/gemini-2.5-flash", "context_length": 1048576},
            ]
        }

    monkeypatch.setattr("letta.llm_api.openai.openai_get_model_list_async", mock_list)
    configs = {c.model: c for c in await provider.list_llm_models_async()}

    flash = configs["google/gemini-3.8-flash"]
    assert flash.reasoning_effort == "high"
    assert flash.enable_reasoner is True
    assert flash.max_tokens == GEMINI_38_FLASH_MAX_OUTPUT_TOKENS

    older = configs["google/gemini-2.5-flash"]
    assert older.reasoning_effort is None
    assert older.max_tokens != GEMINI_38_FLASH_MAX_OUTPUT_TOKENS
