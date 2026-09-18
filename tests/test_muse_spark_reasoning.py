import pytest

from letta.llm_api.openai_client import (
    OpenAIClient,
    _openrouter_reasoning_effort,
)
from letta.schemas.enums import AgentType, MessageRole, ProviderCategory
from letta.schemas.letta_message_content import TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message
from letta.schemas.providers.openrouter import OpenRouterProvider


def _contributor_config(**kwargs) -> LLMConfig:
    defaults = dict(
        model="meta/muse-spark-1.3-contributor",
        model_endpoint_type="openrouter",
        model_endpoint="https://openrouter.ai/api/v1",
        handle="openrouter/meta/muse-spark-1.3-contributor",
        provider_name="openrouter",
        context_window=1048576,
    )
    defaults.update(kwargs)
    return LLMConfig(**defaults)


def test_openrouter_reasoning_effort_contributor_defaults_and_remaps_max():
    model = "meta/muse-spark-1.3-contributor"
    assert _openrouter_reasoning_effort(model, None) == "xhigh"
    assert _openrouter_reasoning_effort(model, "max") == "xhigh"
    assert _openrouter_reasoning_effort(model, "high") == "high"


def test_openrouter_reasoning_effort_leaves_other_models_alone():
    assert _openrouter_reasoning_effort("meta/muse-spark-1.3", None) is None
    assert _openrouter_reasoning_effort("meta/muse-spark-1.3", "max") == "max"
    assert _openrouter_reasoning_effort("moonshotai/kimi-k2.6", None) is None


def test_muse_spark_is_openrouter_reasoning_model():
    config = _contributor_config()
    assert LLMConfig.is_openrouter_reasoning_model(config)


def test_muse_spark_contributor_defaults_to_xhigh():
    config = _contributor_config()
    for agent_type in (AgentType.letta_v1_agent, AgentType.memgpt_v2_agent):
        for reasoning in (True, False):
            updated = LLMConfig.apply_reasoning_setting_to_config(
                config.model_copy(), reasoning=reasoning, agent_type=agent_type
            )
            assert updated.enable_reasoner is True
            assert updated.put_inner_thoughts_in_kwargs is False
            assert updated.reasoning_effort == "xhigh"


def test_muse_spark_contributor_request_sends_xhigh():
    client = OpenAIClient()
    llm_config = _contributor_config(enable_reasoner=True)
    messages = [
        Message(
            role=MessageRole.user,
            content=[TextContent(text="hello")],
            agent_id="agent-abc",
        )
    ]

    request_data = client.build_request_data(
        agent_type=AgentType.letta_v1_agent,
        messages=messages,
        llm_config=llm_config,
        tools=[],
    )

    assert request_data["extra_body"]["reasoning"]["effort"] == "xhigh"


@pytest.mark.asyncio
async def test_openrouter_list_stamps_contributor_xhigh(monkeypatch):
    provider = OpenRouterProvider(
        name="openrouter",
        provider_category=ProviderCategory.base,
        base_url="https://openrouter.ai/api/v1",
    )

    async def mock_list(*_args, **_kwargs):
        return {
            "data": [
                {"id": "meta/muse-spark-1.3-contributor", "context_length": 1048576},
                {"id": "meta/muse-spark-1.3", "context_length": 1048576},
                {"id": "moonshotai/kimi-k2.6", "context_length": 128000},
            ]
        }

    monkeypatch.setattr("letta.llm_api.openai.openai_get_model_list_async", mock_list)
    configs = {c.model: c for c in await provider.list_llm_models_async()}

    contributor = configs["meta/muse-spark-1.3-contributor"]
    assert contributor.reasoning_effort == "xhigh"
    assert contributor.enable_reasoner is True

    standard = configs["meta/muse-spark-1.3"]
    assert standard.reasoning_effort is None
    assert standard.enable_reasoner is True

    kimi = configs["moonshotai/kimi-k2.6"]
    assert kimi.reasoning_effort is None
