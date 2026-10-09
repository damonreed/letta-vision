"""OpenRouter `reasoning_details` (incl. encrypted reasoning) is captured and replayed to the same model."""

from types import SimpleNamespace

from letta.agents.letta_agent_v3 import LettaAgentV3
from letta.helpers.reasoning_details import merge_reasoning_details
from letta.interfaces.openai_streaming_interface import SimpleOpenAIStreamingInterface
from letta.llm_api.openai_client import OpenAIClient
from letta.schemas.enums import AgentType, MessageRole
from letta.schemas.letta_message_content import ReasoningContent, TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message

MUSE = "meta/muse-spark-1.3-contributor"
ENCRYPTED = {"type": "reasoning.encrypted", "data": "Q-PaDg-opaque", "format": "meta-responses-v1", "id": "rs_1", "index": 0}


def _openrouter_config(model: str = MUSE) -> LLMConfig:
    return LLMConfig(
        model=model,
        model_endpoint_type="openrouter",
        model_endpoint="https://openrouter.ai/api/v1",
        handle=f"openrouter/{model}",
        provider_name="openrouter",
        context_window=1048576,
        enable_reasoner=True,
    )


def _history(model: str = MUSE) -> list[Message]:
    return [
        Message(role=MessageRole.user, content=[TextContent(text="stamp the caption")], agent_id="agent-abc"),
        Message(
            role=MessageRole.assistant,
            model=model,
            agent_id="agent-abc",
            content=[ReasoningContent(is_native=True, reasoning="", reasoning_details=[ENCRYPTED])],
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "set_field", "arguments": '{"field":"caption"}'}}],
        ),
        Message(
            role=MessageRole.tool,
            agent_id="agent-abc",
            content=[TextContent(text='{"status": "OK"}')],
            tool_call_id="call_1",
        ),
    ]


def _assistant_rows(request_data: dict) -> list[dict]:
    return [m for m in request_data["messages"] if m["role"] == "assistant"]


def test_merge_concatenates_split_summary_and_keeps_encrypted():
    acc: list[dict] = []
    merge_reasoning_details(acc, [{"type": "reasoning.summary", "summary": "Plan A.", "format": "meta-responses-v1", "index": 0}])
    merge_reasoning_details(acc, [{"type": "reasoning.summary", "summary": "Then B.", "index": 0}])
    merge_reasoning_details(acc, [dict(ENCRYPTED, index=1)])
    assert acc == [
        {"type": "reasoning.summary", "summary": "Plan A.Then B.", "format": "meta-responses-v1", "index": 0},
        dict(ENCRYPTED, index=1),
    ]


def test_merge_keeps_unindexed_items_separate():
    acc: list[dict] = []
    merge_reasoning_details(acc, [{"type": "reasoning.text", "text": "a"}])
    merge_reasoning_details(acc, [{"type": "reasoning.text", "text": "b"}])
    assert [d["text"] for d in acc] == ["a", "b"]


def test_stream_content_keeps_encrypted_only_reasoning():
    interface = SimpleOpenAIStreamingInterface(model=MUSE)
    merge_reasoning_details(interface._reasoning_details, [ENCRYPTED])
    content = interface.get_content()
    assert len(content) == 1
    assert isinstance(content[0], ReasoningContent)
    assert content[0].reasoning == ""
    assert content[0].reasoning_details == [ENCRYPTED]


def test_openrouter_request_replays_details_for_same_model():
    request_data = OpenAIClient().build_request_data(
        agent_type=AgentType.letta_v1_agent, messages=_history(), llm_config=_openrouter_config(), tools=[]
    )
    (assistant,) = _assistant_rows(request_data)
    assert assistant["reasoning_details"] == [ENCRYPTED]
    assert "reasoning_content" not in assistant


def test_openrouter_request_drops_details_from_other_model():
    request_data = OpenAIClient().build_request_data(
        agent_type=AgentType.letta_v1_agent,
        messages=_history(model="z-ai/glm-5.3-flash"),
        llm_config=_openrouter_config(),
        tools=[],
    )
    (assistant,) = _assistant_rows(request_data)
    assert "reasoning_details" not in assistant


def test_non_openrouter_request_never_sends_details():
    config = LLMConfig(
        model=MUSE,
        model_endpoint_type="openai",
        model_endpoint="https://api.openai.com/v1",
        provider_name="openai",
        context_window=128000,
    )
    request_data = OpenAIClient().build_request_data(
        agent_type=AgentType.letta_v1_agent, messages=_history(), llm_config=config, tools=[]
    )
    (assistant,) = _assistant_rows(request_data)
    assert "reasoning_details" not in assistant


def test_final_step_notice_is_hidden_user_heartbeat():
    agent = LettaAgentV3.__new__(LettaAgentV3)
    agent.agent_state = SimpleNamespace(id="agent-abc", timezone="UTC", llm_config=_openrouter_config())
    notice = agent._final_step_notice(run_id="run-1")
    assert notice.role == MessageRole.user
    assert notice.id is not None
    assert "last step of this turn" in notice.content[0].text
