"""Tests for in-band </thinking> split in reasoning_content (Aion/OpenRouter)."""

from letta.helpers.thinking_tags import ThinkingCloseSplitBuffer, split_reasoning_at_thinking_close
from letta.interfaces.openai_streaming_interface import SimpleOpenAIStreamingInterface
from letta.schemas.letta_message import AssistantMessage, ReasoningMessage
from letta.schemas.letta_message_content import ReasoningContent, TextContent
from letta.schemas.message import Message


def test_split_reasoning_at_thinking_close_basic():
    reasoning, response = split_reasoning_at_thinking_close(
        "Plan the reply carefully.</thinking>\n*the first kiss lands*"
    )
    assert reasoning == "Plan the reply carefully."
    assert response == "*the first kiss lands*"


def test_split_reasoning_at_thinking_close_strips_open_tag():
    reasoning, response = split_reasoning_at_thinking_close(
        "<thinking>\nInner monologue.\n</thinking>\nVisible reply."
    )
    assert reasoning == "Inner monologue."
    assert response == "Visible reply."


def test_split_reasoning_at_thinking_close_noop_without_tag():
    text = "Normal reasoning without a close tag."
    reasoning, response = split_reasoning_at_thinking_close(text)
    assert reasoning == text
    assert response is None


def test_thinking_close_split_buffer_across_chunks():
    buf = ThinkingCloseSplitBuffer()
    r1, a1 = buf.feed("Thinking about this.</thin")
    assert a1 == ""
    assert r1 == "Thinking about this."
    r2, a2 = buf.feed("king>\nHello there.")
    assert r2 == ""
    assert a2 == "\nHello there."
    assert buf.closed is True
    r3, a3 = buf.feed(" More text.")
    assert r3 == ""
    assert a3 == " More text."


def test_get_content_splits_leaked_response_from_reasoning():
    interface = SimpleOpenAIStreamingInterface(model="aion-labs/aion-3.0")
    interface.content_messages = [
        ReasoningMessage(
            id=interface.letta_message_id,
            date="2026-07-12T00:00:00+00:00",
            source="reasoner_model",
            reasoning="Rooftop City metaphor.</thinking>\n*the first kiss lands and I don't move away.*",
        )
    ]
    content = interface.get_content()
    assert len(content) == 2
    assert isinstance(content[0], ReasoningContent)
    assert content[0].reasoning == "Rooftop City metaphor."
    assert isinstance(content[1], TextContent)
    assert content[1].text.startswith("*the first kiss")


def test_message_to_letta_messages_splits_leaked_response():
    msg = Message(
        role="assistant",
        content=[
            ReasoningContent(
                is_native=True,
                reasoning="Stay in scene.</thinking>\nI just stay.",
            )
        ],
    )
    letta_msgs = msg.to_letta_messages(use_assistant_message=True)
    types = [m.message_type for m in letta_msgs]
    assert "reasoning_message" in types
    assert "assistant_message" in types
    reasoning = next(m for m in letta_msgs if isinstance(m, ReasoningMessage))
    assistant = next(m for m in letta_msgs if isinstance(m, AssistantMessage))
    assert reasoning.reasoning == "Stay in scene."
    assert assistant.content == "I just stay."
