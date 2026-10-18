"""DeepSeek V4.1 DSML tool calls leaked into assistant text."""

import json

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

from letta.helpers.dsml_tool_calls import (
    DsmlStreamSplitter,
    apply_dsml_recovery,
    is_dsml_only,
    parse_dsml_tool_calls,
    strip_dsml_markup,
)
from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import ReasoningContent, TextContent
from letta.schemas.message import Message
from letta.schemas.openai.chat_completion_response import FunctionCall, ToolCall

# The shape stored for conv-838b0bb0: V4.1 tags, drifted attributes, close-tag spam.
_COMPOSE = """
<｜DSML｜ calls >
<｜DSML｜ invoke name ="compose_image">
<｜DSML｜ parameter name ="prompt" string="true">An oil painting, summer 1907.</｜DSML｜ parameter>
<｜DSML｜ parameter name ="aspect_ratio" string ="3:4">
</｜DSML｜ parameter>
<｜DSML｜ parameter name ="image_urls">
<parameter name ="image_urls">["image-d88a53a5-6474-4f32-a006-adb166a14a5e", "image-bc5a5805-f1e6-48f0-8f1a-02a38a014ce5"]</｜DSML｜ parameter>
</｜DSML｜ parameter>
<｜DSML｜ parameter name ="model">gpt-image-2.5-sunburst</parameter>
<｜DSML｜ parameter name="quality">high</｜DSML｜ parameter>
<｜DSML｜ parameter name="resolution">2k</｜DSML｜ parameter>
</｜DSML｜ invoke>
</compose_image>
</function>
</function>
</antml>
                         </function
"""

_V4 = """<｜DSML｜tool_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>
<｜DSML｜parameter name="count" string="false">5</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>"""


def test_v41_official_shape_parses():
    calls = parse_dsml_tool_calls(
        '<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup">\n'
        '<｜DSML｜ parameter name="query" string="true">value</｜DSML｜ parameter>\n'
        '<｜DSML｜ parameter name="limit" string="false">2</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )
    assert calls == [{"name": "lookup", "arguments": {"query": "value", "limit": 2}}]


def test_v4_shape_without_spaces_parses():
    calls = parse_dsml_tool_calls(_V4)
    assert calls == [{"name": "get_weather", "arguments": {"location": "Beijing", "count": 5}}]


def test_drifted_compose_image_call_parses_and_strips():
    calls = parse_dsml_tool_calls(_COMPOSE)
    assert len(calls) == 1
    call = calls[0]
    assert call["name"] == "compose_image"
    args = call["arguments"]
    assert args["prompt"] == "An oil painting, summer 1907."
    assert args["aspect_ratio"] == "3:4"
    assert args["image_urls"] == [
        "image-d88a53a5-6474-4f32-a006-adb166a14a5e",
        "image-bc5a5805-f1e6-48f0-8f1a-02a38a014ce5",
    ]
    assert args["model"] == "gpt-image-2.5-sunburst"
    assert args["quality"] == "high"
    assert args["resolution"] == "2k"
    assert strip_dsml_markup("Let me write and roll.\n" + _COMPOSE) == "Let me write and roll."
    assert is_dsml_only(_COMPOSE)


def test_unquoted_invoke_name_is_not_a_tool_call():
    text = '<｜DSML｜ calls >\n<｜DSML｜ invoke name_edits>{"handle": "image-1"}</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</function>\n'
    assert parse_dsml_tool_calls(text) == []
    assert strip_dsml_markup(text) == ""


def test_recovery_promotes_markup_when_no_native_tool_call():
    content, tool_calls = apply_dsml_recovery([TextContent(text=_COMPOSE)], [])
    assert content == []
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "compose_image"
    assert json.loads(tool_calls[0].function.arguments)["model"] == "gpt-image-2.5-sunburst"


def test_recovery_does_not_duplicate_native_tool_call():
    native = ToolCall(id="call-native", function=FunctionCall(name="image_edit_text", arguments="{}"))
    markup = '<｜DSML｜ invoke name="image_edit_text">\n<｜DSML｜ parameter name="handle" string="true">image-1</｜DSML｜ parameter>\n</｜DSML｜ invoke>'
    content, tool_calls = apply_dsml_recovery(
        [ReasoningContent(is_native=True, reasoning="stamp it"), TextContent(text=markup)],
        [native],
    )
    assert tool_calls == [native]
    assert content == [ReasoningContent(is_native=True, reasoning="stamp it")]


def test_stream_splitter_holds_a_marker_split_across_chunks():
    splitter = DsmlStreamSplitter()
    assert splitter.feed("Hello") == "Hello"
    assert splitter.feed(" <｜DS") == " "
    assert splitter.feed('ML｜ calls>\n<｜DSML｜ invoke name="lookup">\n') == ""
    assert splitter.feed('<｜DSML｜ parameter name="q" string="true">hi</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n') == ""
    assert splitter.flush_visible() == ""
    assert parse_dsml_tool_calls(splitter.markup) == [{"name": "lookup", "arguments": {"q": "hi"}}]


def test_openai_replay_drops_dsml_text_and_keeps_native_tool_call():
    message = Message(
        role=MessageRole.assistant,
        content=[
            ReasoningContent(is_native=True, reasoning="stamp the print"),
            TextContent(text=_COMPOSE),
        ],
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call-native",
                type="function",
                function=Function(name="image_edit_text", arguments="{}"),
            )
        ],
    )
    serialized = message.to_openai_dict()
    assert serialized["content"] is None
    assert serialized["tool_calls"][0]["function"]["name"] == "image_edit_text"
    assert serialized["reasoning_content"] == "stamp the print"
    assert "DSML" not in json.dumps(serialized)


def test_openai_replay_omits_dsml_only_assistant_turn():
    message = Message(
        role=MessageRole.assistant,
        content=[
            ReasoningContent(is_native=True, reasoning="rerender it"),
            TextContent(text=_COMPOSE),
        ],
    )
    assert message.to_openai_dict() is None


def test_letta_messages_hide_dsml_text():
    message = Message(
        role=MessageRole.assistant,
        content=[
            ReasoningContent(is_native=True, reasoning="rerender it"),
            TextContent(text="Before.\n" + _COMPOSE),
        ],
    )
    letta_messages = message.to_letta_messages(text_is_assistant_message=True)
    assistant = [m for m in letta_messages if m.message_type == "assistant_message"]
    assert len(assistant) == 1
    assert assistant[0].content == "Before."
    assert "DSML" not in assistant[0].content
