from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

from letta.llm_api.openai_client import fill_image_content_in_messages, fill_image_content_in_responses_input
from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import Base64Image, ImageContent, LettaImage, TextContent
from letta.schemas.letta_message import MessageType
from letta.schemas.message import Message, ToolReturn, user_content_to_openai_chat_content
from letta.services.vision.render_policy import RenderTier
from letta.schemas.openai.chat_completion_request import cast_message_to_subtype


def _user_message_with_image_first(text: str) -> Message:
    image = ImageContent(source=Base64Image(media_type="image/png", data="dGVzdA=="))
    return Message(role=MessageRole.user, content=[image, TextContent(text=text)])


def _letta_stored_image_message(text: str | None = None) -> Message:
    """Simulates persisted history: base64 inbound rewritten to LettaImage with inline data."""
    image = ImageContent(
        source=LettaImage(
            file_id="file-test-001",
            data="dGVzdA==",
            media_type="image/jpeg",
        )
    )
    if text:
        return Message(role=MessageRole.user, content=[TextContent(text=text), image])
    return Message(role=MessageRole.user, content=[image])


def _user_content_has_image_url(content) -> bool:
    if isinstance(content, list):
        return any(part.get("type") == "image_url" for part in content)
    return False


def test_to_openai_dicts_include_user_image_url_parts():
    message = _user_message_with_image_first("what is in this image?")
    serialized = Message.to_openai_dicts_from_list([message])
    assert serialized[0]["role"] == "user"
    content = serialized[0]["content"]
    assert isinstance(content, list)
    assert any(part["type"] == "image_url" for part in content)
    assert any(part["type"] == "text" and "what is in this image?" in part["text"] for part in content)


def test_to_openai_dict_text_only_user_unchanged():
    message = Message(role=MessageRole.user, content=[TextContent(text="hello only")])
    serialized = Message.to_openai_dicts_from_list([message])
    assert serialized[0]["content"] == "hello only"


def test_to_openai_dict_image_only_user_has_image_url_not_placeholder():
    message = Message(role=MessageRole.user, content=[ImageContent(source=Base64Image(media_type="image/png", data="dGVzdA=="))])
    serialized = Message.to_openai_dicts_from_list([message])
    content = serialized[0]["content"]
    assert isinstance(content, list)
    assert _user_content_has_image_url(content)
    assert "[Image Here]" not in str(content)


def test_to_openai_dict_letta_image_persisted_in_history():
    message = _letta_stored_image_message("Describe this.")
    serialized = Message.to_openai_dicts_from_list([message])
    content = serialized[0]["content"]
    assert isinstance(content, list)
    assert _user_content_has_image_url(content)
    image_parts = [p for p in content if p["type"] == "image_url"]
    assert "image/jpeg" in image_parts[0]["image_url"]["url"]


def test_to_openai_dict_multiple_images_preserved():
    img1 = ImageContent(source=Base64Image(media_type="image/png", data="Zm9v"))
    img2 = ImageContent(source=Base64Image(media_type="image/png", data="YmFy"))
    message = Message(role=MessageRole.user, content=[img1, TextContent(text="compare"), img2])
    serialized = Message.to_openai_dicts_from_list([message])
    content = serialized[0]["content"]
    assert sum(1 for p in content if p["type"] == "image_url") == 2


def test_multi_turn_history_serializes_each_user_image():
    """Every historical user turn with an image must emit image_url in the LLM payload."""
    turn1 = _letta_stored_image_message("First image.")
    turn2 = Message(role=MessageRole.user, content=[TextContent(text="Follow-up without new image.")])
    turn3 = _letta_stored_image_message("Second image.")
    history = [
        Message(role=MessageRole.system, content=[TextContent(text="system")]),
        turn1,
        Message(role=MessageRole.assistant, content=[TextContent(text="I see the first image.")]),
        turn2,
        Message(role=MessageRole.assistant, content=[TextContent(text="Noted.")]),
        turn3,
    ]
    serialized = Message.to_openai_dicts_from_list(history)
    user_rows = [m for m in serialized if m["role"] == "user"]
    assert len(user_rows) == 3
    assert _user_content_has_image_url(user_rows[0]["content"])
    assert user_rows[1]["content"] == "Follow-up without new image."
    assert _user_content_has_image_url(user_rows[2]["content"])


def test_v030_fill_bailed_on_tool_expand_but_current_serializer_preserves_images():
    """Regression guard for v0.3.0: fill_image bailed when tool rows expanded; user dict used text placeholders."""
    image = ImageContent(source=Base64Image(media_type="image/png", data="dGVzdA=="))
    user = Message(role=MessageRole.user, content=[TextContent(text="see this"), image])
    tool_msg = Message(
        role=MessageRole.tool,
        tool_returns=[
            ToolReturn(tool_call_id="call-a", status="success", func_response="ok"),
            ToolReturn(tool_call_id="call-b", status="success", func_response="done"),
        ],
    )
    pydantic_messages = [tool_msg, user]
    openai_messages = Message.to_openai_dicts_from_list(pydantic_messages)
    assert len(openai_messages) == 3
    assert len(openai_messages) != len(pydantic_messages)

    # Simulate v0.3.0 serialized user row (text placeholder) + old fill bail on length mismatch
    v030_openai = [
        {"role": "tool", "content": "ok", "tool_call_id": "call-a"},
        {"role": "tool", "content": "done", "tool_call_id": "call-b"},
        {"role": "user", "content": "see this [Image omitted]"},
    ]

    def _v030_fill(openai_list, pydantic_list):
        if len(openai_list) != len(pydantic_list):
            return openai_list
        return fill_image_content_in_messages(openai_list, pydantic_list)

    assert _v030_fill(v030_openai, pydantic_messages)[2]["content"] == "see this [Image omitted]"

    # Current path: to_openai_dict emits multimodal user content without relying on fill
    user_rows = [m for m in openai_messages if m.get("role") == "user"]
    assert _user_content_has_image_url(user_rows[0]["content"])


def test_user_content_to_openai_chat_content_letta_dict():
    content = [
        {"type": "text", "text": "hi"},
        {
            "type": "image",
            "source": {
                "type": "letta",
                "file_id": "file-x",
                "data": "dGVzdA==",
                "media_type": "image/jpeg",
            },
        },
    ]
    parts = user_content_to_openai_chat_content(content)
    assert isinstance(parts, list)
    assert any(p["type"] == "image_url" and "image/jpeg" in p["image_url"]["url"] for p in parts)


def _openai_row_role(message):
    if isinstance(message, dict):
        return message.get("role")
    return getattr(message, "role", None)


def test_fill_image_content_in_messages_handles_pydantic_tool_rows():
    """Tool rows from cast_message_to_subtype must accept multimodal re-fill by tool_call_id."""
    image = ImageContent(source=Base64Image(media_type="image/png", data="dGVzdA=="))
    tool_msg = Message(
        role=MessageRole.tool,
        tool_returns=[
            ToolReturn(
                tool_call_id="call-img",
                status="success",
                func_response=[TextContent(text="screenshot"), image],
            )
        ],
    )
    pydantic_messages = [tool_msg]
    openai_messages = [
        cast_message_to_subtype(row)
        for row in Message.to_openai_dicts_from_list(pydantic_messages)
    ]
    filled = fill_image_content_in_messages(openai_messages, pydantic_messages)
    tool_rows = [m for m in filled if _openai_row_role(m) == "tool"]
    assert len(tool_rows) == 1
    content = tool_rows[0]["content"] if isinstance(tool_rows[0], dict) else tool_rows[0].content
    assert any(part["type"] == "image_url" for part in content)


def test_fill_image_content_in_messages_handles_pydantic_openai_rows():
    """openai_message_list entries are ChatMessage pydantic models after cast_message_to_subtype."""
    image = ImageContent(source=Base64Image(media_type="image/png", data="dGVzdA=="))
    user = Message(role=MessageRole.user, content=[TextContent(text="see this"), image])
    pydantic_messages = [
        Message(role=MessageRole.system, content=[TextContent(text="system prompt")]),
        user,
    ]
    openai_messages = [
        cast_message_to_subtype(row)
        for row in Message.to_openai_dicts_from_list(pydantic_messages)
    ]
    filled = fill_image_content_in_messages(openai_messages, pydantic_messages)
    user_rows = [m for m in filled if getattr(m, "role", None) == "user" or (isinstance(m, dict) and m.get("role") == "user")]
    assert len(user_rows) == 1
    content = user_rows[0]["content"] if isinstance(user_rows[0], dict) else user_rows[0].content
    assert any(part["type"] == "image_url" for part in content)


def test_fill_image_content_in_messages_pairs_user_messages_when_tool_rows_expand():
    image = ImageContent(source=Base64Image(media_type="image/png", data="dGVzdA=="))
    user = Message(role=MessageRole.user, content=[TextContent(text="see this"), image])
    tool_msg = Message(
        role=MessageRole.tool,
        tool_returns=[
            ToolReturn(tool_call_id="call-a", status="success", func_response="ok"),
            ToolReturn(tool_call_id="call-b", status="success", func_response="done"),
        ],
    )
    pydantic_messages = [tool_msg, user]
    openai_messages = Message.to_openai_dicts_from_list(pydantic_messages)
    assert len(openai_messages) == 3
    assert len(openai_messages) != len(pydantic_messages)

    filled = fill_image_content_in_messages(openai_messages, pydantic_messages)
    user_rows = [m for m in filled if m.get("role") == "user"]
    assert len(user_rows) == 1
    assert any(part["type"] == "image_url" for part in user_rows[0]["content"])


def test_tool_return_with_image_serializes_multimodal():
    image = ImageContent(source=Base64Image(media_type="image/png", data="dGVzdA=="))
    tool_msg = Message(
        role=MessageRole.tool,
        tool_returns=[
            ToolReturn(
                tool_call_id="call-img",
                status="success",
                func_response=[TextContent(text="screenshot"), image],
            )
        ],
    )
    serialized = Message.to_openai_dicts_from_list([tool_msg])
    assert len(serialized) == 1
    assert serialized[0]["role"] == "tool"
    assert any(p["type"] == "image_url" for p in serialized[0]["content"])


def test_to_openai_responses_dicts_handles_image_first_content():
    message = _user_message_with_image_first("hello world")
    serialized = Message.to_openai_responses_dicts_from_list([message])
    parts = serialized[0]["content"]
    assert any(part["type"] == "input_text" and part["text"] == "hello world" for part in parts)
    assert any(part["type"] == "input_image" for part in parts)


def test_fill_image_content_in_responses_input_includes_image_parts():
    message = _user_message_with_image_first("describe image")
    serialized = Message.to_openai_responses_dicts_from_list([message])
    rewritten = fill_image_content_in_responses_input(serialized, [message])
    assert rewritten == serialized


def test_to_openai_responses_dicts_handles_image_only_content():
    image = ImageContent(source=Base64Image(media_type="image/png", data="dGVzdA=="))
    message = Message(role=MessageRole.user, content=[image])
    serialized = Message.to_openai_responses_dicts_from_list([message])
    parts = serialized[0]["content"]
    assert parts[0]["type"] == "input_image"


def test_to_anthropic_dict_user_letta_image():
    message = _letta_stored_image_message("What is this?")
    serialized = message.to_anthropic_dict(
        current_model="anthropic/claude-sonnet-4-5-20250929",
        put_inner_thoughts_in_kwargs=False,
    )
    assert serialized["role"] == "user"
    assert any(p["type"] == "image" for p in serialized["content"])
    image_block = next(p for p in serialized["content"] if p["type"] == "image")
    assert image_block["source"]["data"] == "dGVzdA=="
    assert image_block["source"]["media_type"] == "image/jpeg"


def test_to_google_dict_user_letta_image():
    message = _letta_stored_image_message("What is this?")
    serialized = message.to_google_dict(current_model="google/gemini-2.5-pro")
    assert serialized["role"] == "user"
    assert any("inline_data" in p for p in serialized["parts"])
    inline = next(p for p in serialized["parts"] if "inline_data" in p)
    assert inline["inline_data"]["data"] == "dGVzdA=="
    assert inline["inline_data"]["mime_type"] == "image/jpeg"


def test_to_anthropic_dict_falls_back_for_malformed_tool_call_arguments():
    malformed_args = '{"message": "unterminated}'
    msg = Message(
        role=MessageRole.assistant,
        content=[TextContent(text="thinking")],
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_test_malformed",
                type="function",
                function=Function(name="send_message", arguments=malformed_args),
            )
        ],
    )

    serialized = msg.to_anthropic_dict(
        current_model="anthropic/claude-sonnet-4-5-20250929",
        inner_thoughts_xml_tag="thinking",
        put_inner_thoughts_in_kwargs=False,
    )

    tool_use_items = [item for item in serialized["content"] if item.get("type") == "tool_use"]
    assert len(tool_use_items) == 1
    assert tool_use_items[0]["input"] == {"_malformed_tool_arguments": malformed_args}


def test_to_google_dict_falls_back_for_malformed_tool_call_arguments():
    malformed_args = '{"message": "unterminated}'
    msg = Message(
        role=MessageRole.assistant,
        content=[],
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_test_malformed_google",
                type="function",
                function=Function(name="send_message", arguments=malformed_args),
            )
        ],
    )

    serialized = msg.to_google_dict(
        current_model="google/gemini-2.5-pro",
    )

    function_calls = [item for item in serialized["parts"] if item.get("functionCall")]
    assert len(function_calls) == 1
    assert function_calls[0]["functionCall"]["args"] == {"_malformed_tool_arguments": malformed_args}


def test_to_google_dict_preserves_thought_signature_on_empty_content():
    """When Gemini returns a function call without reasoning text, the
    thought_signature must still appear on the serialized functionCall part.
    Regression test for LET-8166 / GitHub #3221."""
    sig = "EoQHsomebase64signaturedata=="
    msg = Message(
        role=MessageRole.assistant,
        content=[TextContent(text="", signature=sig)],
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_test_thought_sig",
                type="function",
                function=Function(name="archival_memory_search", arguments='{"query": "test"}'),
            )
        ],
    )

    serialized = msg.to_google_dict(current_model="google/gemini-3-flash")

    function_calls = [p for p in serialized["parts"] if "functionCall" in p]
    assert len(function_calls) == 1
    assert function_calls[0].get("thought_signature") == sig


def test_to_google_dict_no_signature_when_absent():
    """Without a signature, functionCall parts should not include
    thought_signature (no sentinel, no empty string)."""
    msg = Message(
        role=MessageRole.assistant,
        content=[],
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="call_test_no_sig",
                type="function",
                function=Function(name="send_message", arguments='{"message": "hi"}'),
            )
        ],
    )

    serialized = msg.to_google_dict(current_model="google/gemini-3-flash")

    function_calls = [p for p in serialized["parts"] if "functionCall" in p]
    assert len(function_calls) == 1
    assert "thought_signature" not in function_calls[0]


def test_orm_to_pydantic_preserves_multimodal_tool_return_over_legacy_content():
    """Regression: legacy content[0].text must not replace func_response when images are stored."""
    from letta.orm.message import Message as OrmMessage
    from letta.schemas.message import tool_return_has_images

    image = ImageContent(source=Base64Image(media_type="image/png", data="dGVzdA=="))
    multimodal = [TextContent(text='{"images":[{"url":"https://example.com/x.png"}]}'), image]
    legacy_text = '{"status":"OK","message":"{\\"images\\":[]\\"} [Image omitted]"}'

    orm = OrmMessage(
        id="message-orm-multimodal-test",
        role=MessageRole.tool,
        organization_id="org-test",
        agent_id="agent-test",
        tool_returns=[
            ToolReturn(tool_call_id="call-img", status="success", func_response=multimodal),
        ],
        content=[TextContent(text=legacy_text)],
        tool_calls=[],
    )

    assert tool_return_has_images(orm.tool_returns[0].func_response)
    model = orm.to_pydantic()
    assert tool_return_has_images(model.tool_returns[0].func_response)

    letta = model.to_letta_messages()[0]
    assert isinstance(letta.tool_return, list)
    assert any(
        (p.type if hasattr(p, "type") else p.get("type")) == "image"
        for p in letta.tool_return
    )

    openai = Message.to_openai_dicts_from_list([model])
    assert any(p["type"] == "image_url" for p in openai[0]["content"])


def test_dedupe_tool_messages_keeps_same_tool_call_id_across_steps():
    """Provider-reused tool_call_ids must not collapse distinct step executions."""
    reused_id = "functions.scenecraft_inspect_asset:6"
    messages = [
        Message(
            role=MessageRole.tool,
            step_id="step-bronn",
            tool_call_id=reused_id,
            content="bronn",
        ),
        Message(
            role=MessageRole.tool,
            step_id="step-morbiena",
            tool_call_id=reused_id,
            content="morbiena",
        ),
    ]
    deduped = Message.dedupe_tool_messages_for_llm_api(messages)
    assert len(deduped) == 2
    assert [m.content for m in deduped] == ["bronn", "morbiena"]


def test_dedupe_tool_messages_drops_true_duplicate_same_step():
    reused_id = "functions.scenecraft_inspect_asset:6"
    messages = [
        Message(
            role=MessageRole.tool,
            step_id="step-dup",
            tool_call_id=reused_id,
            content="first",
        ),
        Message(
            role=MessageRole.tool,
            step_id="step-dup",
            tool_call_id=reused_id,
            content="duplicate",
        ),
    ]
    deduped = Message.dedupe_tool_messages_for_llm_api(messages)
    assert len(deduped) == 1
    assert deduped[0].content == "first"


def test_letta_image_with_hydrated_data_serializes_to_image_url():
    letta_ref = LettaImage(file_id="image-abc", data="dGVzdA==", media_type="image/png")
    parts = user_content_to_openai_chat_content(
        [ImageContent(source=letta_ref), TextContent(text="what do you see?")],
        image_render_decisions={"image-abc": RenderTier.TEXT},
    )
    assert isinstance(parts, list)
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_to_letta_messages_stamps_sequence_id_and_sub_order():
    tool_call = ChatCompletionMessageToolCall(
        id="call-recall",
        type="function",
        function=Function(name="recall", arguments='{"query": "codeword"}'),
    )
    assistant = Message(
        id="message-seq-test",
        role=MessageRole.assistant,
        sequence_id=2116,
        content=[TextContent(text="searching memory")],
        tool_calls=[tool_call],
    )
    tool_return = Message(
        id="message-seq-tool",
        role=MessageRole.tool,
        sequence_id=2117,
        name="recall",
        tool_call_id="call-recall",
        content='[{"type":"text","text":"hits"}]',
    )

    assistant_letta = assistant.to_letta_messages(reverse=False)
    assert len(assistant_letta) == 2
    assert assistant_letta[0].message_type == MessageType.reasoning_message
    assert assistant_letta[1].message_type == MessageType.tool_call_message
    assert [msg.seq_id for msg in assistant_letta] == [2116, 2116]
    assert [msg.seq_sub for msg in assistant_letta] == [0, 1]

    return_letta = tool_return.to_letta_messages(reverse=False)
    assert len(return_letta) == 1
    assert return_letta[0].seq_id == 2117
    assert return_letta[0].seq_sub == 0
