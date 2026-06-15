import pytest

from letta.schemas.enums import MessageRole
from letta.schemas.message import Message, ToolReturn, tool_return_to_openai_chat_content
from letta.services.image_fetch import build_fetch_image_tool_return, multimodal_tool_return, normalize_image_handle
from letta.services.recall.recall_service import _file_archive_recall_snippet, _recall_snippet


def test_normalize_image_handle():
    assert normalize_image_handle("image-abc") == "image-abc"
    assert normalize_image_handle("abc") == "image-abc"
    assert normalize_image_handle("  image-xyz  ") == "image-xyz"


def test_file_archive_recall_snippet():
    class _Row:
        title = "Connector auth"
        content = "OAuth refresh tokens expire after 24h."
        file_name = "README.md"

    assert _file_archive_recall_snippet(_Row()) == "[Connector auth] OAuth refresh tokens expire after 24h."


def test_recall_snippet_file_archive_layer():
    class _Row:
        title = "Note"
        content = "Body"
        file_name = None

    assert _recall_snippet("file_archive", _Row()) == "[Note] Body"


@pytest.mark.asyncio
async def test_build_fetch_image_tool_return_multimodal(monkeypatch):
    class _Image:
        object_url_full = "images/sha256/test"
        media_type = "image/png"
        file_size_full = 8
        description = "A test image"
        caption = "Short label"
        details = "Longer literal details"

    class _Mgr:
        async def get_by_id_async(self, handle, actor):
            assert handle == "image-abc"
            return _Image()

    monkeypatch.setattr("letta.services.image_fetch.ImageManager", lambda: _Mgr())

    result = await build_fetch_image_tool_return("abc", actor=None)
    assert isinstance(result, list)
    assert result[0]["type"] == "text"
    text = result[0]["text"]
    assert "Caption: Short label" in text
    assert "Description: A test image" in text
    assert "Details: Longer literal details" in text
    assert "8 bytes" in text
    assert result[1]["type"] == "image"
    assert result[1]["source"]["type"] == "letta"
    assert result[1]["source"]["file_id"] == "image-abc"
    assert result[1]["source"]["detail"] == "high"
    assert result[1]["source"]["data"] is None


def test_fetch_image_tool_return_serializes_for_llm():
    from letta.schemas.letta_message_content import Base64Image, ImageContent, TextContent

    blocks = multimodal_tool_return(
        [
            TextContent(text="summary"),
            ImageContent(source=Base64Image(media_type="image/png", data="dGVzdA==", detail="high")),
        ]
    )
    tool_msg = Message(
        role=MessageRole.tool,
        tool_returns=[
            ToolReturn(
                tool_call_id="call-fetch",
                status="success",
                func_response=blocks,
            )
        ],
    )
    serialized = tool_return_to_openai_chat_content(tool_msg.tool_returns[0].func_response)
    assert isinstance(serialized, list)
    assert any(part["type"] == "image_url" for part in serialized)
