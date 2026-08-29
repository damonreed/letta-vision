import base64
import io
import json

from mcp.types import EmbeddedResource as McpEmbeddedResource
from mcp.types import ImageContent as McpImageContent
from mcp.types import TextContent as McpTextContent
from mcp.types import TextResourceContents
from PIL import Image

from letta.schemas.letta_message_content import ImageContent, TextContent
from letta.schemas.message import tool_return_has_images, tool_return_to_openai_chat_content
from letta.services.mcp.tool_result_formatter import (
    format_mcp_result_for_log,
    format_mcp_tool_content,
    mcp_content_to_letta_parts,
)


def _png_b64(width: int, height: int) -> str:
    img = Image.new("RGB", (width, height), color=(200, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_format_omits_image_base64_keeps_json():
    payload = {
        "status": "success",
        "model": "grok-imagine-image",
        "count": 1,
        "images": [{"url": "https://storage.example.com/out.png?sig=abc"}],
    }
    text = json.dumps(payload, indent=2)
    huge = "A" * 100_000
    content = [
        McpTextContent(type="text", text=text),
        McpImageContent(type="image", data=huge, mimeType="image/png"),
    ]
    out = format_mcp_tool_content(content)
    assert "storage.example.com" in out
    assert huge not in out
    assert "omitted" in out.lower()
    assert len(out) < 10_000


def test_mcp_content_to_letta_parts_preserves_image():
    payload = {"images": [{"url": "https://example.com/a.png"}]}
    text = json.dumps(payload)
    huge = "B" * 50_000
    content = [
        McpTextContent(type="text", text=text),
        McpImageContent(type="image", data=huge, mimeType="image/png"),
    ]
    out = mcp_content_to_letta_parts(content)
    assert isinstance(out, list)
    assert tool_return_has_images(out)
    assert any(isinstance(p, ImageContent) for p in out)
    image_part = next(p for p in out if isinstance(p, ImageContent))
    assert image_part.source.data == huge
    assert "example.com" in next(p.text for p in out if isinstance(p, TextContent))


def test_mcp_content_to_letta_parts_preserves_full_resolution():
    """generate_image 2k pixels must reach ingest; do not downscale in the formatter."""
    data = _png_b64(2048, 1365)
    content = [
        McpTextContent(type="text", text='{"status": "success"}'),
        McpImageContent(type="image", data=data, mimeType="image/png"),
    ]
    out = mcp_content_to_letta_parts(content)
    image_part = next(p for p in out if isinstance(p, ImageContent))
    raw = base64.b64decode(image_part.source.data)
    img = Image.open(io.BytesIO(raw))
    assert img.size == (2048, 1365)


def test_mcp_content_to_letta_parts_prepends_inline_visibility_note():
    payload = {"images": [{"url": "https://example.com/a.png"}]}
    content = [
        McpTextContent(type="text", text=json.dumps(payload)),
        McpImageContent(type="image", data="B" * 1000, mimeType="image/png"),
    ]
    out = mcp_content_to_letta_parts(content)
    text_part = next(p.text for p in out if isinstance(p, TextContent))
    assert "directly visible to you" in text_part
    assert text_part.index("directly visible to you") < text_part.index("example.com")


def test_format_mcp_result_for_log_omits_base64():
    huge = "B" * 50_000
    content = [
        McpTextContent(type="text", text='{"ok": true}'),
        McpImageContent(type="image", data=huge, mimeType="image/png"),
    ]
    parts = mcp_content_to_letta_parts(content)
    log_line = format_mcp_result_for_log(parts)
    assert huge not in log_line
    assert "image block" in log_line.lower() or "omitted" in log_line.lower()


def test_tool_return_to_openai_chat_content_includes_data_url():
    huge = "C" * 100
    parts = [
        TextContent(text='{"ok": true}'),
        ImageContent.model_validate(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": huge},
            }
        ),
    ]
    openai_content = tool_return_to_openai_chat_content(parts)
    assert isinstance(openai_content, list)
    assert openai_content[0]["type"] == "text"
    assert openai_content[1]["type"] == "image_url"
    assert huge in openai_content[1]["image_url"]["url"]


def test_mcp_content_to_letta_parts_extracts_embedded_text_resource():
    """GitHub MCP get_file_contents returns TextContent + EmbeddedResource(TextResourceContents)."""
    content = [
        McpTextContent(type="text", text="successfully downloaded text file (SHA: abc)"),
        McpEmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri="repo://damonreed/letta-vision/sha/abc/contents/README.md",
                mimeType="text/plain; charset=utf-8",
                text="# Hello\n\nVision support section.",
            ),
        ),
    ]
    out = mcp_content_to_letta_parts(content)
    assert isinstance(out, str)
    assert "successfully downloaded text file" in out
    assert "# Hello" in out
    assert "Vision support section" in out
    assert "omitted non-text MCP content" not in out


def test_mcp_content_to_letta_parts_preserves_multiple_images_with_shared_prefix():
    """Batch generate_image JPEGs often share identical header bytes in base64."""
    shared_prefix = "A" * 128
    img_a = shared_prefix + "111"
    img_b = shared_prefix + "222"
    content = [
        McpTextContent(type="text", text='{"status": "success", "count": 2}'),
        McpImageContent(type="image", data=img_a, mimeType="image/jpeg"),
        McpImageContent(type="image", data=img_b, mimeType="image/jpeg"),
    ]
    out = mcp_content_to_letta_parts(content)
    assert isinstance(out, list)
    image_parts = [p for p in out if isinstance(p, ImageContent)]
    assert len(image_parts) == 2
    assert image_parts[0].source.data == img_a
    assert image_parts[1].source.data == img_b


def test_mcp_content_to_letta_parts_dedupes_exact_duplicate_images():
    img = "B" * 200
    content = [
        McpTextContent(type="text", text='{"status": "success", "count": 2}'),
        McpImageContent(type="image", data=img, mimeType="image/png"),
        McpImageContent(type="image", data=img, mimeType="image/png"),
    ]
    out = mcp_content_to_letta_parts(content)
    image_parts = [p for p in out if isinstance(p, ImageContent)]
    assert len(image_parts) == 1


def test_mcp_content_to_letta_parts_small_embedded_resource_not_omitted():
    content = [
        McpEmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri="repo://damonreed/letta-vision/sha/abc/contents/.python-version",
                mimeType="text/plain; charset=utf-8",
                text="3.12\n",
            ),
        ),
    ]
    out = mcp_content_to_letta_parts(content)
    assert out == "3.12"
