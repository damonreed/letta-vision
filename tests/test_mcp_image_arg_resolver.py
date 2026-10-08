import pytest

from letta.schemas.enums import ToolType
from letta.schemas.tool import Tool
from letta.services.helpers.tool_parser_helper import runtime_override_tool_json_schema
from letta.services.mcp.image_arg_resolver import resolve_mcp_image_arguments
from letta.services.tool_executor.mcp_tool_executor import ExternalMCPToolExecutor


def _executor() -> ExternalMCPToolExecutor:
    return ExternalMCPToolExecutor(
        message_manager=None,
        agent_manager=None,
        block_manager=None,
        run_manager=None,
        passage_manager=None,
        actor=None,
    )


PNG = b"\x89PNG\r\n\x1a\n"
HANDLE = "image-11111111-1111-1111-1111-111111111111"
SIGNED = "https://storage.googleapis.com/letta-vision-images/image-11111111-1111-1111-1111-111111111111?X-Goog-Signature=abc"


class _Image:
    def __init__(self, object_url_full="sha256/abc", media_type="image/png"):
        self.object_url_full = object_url_full
        self.media_type = media_type


def _patch_lookup(monkeypatch, image=_Image()):
    class _Mgr:
        async def get_by_id_async(self, image_id, actor):
            assert image_id == HANDLE
            return image

    async def _ensure(image_id, *, minio_key, content_type):
        assert image_id == HANDLE
        assert minio_key == image.object_url_full
        return SIGNED

    monkeypatch.setattr("letta.services.mcp.image_arg_resolver.ImageManager", lambda: _Mgr())
    monkeypatch.setattr(
        "letta.services.mcp.image_arg_resolver.get_gcs_image_mirror",
        lambda: object(),
    )
    monkeypatch.setattr(
        "letta.services.mcp.image_arg_resolver.ensure_mirrored_and_sign",
        _ensure,
    )


@pytest.mark.asyncio
async def test_edit_image_handle_becomes_signed_url(monkeypatch):
    _patch_lookup(monkeypatch)
    original = {"prompt": "make the sky orange", "image_url": f"  {HANDLE}  "}

    resolved = await resolve_mcp_image_arguments("edit_image", original, actor=None)

    assert resolved["image_url"] == SIGNED
    assert resolved["prompt"] == "make the sky orange"
    assert original["image_url"] == f"  {HANDLE}  "


@pytest.mark.asyncio
async def test_https_and_data_uri_pass_through(monkeypatch):
    _patch_lookup(monkeypatch)
    url = "https://example.com/pic.png"
    data_uri = "data:image/png;base64,aaaa"

    https_args = await resolve_mcp_image_arguments("edit_image", {"image_url": url}, actor=None)
    data_args = await resolve_mcp_image_arguments("edit_image", {"image_url": data_uri}, actor=None)

    assert https_args["image_url"] == url
    assert data_args["image_url"] == data_uri


@pytest.mark.asyncio
async def test_missing_image_raises(monkeypatch):
    class _Mgr:
        async def get_by_id_async(self, image_id, actor):
            return None

    monkeypatch.setattr("letta.services.mcp.image_arg_resolver.ImageManager", lambda: _Mgr())

    with pytest.raises(ValueError, match="Image not found"):
        await resolve_mcp_image_arguments("edit_image", {"image_url": HANDLE}, actor=None)


@pytest.mark.asyncio
async def test_compose_resolves_handles_and_leaves_urls(monkeypatch):
    _patch_lookup(monkeypatch, image=_Image(media_type="image/jpeg; charset=binary"))
    url = "https://example.com/b.jpg"

    resolved = await resolve_mcp_image_arguments(
        "compose_image",
        {"image_urls": [HANDLE, url]},
        actor=None,
    )

    assert resolved["image_urls"][0] == SIGNED
    assert resolved["image_urls"][1] == url


@pytest.mark.asyncio
async def test_other_tools_are_left_alone(monkeypatch):
    _patch_lookup(monkeypatch)
    args = {"image_url": HANDLE}
    assert await resolve_mcp_image_arguments("generate_image", args, actor=None) is args


def test_schema_note_is_idempotent():
    tool = {
        "name": "edit_image",
        "description": "Edit an image.",
        "parameters": {
            "type": "object",
            "properties": {"image_url": {"type": "string", "description": "Source URL."}},
            "required": ["image_url"],
        },
    }

    once = runtime_override_tool_json_schema([tool], response_format=None, request_heartbeat=False)
    twice = runtime_override_tool_json_schema(once, response_format=None, request_heartbeat=False)

    description = twice[0]["parameters"]["properties"]["image_url"]["description"]
    assert description.count("image-<uuid>") == 1
    assert "Source URL." in description
    assert "signed" in description.lower() or "HTTPS" in description
    assert twice[0]["description"].count("image-<uuid>") == 1


@pytest.mark.asyncio
async def test_executor_sends_signed_url_and_keeps_caller_args(monkeypatch):
    _patch_lookup(monkeypatch)
    captured = {}

    class _Manager:
        async def execute_mcp_server_tool(self, **kwargs):
            captured.update(kwargs)
            return {"status": "ok"}, True

    monkeypatch.setattr("letta.services.tool_executor.mcp_tool_executor.MCPManager", lambda: _Manager())

    tool = Tool(
        name="edit_image",
        tool_type=ToolType.EXTERNAL_MCP,
        tags=["mcp:zapimage"],
        json_schema={"name": "edit_image", "parameters": {"type": "object", "properties": {}}},
    )
    args = {"image_url": HANDLE, "prompt": "sunset"}
    result = await _executor().execute("edit_image", args, tool, actor=None)

    assert result.status == "success"
    assert captured["tool_args"]["image_url"] == SIGNED
    assert args["image_url"] == HANDLE


@pytest.mark.asyncio
async def test_executor_returns_missing_image_as_tool_error(monkeypatch):
    class _Mgr:
        async def get_by_id_async(self, image_id, actor):
            return None

    monkeypatch.setattr("letta.services.mcp.image_arg_resolver.ImageManager", lambda: _Mgr())

    tool = Tool(
        name="edit_image",
        tool_type=ToolType.EXTERNAL_MCP,
        tags=["mcp:zapimage"],
        json_schema={"name": "edit_image", "parameters": {"type": "object", "properties": {}}},
    )
    result = await _executor().execute(
        "edit_image",
        {"image_url": HANDLE},
        tool,
        actor=None,
    )

    assert result.status == "error"
    assert HANDLE in result.func_return
