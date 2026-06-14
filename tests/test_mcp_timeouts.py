import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from letta.functions.mcp_client.types import StreamableHTTPServerConfig
from letta.services.mcp.base_client import (
    AsyncBaseMCPClient,
    format_mcp_timeout_error,
    run_with_mcp_timeout,
)
from letta.services.mcp.fastmcp_client import AsyncFastMCPStreamableHTTPClient


class _StubMCPClient(AsyncBaseMCPClient):
    async def _initialize_connection(self, server_config):
        self.session = MagicMock()
        self.session.initialize = AsyncMock()

    def to_sync_client(self):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_run_with_mcp_timeout_raises_friendly_timeout_error():
    async def slow():
        await asyncio.sleep(5)

    with patch("letta.services.mcp.base_client.tool_settings") as settings:
        settings.mcp_execute_tool_timeout = 0.05
        with pytest.raises(TimeoutError, match="timed out after 0.05s"):
            await run_with_mcp_timeout(
                slow(),
                timeout_seconds=settings.mcp_execute_tool_timeout,
                operation="tool execution",
                tool_name="scenecraft_inspect_asset",
            )


def test_format_mcp_timeout_error_includes_tool_name():
    message = format_mcp_timeout_error("tool execution", 60.0, tool_name="scenecraft_search")
    assert "scenecraft_search" in message
    assert "60s" in message


@pytest.mark.asyncio
async def test_base_client_execute_tool_returns_timeout_as_tool_error():
    client = _StubMCPClient(
        server_config=StreamableHTTPServerConfig(server_name="test", server_url="https://example.com/mcp")
    )
    client.initialized = True
    client.session = MagicMock()

    async def slow_call(*_args, **_kwargs):
        await asyncio.sleep(5)

    client.session.call_tool = AsyncMock(side_effect=slow_call)

    with patch("letta.services.mcp.base_client.tool_settings") as settings:
        settings.mcp_execute_tool_timeout = 0.05
        result, success = await client.execute_tool("demo_tool", {"x": 1})

    assert success is False
    assert "timed out after 0.05s" in result
    assert "demo_tool" in result


@pytest.mark.asyncio
async def test_fastmcp_client_execute_tool_returns_timeout_as_tool_error():
    client = AsyncFastMCPStreamableHTTPClient(
        server_config=StreamableHTTPServerConfig(server_name="test", server_url="https://example.com/mcp")
    )
    client.initialized = True
    client.client = MagicMock()

    async def slow_call(*_args, **_kwargs):
        await asyncio.sleep(5)

    client.client.call_tool = AsyncMock(side_effect=slow_call)

    with patch("letta.services.mcp.fastmcp_client.tool_settings") as settings:
        settings.mcp_execute_tool_timeout = 0.05
        result, success = await client.execute_tool("demo_tool", {"x": 1})

    assert success is False
    assert "timed out after 0.05s" in result
    assert "demo_tool" in result
