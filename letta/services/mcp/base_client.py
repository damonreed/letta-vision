import asyncio
from contextlib import AsyncExitStack
from typing import Awaitable, Optional, Tuple, TypeVar

from mcp import ClientSession, Tool as MCPTool
from mcp.client.auth import OAuthClientProvider
from mcp.types import TextContent

from letta.errors import LettaMCPConnectionError
from letta.functions.mcp_client.types import BaseServerConfig
from letta.log import get_logger
from letta.services.mcp.tool_result_formatter import mcp_content_to_letta_parts
from letta.settings import tool_settings

logger = get_logger(__name__)

T = TypeVar("T")

EXPECTED_MCP_TOOL_ERRORS = (
    "McpError",
    "ToolError",
    "HTTPStatusError",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "LocalProtocolError",
    "ConnectionError",
    "SSLError",
    "MaxRetryError",
    "ProtocolError",
    "BrokenResourceError",
    "TimeoutError",
)


def unwrap_exception_group(exc: Exception) -> Exception:
    if hasattr(exc, "exceptions") and exc.exceptions and len(exc.exceptions) == 1:
        return exc.exceptions[0]
    return exc


def format_mcp_timeout_error(operation: str, timeout_seconds: float, tool_name: Optional[str] = None) -> str:
    target = f"tool '{tool_name}'" if tool_name else operation
    return (
        f"MCP {target} timed out after {timeout_seconds:g}s "
        f"(no response from server). The remote MCP server may be hung or unreachable."
    )


async def run_with_mcp_timeout(
    awaitable: Awaitable[T],
    *,
    timeout_seconds: float,
    operation: str,
    tool_name: Optional[str] = None,
) -> T:
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(format_mcp_timeout_error(operation, timeout_seconds, tool_name)) from exc


def _log_mcp_tool_error(log: "get_logger", tool_name: str, exc: Exception) -> None:
    exc_name = type(exc).__name__
    if exc_name in EXPECTED_MCP_TOOL_ERRORS:
        log.info(f"MCP tool '{tool_name}' execution failed ({exc_name}): {exc}")
    else:
        log.warning(f"MCP tool '{tool_name}' execution failed with unexpected error ({exc_name}): {exc}", exc_info=True)


# TODO: Get rid of Async prefix on this class name once we deprecate old sync code
class AsyncBaseMCPClient:
    # HTTP headers
    AGENT_ID_HEADER = "X-Agent-Id"

    def __init__(
        self, server_config: BaseServerConfig, oauth_provider: Optional[OAuthClientProvider] = None, agent_id: Optional[str] = None
    ):
        self.server_config = server_config
        self.oauth_provider = oauth_provider
        self.agent_id = agent_id
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self.initialized = False

    async def connect_to_server(self):
        try:
            await run_with_mcp_timeout(
                self._connect_and_initialize(),
                timeout_seconds=tool_settings.mcp_connect_to_server_timeout,
                operation="server connection",
            )
            self.initialized = True
        except TimeoutError as e:
            raise LettaMCPConnectionError(
                message=str(e),
                server_name=getattr(self.server_config, "server_name", None),
            ) from e
        except LettaMCPConnectionError:
            raise
        except ConnectionError as e:
            logger.debug(f"MCP connection failed: {str(e)}")
            raise LettaMCPConnectionError(message=str(e), server_name=getattr(self.server_config, "server_name", None)) from e
        except Exception as e:
            logger.warning(
                f"Connecting to MCP server failed. Please review your server config: {self.server_config.model_dump_json(indent=4)}. Error: {str(e)}"
            )
            if hasattr(self.server_config, "server_url") and self.server_config.server_url:
                server_info = f"server URL '{self.server_config.server_url}'"
            elif hasattr(self.server_config, "command") and self.server_config.command:
                server_info = f"command '{self.server_config.command}'"
            else:
                server_info = f"server '{self.server_config.server_name}'"
            raise LettaMCPConnectionError(
                message=f"Failed to connect to MCP {server_info}. Please check your configuration and ensure the server is accessible.",
                server_name=getattr(self.server_config, "server_name", None),
            ) from e

    async def _initialize_connection(self, server_config: BaseServerConfig) -> None:
        raise NotImplementedError("Subclasses must implement _initialize_connection")

    async def _connect_and_initialize(self) -> None:
        await self._initialize_connection(self.server_config)
        await self.session.initialize()

    async def list_tools(self, serialize: bool = False) -> list[MCPTool]:
        self._check_initialized()
        response = await run_with_mcp_timeout(
            self.session.list_tools(),
            timeout_seconds=tool_settings.mcp_list_tools_timeout,
            operation="tool listing",
        )
        if serialize:
            serializable_tools = []
            for tool in response.tools:
                if hasattr(tool, "model_dump"):
                    # Pydantic model - use model_dump
                    serializable_tools.append(tool.model_dump())
                elif hasattr(tool, "dict"):
                    # Older Pydantic model - use dict()
                    serializable_tools.append(tool.dict())
                elif hasattr(tool, "__dict__"):
                    # Regular object - use __dict__
                    serializable_tools.append(tool.__dict__)
                else:
                    # Fallback - convert to string
                    serializable_tools.append(str(tool))
            return serializable_tools
        return response.tools

    async def execute_tool(self, tool_name: str, tool_args: dict) -> Tuple[str | list, bool]:
        self._check_initialized()
        try:
            result = await run_with_mcp_timeout(
                self.session.call_tool(tool_name, tool_args),
                timeout_seconds=tool_settings.mcp_execute_tool_timeout,
                operation="tool execution",
                tool_name=tool_name,
            )
        except Exception as e:
            exception_to_check = unwrap_exception_group(e)
            _log_mcp_tool_error(logger, tool_name, exception_to_check)
            return str(exception_to_check), False

        final_content = mcp_content_to_letta_parts(result.content)
        is_error = getattr(result, "isError", None)
        if is_error is None:
            is_error = getattr(result, "is_error", False)
        return final_content, not is_error

    def _check_initialized(self):
        if not self.initialized:
            logger.error("MCPClient has not been initialized")
            raise RuntimeError("MCPClient has not been initialized")

    async def cleanup(self):
        """Clean up resources used by the MCP client.

        This method handles ExceptionGroup errors that can occur when closing async context managers
        (e.g., from the MCP library's internal TaskGroup usage). Cleanup is a best-effort operation
        and errors are logged but not re-raised to prevent masking the original exception.
        """
        try:
            await self.exit_stack.aclose()
        except* Exception as eg:
            # ExceptionGroup can be raised when closing async context managers that use TaskGroup
            # Log each sub-exception at debug level since cleanup errors are expected in some cases
            # (e.g., connection already closed, server unavailable)
            for exc in eg.exceptions:
                logger.debug(f"MCP client cleanup error (suppressed): {type(exc).__name__}: {exc}")

    def to_sync_client(self):
        raise NotImplementedError("Subclasses must implement to_sync_client")
