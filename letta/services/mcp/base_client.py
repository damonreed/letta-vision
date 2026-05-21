from contextlib import AsyncExitStack
from typing import Optional, Tuple

from mcp import ClientSession, Tool as MCPTool
from mcp.client.auth import OAuthClientProvider
from mcp.types import TextContent

from letta.errors import LettaMCPConnectionError
from letta.functions.mcp_client.types import BaseServerConfig
from letta.log import get_logger
from letta.services.mcp.tool_result_formatter import mcp_content_to_letta_parts

logger = get_logger(__name__)

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
)


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
            await self._initialize_connection(self.server_config)
            await self.session.initialize()
            self.initialized = True
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

    async def list_tools(self, serialize: bool = False) -> list[MCPTool]:
        self._check_initialized()
        response = await self.session.list_tools()
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
            result = await self.session.call_tool(tool_name, tool_args)
        except Exception as e:
            exception_to_check = e
            if hasattr(e, "exceptions") and e.exceptions and len(e.exceptions) == 1:
                exception_to_check = e.exceptions[0]
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
