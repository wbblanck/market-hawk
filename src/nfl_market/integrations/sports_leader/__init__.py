"""Optional Sports Leader MCP provider (github.com/WalrusQuant/sports-leader-mcp).

Safe to import without Node, the server, or the Python ``mcp`` SDK: the SDK is imported
lazily only when a stdio connection is opened. Configure the server launch command with
``NFLMI_SPORTS_LEADER_CMD`` (for example ``node /path/to/sports-leader-mcp/dist/index.js``).
"""
from .adapter import PROVIDER_NAME, SportsLeaderProvider
from .client import ENV_COMMAND, SportsLeaderClient, SportsLeaderToolError

__all__ = ["ENV_COMMAND", "PROVIDER_NAME", "SportsLeaderClient", "SportsLeaderProvider", "SportsLeaderToolError"]
