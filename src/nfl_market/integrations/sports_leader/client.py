"""Thin synchronous client for the Sports Leader MCP server.

The server is a Node/TypeScript MCP server launched as a subprocess and spoken to over
stdio with the Python ``mcp`` SDK. The SDK is imported lazily, so this module imports
without it. Every caller goes through a ``call_tool(name, args) -> dict`` callable, so tests
(and other transports) can inject a fake that returns recorded JSON.

The server returns each tool result as compact JSON in a text block and, for objects, also
as ``structuredContent`` (sports-leader-mcp ``src/server.ts`` ``toTextResult``). Errors come
back as ``isError: true`` with the message in the text block (``toErrorResult``).
"""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import threading
from collections.abc import Callable
from typing import Any, Self

ENV_COMMAND = "NFLMI_SPORTS_LEADER_CMD"
ENV_TIMEOUT = "NFLMI_SPORTS_LEADER_TIMEOUT"
DEFAULT_TIMEOUT_S = 60.0

# The mcp SDK launches servers with a minimal environment (PATH, HOME, ...). These variables are
# passed through so the Node server can reach the network behind a proxy and find its data
# export settings (SPORTSLEADER_DATA_* / BAYES_DATA_*, see sports-leader-mcp src/data/tools.ts).
PASSTHROUGH_PREFIXES = ("SPORTSLEADER_", "BAYES_", "NODE_")
PASSTHROUGH_NAMES = ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
                     "SSL_CERT_FILE")


def passthrough_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k in PASSTHROUGH_NAMES or k.startswith(PASSTHROUGH_PREFIXES)}


CallTool = Callable[[str, dict[str, Any]], dict[str, Any]]


class SportsLeaderToolError(RuntimeError):
    """The server answered a tool call with ``isError`` or an unusable payload."""


def configured_command(command: str | list[str] | None = None) -> list[str] | None:
    """The launch command as argv, from the argument or ``NFLMI_SPORTS_LEADER_CMD``; None if unset."""
    raw = command if command is not None else os.environ.get(ENV_COMMAND, "")
    argv = list(raw) if isinstance(raw, list) else shlex.split(raw)
    return argv or None


def mcp_sdk_available() -> bool:
    """True when the Python ``mcp`` SDK can be imported (checked without importing it)."""
    if "mcp" in sys.modules:
        return sys.modules["mcp"] is not None
    try:
        import importlib.util

        return importlib.util.find_spec("mcp") is not None
    except (ImportError, ValueError):
        return False


def result_to_dict(result: Any) -> dict[str, Any]:
    """Turn an SDK ``CallToolResult`` (1.x camelCase or 2.x snake_case) into the tool's JSON object."""
    is_error = getattr(result, "is_error", None)
    if is_error is None:
        is_error = getattr(result, "isError", False)
    texts = [getattr(block, "text", None) for block in (getattr(result, "content", None) or [])]
    texts = [t for t in texts if isinstance(t, str)]
    if is_error:
        raise SportsLeaderToolError(texts[0] if texts else "tool returned an error with no message")
    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    for text in texts:
        try:
            data = json.loads(text)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"items": data}  # same wrapping the server applies to structuredContent
    raise SportsLeaderToolError("tool result had no JSON object")


class StdioTransport:
    """Keeps one stdio connection to the server open on a private event-loop thread.

    Use as a context manager; ``__call__(name, args)`` is the injectable ``call_tool``.
    """

    def __init__(self, argv: list[str], *, timeout_s: float = DEFAULT_TIMEOUT_S, env: dict[str, str] | None = None):
        if not argv:
            raise ValueError("empty Sports Leader launch command")
        self.argv = argv
        self.timeout_s = timeout_s
        self.env = passthrough_env() if env is None else env
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        try:
            import mcp  # lazy: optional dependency

            params = mcp.StdioServerParameters(command=self.argv[0], args=self.argv[1:], env=self.env)
            if hasattr(mcp, "Client"):  # SDK 2.x high-level client
                async with mcp.Client(params, read_timeout_seconds=self.timeout_s) as client:
                    self._session = client
                    self._ready.set()
                    await self._stop.wait()
            else:  # SDK 1.x
                from mcp.client.stdio import stdio_client

                async with stdio_client(params) as (read, write), mcp.ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()
        except BaseException as e:  # noqa: BLE001 - surfaced to the caller of open()
            self._error = e
        finally:
            self._session = None
            self._ready.set()

    def open(self) -> StdioTransport:
        if self._thread is not None:
            return self
        self._loop = asyncio.new_event_loop()

        def run() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._serve())

        self._thread = threading.Thread(target=run, name="sports-leader-mcp", daemon=True)
        self._thread.start()
        self._ready.wait(self.timeout_s)
        if self._session is None:
            err = self._error
            self.close()
            raise SportsLeaderToolError(f"could not start Sports Leader MCP server {self.argv!r}: {err!r}")
        return self

    def __call__(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self._session is None or self._loop is None:
            raise SportsLeaderToolError("transport is not open")
        future = asyncio.run_coroutine_threadsafe(self._session.call_tool(name, args), self._loop)
        return result_to_dict(future.result(self.timeout_s))

    def close(self) -> None:
        if self._loop is not None and self._stop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(self.timeout_s)
        if self._loop is not None and not self._loop.is_closed() and not (self._thread and self._thread.is_alive()):
            self._loop.close()
        self._thread = None
        self._loop = None

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()


class SportsLeaderClient:
    """Calls named Sports Leader tools. Pass ``call_tool`` to inject a transport (tests use
    a fake returning fixture JSON); otherwise a stdio connection is opened on first use."""

    def __init__(self, call_tool: CallTool | None = None, *, command: str | list[str] | None = None,
                 timeout_s: float | None = None):
        self._injected = call_tool
        self.command = configured_command(command)
        if timeout_s is None:
            try:
                timeout_s = float(os.environ.get(ENV_TIMEOUT, DEFAULT_TIMEOUT_S))
            except ValueError:
                timeout_s = DEFAULT_TIMEOUT_S
        self.timeout_s = timeout_s
        self._transport: StdioTransport | None = None

    def configured(self) -> bool:
        """True when a transport was injected, or the SDK is importable and a command is set."""
        return self._injected is not None or (self.command is not None and mcp_sdk_available())

    def call(self, name: str, **args: Any) -> dict[str, Any]:
        """Call one tool. ``None`` arguments are dropped (the server's zod schemas treat absent as default)."""
        clean = {k: v for k, v in args.items() if v is not None}
        if self._injected is not None:
            result = self._injected(name, clean)
        else:
            if self.command is None:
                raise SportsLeaderToolError(f"{ENV_COMMAND} is not set")
            if self._transport is None:
                self._transport = StdioTransport(self.command, timeout_s=self.timeout_s).open()
            result = self._transport(name, clean)
        if not isinstance(result, dict):
            raise SportsLeaderToolError(f"{name} returned {type(result).__name__}, expected an object")
        if result.get("truncated") is True:
            raise SportsLeaderToolError(f"{name} result was truncated by the server's token budget: "
                                        f"{result.get('message', '')}")
        return result

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
