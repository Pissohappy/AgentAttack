from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MCPToolCall:
    server: str
    method: str
    arguments: dict[str, Any] = field(default_factory=dict)


class MCPRegistry:
    """Lightweight MCP abstraction to keep planner vendor-agnostic."""

    def __init__(self) -> None:
        self._handlers: dict[tuple[str, str], Any] = {}

    def register(self, server: str, method: str, handler: Any) -> None:
        self._handlers[(server, method)] = handler

    def call(self, request: MCPToolCall) -> Any:
        handler = self._handlers.get((request.server, request.method))
        if handler is None:
            raise KeyError(f"No MCP handler for {request.server}.{request.method}")
        return handler(request.arguments)
