"""Plugin/tool registry for Workbench."""

from __future__ import annotations

from typing import Iterable

from ..plugins.base import WorkbenchPlugin


class ToolRegistry:
    """Simple in-memory plugin registry."""

    def __init__(self) -> None:
        self._plugins: dict[str, WorkbenchPlugin] = {}

    def register(self, plugin: WorkbenchPlugin) -> None:
        self._plugins[plugin.id] = plugin

    def get(self, plugin_id: str) -> WorkbenchPlugin:
        return self._plugins[plugin_id]

    def all(self) -> Iterable[WorkbenchPlugin]:
        return self._plugins.values()

    def ids(self) -> list[str]:
        return list(self._plugins.keys())

