"""Builtin plugin registrations for Workbench."""

from __future__ import annotations

from .tool_registry import ToolRegistry
from ..plugins.gui_plugins import (
    DataGuiPlugin,
    IphaseGuiPlugin,
    ImodelGuiPlugin,
    PetrologyLipGuiPlugin,
    Tomo2dGuiPlugin,
    ZplotpyGuiPlugin,
)
from ..plugins.tomo2d_plugin import Tomo2DShellPlugin


def create_builtin_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tomo2DShellPlugin())
    reg.register(Tomo2dGuiPlugin())
    reg.register(ZplotpyGuiPlugin())
    reg.register(ImodelGuiPlugin())
    reg.register(IphaseGuiPlugin())
    reg.register(PetrologyLipGuiPlugin())
    reg.register(DataGuiPlugin())
    return reg

