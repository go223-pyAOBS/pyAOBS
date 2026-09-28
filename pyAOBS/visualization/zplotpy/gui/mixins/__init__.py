# -*- coding: utf-8 -*-
"""QtFastViewer 功能混入（mixin）包。

``QtFastViewer`` 本身是薄壳（``__init__`` + 生命周期），业务方法按主题拆到本目录。
Relocation 等子类继续继承 ``QtFastViewer``，公开方法名不变。

索引（文件 → 职责）
==================

| 模块 | 类 | 职责 |
|------|----|------|
| ``denoise_mixin`` | DenoiseMixin | 去噪参数、选道、相干门控、应用到渲染道 |
| ``denoise_compare_mixin`` | DenoiseCompareMixin | 去噪 A/B 对比图、全道谱剖面 |
| ``location_map_mixin`` | LocationMapMixin | 位置 Map 对话框、色标、光标、跳转 |
| ``location_terrain_mixin`` | LocationTerrainMixin | 地形叠加、坐标转换、``ensure_shared_terrain_loaded`` |
| ``param_panel_mixin`` | ParamPanelMixin | 参数条高度/拖拽/布局 + 折叠与 action bar |
| ``mute_mixin`` | MuteMixin | 多边形 mute |
| ``waveop_mixin`` | WaveopMixin | V 选波、列表停靠、叠加存取 |
| ``pick_mixin`` | PickMixin | 拾取、撤销、TXIN、对齐、自动/插值拾取 |
| ``travel_static_mixin`` | TravelStaticMixin | 理论走时、水层/静校正、减速度时移 |
| ``theme_mixin`` | ThemeMixin | 主题预设、样式表、主题感知对话框 |
| ``help_mixin`` | HelpMixin | 帮助 / 快捷键 / 悬停提示 |
| ``data_info_mixin`` | DataInfoMixin | 数据信息、坐标参数 |
| ``stack_mixin`` | StackMixin | 叠加显示与评价 |
| ``build_ui_mixin`` | BuildUiMixin | 主界面控件与布局构建 |
| ``file_io_mixin`` | FileIoMixin | 选文件、加载、参数/图导出、炮号导航 |
| ``ui_params_mixin`` | UiParamsMixin | UI↔处理参数同步、自动比例、增益预设 |
| ``plot_interaction_mixin`` | PlotInteractionMixin | 事件绑定、鼠标/键盘交互 |
| ``trace_geom_mixin`` | TraceGeomMixin | 道坐标、几何角色推断 |
| ``render_core_mixin`` | RenderCoreMixin | ``request_render`` / ``_render_now``、曲线池/阴影/密度图项 |

约定
----
- Mixin **不要**插在 ``ZplotAttitudeMixin`` 与 ``QtFastViewer`` 之间（见 relocation）。
- 长生命周期子窗保持非模态（``show_modeless_dialog``）。
- 新增功能优先落对应 mixin；跨主题胶水留在壳或 ``_wire_events``。
"""

from .denoise_mixin import DenoiseMixin
from .denoise_compare_mixin import DenoiseCompareMixin
from .location_map_mixin import LocationMapMixin
from .location_terrain_mixin import LocationTerrainMixin
from .param_panel_mixin import ParamPanelMixin
from .mute_mixin import MuteMixin
from .waveop_mixin import WaveopMixin
from .pick_mixin import PickMixin
from .travel_static_mixin import TravelStaticMixin
from .theme_mixin import ThemeMixin
from .help_mixin import HelpMixin
from .data_info_mixin import DataInfoMixin
from .stack_mixin import StackMixin
from .build_ui_mixin import BuildUiMixin
from .file_io_mixin import FileIoMixin
from .ui_params_mixin import UiParamsMixin
from .plot_interaction_mixin import PlotInteractionMixin
from .trace_geom_mixin import TraceGeomMixin
from .render_core_mixin import RenderCoreMixin

__all__ = [
    "DenoiseMixin",
    "DenoiseCompareMixin",
    "LocationMapMixin",
    "LocationTerrainMixin",
    "ParamPanelMixin",
    "MuteMixin",
    "WaveopMixin",
    "PickMixin",
    "TravelStaticMixin",
    "ThemeMixin",
    "HelpMixin",
    "DataInfoMixin",
    "StackMixin",
    "BuildUiMixin",
    "FileIoMixin",
    "UiParamsMixin",
    "PlotInteractionMixin",
    "TraceGeomMixin",
    "RenderCoreMixin",
]
