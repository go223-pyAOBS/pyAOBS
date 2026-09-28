# -*- coding: utf-8 -*-
"""
OBS 广角 / Madagascar 叠前偏移（RTM）PySide6 GUI

入口:
  python -m pyAOBS.visualization.obs_rtm_qt

阶段（与主窗口侧栏一致）:
  1. 数据加载  — SU → 按炮 RSF（子进程 scripts/su_to_shots.py）
  2. 工区几何  — offset/obs/segy、OBS x、网格（services.geometry）
  3. 预处理 / 道集范围 — 带通增益；应用选道 → shots_proc/
  4. 速度模型  — tomo_vel 中间体；vel + bath1d 成像速度（services.velocity）
  5. 偏移作业  — Madagascar awefd2d（scons，默认）或 custom_bin（scripts/rtm_shot_loop.py）

工区 madagascar_obs_rtm/ 仅数据（零脚本，完整分层）；CLI 在 scripts/。
目录约定见 madagascar_obs_rtm/WORKDIR_LAYOUT.txt（meta/inputs/prep/rtm_in/…）。

复用:
  processors/raw2sac/segy_trace_header   道头几何
  visualization/zplotpy/data_processor   带通 / 速度 mute
  多边形 mute 对齐 zplotpy.qt_fast_viewer（M / Shift+M）
  petrology.gui.dialog_utils             非模态对话框
"""

__all__ = ["main"]


def main(argv=None):
    from .app import main as _main
    return _main(argv)
