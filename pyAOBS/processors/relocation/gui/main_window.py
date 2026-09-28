# -*- coding: utf-8 -*-
"""OBS 姿态校正 GUI — 以 zplotpy QtFastViewer 为底座，混入姿态实现。

策略：
  - 剖面交互（分量、拾取、V 选波、叠加）复用 zplotpy
  - 姿态联合校正 / 主图预览 / 结果窗在 ``ZplotAttitudeMixin``（本包内）
  - 独立 zplotpy 不再包含姿态实现，仅跳转本工区
"""

from __future__ import annotations

from pyAOBS.visualization.zplotpy.gui.qt_fast_viewer import QtFastViewer

from .zplot_attitude_mixin import ZplotAttitudeMixin


class RelocationViewer(ZplotAttitudeMixin, QtFastViewer):
    """zplotpy Fast Viewer + 姿态校正工作站模式。"""

    # 姿态工作流不需要的参数面板（名称 = QtFastViewer 中的属性）
    _HIDDEN_PANEL_ATTRS = (
        "group_denoise",  # 去噪
        "group_ttpl",  # 走时模板 / RAYINVR 相关入口
        "group_advcorr",  # 高级校正（水层/静校正等，可按需再打开）
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OBS 姿态校正 — zplotpy")
        self._init_attitude_mixin_state()
        self._apply_relocation_mode_ui()
        try:
            self.lbl_status.setText(
                "姿态模式：完整 zplotpy 交互 | 类型=分量 | P 拾取 | Shift 连续拾取 | "
                "增益/滤波 | V 选波 → 姿态校正"
            )
        except Exception:
            pass

    def _apply_relocation_mode_ui(self) -> None:
        """隐藏无关面板，突出波形操作/拾取/增益。"""
        for attr in self._HIDDEN_PANEL_ATTRS:
            group = getattr(self, attr, None)
            if group is None:
                continue
            try:
                group.setVisible(False)
            except Exception:
                pass

        # 默认强调垂直分量浏览（与 zplotpy 一致）；3C 数据仍在内存，姿态校正可组三分量
        try:
            if hasattr(self, "combo_itype") and self.combo_itype.count() >= 2:
                # 0全部 1垂直 2径向 3横向 4水听器
                if self.combo_itype.currentIndex() == 0:
                    self.combo_itype.setCurrentIndex(1)
        except Exception:
            pass

        try:
            if getattr(self, "group_waveop", None) is not None:
                self.group_waveop.setTitle("波形/姿态")
            if getattr(self, "btn_waveop_att", None) is not None:
                self.btn_waveop_att.setText("姿态")
                self.btn_waveop_att.setToolTip(
                    "姿态校正：基于 V 段三分量+走时联合反演方位/倾角/位置。"
                    "请先用 V 选波；水深优先用工程输入页（与位置 Map 共用）。"
                    "接受解后请保存工区，下次打开工区自动加载为初值。"
                )
        except Exception:
            pass


# 兼容旧名
AttitudeMainWindow = RelocationViewer


def main() -> int:
    """独立工作台入口；工程化主入口请用 ``project_window.main``。"""
    import os
    import sys
    import warnings

    import pyqtgraph as pg
    from PySide6 import QtWidgets

    warnings.filterwarnings(
        "ignore",
        message=r"This figure includes Axes that are not compatible with tight_layout.*",
        category=UserWarning,
    )
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    existing = QtWidgets.QApplication.instance()
    created_here = existing is None
    app = QtWidgets.QApplication(sys.argv) if created_here else existing
    pg.setConfigOptions(antialias=False, useOpenGL=True)
    win = RelocationViewer()
    win.show()
    if created_here:
        return int(app.exec())
    return 0
