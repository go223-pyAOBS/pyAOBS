"""棋盘格预览：非模态三幅图（上扰动、中棋盘后、下棋盘前）。"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from ..dialog_utils import show_modeless_dialog, show_modeless_message
from ..plots.inv_monitor_model import MonitorModelWidget
from ..state.form_state import FormState


class CheckerboardPreviewWindow(QWidget):
    """独立绘图窗：等值线 / DWS 勾选与监视、对比共用表单偏好。"""

    def __init__(
        self,
        state: FormState,
        *,
        pull: Callable[[], None] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self._pull = pull
        self.setWindowTitle("棋盘格预览")
        self.resize(960, 980)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        bar = QHBoxLayout()
        self.ck_contours = QCheckBox("叠加等值线")
        from ..plots.velocity_contours import contours_enabled

        self.ck_contours.setChecked(contours_enabled(self.state))
        self.ck_contours.setToolTip(
            "勾选：中图棋盘后、下图棋盘前按当前色标叠对应等值线"
            "（vp / vs / vpvs）。上图 ΔV 不叠速度等值线。"
        )
        self.ck_dws = QCheckBox("DWS 遮罩")
        from ..services.dws_plot import dws_mask_enabled
        from ..widgets.smesh_cmap_combo import SmeshCmapCombo

        self.cmap_combo = SmeshCmapCombo(
            self.state, on_changed=lambda *_a: self.refresh(reset_home=False)
        )
        self.ck_dws.setChecked(dws_mask_enabled(self.state))
        self.ck_dws.setToolTip(
            "勾选：按背景 smesh 就近找 DWS（同目录 → 该次运行包 outputs/dws/ → "
            "表单 inv.dws_file）。无覆盖留白；有覆盖按 log(DWS) 透明。"
            "三幅图共用同一遮罩。找不到文件则整幅实色。"
        )
        btn_refresh = QPushButton("刷新")
        btn_refresh.setToolTip("按页签当前振幅 / 波长 / 背景 smesh 重算（不写盘）")
        btn_refresh.clicked.connect(lambda: self.refresh(reset_home=True))
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        bar.addWidget(self.ck_contours)
        bar.addWidget(self.cmap_combo)
        bar.addWidget(self.ck_dws)
        bar.addWidget(btn_refresh)
        bar.addStretch(1)
        bar.addWidget(btn_close)
        root.addLayout(bar)

        self.plot = MonitorModelWidget(self, compare_bar=False)
        self.plot.show_empty_stack("等待计算棋盘场…")
        self.plot.set_interaction_hint(
            "上：扰动 ΔV · 中：棋盘后 Vp · 下：棋盘前 Vp · 右键写入表单"
        )
        root.addWidget(self.plot, stretch=1)

        self.ck_contours.toggled.connect(self._on_contours_toggled)
        self.ck_dws.toggled.connect(self._on_dws_toggled)

    def _on_contours_toggled(self, on: bool) -> None:
        from ..plots.velocity_contours import set_contours_enabled

        set_contours_enabled(self.state, bool(on))
        self.refresh(reset_home=False)

    def _on_dws_toggled(self, on: bool) -> None:
        from ..services.dws_plot import set_dws_mask_enabled

        set_dws_mask_enabled(self.state, bool(on))
        self.refresh(reset_home=False)

    def refresh(self, *, reset_home: bool = True) -> None:
        from ..services.paths import resolve_work_dir
        from ..services.qc_workflows import paint_checkerboard_preview

        if callable(self._pull):
            self._pull()
        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
            paint_checkerboard_preview(
                self.plot, self.state, work, reset_home=reset_home
            )
        except Exception as e:
            show_modeless_message("棋盘预览", str(e))


def open_checkerboard_preview(
    state: FormState,
    *,
    pull: Callable[[], None] | None = None,
    existing: CheckerboardPreviewWindow | None = None,
) -> CheckerboardPreviewWindow | None:
    """打开或刷新非模态预览窗。参数无效时提示并返回 None。"""
    from ..services.paths import resolve_work_dir
    from ..services.qc_workflows import checkerboard_preview_params

    if callable(pull):
        pull()
    try:
        work = resolve_work_dir(state.get_str("work_dir"))
    except Exception as e:
        show_modeless_message("棋盘预览", str(e))
        return existing
    _bg, bg_abs, amp, h_len, v_len = checkerboard_preview_params(state, work)
    if bg_abs is None or not bg_abs.is_file():
        show_modeless_message(
            "棋盘预览",
            "请先指定存在的背景 smesh（空则回退 inv.mesh / fwd.smesh）。",
        )
        return existing
    if amp is None or h_len is None or v_len is None:
        show_modeless_message(
            "棋盘预览", "振幅 A、水平波长 h、垂向波长 v 须为数值。"
        )
        return existing

    win = existing
    try:
        if win is not None and win.isVisible():
            win.refresh(reset_home=True)
            win.raise_()
            win.activateWindow()
            return win
    except RuntimeError:
        win = None

    win = CheckerboardPreviewWindow(state, pull=pull)
    try:
        win.refresh(reset_home=True)
    except Exception as e:
        show_modeless_message("棋盘预览", str(e))
        win.deleteLater()
        return None
    show_modeless_dialog(win, activate=True)
    return win
