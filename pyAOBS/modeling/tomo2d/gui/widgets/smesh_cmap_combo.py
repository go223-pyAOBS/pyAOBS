"""模型绘图共用色标下拉：vpvs / vs / vp（内置 GMT CPT）。"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from ...param_hints import apply_param_tooltip
from ..services.smesh_plot_core import (
    SMESH_CMAP_KEY,
    get_smesh_cmap_id,
    list_builtin_smesh_cmap_ids,
    set_smesh_cmap_id,
)
from ..state.form_state import FormState


class SmeshCmapCombo(QWidget):
    """标签 + 下拉；写入 ``gui.plot_smesh_cmap``，切换后 ``changed``。"""

    changed = Signal(str)

    def __init__(
        self,
        state: FormState,
        parent=None,
        *,
        on_changed: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self._on_changed = on_changed
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self._label = QLabel("色标")
        self.combo = QComboBox()
        for cid in list_builtin_smesh_cmap_ids():
            self.combo.addItem(cid, cid)
        self.combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.combo.setToolTip(
            "速度场色标：vp = scale_p.cpt，vs = scale_s.cpt，"
            "vpvs = scale_vpvs.cpt（Haiti vpvs1，jet 1.65–2.00），"
            "water = scale_water.cpt（水层 Vp 1.35–1.65：浅暖深冷，空气无色）。"
            "各绘图窗与顶栏共用。"
        )
        apply_param_tooltip(self, SMESH_CMAP_KEY)
        apply_param_tooltip(self._label, SMESH_CMAP_KEY)
        apply_param_tooltip(self.combo, SMESH_CMAP_KEY)
        lay.addWidget(self._label)
        lay.addWidget(self.combo)
        self.sync_from_state()
        self.combo.currentIndexChanged.connect(self._on_index)

    def current_id(self) -> str:
        data = self.combo.currentData()
        return str(data or get_smesh_cmap_id(self.state))

    def sync_from_state(self) -> None:
        want = get_smesh_cmap_id(self.state)
        idx = self.combo.findData(want)
        if idx < 0:
            idx = self.combo.findData("vp")
        self.combo.blockSignals(True)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)
        self.combo.blockSignals(False)

    def _on_index(self, _idx: int) -> None:
        cid = self.current_id()
        set_smesh_cmap_id(self.state, cid)
        self.changed.emit(cid)
        if callable(self._on_changed):
            self._on_changed(cid)
