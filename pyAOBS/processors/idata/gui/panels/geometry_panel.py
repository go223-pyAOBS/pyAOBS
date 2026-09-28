# -*- coding: utf-8 -*-
"""工区几何 Map 与检查。"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

try:
    from pyAOBS.utils.qt_combo import connect_combo_deferred
except ImportError:
    def connect_combo_deferred(combo, slot):  # type: ignore
        combo.currentIndexChanged.connect(slot)

from ...project import DEFAULT_GEOM
from ..services import header_edit
from ..services.raw2sac_paths import ensure_raw2sac_on_path
from ..services.segy_dataset import SegyDataset
from ..widgets.geom_canvas import GeomCanvas

try:
    from pyAOBS.geometry_roles import physical_shot_obs_xyz, resolve_geom
except ImportError:
    physical_shot_obs_xyz = None  # type: ignore
    resolve_geom = None  # type: ignore

ensure_raw2sac_on_path()
from segy_trace_header import resolve_offset_m  # type: ignore  # noqa: E402


def _slot_labels(geom: str) -> Tuple[str, str, str]:
    """返回 (炮图例, OBS图例, 说明文字)。"""
    mode = (geom or DEFAULT_GEOM).lower()
    if mode == "obs":
        return (
            "炮点 ← gx,gy",
            "OBS ← sx,sy",
            "旧对调：炮=gx/gy，OBS=sx/sy",
        )
    return (
        "炮点 ← sx,sy",
        "OBS ← gx,gy",
        "约定：炮=sx/sy，OBS=gx/gy",
    )


class GeometryPanel(QWidget):
    trace_selected = Signal(int)
    geom_mode_changed = Signal(str)
    log_message = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ds = SegyDataset()
        self._geom = DEFAULT_GEOM
        self._build_ui()

    def _log(self, text: str) -> None:
        self.log_message.emit(text)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        row = QHBoxLayout()
        self.geom_combo = QComboBox()
        self.geom_combo.addItem("约定：炮←sx,sy，OBS←gx,gy", DEFAULT_GEOM)
        self.geom_combo.addItem("旧对调：炮←gx,gy，OBS←sx,sy", "obs")
        self.geom_combo.setCurrentIndex(0)  # 默认约定
        connect_combo_deferred(self.geom_combo, self._on_geom_combo)
        row.addWidget(QLabel("解释模式"))
        row.addWidget(self.geom_combo, stretch=1)
        self.btn_refresh = QPushButton("刷新几何")
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        row.addWidget(QLabel("offset容差(m)"))
        self.offset_tol = QDoubleSpinBox()
        self.offset_tol.setRange(0.0, 1.0e9)
        self.offset_tol.setDecimals(1)
        self.offset_tol.setValue(5.0)
        self.offset_tol.setMaximumWidth(100)
        row.addWidget(self.offset_tol)
        self.btn_offset = QPushButton("offset 检查")
        self.btn_offset.clicked.connect(self.run_offset_check)
        self.btn_check = QPushButton("运行检查")
        self.btn_check.clicked.connect(self.run_checks)
        row.addWidget(self.btn_refresh)
        row.addWidget(self.btn_offset)
        row.addWidget(self.btn_check)
        root.addLayout(row)

        self.map_hint = QLabel("")
        self.map_hint.setStyleSheet("color:#475569;")
        self.map_hint.setWordWrap(True)
        root.addWidget(self.map_hint)

        self.canvas = GeomCanvas()
        self.canvas.trace_clicked.connect(self.trace_selected.emit)
        root.addWidget(self.canvas, stretch=3)

        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText("几何检查报告…")
        self.report.setMaximumBlockCount(2000)
        root.addWidget(self.report, stretch=1)

    def set_dataset(self, ds: SegyDataset) -> None:
        self._ds = ds
        self.canvas.clear()
        self.refresh(reset_view=True)

    def _on_refresh_clicked(self) -> None:
        self.refresh(reset_view=True)

    def geom_mode(self) -> str:
        return str(self.geom_combo.currentData() or self._geom or DEFAULT_GEOM)

    def set_geom_mode(self, mode: str, *, reset_view: bool = False) -> None:
        mode = (mode or DEFAULT_GEOM).lower()
        if mode in ("literal_segy", "约定"):
            mode = DEFAULT_GEOM
        idx = self.geom_combo.findData(mode)
        if idx < 0:
            idx = 0
            mode = DEFAULT_GEOM
        changed = mode != self._geom or self.geom_combo.currentIndex() != idx
        self._geom = mode
        if self.geom_combo.currentIndex() != idx:
            self.geom_combo.blockSignals(True)
            self.geom_combo.setCurrentIndex(idx)
            self.geom_combo.blockSignals(False)
        if changed or reset_view:
            self.refresh(reset_view=reset_view)

    def _on_geom_combo(self, _idx: int = 0) -> None:
        mode = self.geom_mode()
        if mode == self._geom:
            self.refresh(reset_view=False)
            return
        self._geom = mode
        n_shot, n_obs, _n_anc = self.refresh(reset_view=False)
        self.geom_mode_changed.emit(mode)
        shot_lab, obs_lab, hint = _slot_labels(mode)
        self._log(
            f"[几何] 解释模式 → {mode}（{hint}；{shot_lab} / {obs_lab}）  "
            f"炮点数={n_shot}  OBS数={n_obs}"
        )

    def _collect_points(self):
        shots: List[Tuple[float, float]] = []
        obs: List[Tuple[float, float]] = []
        anchors: List[Tuple[float, float, int]] = []
        shot_set: Set[Tuple[float, float]] = set()
        obs_set: Set[Tuple[float, float]] = set()
        if not self._ds.is_open:
            return shots, obs, anchors

        for i, th0 in enumerate(self._ds.headers):
            phy = self._ds.physical_xy(i)
            th = dict(th0)
            th.update(phy)
            if physical_shot_obs_xyz is not None:
                mode = resolve_geom(self._geom, [th]) if resolve_geom else self._geom
                shot, ob = physical_shot_obs_xyz(th, geom=mode, use_utm=True)
                sx, sy = float(shot[0]), float(shot[1])
                ox, oy = float(ob[0]), float(ob[1])
            else:
                # fallback：约定 shot=s*，OBS=g*；旧对调相反
                if self._geom == "obs":
                    sx, sy = phy["gx"], phy["gy"]
                    ox, oy = phy["sx"], phy["sy"]
                else:
                    sx, sy = phy["sx"], phy["sy"]
                    ox, oy = phy["gx"], phy["gy"]
            sk = (round(sx, 2), round(sy, 2))
            ok = (round(ox, 2), round(oy, 2))
            if abs(sx) + abs(sy) > 1e-6 and sk not in shot_set:
                shot_set.add(sk)
                shots.append((sx, sy))
            if abs(ox) + abs(oy) > 1e-6 and ok not in obs_set:
                obs_set.add(ok)
                obs.append((ox, oy))
            if abs(sx) + abs(sy) > 1e-6:
                anchors.append((sx, sy, i))
        return shots, obs, anchors

    def refresh(self, *, reset_view: bool = False):
        shot_lab, obs_lab, hint = _slot_labels(self._geom)
        self.map_hint.setText(
            f"{hint}　|　图例：红三角=炮点，蓝菱形=OBS，绿圈=选中炮"
        )
        shots, obs, anchors = self._collect_points()
        self.canvas.set_points(
            shots,
            obs,
            trace_anchors=anchors,
            title=(
                f"{hint}  |  炮点数={len(shots)}  OBS数={len(obs)}  "
                f"（点击炮点→道集高亮；道集选道→炮点高亮）"
            ),
            shot_label=shot_lab,
            obs_label=obs_lab,
            reset_view=reset_view,
        )
        return len(shots), len(obs), len(anchors)

    def highlight_trace(self, row: int) -> None:
        if not self._ds.is_open or row < 0 or row >= self._ds.ntraces:
            return
        phy = self._ds.physical_xy(row)
        th = dict(self._ds.get_header(row))
        th.update(phy)
        if physical_shot_obs_xyz is not None:
            mode = resolve_geom(self._geom, [th]) if resolve_geom else self._geom
            shot, ob = physical_shot_obs_xyz(th, geom=mode, use_utm=True)
            if abs(float(shot[0])) + abs(float(shot[1])) > 1e-6:
                self.canvas.set_selected_trace(row, (float(shot[0]), float(shot[1])))
            else:
                self.canvas.set_selected_trace(row, (float(ob[0]), float(ob[1])))
        else:
            if self._geom == "obs":
                self.canvas.set_selected_trace(row, (phy["gx"], phy["gy"]))
            else:
                self.canvas.set_selected_trace(row, (phy["sx"], phy["sy"]))

    def run_offset_check(self) -> None:
        """一键：道头 offset vs sx/sy/gx/gy 计算距离（报告写入本页，不弹模态窗）。"""
        if not self._ds.is_open:
            self.report.setPlainText("未打开数据。")
            return
        tol = float(self.offset_tol.value())
        result = header_edit.check_offset_vs_xy(self._ds, tol_m=tol)
        self.report.setPlainText(header_edit.format_offset_check_report(result))
        self._log(
            f"[几何] offset 检查 → 一致 {result.get('ok', 0)} / "
            f"不一致 {result.get('bad', 0)} / 共 {result.get('n', 0)}  "
            f"tol={tol:g}m  Δ均={float(result.get('mean_diff', 0)):.3f}m  "
            f"Δ最大={float(result.get('max_diff', 0)):.3f}m"
        )

    def run_checks(self) -> None:
        lines: List[str] = []
        if not self._ds.is_open:
            self.report.setPlainText("未打开数据。")
            return
        n = self._ds.ntraces
        missing_xy = 0
        bad_scalco = 0
        offset_mismatch = 0
        warn_msgs: List[str] = []

        def _warn(msg: str) -> None:
            warn_msgs.append(msg)

        for i, th0 in enumerate(self._ds.headers):
            sc = int(th0.get("scalco", 0) or 0)
            if abs(sc) not in (0, 1, 10, 100, 1000, 10000) and sc != 0:
                # 允许常见值；其它记一次
                if abs(sc) > 100000:
                    bad_scalco += 1
            phy = self._ds.physical_xy(i)
            if abs(phy["sx"]) + abs(phy["sy"]) + abs(phy["gx"]) + abs(phy["gy"]) < 1e-6:
                missing_xy += 1
                continue
            th = {
                "sx": phy["sx"],
                "sy": phy["sy"],
                "gx": phy["gx"],
                "gy": phy["gy"],
                "offset": th0.get("offset", 0),
            }
            try:
                used, src = resolve_offset_m(th, tol_m=5.0, warn=_warn)
                if src == "header":
                    # mismatch already warned
                    pass
            except Exception:
                offset_mismatch += 1

        # count unique warn
        offset_mismatch = len(warn_msgs)
        lines.append(f"traces={n}  geom={self._geom}")
        lines.append(f"missing_xy≈0: {missing_xy}")
        lines.append(f"suspicious_scalco: {bad_scalco}")
        lines.append(f"offset vs xy warnings: {offset_mismatch}")
        if warn_msgs[:20]:
            lines.append("--- samples ---")
            lines.extend(warn_msgs[:20])
        if offset_mismatch > 20:
            lines.append(f"... and {offset_mismatch - 20} more")
        self.report.setPlainText("\n".join(lines))
        self._log(
            f"[几何] 检查完成  missing_xy={missing_xy}  "
            f"suspicious_scalco={bad_scalco}  "
            f"offset_warnings={offset_mismatch}"
        )
