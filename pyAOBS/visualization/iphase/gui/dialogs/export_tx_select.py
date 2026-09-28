# -*- coding: utf-8 -*-
"""预览 / 筛选 tx.in：对齐 tomo2d「预览 tx.in」——左侧勾选 OBS/震相，右侧即时预览。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..widgets.simple_check_list import SimpleCheckList

_PALETTE = [
    QColor("#E60000"),
    QColor("#0055FF"),
    QColor("#009933"),
    QColor("#FF8800"),
    QColor("#9900CC"),
    QColor("#008888"),
    QColor("#FF1493"),
    QColor("#886600"),
    QColor("#222222"),
    QColor("#00AACC"),
]


def _color(i: int) -> QColor:
    return _PALETTE[int(i) % len(_PALETTE)]


def _stable_phase_color(phase_id: int) -> QColor:
    """震相色按 phase_id 固定，与当前勾选集合无关。"""
    return _color(abs(int(phase_id)))


def _stable_obs_color(obs_key: str, catalog_keys: Sequence[str]) -> QColor:
    """OBS 色按完整台站目录序号固定。"""
    try:
        idx = list(catalog_keys).index(str(obs_key))
    except ValueError:
        idx = abs(hash(str(obs_key))) % len(_PALETTE)
    return _color(idx)


@dataclass
class TxPreviewPick:
    x: float
    t: float
    phase_id: int
    obs_key: str  # 台站模型距离，如 "30.818"（同 tomo2d 按 xobs 合并）
    obs_tag: str  # station.lis / 深度表站号，无则 "—"
    xobs: float
    source: str = ""  # 来源文件名（多文件时提示）


def _stations_from_depth_arrays(
    xs: np.ndarray | None,
    zs: np.ndarray | None,
    ids: list[str] | None,
) -> list[tuple[int, float, float]]:
    """把 OBS/炮点深度表转成 (obs_id, x, z)；无站号时用 1..N。"""
    if xs is None or np.size(xs) == 0:
        return []
    x = np.asarray(xs, dtype=float)
    z = (
        np.asarray(zs, dtype=float)
        if zs is not None and np.size(zs) == np.size(x)
        else np.zeros_like(x)
    )
    out: list[tuple[int, float, float]] = []
    for i in range(int(x.size)):
        if not np.isfinite(x[i]):
            continue
        oid = i + 1
        if ids is not None and i < len(ids):
            s = str(ids[i]).strip()
            if s.isdigit():
                oid = int(s)
        out.append((int(oid), float(x[i]), float(z[i]) if np.isfinite(z[i]) else 0.0))
    return out


def _nearest_obs_id(
    x: float,
    stations: Sequence[tuple[int, float, float]] | None,
    *,
    tol: float = 0.001,
) -> int | None:
    if not stations:
        return None
    best: int | None = None
    best_d = float(tol)
    for oid, sx, _sz in stations:
        d = abs(float(x) - float(sx))
        if d <= best_d:
            best_d = d
            best = int(oid)
    return best


def build_preview_picks_from_results(
    results: Sequence,
    *,
    stations: Sequence[tuple[int, float, float]] | None = None,
    x_decimals: int = 3,
) -> list[TxPreviewPick]:
    """
    从已加载 FileResult 抽取预览点。

    按每个 shot 的 ``xshot``（台站模型距离）分 OBS，与 tomo2d「预览 tx.in」一致；
    合并文件（如 ``tx_st1_65.in``）会列出多个台站，而不是整文件一行。
    """
    out: list[TxPreviewPick] = []
    nd = int(x_decimals)
    for i, r in enumerate(results):
        path = getattr(r, "path", None)
        src = Path(path).name if path else f"idx:{i}"
        for shot in r.ds.shots:
            xobs_f = float(shot.xshot)
            xobs_r = round(xobs_f, nd)
            key = f"{xobs_r:.{nd}f}"
            oid = _nearest_obs_id(xobs_r, stations, tol=10.0 ** (-nd))
            tag = str(oid) if oid is not None else "—"
            for p in shot.picks:
                out.append(
                    TxPreviewPick(
                        x=float(p.x),
                        t=float(p.t),
                        phase_id=int(p.phase_id),
                        obs_key=key,
                        obs_tag=tag,
                        xobs=xobs_r,
                        source=src,
                    )
                )
    return out


def build_obs_records(picks: Sequence[TxPreviewPick]) -> list[dict]:
    """按台站 xobs 合并左右支，标签对齐 tomo2d：``站号  x=… (n=…)  L…/R…``。"""
    from pyAOBS.modeling.rayinvr.tx_obs_catalog import format_obs_catalog_label

    by: dict[str, dict] = {}
    for p in picks:
        rec = by.get(p.obs_key)
        if rec is None:
            by[p.obs_key] = {
                "key": p.obs_key,
                "id": p.obs_key,
                "obs_tag": p.obs_tag,
                "xobs": float(p.xobs),
                "n": 1,
                "n_left": 1 if p.x < p.xobs - 1e-6 else 0,
                "n_right": 1 if p.x > p.xobs + 1e-6 else 0,
                "sources": {p.source} if p.source else set(),
            }
        else:
            rec["n"] = int(rec["n"]) + 1
            if p.x < float(rec["xobs"]) - 1e-6:
                rec["n_left"] = int(rec["n_left"]) + 1
            elif p.x > float(rec["xobs"]) + 1e-6:
                rec["n_right"] = int(rec["n_right"]) + 1
            if p.obs_tag != "—" and (
                rec["obs_tag"] == "—" or not str(rec["obs_tag"]).isdigit()
            ):
                rec["obs_tag"] = p.obs_tag
            if p.source:
                rec["sources"].add(p.source)
    records: list[dict] = []
    for key, rec in sorted(by.items(), key=lambda kv: float(kv[1]["xobs"])):
        tag = str(rec["obs_tag"])
        obs_id: int | None = int(tag) if tag.isdigit() else None
        label = format_obs_catalog_label(
            obs_id=obs_id,
            xobs=float(rec["xobs"]),
            n=int(rec["n"]),
            n_left=int(rec.get("n_left") or 0),
            n_right=int(rec.get("n_right") or 0),
        )
        records.append(
            {
                "key": key,
                "id": key,
                "label": label,
                "obs_tag": tag,
                "xobs": float(rec["xobs"]),
            }
        )
    return records


def build_phase_records(picks: Sequence[TxPreviewPick]) -> list[dict]:
    counts: dict[int, int] = {}
    for p in picks:
        counts[p.phase_id] = counts.get(p.phase_id, 0) + 1
    return [
        {"id": pid, "key": str(pid), "label": f"phase {pid}  (n={counts[pid]})"}
        for pid in sorted(counts)
    ]


class ExportTxSelectDialog(QDialog):
    """非模态：左侧勾选 OBS/震相（同 tomo2d），右侧折合走时预览，可导出。"""

    export_requested = Signal(list, list)  # obs_keys, phase_ids

    def __init__(
        self,
        parent=None,
        *,
        picks: Sequence[TxPreviewPick] | None = None,
        obs_items: Sequence[tuple[str, str]] | None = None,
        phase_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("预览 / 筛选 tx.in")
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(1100, 720)
        self.setMinimumSize(720, 480)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self._picks: list[TxPreviewPick] = list(picks or [])
        self._keep_view = False

        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("折合速度 vred (km/s)："))
        self.spin_vred = QDoubleSpinBox()
        self.spin_vred.setRange(0.0, 20.0)
        self.spin_vred.setDecimals(3)
        self.spin_vred.setSingleStep(0.1)
        self.spin_vred.setValue(7.0)
        self.spin_vred.setToolTip(
            "t′ = t − |x−xobs|/vred；x 为模型距离；设为 0 显示真走时"
        )
        bar.addWidget(self.spin_vred)
        self.chk_by_obs = QCheckBox("按 OBS 着色")
        self.chk_by_obs.setChecked(False)
        self.chk_by_obs.setToolTip("开启：按 OBS 着色；关闭：按震相号着色")
        bar.addWidget(self.chk_by_obs)
        self.chk_mark_obs = QCheckBox("标 OBS 位置")
        self.chk_mark_obs.setChecked(True)
        bar.addWidget(self.chk_mark_obs)
        bar.addStretch(1)
        self.btn_reset = QPushButton("复位")
        self.btn_reset.clicked.connect(self._reset_view)
        self.btn_export = QPushButton("导出筛选 tx.in…")
        self.btn_export.setToolTip(
            "按左侧勾选写出每个 OBS 的 tx_*_sel.in（默认工区 outputs/）"
        )
        self.btn_export.clicked.connect(self._on_export)
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.close)
        bar.addWidget(self.btn_reset)
        bar.addWidget(self.btn_export)
        bar.addWidget(self.btn_close)
        root.addLayout(bar)

        split = QSplitter(Qt.Orientation.Horizontal)
        side = QWidget()
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(0, 0, 0, 0)
        side_lay.setSpacing(8)

        self.obs_list = SimpleCheckList(heading="显示 OBS（按台站 x 合并，可多选）")
        self.obs_list.setToolTip(
            "按炮头 xshot（台站模型距离）合并左右支，与 tomo2d「预览 tx.in」一致；"
            "若已加载 station.lis / OBS 深度表则显示站号。"
        )
        self.phase_list = SimpleCheckList(heading="显示震相（tx.in phase）")
        self.phase_list.setToolTip("只绘制 / 导出勾选的震相号")

        side_split = QSplitter(Qt.Orientation.Vertical)
        side_split.setChildrenCollapsible(False)
        side_split.addWidget(self.obs_list)
        side_split.addWidget(self.phase_list)
        side_split.setStretchFactor(0, 3)
        side_split.setStretchFactor(1, 2)
        side_split.setSizes([420, 280])
        side_lay.addWidget(side_split, stretch=1)
        split.addWidget(side)

        self.plot = pg.PlotWidget(background="w")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "Model distance", units="km")
        self.plot.setLabel("left", "Traveltime", units="s")
        self.plot.invertY(True)
        try:
            self.plot.setMenuEnabled(False)
            vb = self.plot.getViewBox()
            vb.setMenuEnabled(False)
            vb.disableAutoRange()
        except Exception:
            pass
        split.addWidget(self.plot)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([280, 820])
        root.addWidget(split, stretch=1)

        self._tip = QLabel(
            "左侧勾选 OBS / 震相即时预览 · 滚轮缩放 · 左拖平移 · 右拖缩放 · 双击复位"
        )
        self._tip.setStyleSheet("color:#666; font-size:11px;")
        root.addWidget(self._tip)

        self.obs_list.selection_changed.connect(self._on_param_changed)
        self.phase_list.selection_changed.connect(self._on_param_changed)
        self.spin_vred.valueChanged.connect(self._on_param_changed)
        self.chk_by_obs.toggled.connect(self._on_param_changed)
        self.chk_mark_obs.toggled.connect(self._on_param_changed)

        if self._picks:
            self._reload_lists_from_picks()
        elif obs_items is not None and phase_ids is not None:
            # 兼容旧调用：仅列表、无预览点
            self.obs_list.set_records(
                [{"key": k, "id": k, "label": lab} for k, lab in obs_items]
            )
            self.phase_list.set_records(
                [{"id": int(p), "key": str(int(p)), "label": f"phase {int(p)}"} for p in phase_ids]
            )
        self._redraw(fit=True)

    def set_picks(self, picks: Sequence[TxPreviewPick]) -> None:
        self._picks = list(picks)
        self._reload_lists_from_picks()
        self._redraw(fit=True)

    def set_items(
        self,
        *,
        obs_items: Sequence[tuple[str, str]],
        phase_ids: Sequence[int],
    ) -> None:
        """兼容旧接口：仅刷新勾选列表（无新预览点时）。"""
        self.obs_list.set_records(
            [{"key": k, "id": k, "label": lab} for k, lab in obs_items]
        )
        self.phase_list.set_records(
            [{"id": int(p), "key": str(int(p)), "label": f"phase {int(p)}"} for p in phase_ids]
        )
        self._redraw(fit=False)

    def _reload_lists_from_picks(self) -> None:
        prev_obs = self.obs_list.checked_keys()
        prev_ph = self.phase_list.checked_ids()
        self.obs_list.set_records(build_obs_records(self._picks))
        self.phase_list.set_records(build_phase_records(self._picks))
        if prev_obs:
            # 尽量保留勾选
            want = set(prev_obs)
            self.obs_list._applying = True
            self.obs_list.list_w.blockSignals(True)
            for i in range(self.obs_list.list_w.count()):
                item = self.obs_list.list_w.item(i)
                if item is None:
                    continue
                rec = item.data(Qt.ItemDataRole.UserRole) or {}
                on = str(rec.get("key", "")) in want
                item.setCheckState(
                    Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
                )
            self.obs_list.list_w.blockSignals(False)
            self.obs_list._applying = False
        if prev_ph:
            want_p = set(prev_ph)
            self.phase_list._applying = True
            self.phase_list.list_w.blockSignals(True)
            for i in range(self.phase_list.list_w.count()):
                item = self.phase_list.list_w.item(i)
                if item is None:
                    continue
                rec = item.data(Qt.ItemDataRole.UserRole) or {}
                on = int(rec.get("id", -1)) in want_p
                item.setCheckState(
                    Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
                )
            self.phase_list.list_w.blockSignals(False)
            self.phase_list._applying = False

    def _on_export(self) -> None:
        obs_keys = self.obs_list.checked_keys()
        phase_ids = self.phase_list.checked_ids()
        self.export_requested.emit(obs_keys, phase_ids)

    def _on_param_changed(self, *_a) -> None:
        self._redraw(fit=False)

    def _reset_view(self) -> None:
        self._redraw(fit=True)

    def _view_range(self):
        try:
            xr, yr = self.plot.getViewBox().viewRange()
            return [float(xr[0]), float(xr[1])], [float(yr[0]), float(yr[1])]
        except Exception:
            return None

    def _apply_view(self, *, fit: bool, rng) -> None:
        vb = self.plot.getViewBox()
        try:
            self.plot.invertY(True)
        except Exception:
            pass
        if fit or rng is None:
            vb.enableAutoRange(enable=True)
            self.plot.autoRange()
        else:
            vb.setRange(xRange=rng[0], yRange=rng[1], padding=0)
        vb.disableAutoRange()
        self._keep_view = True

    def _filtered_picks(self) -> list[TxPreviewPick]:
        obs_sel = self.obs_list.selected_keys()
        ph_sel = self.phase_list.selected_ids()
        out: list[TxPreviewPick] = []
        for p in self._picks:
            if obs_sel is not None and p.obs_key not in obs_sel:
                continue
            if ph_sel is not None and p.phase_id not in ph_sel:
                continue
            out.append(p)
        return out

    def _redraw(self, *, fit: bool) -> None:
        rng = None if fit else self._view_range()
        self.plot.clear()
        vred = float(self.spin_vred.value())
        by_obs = bool(self.chk_by_obs.isChecked())
        picks = self._filtered_picks()
        if not picks:
            self.plot.setTitle("未选择 OBS / 震相")
            self._tip.setText("左侧勾选 OBS 与震相后显示预览点")
            self._apply_view(fit=True, rng=None)
            return

        # group key → points；颜色按震相号 / 全量 OBS 目录固定，勾选变化不改色
        groups: dict[str, list[TxPreviewPick]] = {}
        for p in picks:
            g = p.obs_key if by_obs else f"ph:{p.phase_id}"
            groups.setdefault(g, []).append(p)

        obs_catalog_keys = [str(r["key"]) for r in build_obs_records(self._picks)]
        keys = sorted(groups.keys())
        all_ys: list[float] = []
        for g in keys:
            pts = groups[g]
            xs = np.asarray([p.x for p in pts], dtype=float)
            ts = np.asarray([p.t for p in pts], dtype=float)
            xobs = np.asarray([p.xobs for p in pts], dtype=float)
            if vred > 1e-9:
                ys = ts - np.abs(xs - xobs) / vred
            else:
                ys = ts
            all_ys.extend(float(v) for v in ys if np.isfinite(v))
            if by_obs:
                brush = _stable_obs_color(g, obs_catalog_keys)
            else:
                # g == "ph:{id}"
                try:
                    pid = int(str(g).split(":", 1)[1])
                except Exception:
                    pid = abs(hash(g)) % 97
                brush = _stable_phase_color(pid)
            scatter = pg.ScatterPlotItem(
                x=xs,
                y=ys,
                size=6,
                pen=None,
                brush=brush,
                tip=None,
            )
            self.plot.addItem(scatter)

        y_mark = float(np.nanmin(all_ys)) if all_ys else 0.0
        if self.chk_mark_obs.isChecked():
            seen: set[float] = set()
            for p in picks:
                xo = round(float(p.xobs), 3)
                if xo in seen:
                    continue
                seen.add(xo)
                line = pg.InfiniteLine(
                    pos=xo,
                    angle=90,
                    pen=pg.mkPen("#888888", width=1, style=Qt.PenStyle.DashLine),
                )
                self.plot.addItem(line)
                tag = p.obs_tag
                label = tag if tag not in ("", "—") else f"{p.xobs:.3f}"
                text = pg.TextItem(label, color="#444444", anchor=(0.5, 1.0))
                text.setPos(xo, y_mark)
                self.plot.addItem(text)

        ylab = f"t' (vred={vred:g})" if vred > 1e-9 else "t (s)"
        self.plot.setLabel("left", ylab, units="s")
        self.plot.setTitle(f"{len(picks)} picks · {len(groups)} series")
        self._tip.setText(
            f"显示 {len(picks)} 点 · 勾选 OBS={len(self.obs_list.checked_keys())} "
            f"震相={len(self.phase_list.checked_ids())} · 滚轮/左拖/右拖/双击复位"
        )
        self._apply_view(fit=fit, rng=rng)
