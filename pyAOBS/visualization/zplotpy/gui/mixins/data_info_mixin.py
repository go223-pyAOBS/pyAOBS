# -*- coding: utf-8 -*-
"""Data / coordinate info dialogs mixed into QtFastViewer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover
    raise RuntimeError("未安装 PySide6") from exc


class DataInfoMixin:
    """数据信息与坐标参数查阅。"""

    def _compose_data_info_text(self) -> str:
        if self.loaded is None:
            return "当前尚未加载数据。"
        header = self.loaded.get("header")
        traces = self.loaded.get("traces", [])
        offsets = self.loaded.get("offsets", [])
        times = self.loaded.get("times", [])
        trace_headers = self.loaded.get("trace_headers", []) or []
        picks_count = 0
        if self.pick_manager is not None:
            picks_count = int(self.pick_manager.count_picks())

        # 文件信息
        def _file_size_text(path: Optional[str]) -> str:
            if not path:
                return "-"
            try:
                p = Path(path)
                if not p.exists():
                    return "不存在"
                sz = float(p.stat().st_size)
                if sz < 1024:
                    return f"{int(sz)} B"
                if sz < 1024 * 1024:
                    return f"{sz / 1024.0:.1f} KB"
                if sz < 1024 * 1024 * 1024:
                    return f"{sz / (1024.0 * 1024.0):.2f} MB"
                return f"{sz / (1024.0 * 1024.0 * 1024.0):.2f} GB"
            except Exception:
                return "未知"

        # 采样与范围信息
        dt_ms = 0.0
        if len(times) >= 2:
            try:
                dt_ms = (float(times[1]) - float(times[0])) * 1000.0
            except Exception:
                dt_ms = 0.0
        tmin = float(np.min(times)) if len(times) > 0 else 0.0
        tmax = float(np.max(times)) if len(times) > 0 else 0.0
        xmin = float(np.min(offsets)) if len(offsets) > 0 else 0.0
        xmax = float(np.max(offsets)) if len(offsets) > 0 else 0.0

        # 道头统计
        rec_ids = sorted(
            {
                int(getattr(th, "ishoti", 0) or 0)
                for th in trace_headers
                if int(getattr(th, "ishoti", 0) or 0) > 0
            }
        )
        dead_count = 0
        comp_counter: Dict[int, int] = {}
        for th in trace_headers:
            try:
                if int(getattr(th, "iflagi", 1) or 1) != 1:
                    dead_count += 1
            except Exception:
                pass
            c = int(getattr(th, "itypei", 0) or 0)
            comp_counter[c] = int(comp_counter.get(c, 0)) + 1

        comp_name = {0: "全部/未标注", 1: "垂直", 2: "径向", 3: "横向", 4: "水听器"}
        comp_text = ", ".join(
            f"{comp_name.get(k, str(k))}:{v}" for k, v in sorted(comp_counter.items(), key=lambda kv: kv[0])
        ) or "-"

        # 拾取统计
        pick_word_max = int(getattr(header, "npick", 0) or 0)
        active_word = int(self.spin_apick.value()) if hasattr(self, "spin_apick") else 1
        active_word_count = 0
        if self.pick_manager is not None:
            by_word = self.pick_manager.get_picks_by_word(active_word)
            active_word_count = len(by_word)

        # 头信息中的真实参数（来自数据，不依赖当前界面控件）
        def _header_value(name: str) -> str:
            if header is None:
                return "-"
            try:
                v = getattr(header, name)
                if v is None:
                    return "-"
                if isinstance(v, float):
                    return f"{v:.6g}"
                return str(v)
            except Exception:
                return "-"

        msg = (
            "[文件]\n"
            f"dfile: {self._dfile or '-'}\n"
            f"  大小: {_file_size_text(self._dfile)}\n"
            f"hfile: {self._hfile or '-'}\n"
            f"  大小: {_file_size_text(self._hfile)}\n"
            f"rfile: {self._rfile or '-'}\n"
            f"  大小: {_file_size_text(self._rfile)}\n\n"
            "[数据规模]\n"
            f"ntraces(header): {int(getattr(header, 'ntraces', 0) or 0)}\n"
            f"npts(header): {int(getattr(header, 'npts', 0) or 0)}\n"
            f"npick(header): {pick_word_max}\n"
            f"trace arrays: {len(traces)}\n"
            f"offset count: {len(offsets)}\n"
            f"time samples: {len(times)}\n\n"
            "[采样与范围]\n"
            f"采样间隔 dt: {dt_ms:.3f} ms\n"
            f"时间范围: [{tmin:.4f}, {tmax:.4f}] s\n"
            f"偏移范围: [{xmin:.4f}, {xmax:.4f}] km\n\n"
            "[道头统计]\n"
            f"记录(shot)数量: {len(rec_ids)}\n"
            f"记录号范围: {rec_ids[0] if rec_ids else '-'} ~ {rec_ids[-1] if rec_ids else '-'}\n"
            f"死道数量: {dead_count}\n"
            f"分量统计: {comp_text}\n\n"
            "[拾取统计]\n"
            f"全部拾取点: {picks_count}\n"
            f"当前活动字(apick={active_word})拾取道数: {active_word_count}\n\n"
            "[数据头真实参数]\n"
            f"vredf(header): {_header_value('vredf')}\n"
            f"nrec(header): {_header_value('nrec')}\n"
            f"ntraces(header): {_header_value('ntraces')}\n"
            f"npts(header): {_header_value('npts')}\n"
            f"npick(header): {_header_value('npick')}\n"
            f"f1(header): {_header_value('f1')}\n"
            f"dt(header): {_header_value('dt')}"
        )
        return msg


    def _show_data_info(self) -> None:
        self._show_data_information(initial_tab=0)


    def _show_coordinate_parameters(self) -> None:
        self._show_data_information(initial_tab=1)


    def _show_data_information(self, initial_tab: int = 0) -> None:
        if self.loaded is None:
            self._show_themed_info("数据信息", "当前尚未加载数据。")
            return
        msg = self._compose_data_info_text()
        if getattr(self, "_data_info_dialog", None) is not None:
            try:
                self._data_info_dialog.close()
            except Exception:
                pass
            self._data_info_dialog = None

        dialog = QtWidgets.QDialog(self)
        self._data_info_dialog = dialog
        dialog.setWindowTitle("数据信息")
        dialog.resize(1220, 680)
        dialog.setModal(False)
        dialog.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.destroyed.connect(lambda *_: setattr(self, "_data_info_dialog", None))
        layout = QtWidgets.QVBoxLayout(dialog)
        tabs = QtWidgets.QTabWidget(dialog)
        layout.addWidget(tabs, stretch=1)

        overview = QtWidgets.QPlainTextEdit(dialog)
        overview.setReadOnly(True)
        overview.setPlainText(msg)
        tabs.addTab(overview, "数据概览")

        header_page = QtWidgets.QWidget(dialog)
        header_lay = QtWidgets.QVBoxLayout(header_page)
        tip = QtWidgets.QLabel("以下为 .z/.hdr 的文件头参数与全部道头参数统计（含参数说明）。", header_page)
        tip.setWordWrap(True)
        header_lay.addWidget(tip)
        table = self._build_coord_params_table(header_page)
        header_lay.addWidget(table, stretch=1)
        tabs.addTab(header_page, "道头参数")

        tabs.setCurrentIndex(0 if int(initial_tab) <= 0 else 1)
        close_btn = QtWidgets.QPushButton("关闭", dialog)
        close_btn.clicked.connect(dialog.close)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close_btn)
        layout.addLayout(row)
        self._register_floating_dialog(dialog)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()


    def _build_coord_params_table(self, parent: QtWidgets.QWidget) -> QtWidgets.QTableWidget:
        header = self.loaded.get("header") if self.loaded is not None else None
        trace_headers = (self.loaded.get("trace_headers", []) or []) if self.loaded is not None else []

        # 字段说明与单位（未知字段会自动给默认描述）
        field_desc: Dict[str, str] = {
            # .z 文件头（52字节）字段说明（按 data_loader.ZFormatHeader / su2z_hhb 写入顺序）
            "ntraces": "总道数（文件中道记录数量）",
            "npts": "每道采样点数",
            "sint": "采样间隔（微秒）",
            "tstart": "起始时间（毫秒）",
            "tend": "结束时间（毫秒；若<=0可由npts与sint推算）",
            "nrec": "记录数（炮集数）",
            "npick": "每道拾取字数量（最大40）",
            "vredf": "折合速度（km/s）",
            "ifmt": "道数据格式标识（1=float32, 0=int16）",
            "xlatlong": "经纬度缩放因子（头参数）",
            "xelev": "高程缩放因子（头参数）",
            "xutm": "UTM坐标缩放因子（头参数）",
            "cm": "坐标参考参数（转换程序中默认0.0，常作保留/扩展字段）",
            "nreci": "记录号（标准化后记录索引）",
            "itsn": "记录内道序号",
            "ireci": "接收站号",
            "itypei": "分量类型编号（1垂直/2径向/3横向/4水听器）",
            "iflagi": "道有效标志（1有效）",
            "offsti": "炮检距",
            "azi": "方位角",
            "igaini": "道增益因子（来自道头）",
            "texact": "精确时间修正项",
            "slat": "震源纬度",
            "slong": "震源经度",
            "selev": "震源高程",
            "swdepth": "震源水深",
            "rlat": "接收点纬度",
            "rlong": "接收点经度",
            "relev": "接收点高程",
            "sxutm": "震源UTM X",
            "syutm": "震源UTM Y",
            "sz": "震源深度/高程Z",
            "rxutm": "接收点UTM X",
            "ryutm": "接收点UTM Y",
            "rz": "接收点深度/高程Z",
            "ishoti": "炮号（shot id）",
            "picks": "该道各拾取字走时数组",
        }
        field_unit: Dict[str, str] = {
            "ntraces": "-",
            "npts": "-",
            "sint": "us",
            "tstart": "ms",
            "tend": "ms",
            "nrec": "-",
            "npick": "-",
            "offsti": "km",
            "azi": "deg",
            "slat": "deg",
            "slong": "deg",
            "rlat": "deg",
            "rlong": "deg",
            "selev": "m",
            "swdepth": "m",
            "relev": "m",
            "sxutm": "m",
            "syutm": "m",
            "sz": "m",
            "rxutm": "m",
            "ryutm": "m",
            "rz": "m",
            "vredf": "km/s",
        }

        def _header_summary(value: object) -> str:
            if value is None:
                return "-"
            if isinstance(value, float):
                return f"{value:.6g}"
            return str(value)

        def _value_summary(values: List[object]) -> str:
            if len(values) == 0:
                return "-"
            numeric_vals: List[float] = []
            seq_count = 0
            seq_nonzero = 0
            for v in values:
                if isinstance(v, (list, tuple, np.ndarray)):
                    seq_count += 1
                    arr = np.asarray(v, dtype=float) if len(v) > 0 else np.asarray([], dtype=float)
                    if arr.size > 0:
                        seq_nonzero += int(np.sum(np.abs(arr) > 1e-12))
                    continue
                try:
                    numeric_vals.append(float(v))
                except Exception:
                    pass
            if numeric_vals:
                arr = np.asarray(numeric_vals, dtype=float)
                finite = arr[np.isfinite(arr)]
                if finite.size > 0:
                    nz = int(np.sum(np.abs(finite) > 1e-12))
                    return (
                        f"min={float(np.min(finite)):.6g}, "
                        f"max={float(np.max(finite)):.6g}, "
                        f"mean={float(np.mean(finite)):.6g}, "
                        f"非零={nz}/{int(finite.size)}"
                    )
            if seq_count > 0:
                lengths = [len(v) for v in values if isinstance(v, (list, tuple, np.ndarray))]
                avg_len = float(np.mean(lengths)) if lengths else 0.0
                return f"序列字段: 道数={seq_count}, 平均长度={avg_len:.1f}, 非零总数={seq_nonzero}"
            # 非数值字段：显示去重样本
            uniq = []
            for v in values:
                s = str(v)
                if s not in uniq:
                    uniq.append(s)
                if len(uniq) >= 6:
                    break
            return f"样本值: {', '.join(uniq)}"

        rows: List[Tuple[str, str, str, str, str]] = []

        # 文件头字段：严格按 .z 52字节头结构展示，避免混入派生/无关属性
        if header is not None:
            header_keys = [
                "ntraces", "npts", "sint", "tstart", "tend", "nrec", "npick",
                "vredf", "ifmt", "xlatlong", "xelev", "xutm", "cm"
            ]
            for key in header_keys:
                try:
                    v = getattr(header, key)
                except Exception:
                    v = None
                desc = field_desc.get(key, "文件头参数")
                unit = field_unit.get(key, "-")
                rows.append(("文件头(.z)", key, unit, desc, _header_summary(v)))

        # 所有道头字段（全量）
        header_field_values: Dict[str, List[object]] = {}
        for th in trace_headers:
            try:
                th_items = vars(th).items()
            except Exception:
                th_items = []
                for k in dir(th):
                    if str(k).startswith("_"):
                        continue
                    try:
                        v = getattr(th, k)
                    except Exception:
                        continue
                    if callable(v):
                        continue
                    th_items.append((k, v))
            for k, v in th_items:
                if str(k).startswith("_") or callable(v):
                    continue
                header_field_values.setdefault(str(k), []).append(v)

        for name in sorted(header_field_values.keys()):
            vals = header_field_values[name]
            desc = field_desc.get(name, "道头参数（自动识别）")
            unit = field_unit.get(name, "-")
            rows.append(("道头(.hdr/.z)", name, unit, desc, _value_summary(vals)))

        table = QtWidgets.QTableWidget(len(rows), 5, parent)
        table.setHorizontalHeaderLabels(["来源", "参数名", "单位", "说明", "值/统计"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        for r, (src, name, unit, desc, val) in enumerate(rows):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(src))
            table.setItem(r, 1, QtWidgets.QTableWidgetItem(name))
            table.setItem(r, 2, QtWidgets.QTableWidgetItem(unit))
            table.setItem(r, 3, QtWidgets.QTableWidgetItem(desc))
            table.setItem(r, 4, QtWidgets.QTableWidgetItem(val))
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        return table


    def _trace_header_source_label(self) -> str:
        """返回当前道头来源标签。"""
        if self.loaded is None:
            return "-"
        trace_headers = self.loaded.get("trace_headers", []) or []
        if not trace_headers:
            return "无道头"
        from_header_flags = [
            bool(getattr(th, "picks_from_header", False))
            for th in trace_headers
            if hasattr(th, "picks_from_header")
        ]
        if from_header_flags:
            if all(from_header_flags):
                return ".hdr"
            if not any(from_header_flags):
                return ".z内嵌"
            return "混合(.hdr/.z)"
        # 兼容缺少标记的情况：根据是否加载了有效 hfile 推断
        if self._hfile and Path(self._hfile).exists():
            return ".hdr(推断)"
        return ".z/默认(推断)"

