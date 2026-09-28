"""多日志图：右击某次日志的点/条/曲线，打开操作菜单。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QCursor, QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QMenu, QVBoxLayout, QWidget

from pyAOBS.modeling.tomo2d.tt_inverse_log_analysis import (
    composite_score,
    format_params_compact,
    format_tt_inverse_run_params,
    last_row_metrics,
    parse_tt_inverse_log_header,
)

from ..dialog_utils import show_modeless_dialog, show_modeless_message, show_modeless_text
from ..services.paths import resolve_work_dir
from ..services.result_nav import (
    find_smesh_for_inverse_log,
    infer_run_dir_from_smesh,
    monitor_spec_for_run,
)
from ..state.form_state import FormState


def pareto_hit_index(fig, event) -> int | None:
    """命中散点、得分条或叠画曲线时返回 series 下标（与 ``names`` 一致）。"""
    meta = getattr(fig, "_pyaobs_pareto", None)
    if meta is None or event is None:
        return None
    scatters = [s for s in ([meta.get("scatter")] + list(meta.get("scatters") or [])) if s is not None]
    for scatter in scatters:
        if event.inaxes is not getattr(scatter, "axes", None):
            continue
        try:
            cont, info = scatter.contains(event)
            inds = list((info or {}).get("ind", []) or [])
            if cont and inds:
                return int(inds[0])
        except Exception:
            pass
        near = _nearest_scatter_index(scatter, event)
        if near is not None:
            return near
    bars = meta.get("bars")
    mapping = list(meta.get("bar_series_index") or [])
    patches = getattr(bars, "patches", None) or []
    for i, patch in enumerate(patches):
        try:
            if event.inaxes is patch.axes and patch.contains(event)[0]:
                if 0 <= i < len(mapping):
                    return int(mapping[i])
        except Exception:
            continue
    ax_b = getattr(bars, "axes", None) if bars is not None else None
    if ax_b is not None and event.inaxes is ax_b and getattr(event, "ydata", None) is not None:
        y = float(event.ydata)
        i_bar = int(round(y))
        if 0 <= i_bar < len(mapping) and abs(y - i_bar) <= 0.45:
            return int(mapping[i_bar])
    lines = list(meta.get("lines") or [])
    line_map = list(meta.get("line_series_index") or range(len(lines)))
    hit_line = _hit_line_index(lines, line_map, event)
    if hit_line is not None:
        return hit_line
    tbl = meta.get("table")
    names = list(meta.get("names") or [])
    hit_tbl = _hit_table_index(tbl, names, event)
    if hit_tbl is not None:
        return hit_tbl
    return None


def _nearest_scatter_index(scatter, event, *, max_px: float = 12.0) -> int | None:
    x, y = getattr(event, "x", None), getattr(event, "y", None)
    if x is None or y is None:
        return None
    try:
        import numpy as np

        offs = scatter.get_offsets()
        pts = scatter.axes.transData.transform(offs)
    except Exception:
        return None
    if len(pts) == 0:
        return None
    d2 = (pts[:, 0] - float(x)) ** 2 + (pts[:, 1] - float(y)) ** 2
    finite = np.isfinite(d2)
    if not np.any(finite):
        return None
    j = int(np.nanargmin(d2))
    if float(d2[j]) <= max_px * max_px:
        return j
    return None


def _hit_line_index(lines, mapping, event, *, max_px: float = 12.0) -> int | None:
    x, y = getattr(event, "x", None), getattr(event, "y", None)
    if x is None or y is None:
        return None
    import numpy as np

    best_i: int | None = None
    best_d: float | None = None
    for line, idx in zip(lines, mapping):
        if event.inaxes is not getattr(line, "axes", None):
            continue
        try:
            cont, _info = line.contains(event)
            if cont:
                return int(idx)
        except Exception:
            pass
        try:
            xd, yd = line.get_data()
            if len(xd) == 0:
                continue
            pts = line.axes.transData.transform(np.column_stack([xd, yd]))
            d2 = (pts[:, 0] - float(x)) ** 2 + (pts[:, 1] - float(y)) ** 2
            d2 = d2[np.isfinite(d2)]
            if d2.size == 0:
                continue
            d = float(np.min(d2))
        except Exception:
            continue
        if best_d is None or d < best_d:
            best_d = d
            best_i = int(idx)
    if best_i is not None and best_d is not None and best_d <= max_px * max_px:
        return best_i
    return None


def _hit_table_index(tbl, names, event) -> int | None:
    if tbl is None or event is None:
        return None
    x, y = getattr(event, "x", None), getattr(event, "y", None)
    if x is None or y is None:
        return None
    try:
        cells = tbl.get_celld()
    except Exception:
        return None
    for (r, _c), cell in cells.items():
        if r <= 0:
            continue
        try:
            bbox = cell.get_window_extent()
        except Exception:
            continue
        if bbox.x0 <= float(x) <= bbox.x1 and bbox.y0 <= float(y) <= bbox.y1:
            i = int(r) - 1
            if 0 <= i < len(names):
                return i
    return None


def _pick_scatters(meta) -> list:
    out = []
    if meta.get("scatter") is not None:
        out.append(meta["scatter"])
    out.extend(s for s in (meta.get("scatters") or []) if s is not None)
    return out


def _rgba_n(colors, n: int):
    import numpy as np

    a = np.asarray(colors, dtype=float)
    if a.size == 0:
        return np.zeros((max(n, 0), 4), dtype=float)
    if a.ndim == 1:
        if a.size == 3:
            a = np.append(a, 1.0)
        a = a.reshape(1, -1)
    if a.shape[-1] == 3:
        a = np.hstack([a, np.ones((a.shape[0], 1), dtype=float)])
    if n <= 0:
        return a
    if a.shape[0] == 1 and n > 1:
        a = np.repeat(a, n, axis=0)
    if a.shape[0] != n:
        out = np.zeros((n, 4), dtype=float)
        out[:, 3] = 1.0
        m = min(n, a.shape[0])
        out[:m] = a[:m]
        return out
    return a.copy()


def _snapshot_highlight_styles(meta) -> dict:
    import numpy as np

    scatters: dict[int, dict] = {}
    for sc in _pick_scatters(meta):
        n = len(sc.get_offsets())
        sizes = np.asarray(sc.get_sizes(), dtype=float)
        if sizes.size == 1 and n > 1:
            sizes = np.repeat(sizes, n)
        elif sizes.size != n:
            sizes = np.resize(sizes, n)
        scatters[id(sc)] = {
            "sizes": sizes.copy(),
            "fc": _rgba_n(sc.get_facecolors(), n),
            "ec": _rgba_n(sc.get_edgecolors(), n),
            "lw": np.resize(np.asarray(sc.get_linewidths(), dtype=float), n).copy(),
        }
    bars: dict[int, dict] = {}
    patches = getattr(meta.get("bars"), "patches", None) or []
    for patch in patches:
        bars[id(patch)] = {
            "fc": patch.get_facecolor(),
            "ec": patch.get_edgecolor(),
            "lw": patch.get_linewidth(),
            "alpha": patch.get_alpha(),
        }
    lines: dict[int, dict] = {}
    artists = list(meta.get("lines") or [])
    for line in artists:
        try:
            lines[id(line)] = {
                "color": line.get_color(),
                "lw": line.get_linewidth(),
                "alpha": line.get_alpha(),
                "ms": line.get_markersize(),
                "zorder": line.get_zorder(),
                "mec": line.get_markeredgecolor(),
                "mew": line.get_markeredgewidth(),
            }
        except Exception:
            continue
    table_cells: dict = {}
    tbl = meta.get("table")
    if tbl is not None:
        try:
            for key, cell in tbl.get_celld().items():
                txt = cell.get_text()
                table_cells[key] = {
                    "fc": cell.get_facecolor(),
                    "ec": cell.get_edgecolor(),
                    "lw": cell.get_linewidth(),
                    "tc": txt.get_color(),
                    "fw": txt.get_fontweight(),
                }
        except Exception:
            table_cells = {}
    return {
        "scatters": scatters,
        "bars": bars,
        "lines": lines,
        "table": table_cells,
    }


def _normalize_indices(index) -> tuple[int, ...]:
    if index is None:
        return ()
    if isinstance(index, bool):
        return ()
    if isinstance(index, int):
        return (int(index),)
    try:
        return tuple(int(i) for i in index)
    except TypeError:
        return (int(index),)


def apply_series_highlight(fig, index=None) -> None:
    """同一 figure 内高亮一个或多个系列下标（``None`` / 空则恢复）。"""
    import numpy as np

    meta = getattr(fig, "_pyaobs_pareto", None)
    if not meta:
        return
    names = list(meta.get("names") or [])
    n_names = len(names)
    raw = _normalize_indices(index)
    indices = tuple(i for i in raw if 0 <= i < n_names) if n_names else raw
    orig = meta.get("_hl_orig")
    if orig is None:
        orig = _snapshot_highlight_styles(meta)
        meta["_hl_orig"] = orig
    meta["highlight_indices"] = indices
    meta["highlight_index"] = None if not indices else (
        indices[0] if len(indices) == 1 else indices[0]
    )
    sel_set = set(indices)
    for sc in _pick_scatters(meta):
        snap = orig["scatters"].get(id(sc))
        if snap is None:
            continue
        n = len(sc.get_offsets())
        sizes = np.array(snap["sizes"], copy=True)
        fc = np.array(snap["fc"], copy=True)
        ec = np.array(snap["ec"], copy=True)
        lw = np.array(snap["lw"], copy=True)
        if n != sizes.size:
            continue
        if indices:
            sel = np.zeros(n, dtype=bool)
            for i in sel_set:
                if 0 <= int(i) < n:
                    sel[int(i)] = True
            sizes = np.where(sel, np.maximum(sizes, 64.0) * 1.85, sizes * 0.82)
            if fc.shape[0] == n and fc.shape[1] == 4:
                fc[:, 3] = np.where(sel, np.maximum(fc[:, 3], 0.92), np.maximum(fc[:, 3] * 0.7, 0.55))
            if ec.shape[0] != n:
                ec = np.zeros((n, 4), dtype=float)
            else:
                ec = np.array(ec, copy=True)
                ec[:, 3] = np.where(sel, 1.0, 0.0)
            for i in indices:
                if i < n:
                    ec[i] = (0.08, 0.08, 0.08, 1.0)
            lw = np.where(sel, 1.8, 0.0)
        sc.set_sizes(sizes)
        sc.set_facecolors(fc)
        sc.set_edgecolors(ec)
        sc.set_linewidths(lw)
    mapping = list(meta.get("bar_series_index") or [])
    patches = getattr(meta.get("bars"), "patches", None) or []
    for i, patch in enumerate(patches):
        snap = orig["bars"].get(id(patch))
        if snap is None:
            continue
        patch.set_facecolor(snap["fc"])
        patch.set_edgecolor(snap["ec"])
        patch.set_linewidth(snap["lw"])
        patch.set_alpha(snap["alpha"] if snap["alpha"] is not None else 1.0)
        if not indices:
            continue
        si = mapping[i] if i < len(mapping) else None
        if si in sel_set:
            patch.set_edgecolor((0.08, 0.08, 0.08, 1.0))
            patch.set_linewidth(2.0)
            patch.set_alpha(1.0)
        else:
            a = snap["alpha"]
            patch.set_alpha((0.58 if a is None else max(float(a) * 0.72, 0.5)))
            patch.set_linewidth(0.4)
    line_map = list(meta.get("line_series_index") or [])
    line_arts = list(meta.get("lines") or [])
    if not line_map:
        line_map = list(range(len(line_arts)))
    for line, si in zip(line_arts, line_map):
        snap = orig.get("lines", {}).get(id(line))
        if snap is None:
            continue
        line.set_color(snap["color"])
        line.set_linewidth(snap["lw"])
        line.set_markersize(snap["ms"])
        line.set_zorder(snap["zorder"])
        line.set_markeredgecolor(snap["mec"])
        line.set_markeredgewidth(snap["mew"])
        a = snap["alpha"]
        line.set_alpha(1.0 if a is None else a)
        if not indices:
            continue
        if int(si) in sel_set:
            line.set_linewidth(max(float(snap["lw"]), 1.0) * 2.3)
            line.set_markersize(max(float(snap["ms"]), 3.0) * 1.45)
            line.set_zorder(6)
            line.set_alpha(1.0)
            line.set_markeredgecolor((0.08, 0.08, 0.08, 1.0))
            line.set_markeredgewidth(0.9)
        else:
            line.set_alpha(0.58)
            line.set_zorder(2)
    tbl = meta.get("table")
    tsnap = orig.get("table") or {}
    if tbl is not None and tsnap:
        try:
            cells = tbl.get_celld()
        except Exception:
            cells = {}
        for key, cell in cells.items():
            snap = tsnap.get(key)
            if snap is None:
                continue
            cell.set_facecolor(snap["fc"])
            cell.set_edgecolor(snap["ec"])
            cell.set_linewidth(snap["lw"])
            txt = cell.get_text()
            txt.set_color(snap["tc"])
            txt.set_fontweight(snap["fw"])
            r = key[0] if isinstance(key, tuple) else 0
            if not indices or r <= 0:
                continue
            if (r - 1) in sel_set:
                cell.set_facecolor((1.0, 0.92, 0.52, 1.0))
                txt.set_fontweight("bold")
            else:
                cell.set_facecolor((0.96, 0.96, 0.96, 0.42))
    canvas = getattr(fig, "canvas", None)
    if canvas is not None:
        try:
            canvas.draw_idle()
        except Exception:
            pass


def apply_series_highlight_by_name(fig, name: str | None) -> None:
    apply_series_highlight_by_names(fig, None if name is None else [name])


def apply_series_highlight_by_names(fig, names_sel: list[str] | None) -> None:
    meta = getattr(fig, "_pyaobs_pareto", None)
    if not meta:
        return
    names = list(meta.get("names") or [])
    if not names_sel:
        apply_series_highlight(fig, None)
        return
    idxs = [names.index(n) for n in names_sel if n in names]
    apply_series_highlight(fig, idxs)


def _host_names(host) -> list[str]:
    names = getattr(host, "names", None)
    if names:
        return list(names)
    meta = getattr(host, "_pyaobs_pareto", None)
    if meta:
        return list(meta.get("names") or [])
    fig = getattr(host, "_fig", None)
    meta = getattr(fig, "_pyaobs_pareto", None) or {}
    return list(meta.get("names") or [])


class SeriesHighlightGroup:
    """一次多日志分析弹出的各图窗：按日志名同步高亮（可多选）。"""

    def __init__(self) -> None:
        self.selected_names: list[str] = []
        self.selected_name: str | None = None
        self._anchor: str | None = None
        self._wins: list = []

    def register(self, win) -> None:
        import weakref

        self._wins.append(weakref.ref(win))

    def select_from_figure(self, fig, index: int | None) -> None:
        names = _host_names(fig)
        name = None
        if index is not None and 0 <= int(index) < len(names):
            name = names[int(index)]
        self.select_name(name)

    def toggle_from_figure(self, fig, index: int) -> None:
        names = _host_names(fig)
        if not (0 <= int(index) < len(names)):
            return
        name = names[int(index)]
        cur = [n for n in self.selected_names if n != name]
        if name not in self.selected_names:
            cur.append(name)
        self._anchor = name
        self.select_names(cur)

    def select_range_to(
        self, fig, index: int, visual_order: list[str] | None = None
    ) -> None:
        names = _host_names(fig)
        if not names or not (0 <= int(index) < len(names)):
            return
        target = names[int(index)]
        order = [n for n in (visual_order or []) if n in names]
        if not order:
            order = names
        i = order.index(target) if target in order else int(index)
        if self._anchor in order:
            a = order.index(self._anchor)
        elif self.selected_names:
            last = self.selected_names[-1]
            a = order.index(last) if last in order else i
        else:
            a = i
            self._anchor = target
        lo, hi = (a, i) if a <= i else (i, a)
        self.select_names(order[lo : hi + 1])

    def select_indices(
        self, fig, indices: list[int], *, additive: bool = False
    ) -> None:
        names = _host_names(fig)
        picked = [names[i] for i in indices if 0 <= int(i) < len(names)]
        if not picked:
            return
        if additive:
            seen = set(self.selected_names)
            cur = list(self.selected_names)
            for n in picked:
                if n not in seen:
                    cur.append(n)
                    seen.add(n)
            self.select_names(cur)
        else:
            self.select_names(picked)
        self._anchor = picked[-1]

    def select_name(self, name: str | None) -> None:
        if name is None:
            self._anchor = None
            self.select_names([])
        else:
            self._anchor = name
            self.select_names([name])

    def select_names(self, names: list[str] | None) -> None:
        uniq: list[str] = []
        seen: set[str] = set()
        for n in names or []:
            if n and n not in seen:
                uniq.append(n)
                seen.add(n)
        self.selected_names = uniq
        self.selected_name = uniq[-1] if uniq else None
        alive: list = []
        for ref in self._wins:
            win = ref()
            if win is None:
                continue
            alive.append(ref)
            fig = getattr(win, "_fig", None)
            setter_hl = getattr(win, "set_highlight_names", None)
            if callable(setter_hl):
                try:
                    setter_hl(uniq)
                except Exception:
                    pass
            else:
                apply_series_highlight_by_names(fig, uniq)
            setter = getattr(win, "_hl_set_hint", None)
            if callable(setter):
                try:
                    setter(uniq)
                except Exception:
                    pass
        self._wins = alive


def _figure_can_cross_highlight(meta) -> bool:
    return bool(
        _pick_scatters(meta)
        or meta.get("bars")
        or meta.get("lines")
        or meta.get("table")
    )


def _click_unmoved(x0, y0, event, *, max_px: float = 8.0) -> bool:
    x1, y1 = getattr(event, "x", None), getattr(event, "y", None)
    if None in (x0, y0, x1, y1):
        return True
    return (float(x1) - float(x0)) ** 2 + (float(y1) - float(y0)) ** 2 <= max_px * max_px


def _box_norm(x0, y0, x1, y1) -> tuple[float, float, float, float]:
    return (
        min(float(x0), float(x1)),
        min(float(y0), float(y1)),
        max(float(x0), float(x1)),
        max(float(y0), float(y1)),
    )


def _px_in_box(px, py, x0, y0, x1, y1) -> bool:
    lo_x, lo_y, hi_x, hi_y = _box_norm(x0, y0, x1, y1)
    return lo_x <= float(px) <= hi_x and lo_y <= float(py) <= hi_y


def _bbox_hits_box(bbox, x0, y0, x1, y1) -> bool:
    lo_x, lo_y, hi_x, hi_y = _box_norm(x0, y0, x1, y1)
    return not (
        float(bbox.x1) < lo_x
        or float(bbox.x0) > hi_x
        or float(bbox.y1) < lo_y
        or float(bbox.y0) > hi_y
    )


def _seg_hits_box(px0, py0, px1, py1, x0, y0, x1, y1) -> bool:
    if _px_in_box(px0, py0, x0, y0, x1, y1) or _px_in_box(px1, py1, x0, y0, x1, y1):
        return True
    lo_x, lo_y, hi_x, hi_y = _box_norm(x0, y0, x1, y1)
    dx = float(px1) - float(px0)
    dy = float(py1) - float(py0)
    p = (-dx, dx, -dy, dy)
    q = (
        float(px0) - lo_x,
        hi_x - float(px0),
        float(py0) - lo_y,
        hi_y - float(py0),
    )
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
            continue
        t = qi / pi
        if pi < 0:
            u1 = max(u1, t)
        else:
            u2 = min(u2, t)
        if u1 > u2:
            return False
    return True


def pareto_box_indices(fig, x0, y0, x1, y1, *, min_px: float = 8.0) -> list[int]:
    """显示坐标框选命中的系列下标（与 ``names`` 一致）。过小的框视为单击，返回空。"""
    if None in (x0, y0, x1, y1):
        return []
    if (float(x1) - float(x0)) ** 2 + (float(y1) - float(y0)) ** 2 <= min_px * min_px:
        return []
    meta = getattr(fig, "_pyaobs_pareto", None)
    if not meta:
        return []
    names = list(meta.get("names") or [])
    hit: set[int] = set()
    import numpy as np

    for scatter in _pick_scatters(meta):
        try:
            offs = scatter.get_offsets()
            pts = scatter.axes.transData.transform(offs)
        except Exception:
            continue
        for j, (px, py) in enumerate(pts):
            if _px_in_box(px, py, x0, y0, x1, y1) and 0 <= j < len(names):
                hit.add(int(j))
    mapping = list(meta.get("bar_series_index") or [])
    patches = getattr(meta.get("bars"), "patches", None) or []
    for i, patch in enumerate(patches):
        try:
            if _bbox_hits_box(patch.get_window_extent(), x0, y0, x1, y1):
                if 0 <= i < len(mapping):
                    hit.add(int(mapping[i]))
        except Exception:
            continue
    lines = list(meta.get("lines") or [])
    line_map = list(meta.get("line_series_index") or range(len(lines)))
    for line, idx in zip(lines, line_map):
        try:
            xd, yd = line.get_data()
            if len(xd) == 0:
                continue
            pts = line.axes.transData.transform(np.column_stack([xd, yd]))
        except Exception:
            continue
        inside = False
        for j, (px, py) in enumerate(pts):
            if _px_in_box(px, py, x0, y0, x1, y1):
                inside = True
                break
            if j > 0 and _seg_hits_box(
                pts[j - 1][0], pts[j - 1][1], px, py, x0, y0, x1, y1
            ):
                inside = True
                break
        if inside:
            hit.add(int(idx))
    tbl = meta.get("table")
    if tbl is not None:
        try:
            cells = tbl.get_celld()
        except Exception:
            cells = {}
        for (r, _c), cell in cells.items():
            if r <= 0:
                continue
            try:
                bbox = cell.get_window_extent()
            except Exception:
                continue
            if _bbox_hits_box(bbox, x0, y0, x1, y1):
                i = int(r) - 1
                if 0 <= i < len(names):
                    hit.add(i)
    return sorted(hit)


def _qt_mods() -> tuple[bool, bool]:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            return False, False
        m = app.keyboardModifiers()
        ctrl = bool(
            m
            & (
                Qt.KeyboardModifier.ControlModifier
                | Qt.KeyboardModifier.MetaModifier
            )
        )
        shift = bool(m & Qt.KeyboardModifier.ShiftModifier)
        return ctrl, shift
    except Exception:
        return False, False


def _disp_to_fig(fig, x, y) -> tuple[float, float]:
    inv = fig.transFigure.inverted()
    a, b = inv.transform((float(x), float(y)))
    return float(a), float(b)


def _hint_selected_text(sel, *, base_hint: str) -> str:
    if isinstance(sel, str):
        sel = [sel] if sel else []
    sel = list(sel or [])
    if not sel:
        return base_hint
    if len(sel) == 1:
        return (
            f"已选：{sel[0]} · Ctrl+点加减 · Shift+点连选 · Shift+拖框选 · 右击菜单"
        )
    shown = "、".join(sel[:4])
    extra = f" 等{len(sel)}个" if len(sel) > 4 else ""
    return f"已选 {len(sel)} 个：{shown}{extra} · 右击菜单"


def _wire_pg_analysis_menu(
    win,
    *,
    state: FormState,
    path_by_name: dict[str, Path],
    rows_by_name: dict[str, list],
    rough_weight: float,
    group: SeriesHighlightGroup | None,
) -> None:
    names = list(getattr(win, "names", None) or [])
    is_table = type(win).__name__ == "SummaryTableWindow"
    base_hint = (
        "Ctrl/Shift 多选 · 右击菜单"
        if is_table
        else (
            "滚轮缩放 · 左拖平移 · 左击选中 · Ctrl+点加减 · Shift+点连选 · "
            "Shift+拖框选 · 右击菜单 · 右拖空白缩放"
        )
    )
    try:
        if getattr(win, "hint", None) is not None:
            win.hint.setText(base_hint)
    except Exception:
        pass

    def _hint(sel) -> None:
        try:
            if getattr(win, "hint", None) is not None:
                win.hint.setText(_hint_selected_text(sel, base_hint=base_hint))
        except Exception:
            pass

    def _menu(picked: list[str]) -> None:
        if not picked:
            return
        _popup_log_menu(
            win,
            names=picked,
            state=state,
            path_by_name=path_by_name,
            rows_by_name=rows_by_name,
            rough_weight=rough_weight,
            group=group,
        )

    def _on_pick(kind: str, **kw) -> None:
        if kind == "table":
            picked = list(kw.get("names") or [])
            if group is not None:
                group.select_names(picked)
            else:
                _hint(picked)
            return
        if kind == "menu":
            _menu(list(kw.get("names") or []))
            return
        if kind == "box":
            indices = list(kw.get("indices") or [])
            if not indices:
                return
            if group is not None:
                group.select_indices(win, indices, additive=bool(kw.get("ctrl")))
            _hint(group.selected_names if group is not None else [names[i] for i in indices if 0 <= i < len(names)])
            return
        if kind != "click":
            return
        btn = kw.get("button")
        i = kw.get("index")
        ctrl = bool(kw.get("ctrl"))
        shift = bool(kw.get("shift"))
        if btn == Qt.MouseButton.RightButton:
            if i is None or not (0 <= int(i) < len(names)):
                return
            nm = names[int(i)]
            if group is not None and nm in group.selected_names:
                picked = list(group.selected_names)
            elif group is not None:
                group.select_name(nm)
                picked = [nm]
            else:
                picked = [nm]
            _menu(picked)
            return
        if btn != Qt.MouseButton.LeftButton:
            return
        if i is None:
            if group is not None and not shift and not ctrl:
                group.select_name(None)
            return
        i = int(i)
        if group is not None:
            if shift:
                group.select_range_to(
                    win, i, visual_order=kw.get("visual_order")
                )
            elif ctrl:
                group.toggle_from_figure(win, i)
            else:
                group.select_from_figure(win, i)
            return
        _hint([names[i]] if 0 <= i < len(names) else [])

    win.user_pick = _on_pick
    win._hl_set_hint = _hint
    if group is not None:
        win._hl_group = group
        group.register(win)


def wire_pareto_log_menu(
    win,
    *,
    state: FormState,
    path_by_name: dict[str, Path],
    rows_by_name: dict[str, list],
    rough_weight: float,
    group: SeriesHighlightGroup | None = None,
) -> None:
    if getattr(win, "_analysis_pg", False):
        _wire_pg_analysis_menu(
            win,
            state=state,
            path_by_name=path_by_name,
            rows_by_name=rows_by_name,
            rough_weight=rough_weight,
            group=group,
        )
        return
    fig = getattr(win, "_fig", None)
    host = getattr(win, "host", None)
    if fig is None or host is None:
        return
    canvas = host.canvas
    nav = host._nav
    meta = getattr(fig, "_pyaobs_pareto", None)
    if not meta:
        return
    can_hl = _figure_can_cross_highlight(meta)
    base_hint = (
        "滚轮缩放 · 左拖平移 · 左击选中 · Ctrl+点加减 · Shift+点连选 · "
        "Shift+拖框选 · 右击菜单 · 右拖空白缩放"
        if can_hl
        else "滚轮缩放 · 左拖平移 · 右拖空白处缩放 · 右击点/曲线：该次日志"
    )
    try:
        host.hint.setText(base_hint)
    except Exception:
        pass

    try:
        canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
    except Exception:
        pass

    names = list(meta.get("names") or [])
    pending: dict[str, int | float | None] = {
        "i": None,
        "x": None,
        "y": None,
        "btn": None,
        "shift": None,
        "ctrl": None,
    }
    rubber = {"art": None}

    def _hit(event) -> int | None:
        return pareto_hit_index(fig, event)

    def _claim_right(event=None) -> bool:
        return _hit(event) is not None

    def _claim_left(event=None) -> bool:
        _ctrl, shift = _qt_mods()
        return bool(shift and getattr(event, "inaxes", None) is not None)

    nav.claim_right = _claim_right
    nav.claim_left = _claim_left

    def _hint_names(sel) -> None:
        if not can_hl:
            return
        try:
            if isinstance(sel, str):
                sel = [sel] if sel else []
            sel = list(sel or [])
            if not sel:
                host.hint.setText(base_hint)
            elif len(sel) == 1:
                host.hint.setText(
                    f"已选：{sel[0]} · Ctrl+点加减 · Shift+点连选 · "
                    "Shift+拖框选 · 右击菜单"
                )
            else:
                shown = "、".join(sel[:4])
                extra = f" 等{len(sel)}个" if len(sel) > 4 else ""
                host.hint.setText(f"已选 {len(sel)} 个：{shown}{extra} · 右击菜单")
        except Exception:
            pass

    def _clear_rubber() -> None:
        art = rubber.get("art")
        rubber["art"] = None
        if art is None:
            return
        try:
            art.remove()
        except Exception:
            pass
        try:
            canvas.draw_idle()
        except Exception:
            pass

    def _update_rubber(x0, y0, x1, y1) -> None:
        from matplotlib.patches import Rectangle

        art = rubber.get("art")
        if art is None:
            art = Rectangle(
                (0.0, 0.0),
                0.0,
                0.0,
                fill=True,
                alpha=0.18,
                lw=1.0,
                edgecolor="#1d4ed8",
                facecolor="#93c5fd",
                transform=fig.transFigure,
                zorder=20,
            )
            fig.add_artist(art)
            rubber["art"] = art
        p0 = _disp_to_fig(fig, x0, y0)
        p1 = _disp_to_fig(fig, x1, y1)
        art.set_xy((min(p0[0], p1[0]), min(p0[1], p1[1])))
        art.set_width(abs(p1[0] - p0[0]))
        art.set_height(abs(p1[1] - p0[1]))
        try:
            canvas.draw_idle()
        except Exception:
            pass

    def _on_press(event) -> None:
        btn = getattr(event, "button", None)
        if btn not in (1, 3) or getattr(event, "dblclick", False):
            return
        ctrl, shift = _qt_mods()
        pending["btn"] = btn
        pending["i"] = _hit(event)
        pending["x"] = getattr(event, "x", None)
        pending["y"] = getattr(event, "y", None)
        pending["shift"] = 1 if shift else 0
        pending["ctrl"] = 1 if ctrl else 0

    def _on_motion(event) -> None:
        if pending.get("btn") != 1 or not pending.get("shift"):
            return
        x0, y0 = pending.get("x"), pending.get("y")
        x1, y1 = getattr(event, "x", None), getattr(event, "y", None)
        if None in (x0, y0, x1, y1):
            return
        if _click_unmoved(x0, y0, event):
            return
        _update_rubber(x0, y0, x1, y1)

    def _apply_left_hit(i: int, *, ctrl: bool, shift: bool) -> None:
        if group is not None:
            if shift:
                group.select_range_to(fig, i)
            elif ctrl:
                group.toggle_from_figure(fig, i)
            else:
                group.select_from_figure(fig, i)
            return
        if shift or ctrl:
            cur = list(getattr(fig, "_pyaobs_pareto", {}).get("highlight_indices") or ())
            if shift:
                a = cur[-1] if cur else i
                lo, hi = (a, i) if a <= i else (i, a)
                apply_series_highlight(fig, list(range(lo, hi + 1)))
            else:
                s = set(cur)
                if i in s:
                    s.discard(i)
                else:
                    s.add(i)
                apply_series_highlight(fig, sorted(s) or None)
            _hint_names(
                [
                    names[j]
                    for j in (
                        getattr(fig, "_pyaobs_pareto", {}).get("highlight_indices")
                        or ()
                    )
                    if 0 <= j < len(names)
                ]
            )
            return
        apply_series_highlight(fig, i)
        _hint_names([names[i]] if 0 <= i < len(names) else [])

    def _clear_sel() -> None:
        if group is not None:
            group.select_name(None)
        else:
            apply_series_highlight(fig, None)
            _hint_names([])

    def _on_release(event) -> None:
        btn = pending.get("btn")
        i = pending["i"]
        x0, y0 = pending["x"], pending["y"]
        was_shift = bool(pending.get("shift"))
        was_ctrl = bool(pending.get("ctrl"))
        pending["btn"] = pending["i"] = pending["x"] = pending["y"] = None
        pending["shift"] = pending["ctrl"] = None
        if getattr(event, "button", None) != btn:
            _clear_rubber()
            return
        x1, y1 = getattr(event, "x", None), getattr(event, "y", None)
        if btn == 1 and was_shift and not _click_unmoved(x0, y0, event):
            box = pareto_box_indices(fig, x0, y0, x1, y1)
            _clear_rubber()
            if box:
                if group is not None:
                    group.select_indices(fig, box, additive=was_ctrl)
                else:
                    apply_series_highlight(fig, box)
                    _hint_names([names[j] for j in box if 0 <= j < len(names)])
            return
        _clear_rubber()
        if not _click_unmoved(x0, y0, event):
            return
        hit = _hit(event)
        if btn == 3:
            if i is None or hit != i or not (0 <= i < len(names)):
                return
            picked = [names[i]]
            if group is not None and names[i] in group.selected_names:
                picked = list(group.selected_names)
            elif group is not None:
                group.select_name(names[i])
            else:
                apply_series_highlight(fig, i)
                _hint_names(picked)
            _popup_log_menu(
                win,
                names=picked,
                state=state,
                path_by_name=path_by_name,
                rows_by_name=rows_by_name,
                rough_weight=rough_weight,
                group=group,
            )
            return
        if btn != 1 or not can_hl:
            return
        if i is not None and hit == i:
            _apply_left_hit(i, ctrl=was_ctrl, shift=was_shift)
        elif (
            i is None
            and hit is None
            and getattr(event, "inaxes", None) is not None
            and not was_shift
            and not was_ctrl
        ):
            _clear_sel()

    canvas.mpl_connect("button_press_event", _on_press)
    canvas.mpl_connect("button_release_event", _on_release)
    canvas.mpl_connect("motion_notify_event", _on_motion)
    if group is not None:
        win._hl_group = group
        win._hl_set_hint = _hint_names
        group.register(win)


def _popup_log_menu(
    win,
    *,
    names: list[str],
    state: FormState,
    path_by_name: dict[str, Path],
    rows_by_name: dict[str, list],
    rough_weight: float,
    group: SeriesHighlightGroup | None = None,
) -> None:
    names = [n for n in names if n]
    if not names:
        return
    n = len(names)
    one = names[0]
    path = path_by_name.get(one)
    menu = QMenu(win)
    if n == 1:
        head = menu.addAction(path.name if path is not None else one)
    else:
        head = menu.addAction(f"{n} 个已选")
    head.setEnabled(False)
    menu.addSeparator()
    act_model = menu.addAction("绘制模型（速度场 + 走时拟合）")
    act_dir = menu.addAction("打开目录")
    act_curve = menu.addAction("绘制迭代曲线")
    act_cmp = menu.addAction("添加至模型对比")
    act_params = menu.addAction("反演参数列表…")
    if n != 1:
        act_model.setEnabled(False)
        act_dir.setEnabled(False)
        act_model.setToolTip("请只选一个")
        act_dir.setToolTip("请只选一个")
    win._pareto_ctx_menu = menu
    act_model.triggered.connect(lambda: _plot_model(state, path, one))
    act_dir.triggered.connect(lambda: _open_dir(path, one))
    act_curve.triggered.connect(
        lambda: _plot_curves(
            names,
            path_by_name=path_by_name,
            rows_by_name=rows_by_name,
            state=state,
            rough_weight=rough_weight,
            group=group,
        )
    )
    act_cmp.triggered.connect(
        lambda: _add_selected_to_compare(names, path_by_name, state)
    )
    act_params.triggered.connect(
        lambda: _show_params_selected(
            names, path_by_name, rows_by_name, rough_weight
        )
    )
    menu.popup(QCursor.pos())


def _open_dir(path: Path | None, name: str) -> None:
    from PySide6.QtWidgets import QMessageBox

    if path is None:
        show_modeless_message(
            "反演分析", f"无路径：{name}", icon=QMessageBox.Icon.Warning
        )
        return
    folder = path if path.is_dir() else path.parent
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))


def _plot_curve(path: Path | None, rows: list, name: str) -> None:
    if not rows:
        show_modeless_message("反演分析", f"无迭代数据：{name}")
        return
    from .inv_analysis_pg import show_single_log_window

    title = path.name if path is not None else name
    show_single_log_window(
        rows,
        title=title,
        save_dir=str(path.parent) if path is not None else None,
    )


def _plot_curves(
    names: list[str],
    *,
    path_by_name: dict[str, Path],
    rows_by_name: dict[str, list],
    state: FormState,
    rough_weight: float,
    group: SeriesHighlightGroup | None,
) -> None:
    if len(names) == 1:
        n = names[0]
        _plot_curve(path_by_name.get(n), rows_by_name.get(n) or [], n)
        return
    from .inv_analysis_pg import show_overlay_window

    series = {n: rows_by_name.get(n) or [] for n in names if rows_by_name.get(n)}
    if len(series) < 2:
        show_modeless_message("反演分析", "所选日志没有足够的迭代数据。")
        return
    first = path_by_name.get(names[0])
    win = show_overlay_window(
        series,
        title=f"所选迭代曲线 n={len(series)}",
        save_dir=str(first.parent) if first is not None else None,
    )
    wire_pareto_log_menu(
        win,
        state=state,
        path_by_name=path_by_name,
        rows_by_name=rows_by_name,
        rough_weight=rough_weight,
        group=group,
    )


def _add_selected_to_compare(
    names: list[str],
    path_by_name: dict[str, Path],
    state: FormState,
) -> None:
    from ..services.model_compare import add_smeshes_to_compare

    smeshes: list[Path] = []
    errs: list[str] = []
    for name in names:
        path = path_by_name.get(name)
        if path is None:
            errs.append(f"{name}：无日志路径")
            continue
        try:
            smeshes.append(find_smesh_for_inverse_log(path))
        except Exception as e:
            errs.append(f"{name}：{e}")
    if not smeshes:
        show_modeless_message(
            "反演分析",
            "未能定位任何对应 smesh。\n" + "\n".join(errs[:8]),
        )
        return
    msg = add_smeshes_to_compare(smeshes, state=state)
    if errs:
        show_modeless_message(
            "反演分析",
            msg + "\n部分未加入：\n" + "\n".join(errs[:8]),
        )


def format_logs_params_table(
    items: list[tuple[str, Path | None, list]],
    rough_weight: float,
) -> str:
    """多日志参数对照表（等宽文本）。"""
    rows_out: list[list[str]] = []
    headers = [
        "标签",
        "sv",
        "sd",
        "dv",
        "dd",
        "predχ²",
        "RMS",
        "R",
        "score",
        "命令摘要",
    ]
    for name, path, rows in items:
        m = last_row_metrics(rows)
        hdr = ""
        if path is not None:
            try:
                hdr = format_tt_inverse_run_params(parse_tt_inverse_log_header(path))
            except Exception:
                hdr = ""
        if m is None:
            rows_out.append(
                [name, "—", "—", "—", "—", "—", "—", "—", "—", hdr or "—"]
            )
            continue
        sc = composite_score(m["pred_chi"], m["roughness_sum"], rough_weight)
        rows_out.append(
            [
                name,
                f"{m['w_sv']:.4g}",
                f"{m['w_sd']:.4g}",
                f"{m['w_dv']:.4g}",
                f"{m['w_dd']:.4g}",
                f"{m['pred_chi']:.4g}",
                f"{m['rms_total']:.4g}",
                f"{m['roughness_sum']:.4g}",
                f"{sc:.4g}",
                hdr or "—",
            ]
        )
    widths = [len(h) for h in headers]
    for rec in rows_out:
        for i, cell in enumerate(rec):
            widths[i] = max(widths[i], len(cell))
    def _fmt(rec: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(rec))

    lines = [_fmt(headers), _fmt(["—" * w for w in widths])]
    lines.extend(_fmt(r) for r in rows_out)
    lines.append("")
    lines.append(f"score = predχ² × (1 + w×R)，w={rough_weight:g}（越小越好）")
    return "\n".join(lines)


def _show_params_selected(
    names: list[str],
    path_by_name: dict[str, Path],
    rows_by_name: dict[str, list],
    rough_weight: float,
) -> None:
    if len(names) == 1:
        n = names[0]
        _show_params(
            path_by_name.get(n),
            rows_by_name.get(n) or [],
            n,
            rough_weight,
        )
        return
    items = [
        (n, path_by_name.get(n), rows_by_name.get(n) or []) for n in names
    ]
    body = format_logs_params_table(items, rough_weight)
    extras: list[str] = []
    for n, path, _rows in items:
        if path is None:
            continue
        extras.append(f"{n}  {path}")
    if extras:
        body = body + "\n\n路径：\n" + "\n".join(extras)
    show_modeless_text(
        "反演参数列表",
        body,
        summary=f"{len(names)} 个已选",
        width=960,
        height=520,
        monospace=True,
    )


def _show_params(
    path: Path | None, rows: list, name: str, rough_weight: float
) -> None:
    lines = [f"标签：{name}"]
    if path is not None:
        lines.append(f"日志：{path}")
        rd = infer_run_dir_from_smesh(path)
        if rd is not None:
            lines.append(f"运行包：{rd}")
        try:
            info = parse_tt_inverse_log_header(path)
            hdr = format_tt_inverse_run_params(info)
            if hdr:
                lines.append("")
                lines.append("命令/头信息：")
                lines.append(hdr)
        except Exception as e:
            lines.append(f"（读 -L 头失败：{e}）")
    m = last_row_metrics(rows)
    if m is not None:
        sc = composite_score(m["pred_chi"], m["roughness_sum"], rough_weight)
        lines.append("")
        lines.append("末步指标：")
        lines.append(format_params_compact(m))
        lines.append(
            f"RMS={m['rms_total']:.6g}  χ²={m['chi_total']:.6g}  "
            f"pred χ²={m['pred_chi']:.6g}"
        )
        lines.append(
            f"R={m['roughness_sum']:.6g}  "
            f"score(w={rough_weight:g})={sc:.6g}（越小越好）"
        )
    show_modeless_text(
        "反演参数列表", "\n".join(lines), summary=name, width=720, height=520
    )


def _plot_model(state: FormState, path: Path | None, name: str) -> None:
    if path is None:
        show_modeless_message("反演分析", f"无日志路径：{name}")
        return
    try:
        smesh = find_smesh_for_inverse_log(path)
    except Exception as e:
        show_modeless_message("反演分析", str(e))
        return
    try:
        from .inv_monitor_model import MonitorModelWidget
        from .velocity_contours import contour_specs_for_state, contours_enabled
        from ..services.dws_plot import dws_mask_enabled, dws_xyz_for_plot
        from ..services.obs_stations import load_obs_context, resolve_inv_geometry_path
        from ..services.smesh_ops import parse_inverse_smesh_name
        from ..services.smesh_plot_core import (
            load_smesh_plot_data,
            resolve_plot_refl_for_smesh,
            resolve_plot_smesh_cmap,
        )
        from ..services.tres_sample import (
            load_outliers_for_monitor,
            load_tres_for_monitor,
            tres_out_root_candidates,
        )

        from ..widgets.smesh_cmap_combo import SmeshCmapCombo

        work = resolve_work_dir(state.get_str("work_dir"))
        rd = infer_run_dir_from_smesh(path)
        spec = monitor_spec_for_run(rd) if rd is not None else None
        host = QWidget()
        host.resize(960, 760)
        host.setWindowTitle(f"模型 · {smesh.name}（末轮写出）")
        root = QVBoxLayout(host)
        root.setContentsMargins(8, 8, 8, 8)
        bar = QHBoxLayout()
        win = MonitorModelWidget(host)
        win.set_save_dir(work)
        refl = resolve_plot_refl_for_smesh(smesh, state, work)
        mesh, ds, extra = load_smesh_plot_data(smesh, refl)
        dws_xyz = None
        if dws_mask_enabled(state):
            dws_xyz = dws_xyz_for_plot(
                state,
                work,
                smesh,
                out_root=spec.out_root if spec else None,
                run_dir=spec.run_dir if spec else None,
                enabled=True,
            )
        draw_c = contours_enabled(state)

        def _paint_vel(*, reset_home: bool) -> tuple[float, float]:
            return win.set_velocity(
                ds,
                mesh,
                extra,
                resolve_plot_smesh_cmap(state, work),
                smesh.name,
                reset_home=reset_home,
                contour_specs=contour_specs_for_state(state) if draw_c else None,
                dws_xyz=dws_xyz,
            )

        cmap_combo = SmeshCmapCombo(
            state, on_changed=lambda *_a: _paint_vel(reset_home=False)
        )
        bar.addWidget(cmap_combo)
        bar.addStretch(1)
        root.addLayout(bar)
        root.addWidget(win, stretch=1)
        xmin, xmax = _paint_vel(reset_home=True)
        win.set_model_source(smesh, state)
        ctx = load_obs_context(state, work)
        data_file = ctx.data_path or resolve_inv_geometry_path(state, work)
        win.add_stations(
            ctx.stations,
            x_range=(xmin, xmax),
            label_ids=set(ctx.isrc_to_obs.values()) if ctx.isrc_to_obs else None,
        )
        key = parse_inverse_smesh_name(smesh)
        groups: list = []
        note = ""
        outliers: list = []
        used_root = None
        for root in tres_out_root_candidates(smesh=smesh, log=path, spec=spec):
            try:
                groups, note = load_tres_for_monitor(
                    root,
                    iter_prefer=key[0] if key else None,
                    run_dir=spec.run_dir if spec else path.parent,
                    data_file=data_file,
                )
            except Exception as e:
                groups, note = [], str(e)
            if groups:
                used_root = root
                break
        if used_root is not None:
            try:
                outliers, onote = load_outliers_for_monitor(
                    used_root,
                    iter_prefer=key[0] if key else None,
                    iset_prefer=key[1] if key else None,
                )
            except Exception:
                outliers, onote = [], ""
            if onote:
                note = f"{note} · {onote}" if note else onote
        win.set_residuals(
            groups or None,
            ctx,
            note=note or "走时拟合（无 .tres：需 -O 且 out_level≥1）",
            stations=ctx.stations,
            outliers=outliers or None,
        )
        show_modeless_dialog(host, activate=True)
    except Exception as e:
        show_modeless_message("反演分析", f"绘制模型失败: {e}")
