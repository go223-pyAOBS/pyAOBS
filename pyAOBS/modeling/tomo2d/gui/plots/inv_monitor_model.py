"""监视窗右侧：走时拟合 + 速度场。

速度场用 Matplotlib ``imshow``（本窗唯一确认能上色的路径）。
pyqtgraph ``ImageItem`` 在此窗 Windows/PySide6 上会整幅发黑，轴和 OBS 仍正常。
交互挂 ``PyqtgraphStyleNav``（滚轮 / 左拖 / 右拖 / 双击复位）。左侧 χ²/RMS 仍是 pyqtgraph。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from pyAOBS.modeling.tomo2d.tt_inverse_log_analysis import ensure_matplotlib_cjk_font
from pyAOBS.utils.mpl_plot_nav import PyqtgraphStyleNav

from ..services.ray_sample import obs_ray_color
from ..services.tres_sample import RAYTYPE_REFL, RAYTYPE_REFR, RAYTYPE_UNKNOWN
from .export_figure import save_mpl_figure
from .inv_monitor_pg import _PgHintBar

_PHASE_STYLE = {
    RAYTYPE_REFR: {
        "marker": "o",
        "color": "#1f77b4",
        "ms": 4.0,
        "label": "折射 (code=0)",
        "zorder": 3,
    },
    RAYTYPE_REFL: {
        "marker": "^",
        "color": "#d62728",
        "ms": 5.0,
        "label": "反射 (code=1)",
        "zorder": 4,
    },
}
_UNK_STYLE = {
    "marker": "x",
    "color": "#64748b",
    "ms": 4.0,
    "label": "未知震相",
    "zorder": 2,
}
_OTHER_STYLE = {
    "marker": "s",
    "color": "#7c3aed",
    "ms": 3.5,
    "label": "其他震相",
    "zorder": 2,
}


def _unpack_tres_group(item) -> tuple[int, list[float], list[float], list[int]]:
    if len(item) >= 4:
        isrc, xs, rs, codes = item[0], item[1], item[2], item[3]
    else:
        isrc, xs, rs = item[0], item[1], item[2]
        codes = [RAYTYPE_UNKNOWN] * len(xs)
    return int(isrc), list(xs), list(rs), list(codes)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:  # pragma: no cover
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


def _cmap_zero_white(base_name: str, n: int = 256):
    """顺序色标：0 为纯白，再接到 ``base_name``（避免 YlGnBu 浅黄当 0）。"""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    base = None
    try:
        from matplotlib import colormaps

        base = colormaps[base_name]
    except Exception:
        from matplotlib import cm

        base = cm.get_cmap(base_name)
    rest = [tuple(base(float(t))) for t in np.linspace(0.12, 1.0, n - 1)]
    return LinearSegmentedColormap.from_list(
        f"{base_name}_0white", [(1.0, 1.0, 1.0, 1.0), *rest], N=n
    )


def _mpl_cmap(spec: str | None):
    """返回 (matplotlib colormap, optional (vmin, vmax))。"""
    name = (spec or "jet").strip() or "jet"
    if name.lower().endswith("_0white"):
        base = name[: -len("_0white")].strip() or "YlGnBu"
        try:
            return _cmap_zero_white(base), None
        except Exception:
            name = "YlGnBu"
    if name.lower().endswith(".cpt") and Path(name).is_file():
        try:
            from pyAOBS.visualization.gmt_cpt import parse_gmt_cpt_for_matplotlib

            cmap, zmin, zmax = parse_gmt_cpt_for_matplotlib(name)
            return cmap, (float(zmin), float(zmax))
        except Exception:
            name = "jet"
    elif name.lower().endswith(".cpt"):
        name = "jet"
    try:
        from matplotlib import colormaps

        return colormaps[name], None
    except Exception:
        from matplotlib import cm

        try:
            return cm.get_cmap(name), None
        except Exception:
            return cm.get_cmap("jet"), None


def _robust_levels(
    data: np.ndarray, *, weight: np.ndarray | None = None, min_weight: float = 0.35
) -> tuple[float, float]:
    arr = np.asarray(data, dtype=float)
    finite = np.isfinite(arr)
    if weight is not None:
        sel = finite & (np.asarray(weight, dtype=float) >= float(min_weight))
        if int(np.count_nonzero(sel)) >= 16:
            finite = sel
    if not np.any(finite):
        return 1.5, 8.0
    a = float(np.nanpercentile(arr[finite], 1))
    b = float(np.nanpercentile(arr[finite], 99))
    if b <= a:
        b = a + 1e-6
    return a, b


def _fmt_colorbar_tick(v: float) -> str:
    s = f"{float(v):.4g}"
    if s.endswith(".0") and "." in s and s.count(".") == 1:
        try:
            if abs(float(s) - float(v)) < 1e-12:
                return s[:-2] if abs(float(v) - int(float(v))) < 1e-12 else s
        except ValueError:
            pass
    return s


def _apply_colorbar_limits(cb, cax, lo: float, hi: float, *, cb_label: str) -> None:
    """换 CPT 后必须重设 clim / ylim，否则旧范围会裁掉新色标两端。"""
    lo_f, hi_f = float(lo), float(hi)
    if not np.isfinite(lo_f) or not np.isfinite(hi_f) or hi_f <= lo_f:
        hi_f = lo_f + 1e-6
    try:
        cb.mappable.set_clim(lo_f, hi_f)
    except Exception:
        pass
    try:
        cax.set_autoscaley_on(False)
        cax.set_ylim(lo_f, hi_f)
    except Exception:
        pass
    ticks = np.linspace(lo_f, hi_f, 6)
    cb.set_ticks(ticks)
    try:
        cb.set_ticklabels([_fmt_colorbar_tick(t) for t in ticks])
    except Exception:
        pass
    cb.set_label(cb_label, color="black")
    cb.ax.tick_params(colors="black", labelsize=8)
    cb.ax.yaxis.label.set_color("black")
    try:
        cb.ax.yaxis.set_tick_params(which="both", right=True, pad=2)
    except Exception:
        pass


def finish_figure_layout(fig) -> None:
    """constrained layout 在换色标后按新刻度重排，避免右侧范围被裁。"""
    engine = getattr(fig, "get_layout_engine", lambda: None)()
    if engine is None:
        return
    try:
        engine.execute(fig)
    except Exception:
        pass


def _value_norm(lo: float, hi: float, gamma: float | None):
    from matplotlib.colors import Normalize, PowerNorm

    vmin, vmax = float(lo), float(hi)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    try:
        g = float(gamma) if gamma is not None else 1.0
    except (TypeError, ValueError):
        g = 1.0
    if 0.0 < g < 0.999:
        return PowerNorm(gamma=g, vmin=vmin, vmax=vmax, clip=False)
    return Normalize(vmin=vmin, vmax=vmax, clip=False)


def imshow_velocity_field(
    ax,
    cax,
    fig,
    data: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    cmap,
    lo: float,
    hi: float,
    *,
    alpha: np.ndarray | None = None,
    cb_label: str = "km/s",
    norm_gamma: float | None = None,
):
    """速度 imshow。有 ``alpha`` 时用 RGBA（色标仍按 ``cb_label``）。返回 extent 与 colorbar。"""
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    extent = (xmin, xmax, zmax, zmin)
    try:
        cax.set_ylim(float(lo), float(hi))
    except Exception:
        pass
    norm = _value_norm(lo, hi, norm_gamma)
    if alpha is None:
        im = ax.imshow(
            data,
            extent=extent,
            cmap=cmap,
            aspect="auto",
            norm=norm,
            interpolation="nearest",
            zorder=0,
        )
        cb = fig.colorbar(im, cax=cax)
    else:
        from matplotlib.cm import ScalarMappable

        rgba = np.asarray(cmap(norm(np.asarray(data, dtype=float))), dtype=float)
        a = np.asarray(alpha, dtype=float)
        if a.shape != rgba.shape[:2]:
            a = np.ones(rgba.shape[:2], dtype=float)
        bad = ~np.isfinite(data)
        rgba[..., 3] = np.where(bad, 0.0, np.clip(a, 0.0, 1.0))
        ax.imshow(
            rgba,
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            zorder=0,
        )
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
    _apply_colorbar_limits(cb, cax, lo, hi, cb_label=cb_label)
    return xmin, xmax, zmin, zmax, cb


def _prepare_velocity_arrays(
    ds: Any,
    cmap_spec: str,
    *,
    dws_xyz=None,
    mesh: Any = None,
    vlim: tuple[float, float] | None = None,
) -> dict[str, Any]:
    data = np.asarray(ds["velocity"].values, dtype=float)
    x = np.asarray(ds["x"].values, dtype=float)
    z = np.asarray(ds["z"].values, dtype=float)
    dims = tuple(getattr(ds["velocity"], "dims", ()))
    if "z" in dims and "x" in dims:
        data = np.asarray(ds["velocity"].transpose("z", "x").values, dtype=float)
    elif data.ndim == 2 and data.shape == (x.size, z.size):
        data = data.T
    cmap, cpt_lv = _mpl_cmap(cmap_spec)
    from ..services.smesh_plot_core import (
        alias_builtin_smesh_cmap_id,
        cmap_blank_air,
        mask_air_layer_for_plot,
    )

    if alias_builtin_smesh_cmap_id(str(cmap_spec)) == "water":
        data = mask_air_layer_for_plot(data, x, z, mesh)
        cmap = cmap_blank_air(cmap)
    alpha = None
    contour_data = data
    if dws_xyz is not None:
        from ..services.dws_plot import (
            alpha_from_dws,
            cmap_blank_uncovered,
            dws_grid_for_plot,
        )

        cmap = cmap_blank_uncovered(cmap)
        dws = dws_grid_for_plot(x, z, dws_xyz, mesh=mesh)
        if dws is not None:
            alpha = alpha_from_dws(dws)
            contour_data = np.array(data, dtype=float, copy=True)
            contour_data[alpha <= 0.0] = np.nan
    lo, hi = (
        vlim
        if vlim is not None
        else (cpt_lv if cpt_lv is not None else _robust_levels(data, weight=alpha))
    )
    return {
        "data": data,
        "x": x,
        "z": z,
        "cmap": cmap,
        "cpt_lv": cpt_lv,
        "alpha": alpha,
        "contour_data": contour_data,
        "lo": lo,
        "hi": hi,
    }


def overlay_line_color(raw, default: str = "#dc143c"):
    """界面叠线颜色：hex 原样；RGBA 元组收成 Python float，避免 ``str(np.float64)``。"""
    if raw is None:
        return default
    if isinstance(raw, str):
        return raw
    try:
        vals = tuple(float(x) for x in raw)
    except (TypeError, ValueError):
        return default
    if len(vals) >= 4:
        return vals[:4]
    if len(vals) == 3:
        return vals
    return default


def _overlay_mesh_geometry(ax, mesh: Any, extra_interfaces: list | None) -> None:
    if mesh is not None and getattr(mesh, "xpos", None) is not None:
        ax.plot(
            np.asarray(mesh.xpos, dtype=float),
            np.asarray(mesh.topo, dtype=float),
            color="k",
            lw=0.9,
            zorder=2,
        )
    if extra_interfaces:
        for iface in extra_interfaces:
            x = np.asarray(iface["x"], dtype=float)
            color = overlay_line_color(iface.get("color"))
            z_lo, z_hi = iface.get("z_lo"), iface.get("z_hi")
            if z_lo is not None and z_hi is not None:
                ax.fill_between(
                    x,
                    np.asarray(z_lo, dtype=float),
                    np.asarray(z_hi, dtype=float),
                    color=color,
                    alpha=float(iface.get("fill_alpha") or 0.22),
                    linewidth=0,
                    zorder=2.4,
                )
            if iface.get("draw_line", True):
                ax.plot(
                    x,
                    np.asarray(iface["z"], dtype=float),
                    color=color,
                    lw=float(iface.get("linewidth") or 1.6),
                    ls=str(iface.get("linestyle") or "--"),
                    zorder=3,
                )
            text = str(iface.get("text") or "").strip()
            if text and x.size:
                zz = np.asarray(iface.get("z"), dtype=float)
                if z_lo is not None and z_hi is not None:
                    zz = 0.5 * (
                        np.asarray(z_lo, dtype=float) + np.asarray(z_hi, dtype=float)
                    )
                mid = int(x.size // 2)
                ax.annotate(
                    text,
                    (float(x[mid]), float(zz[min(mid, zz.size - 1)])),
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                    zorder=5,
                    xytext=(4, 2),
                    textcoords="offset points",
                )


def _style_depth_ax(ax, title: str, *, xlabel: bool) -> None:
    ax.set_title(title, color="black")
    ax.set_ylabel("深度 (km)", color="black")
    if xlabel:
        ax.set_xlabel("模型距离 (km)", color="black")
        ax.tick_params(labelbottom=True, colors="black")
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False, colors="black")
    ax.yaxis.label.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.grid(True, alpha=0.3)


def overlay_ray_groups(ax, groups, *, color_for_isrc=None, zorder: int = 4) -> int:
    """在速度场 axes 上叠加抽样射线（按 OBS/炮着色）。返回条数。"""
    n_segs = sum(len(s) for _, s in groups)
    if n_segs <= 0:
        return 0
    alpha = 0.20 if n_segs > 400 else (0.26 if n_segs > 80 else 0.32)
    for isrc, segs in groups:
        if not segs:
            continue
        oid = color_for_isrc(isrc) if color_for_isrc is not None else int(isrc)
        color = obs_ray_color(oid)
        xs: list[float] = []
        zs: list[float] = []
        for sx, sz in segs:
            xs.extend(sx)
            zs.extend(sz)
            xs.append(float("nan"))
            zs.append(float("nan"))
        ax.plot(xs, zs, color=color, lw=0.6, alpha=alpha, zorder=zorder)
    return n_segs


class MonitorModelWidget(QWidget):
    """单模型：上两行拟合、下速度场。差值：ΔV / B / A。统计：均值 / σ。"""

    def __init__(self, parent=None, *, compare_bar: bool = True) -> None:
        super().__init__(parent)
        _ = compare_bar  # 兼容旧调用；对比改走图上右键，不再显示工具条按钮。
        ensure_matplotlib_cjk_font()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self._save_dir = ""
        self._smesh_path: Path | None = None
        self._stack_path_a: Path | None = None
        self._stack_path_b: Path | None = None
        self._form_state = None
        self._allow_clear_compare = False
        self._hint_bar = _PgHintBar(self.reset_view, self.save_png, self)
        lay.addWidget(self._hint_bar)

        self.fig = Figure(figsize=(8.0, 7.0), facecolor="w", layout="constrained")
        try:
            self.fig.set_constrained_layout_pads(
                w_pad=0.04, h_pad=0.04, hspace=0.04, wspace=0.05
            )
        except Exception:
            pass
        self._layout_mode = None
        self._cb = None
        self._build_fit_layout()
        self._setup_empty_axes("等待模型…")

        self.canvas = FigureCanvasQTAgg(self.fig)
        lay.addWidget(self.canvas, stretch=1)
        self._on_stats_context = None
        self._nav = PyqtgraphStyleNav(
            self.canvas, on_right_click=self._on_mpl_right_click
        )
        self._home: tuple[float, float, float, float] | None = None

    def set_interaction_hint(self, extra: str = "") -> None:
        base = "滚轮缩放 · 左拖平移 · 右拖缩放 · 双击复位 · 右键菜单"
        if extra:
            self._hint_bar.hint.setText(f"{base} · {extra}")
        else:
            self._hint_bar.hint.setText(base)

    def set_stats_context_handler(self, fn) -> None:
        self._on_stats_context = fn

    def stats_panel_of(self, ax) -> str | None:
        """统计布局：上图 mean，下图 std；色标算作同行。"""
        if ax is None:
            return None
        if ax is self.ax_diff or ax is self._cax_diff:
            return "mean"
        if ax is self.ax_b or ax is self._cax_b:
            return "std"
        return None

    def _on_mpl_right_click(self, event) -> None:
        from ..services.model_compare import popup_model_context_menu

        ax = getattr(event, "inaxes", None)
        extra = None
        path = self._path_for_axes(ax)
        if self._layout_mode == "stats":
            fn = self._on_stats_context
            panel = self.stats_panel_of(ax)
            if callable(fn) and panel is not None:

                def extra(menu, p=panel):
                    fn(p, menu)

                path = None
        popup_model_context_menu(
            self,
            path=path,
            state=self._form_state,
            extra_actions=extra,
            allow_clear=self._allow_clear_compare,
        )

    def _path_for_axes(self, ax) -> Path | None:
        """差值三图：中 B、下 A 各用对应 smesh；ΔV 不加入对比。

        棋盘预览等同三图但只有一个背景 smesh 时，三幅都可添加该背景。
        """
        if self._layout_mode == "compare":
            split = self._stack_path_a is not None or self._stack_path_b is not None
            if ax is self.ax_a or ax is self._cax_a:
                return self._stack_path_a or self._smesh_path
            if ax is self.ax_b or ax is self._cax_b:
                return self._stack_path_b or self._smesh_path
            if ax is self.ax_diff or ax is self._cax_diff:
                return None if split else self._smesh_path
        return self._smesh_path

    def set_model_source(
        self,
        path: Path | str | None,
        state=None,
        *,
        path_a: Path | str | None = None,
        path_b: Path | str | None = None,
    ) -> None:
        """供右键「添加到对比模型」读取当前 smesh 与工区表单。"""
        self._smesh_path = Path(path) if path is not None else None
        self._stack_path_a = Path(path_a) if path_a is not None else None
        self._stack_path_b = Path(path_b) if path_b is not None else None
        if state is not None:
            self._form_state = state

    def _build_fit_layout(self) -> None:
        self.fig.clear()
        gs = self.fig.add_gridspec(
            3, 2, height_ratios=[0.9, 0.9, 2.2], width_ratios=[28, 1.85]
        )
        self.ax_fit_refr = self.fig.add_subplot(gs[0, 0])
        self.ax_fit_refl = self.fig.add_subplot(gs[1, 0], sharex=self.ax_fit_refr)
        self.ax_fit = self.ax_fit_refr
        self.ax_mesh = self.fig.add_subplot(gs[2, 0], sharex=self.ax_fit_refr)
        self._cax = self.fig.add_subplot(gs[2, 1])
        self._cax.set_visible(False)
        self.ax_diff = self.ax_b = self.ax_a = None
        self._cax_diff = self._cax_b = self._cax_a = None
        self._cb = None
        self._layout_mode = "fit"

    def _build_compare_layout(self, n: int = 3) -> None:
        n = 2 if int(n) == 2 else 3
        self.fig.clear()
        gs = self.fig.add_gridspec(
            n, 2, height_ratios=[1.0] * n, width_ratios=[28, 1.85]
        )
        self.ax_diff = self.fig.add_subplot(gs[0, 0])
        self.ax_b = self.fig.add_subplot(
            gs[1, 0], sharex=self.ax_diff, sharey=self.ax_diff
        )
        self._cax_diff = self.fig.add_subplot(gs[0, 1])
        self._cax_b = self.fig.add_subplot(gs[1, 1])
        if n >= 3:
            self.ax_a = self.fig.add_subplot(
                gs[2, 0], sharex=self.ax_diff, sharey=self.ax_diff
            )
            self._cax_a = self.fig.add_subplot(gs[2, 1])
        else:
            self.ax_a = None
            self._cax_a = None
        self.ax_mesh = self.ax_diff
        self._cax = self._cax_diff
        self.ax_fit_refr = None
        self.ax_fit_refl = None
        self.ax_fit = None
        self._cb = None
        self._layout_mode = "stats" if n == 2 else "compare"

    def _ensure_layout(self, mode: str) -> None:
        if self._layout_mode == mode:
            return
        if mode == "compare":
            self._build_compare_layout(3)
        elif mode == "stats":
            self._build_compare_layout(2)
        else:
            self._build_fit_layout()
        self._home = None
        self._nav.clear_saved_views()

    def _snapshot_plot_views(
        self,
    ) -> list[tuple[Any, tuple[float, float], tuple[float, float]]]:
        """记下当前数据轴视窗（不含色标），勾选 DWS/等值线重绘时还原。"""
        out: list[tuple[Any, tuple[float, float], tuple[float, float]]] = []
        seen: set[int] = set()
        for ax in (
            self.ax_diff,
            self.ax_b,
            self.ax_a,
            self.ax_mesh,
            self.ax_fit_refr,
            self.ax_fit_refl,
        ):
            if ax is None or id(ax) in seen:
                continue
            seen.add(id(ax))
            try:
                out.append((ax, tuple(ax.get_xlim()), tuple(ax.get_ylim())))
            except Exception:
                continue
        return out

    def _restore_plot_views(
        self,
        snap: list[tuple[Any, tuple[float, float], tuple[float, float]]],
    ) -> None:
        for ax, xlim, ylim in snap:
            try:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            except Exception:
                continue
        self._nav.remember_current_views()

    def _setup_empty_axes(self, mesh_title: str) -> None:
        self._style_fit_ax(self.ax_fit_refr, "折射残差 (s)", "走时拟合 · 折射")
        self._style_fit_ax(self.ax_fit_refl, "反射残差 (s)", "走时拟合 · 反射")
        self.ax_mesh.set_title(mesh_title, color="black")
        self.ax_mesh.set_xlabel("模型距离 (km)", color="black")
        self.ax_mesh.set_ylabel("深度 (km)", color="black")
        self.ax_mesh.tick_params(colors="black")
        self.ax_mesh.grid(True, alpha=0.3)
        self._cax.set_visible(False)

    def _paint_velocity_panel(
        self,
        ax,
        cax,
        prep: dict[str, Any],
        mesh: Any,
        extra_interfaces: list | None,
        title: str,
        *,
        cb_label: str,
        draw_contours: bool,
        contour_specs: list | None,
        xlabel: bool,
        note: str = "",
    ) -> tuple[float, float, float, float]:
        ax.clear()
        cax.clear()
        try:
            cax.set_autoscaley_on(True)
        except Exception:
            pass
        xmin, xmax, zmin, zmax, cb = imshow_velocity_field(
            ax,
            cax,
            self.fig,
            prep["data"],
            prep["x"],
            prep["z"],
            prep["cmap"],
            prep["lo"],
            prep["hi"],
            alpha=prep["alpha"],
            cb_label=cb_label,
            norm_gamma=prep.get("norm_gamma"),
        )
        cax.set_visible(True)
        self._cb = cb
        ax.set_facecolor("white")
        if draw_contours:
            from .velocity_contours import overlay_velocity_contours

            overlay_velocity_contours(
                ax, prep["contour_data"], prep["x"], prep["z"], contour_specs, zorder=1
            )
        _overlay_mesh_geometry(ax, mesh, extra_interfaces)
        _style_depth_ax(ax, title, xlabel=xlabel)
        text = str(note or "").strip()
        if text:
            ax.text(
                0.012,
                0.035,
                text,
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                ha="left",
                color="black",
                zorder=6,
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": "white",
                    "edgecolor": "#94a3b8",
                    "alpha": 0.86,
                },
            )
        return xmin, xmax, zmin, zmax

    @staticmethod
    def _style_fit_ax(ax, ylabel: str, title: str) -> None:
        ax.set_title(title, color="black")
        ax.set_ylabel(ylabel, color="black")
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="#94a3b8", lw=0.9, zorder=1)
        ax.tick_params(labelbottom=False, colors="black")

    def reset_view(self) -> None:
        self._nav.reset_view()

    def set_save_dir(self, path: str | Path | None) -> None:
        self._save_dir = str(path) if path else ""

    def save_png(self) -> None:
        save_mpl_figure(
            self, self.fig, start_dir=self._save_dir, default_name="inv_monitor_model.png"
        )

    def show_empty(self, mesh_title: str) -> None:
        self._ensure_layout("fit")
        self.ax_fit_refr.clear()
        self.ax_fit_refl.clear()
        self.ax_mesh.clear()
        self._cax.clear()
        self._cb = None
        self._setup_empty_axes(mesh_title)
        self._home = None
        self._nav.clear_saved_views()
        self.canvas.draw_idle()

    def show_empty_stack(self, message: str) -> None:
        """三行速度图占位（无走时拟合轴），供棋盘格预览等。"""
        self._ensure_layout("compare")
        for ax, cax, title, xlabel in (
            (self.ax_diff, self._cax_diff, message, False),
            (self.ax_b, self._cax_b, "棋盘后 Vp", False),
            (self.ax_a, self._cax_a, "棋盘前 Vp（背景）", True),
        ):
            ax.clear()
            cax.clear()
            cax.set_visible(False)
            ax.set_title(title, color="black")
            ax.set_ylabel("深度 (km)", color="black")
            if xlabel:
                ax.set_xlabel("模型距离 (km)", color="black")
            else:
                ax.set_xlabel("")
            ax.tick_params(colors="black")
            ax.grid(True, alpha=0.3)
        self._cb = None
        self._home = None
        self._nav.clear_saved_views()
        self.canvas.draw_idle()

    def set_velocity(
        self,
        ds: Any,
        mesh: Any,
        extra_interfaces: list | None,
        cmap_spec: str,
        title: str,
        *,
        reset_home: bool = True,
        contour_specs: list | None = None,
        dws_xyz=None,
        vlim: tuple[float, float] | None = None,
        cb_label: str | None = None,
        draw_contours: bool = True,
    ) -> tuple[float, float]:
        self._ensure_layout("fit")
        if not cb_label:
            from ..services.smesh_plot_core import colorbar_label_for_cmap

            cb_label = colorbar_label_for_cmap(cmap_spec)
        snap = [] if reset_home else self._snapshot_plot_views()
        prep = _prepare_velocity_arrays(
            ds, cmap_spec, dws_xyz=dws_xyz, mesh=mesh, vlim=vlim
        )

        xmin, xmax, zmin, zmax = self._paint_velocity_panel(
            self.ax_mesh,
            self._cax,
            prep,
            mesh,
            extra_interfaces,
            title,
            cb_label=cb_label,
            draw_contours=draw_contours,
            contour_specs=contour_specs,
            xlabel=True,
        )
        if snap:
            self._restore_plot_views(snap)
        else:
            self.ax_mesh.set_xlim(xmin, xmax)
            from ..services.smesh_plot_core import (
                air_axis_zlim,
                alias_builtin_smesh_cmap_id,
            )

            if alias_builtin_smesh_cmap_id(str(cmap_spec)) == "water":
                ylim = air_axis_zlim(zmin, zmax)
                self.ax_mesh.set_ylim(*ylim)
                self._home = (xmin, xmax, ylim[1], ylim[0])
            else:
                self.ax_mesh.set_ylim(zmax, zmin)
                self._home = (xmin, xmax, zmin, zmax)
            self._nav.schedule_home_refresh()
        finish_figure_layout(self.fig)
        self.canvas.draw_idle()
        return xmin, xmax

    def set_velocity_stack(
        self,
        panels: list[dict[str, Any]],
        *,
        dws_xyz=None,
        reset_home: bool = True,
    ) -> tuple[float, float]:
        """自上而下速度图：差值 3 块（ΔV / B / A），统计 2 块（均值 / σ）。"""
        n = len(panels)
        if n not in (2, 3):
            raise ValueError("set_velocity_stack 需要 2 或 3 个面板")
        self._ensure_layout("stats" if n == 2 else "compare")
        snap = [] if reset_home else self._snapshot_plot_views()
        preps: list[dict[str, Any]] = []
        for p in panels:
            preps.append(
                _prepare_velocity_arrays(
                    p["ds"],
                    str(p.get("cmap_spec") or "jet"),
                    dws_xyz=dws_xyz,
                    mesh=p.get("mesh"),
                    vlim=p.get("vlim"),
                )
            )
            preps[-1]["norm_gamma"] = p.get("norm_gamma")
        if (
            n == 3
            and panels[1].get("vlim") is None
            and panels[2].get("vlim") is None
        ):
            if preps[1]["cpt_lv"] is None and preps[2]["cpt_lv"] is None:
                lo, hi = _robust_levels(
                    np.concatenate(
                        [preps[1]["data"].ravel(), preps[2]["data"].ravel()]
                    )
                )
                preps[1]["lo"] = preps[2]["lo"] = lo
                preps[1]["hi"] = preps[2]["hi"] = hi

        axes = [
            (self.ax_diff, self._cax_diff, False),
            (self.ax_b, self._cax_b, n == 2),
        ]
        if n == 3:
            axes.append((self.ax_a, self._cax_a, True))
        extents: list[tuple[float, float, float, float]] = []
        for prep, panel, (ax, cax, xlabel) in zip(preps, panels, axes):
            extents.append(
                self._paint_velocity_panel(
                    ax,
                    cax,
                    prep,
                    panel.get("mesh"),
                    panel.get("extra"),
                    str(panel.get("title") or ""),
                    cb_label=str(panel.get("cb_label") or "km/s"),
                    draw_contours=bool(panel.get("draw_contours", False)),
                    contour_specs=panel.get("contour_specs"),
                    xlabel=xlabel,
                    note=str(panel.get("note") or ""),
                )
            )
        xmin = min(e[0] for e in extents)
        xmax = max(e[1] for e in extents)
        zmin = min(e[2] for e in extents)
        zmax = max(e[3] for e in extents)
        if snap:
            self._restore_plot_views(snap)
        else:
            self.ax_diff.set_xlim(xmin, xmax)
            from ..services.smesh_plot_core import (
                air_axis_zlim,
                alias_builtin_smesh_cmap_id,
            )

            water = any(
                alias_builtin_smesh_cmap_id(str(p.get("cmap_spec") or ""))
                == "water"
                for p in panels
            )
            ylim = air_axis_zlim(zmin, zmax) if water else (zmax, zmin)
            self.ax_diff.set_ylim(*ylim)
            self._home = (xmin, xmax, ylim[1], ylim[0])
            self._nav.schedule_home_refresh()
        finish_figure_layout(self.fig)
        self.canvas.draw_idle()
        return xmin, xmax

    def add_rays(self, groups: list, ctx) -> None:
        for isrc, segs in groups:
            if not segs:
                continue
            x0 = segs[0][0][0] if segs[0][0] else None
            if x0 is not None:
                ctx.enrich_from_ray_x(isrc, float(x0))
        overlay_ray_groups(
            self.ax_mesh, groups, color_for_isrc=ctx.obs_id
        )
        self.canvas.draw_idle()

    def add_stations(
        self,
        stations: list[tuple[int, float, float]],
        *,
        x_range: tuple[float, float] | None = None,
        label_ids: set[int] | None = None,
    ) -> int:
        if not stations:
            return 0
        xmin = xmax = None
        if x_range is not None:
            xmin, xmax = float(min(x_range)), float(max(x_range))
            pad = max(1.0, 0.02 * (xmax - xmin)) if xmax > xmin else 1.0
            xmin -= pad
            xmax += pad
        xs: list[float] = []
        zs: list[float] = []
        ids: list[int] = []
        for oid, x, z in stations:
            if xmin is not None and not (xmin <= float(x) <= xmax):
                continue
            xs.append(float(x))
            zs.append(float(z))
            ids.append(int(oid))
        if not xs:
            return 0
        self.ax_mesh.plot(
            xs,
            zs,
            "o",
            ms=7,
            mfc="white",
            mec="k",
            mew=1.1,
            zorder=6,
            linestyle="none",
        )
        show_all = len(ids) <= 24
        for oid, x, z in zip(ids, xs, zs):
            if not show_all and label_ids is not None and oid not in label_ids:
                continue
            self.ax_mesh.annotate(
                str(oid),
                (x, z),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                va="bottom",
                fontsize=8,
                color="k",
                zorder=7,
            )
        self.canvas.draw_idle()
        return len(ids)

    def set_residuals(
        self,
        groups: list | None,
        ctx,
        *,
        note: str = "",
        stations: list | None = None,
        outliers: list | None = None,
    ) -> None:
        if self._layout_mode != "fit" or self.ax_fit_refr is None:
            return
        xlim = self.ax_fit_refr.get_xlim()
        self.ax_fit_refr.clear()
        self.ax_fit_refl.clear()
        self._style_fit_ax(self.ax_fit_refr, "折射残差 (s)", "折射")
        self._style_fit_ax(self.ax_fit_refl, "反射残差 (s)", "反射")
        if not groups:
            self.ax_fit_refr.set_title(note or "走时拟合 · 折射")
            self.ax_fit_refr.set_xlim(xlim)
            self.ax_fit_refl.set_xlim(xlim)
            self.canvas.draw_idle()
            return
        used: set[int] = set()
        all_r: list[float] = []
        by_code: dict[int, tuple[list[float], list[float]]] = {}
        unpacked: list[tuple[int, list[float], list[float], list[int]]] = []
        for item in groups:
            isrc, xs, rs, codes = _unpack_tres_group(item)
            unpacked.append((isrc, xs, rs, codes))
            all_r.extend(rs)
            if ctx is not None:
                used.add(int(ctx.obs_id(isrc)))
            for x, r, c in zip(xs, rs, codes):
                px, pr = by_code.setdefault(int(c), ([], []))
                px.append(x)
                pr.append(r)
        by_phase = any(c in (RAYTYPE_REFR, RAYTYPE_REFL) for c in by_code)
        if by_phase:
            refr = by_code.get(RAYTYPE_REFR)
            if refr and refr[0]:
                st = _PHASE_STYLE[RAYTYPE_REFR]
                self.ax_fit_refr.plot(refr[0], refr[1], linestyle="none", alpha=0.85, **st)
            unk = by_code.get(RAYTYPE_UNKNOWN)
            if unk and unk[0]:
                self.ax_fit_refr.plot(unk[0], unk[1], linestyle="none", alpha=0.7, **_UNK_STYLE)
            refl = by_code.get(RAYTYPE_REFL)
            if refl and refl[0]:
                st = _PHASE_STYLE[RAYTYPE_REFL]
                self.ax_fit_refl.plot(refl[0], refl[1], linestyle="none", alpha=0.85, **st)
            other_x: list[float] = []
            other_r: list[float] = []
            for code, pts in by_code.items():
                if code in (RAYTYPE_REFR, RAYTYPE_REFL, RAYTYPE_UNKNOWN):
                    continue
                other_x.extend(pts[0])
                other_r.extend(pts[1])
            if other_x:
                self.ax_fit_refl.plot(
                    other_x, other_r, linestyle="none", alpha=0.75, **_OTHER_STYLE
                )
        else:
            for isrc, xs, rs, _codes in unpacked:
                color = "#1f77b4"
                if ctx is not None:
                    color = obs_ray_color(ctx.obs_id(isrc))
                self.ax_fit_refr.plot(
                    xs,
                    rs,
                    "o",
                    ms=3.5,
                    color=color,
                    alpha=0.75,
                    linestyle="none",
                    zorder=2,
                )
            self.ax_fit_refl.set_title("反射（无震相信息）")
        self._draw_obs_guides(self.ax_fit_refr, stations, used)
        self._draw_obs_guides(self.ax_fit_refl, stations, used)
        n_x0 = 0
        n_x1 = 0
        if outliers:
            ox0: list[float] = []
            or0: list[float] = []
            ox1: list[float] = []
            or1: list[float] = []
            for item in outliers:
                if len(item) >= 6:
                    x, r, code = float(item[3]), float(item[5]), int(item[4])
                else:
                    x, r, code = float(item[0]), float(item[1]), int(item[2])
                if int(code) == RAYTYPE_REFL:
                    ox1.append(x)
                    or1.append(r)
                else:
                    ox0.append(x)
                    or0.append(r)
                all_r.append(r)
            n_x0 = len(ox0)
            n_x1 = len(ox1)
            if ox0:
                self.ax_fit_refr.plot(
                    ox0, or0, "x", ms=6, mew=1.2, color="#111827", zorder=6, linestyle="none"
                )
            if ox1:
                self.ax_fit_refl.plot(
                    ox1, or1, "x", ms=6, mew=1.2, color="#111827", zorder=6, linestyle="none"
                )
        arr = np.asarray(all_r, dtype=float)
        span = float(np.nanpercentile(np.abs(arr), 98)) if arr.size else 0.05
        ylim = max(0.05, 1.15 * span)
        self.ax_fit_refr.set_ylim(-ylim, ylim)
        self.ax_fit_refl.set_ylim(-ylim, ylim)
        self.ax_fit_refr.set_xlim(xlim)
        self.ax_fit_refl.set_xlim(xlim)
        n_refr = len(by_code.get(RAYTYPE_REFR, ([], []))[0]) if by_phase else len(all_r)
        n_refl = len(by_code.get(RAYTYPE_REFL, ([], []))[0]) if by_phase else 0
        t_refr = f"折射  n={n_refr}"
        t_refl = f"反射  n={n_refl}"
        if n_x0:
            t_refr = f"{t_refr}  · ×={n_x0} 本轮-R剔除"
        if n_x1:
            t_refl = f"{t_refl}  · ×={n_x1} 本轮-R剔除"
        self.ax_fit_refr.set_title(t_refr)
        self.ax_fit_refl.set_title(t_refl)
        self.canvas.draw_idle()

    @staticmethod
    def _draw_obs_guides(ax, stations: list | None, used: set[int]) -> None:
        if not stations:
            return
        sx: list[float] = []
        for oid, x, _z in stations:
            if int(oid) not in used:
                continue
            ax.axvline(float(x), color="#94a3b8", lw=0.6, ls=":", zorder=0)
            sx.append(float(x))
        if sx:
            ax.plot(
                sx,
                [0.0] * len(sx),
                "o",
                ms=6,
                mfc="white",
                mec="k",
                mew=1.0,
                zorder=4,
                linestyle="none",
            )
