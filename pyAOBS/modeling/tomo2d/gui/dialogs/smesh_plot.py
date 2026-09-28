"""绘制 smesh 速度模型（Matplotlib imshow，与监视窗同一套）。

pyqtgraph ImageItem 在 Windows/PySide6 上会整幅发黑；本窗与监视窗一样用
``ax.imshow``，交互挂 ``PyqtgraphStyleNav``。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import (
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pyAOBS.modeling.tomo2d.tt_inverse_log_analysis import ensure_matplotlib_cjk_font

from ..dialog_utils import file_dialog_options, show_modeless_dialog, show_modeless_message
from ..plots.inv_monitor_model import (
    _mpl_cmap,
    _robust_levels,
    finish_figure_layout,
    imshow_velocity_field,
    overlay_line_color,
    overlay_ray_groups,
)
from ..plots.mpl_figure_window import MplNavCanvas
from ..services.file_filters import DWS_OPEN_FILTERS, MODEL_OPEN_FILTERS, RAY_OPEN_FILTERS, REFL_OPEN_FILTERS
from ..services.paths import resolve_work_dir
from ..services.smesh_plot_core import (
    alias_builtin_smesh_cmap_id,
    cmap_blank_air,
    colorbar_label_for_cmap,
    load_interface_overlays,
    load_model_plot_data,
    looks_like_interface_name,
    looks_like_model_name,
    looks_like_smesh_name,
    normalize_dropped_path,
    parse_clipboard_path_text,
    read_windows_explorer_clipboard_paths,
    resolve_plot_refl_for_smesh,
    resolve_plot_smesh_cmap,
    running_in_wsl,
)
from ..state.form_state import FormState


def draw_smesh_velocity_figure(
    ds,
    mesh,
    extra_interfaces: list | None,
    cmap_spec: str,
    title: str,
    *,
    contour_specs: list | None = None,
    dws_xyz=None,
    ray_groups=None,
) -> Figure:
    """与 ``MonitorModelWidget.set_velocity`` 相同的 imshow / 色标 / 界面。"""
    ensure_matplotlib_cjk_font()
    data = np.asarray(ds["velocity"].values, dtype=float)
    x = np.asarray(ds["x"].values, dtype=float)
    z = np.asarray(ds["z"].values, dtype=float)
    dims = tuple(getattr(ds["velocity"], "dims", ()))
    if "z" in dims and "x" in dims:
        data = np.asarray(ds["velocity"].transpose("z", "x").values, dtype=float)
    elif data.ndim == 2 and data.shape == (x.size, z.size):
        data = data.T
    cmap, cpt_lv = _mpl_cmap(cmap_spec)
    if alias_builtin_smesh_cmap_id(str(cmap_spec)) == "water":
        from ..services.smesh_plot_core import mask_air_layer_for_plot

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
    lo, hi = cpt_lv if cpt_lv is not None else _robust_levels(data, weight=alpha)

    fig = Figure(figsize=(9.6, 5.2), facecolor="w", layout="constrained")
    try:
        fig.set_constrained_layout_pads(
            w_pad=0.04, h_pad=0.04, hspace=0.04, wspace=0.05
        )
    except Exception:
        pass
    gs = fig.add_gridspec(1, 2, width_ratios=[28, 1.85])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    xmin, xmax, zmin, zmax, _cb = imshow_velocity_field(
        ax,
        cax,
        fig,
        data,
        x,
        z,
        cmap,
        lo,
        hi,
        alpha=alpha,
        cb_label=colorbar_label_for_cmap(cmap_spec),
    )
    ax.set_facecolor("white")
    if alias_builtin_smesh_cmap_id(str(cmap_spec)) == "water":
        from ..services.smesh_plot_core import air_axis_zlim

        ax.set_ylim(*air_axis_zlim(zmin, zmax))
    cax.set_visible(True)
    from ..plots.velocity_contours import overlay_velocity_contours

    overlay_velocity_contours(ax, contour_data, x, z, contour_specs, zorder=1)
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
            ax.plot(
                np.asarray(iface["x"], dtype=float),
                np.asarray(iface["z"], dtype=float),
                color=overlay_line_color(iface.get("color")),
                lw=float(iface.get("linewidth", 1.6)),
                ls=str(iface.get("linestyle", "--")),
                zorder=3,
            )
    if ray_groups:
        overlay_ray_groups(ax, ray_groups)
    ax.set_title(title, color="black")
    ax.set_xlabel("模型距离 (km)", color="black")
    ax.set_ylabel("深度 (km)", color="black")
    ax.tick_params(colors="black")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmax, zmin)
    finish_figure_layout(fig)
    return fig


def drop_local_paths(event) -> list[Path]:
    """从拖放事件取出本地路径（含 WSL UNC / file://wsl.localhost）。"""
    md = event.mimeData() if event is not None else None
    if md is None:
        return []
    raw: list[str] = []
    if md.hasUrls():
        for url in md.urls():
            local = ""
            try:
                local = url.toLocalFile() or ""
            except Exception:
                local = ""
            if local:
                raw.append(local)
                continue
            try:
                s = url.toString()
            except Exception:
                s = ""
            if s:
                raw.append(s)
    if not raw:
        try:
            if md.hasFormat("text/uri-list"):
                blob = bytes(md.data("text/uri-list")).decode("utf-8", errors="replace")
                for line in blob.splitlines():
                    t = line.strip()
                    if t and not t.startswith("#"):
                        raw.append(t)
        except Exception:
            pass
    if not raw and md.hasText():
        text = md.text().replace("file:///", "").replace("file://", "")
        for line in text.splitlines():
            t = line.strip().strip('"')
            if t:
                raw.append(t)
    out: list[Path] = []
    seen: set[str] = set()
    for item in raw:
        p = normalize_dropped_path(item)
        key = str(p)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def dropped_smesh_path(event) -> Path | None:
    """拖放中第一个像 smesh 的文件。"""
    for p in drop_local_paths(event):
        if looks_like_model_name(p):
            return p
    return None


def classify_dropped_plot_files(
    event, *, loose_iface: bool = False
) -> tuple[Path | None, list[Path]]:
    """拆出 smesh 与界面文件。"""
    return classify_plot_paths(drop_local_paths(event), loose_iface=loose_iface)


def peel_dws_paths(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """把文件名含 dws 的项拆出，避免被当成界面文本。"""
    from ..services.dws_plot import looks_like_dws_name

    dws: list[Path] = []
    rest: list[Path] = []
    for p in paths:
        if looks_like_dws_name(p):
            dws.append(p)
        else:
            rest.append(p)
    return dws, rest


def peel_ray_paths(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """把射线文件 / rays 目录拆出，避免被当成界面文本。"""
    from ..services.ray_sample import looks_like_ray_name

    rays: list[Path] = []
    rest: list[Path] = []
    for p in paths:
        if looks_like_ray_name(p):
            rays.append(p)
        else:
            rest.append(p)
    return rays, rest


def classify_plot_paths(
    paths: list[Path], *, loose_iface: bool = False
) -> tuple[Path | None, list[Path]]:
    smesh: Path | None = None
    ifaces: list[Path] = []
    for p in paths:
        if looks_like_model_name(p):
            if smesh is None:
                smesh = p
        elif looks_like_interface_name(p) or loose_iface:
            ifaces.append(p)
    return smesh, ifaces


def clipboard_plot_paths() -> list[Path]:
    """Qt 剪贴板 +（WSL 下）资源管理器 Ctrl+C 的文件列表。"""
    from PySide6.QtWidgets import QApplication

    class _Md:
        def __init__(self, md) -> None:
            self._md = md

        def mimeData(self):
            return self._md

    paths: list[Path] = []
    app = QApplication.instance()
    if app is not None:
        clip = app.clipboard()
        try:
            paths = drop_local_paths(_Md(clip.mimeData()))
        except Exception:
            paths = []
        if not paths:
            try:
                paths = parse_clipboard_path_text(clip.text() or "")
            except Exception:
                paths = []
    if not paths:
        paths = read_windows_explorer_clipboard_paths()
    return paths


def _drop_mime_maybe_files(event) -> bool:
    md = event.mimeData() if event is not None else None
    if md is None:
        return False
    try:
        return md.hasUrls() or md.hasText() or md.hasFormat("text/uri-list")
    except Exception:
        return False


class SmeshPlotWindow(QWidget):
    """非模态图窗：打开… 或把 smesh / 界面文件拖进来。"""

    def __init__(self, state: FormState) -> None:
        super().__init__(None)
        self.state = state
        self.setWindowTitle("smesh 速度模型")
        self.resize(960, 640)
        self.setAcceptDrops(True)
        self._path: Path | None = None
        self._browse_start: str = ""
        self._dws_abs: Path | None = None
        self._ray_hint: Path | None = None
        self._overlay_ifaces: list[Path] | None = None  # None=自动配套/表单 -F
        self._fig = None
        self._host: MplNavCanvas | None = None

        self._lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        btn_open = QPushButton("打开…")
        btn_open.setToolTip("浏览 smesh / v.in / .grd；也可把文件拖到本窗口或主窗口")
        btn_open.clicked.connect(self._browse)
        btn_reload = QPushButton("刷新")
        btn_reload.setToolTip("从磁盘重新读取当前 smesh（修改文件后点此更新图）")
        btn_reload.clicked.connect(self._redraw_keeping_view)
        btn_paste = QPushButton("粘贴")
        btn_paste.setToolTip(
            "资源管理器里复制文件（Ctrl+C）或「复制为路径」，再到本窗粘贴（Ctrl+V）"
        )
        btn_paste.clicked.connect(self._paste_from_clipboard)
        btn_iface = QPushButton("叠加界面…")
        btn_iface.setToolTip("浏览或拖入反射/界面文件（每行 x z）；可叠多条")
        btn_iface.clicked.connect(self._browse_iface)
        btn_clear = QPushButton("清除界面")
        btn_clear.setToolTip("去掉叠加的界面线（地形仍画）")
        btn_clear.clicked.connect(self._clear_ifaces)
        self.ck_contours = QCheckBox("叠加等值线")
        from ..plots.velocity_contours import contours_enabled

        self.ck_contours.setChecked(contours_enabled(self.state))
        self.ck_contours.setToolTip(
            "勾选：按当前色标叠对应等值线（vp / vs / vpvs；A 粗实线+标注，C 细虚线）。"
            "不勾选则只画色块、地形与界面。"
        )
        self.ck_contours.toggled.connect(self._on_contours_toggled)
        from ..widgets.smesh_cmap_combo import SmeshCmapCombo

        self.cmap_combo = SmeshCmapCombo(
            self.state, on_changed=lambda *_a: self._redraw_keeping_view()
        )
        self.ck_dws = QCheckBox("DWS 遮罩")
        from ..services.dws_plot import dws_mask_enabled, get_explicit_dws_path

        try:
            work0 = resolve_work_dir(self.state.get_str("work_dir"))
        except Exception:
            work0 = Path.cwd()
        self._dws_abs = get_explicit_dws_path(self.state, work0)
        has_dws = self._dws_abs is not None
        self.ck_dws.setChecked(bool(dws_mask_enabled(self.state) and has_dws))
        self.ck_dws.setToolTip(
            "勾选：用指定的 DWS 文件做透明遮罩。"
            "无覆盖（DWS≤0）留白；有覆盖按 log(DWS) 越大越实、越小越淡。"
            "本窗不自动查找，请点「指定 DWS…」或拖入/粘贴。"
            "水层与地形线仍实色。"
        )
        self.ck_dws.toggled.connect(self._on_dws_toggled)
        btn_dws = QPushButton("指定 DWS…")
        btn_dws.setToolTip(
            "选择 tt_inverse -K 写出的覆盖权重文件（每行 x z DWS）。"
            "文件名不限，不必叫 dws.dat。"
        )
        btn_dws.clicked.connect(self._browse_dws)
        btn_dws_clear = QPushButton("清除 DWS")
        btn_dws_clear.setToolTip("去掉已指定的 DWS，恢复整幅速度着色")
        btn_dws_clear.clicked.connect(self._clear_dws)
        self.lbl_dws = QLabel("")
        self.lbl_dws.setStyleSheet("color:#555;font-size:11px;")
        self.lbl_dws.setMaximumWidth(180)
        self._refresh_dws_label()
        self.ck_rays = QCheckBox("叠加射线")
        from ..services.ray_sample import (
            get_explicit_ray_hint,
            rays_overlay_enabled,
        )

        self._ray_hint = get_explicit_ray_hint(self.state, work0)
        has_ray = self._ray_hint is not None
        self.ck_rays.setChecked(bool(rays_overlay_enabled(self.state) and has_ray))
        self.ck_rays.setToolTip(
            "勾选：在速度场上叠加抽样射线（与监视窗同套：按 OBS/炮着色）。"
            "本窗不自动查找，请点「指定射线…」或拖入/粘贴任一 "
            "stem.ray.<iter>.<isrc>、正演 .ray，或 rays 目录。"
            "反演射线需 out_level(-o)≥2。"
        )
        self.ck_rays.toggled.connect(self._on_rays_toggled)
        btn_ray = QPushButton("指定射线…")
        btn_ray.setToolTip(
            "选任一 .ray.<iter>.<isrc> 即可加载同套抽样射线；"
            "也可选正演 tt_forward -R 的单文件。"
        )
        btn_ray.clicked.connect(self._browse_rays)
        btn_ray_clear = QPushButton("清除射线")
        btn_ray_clear.setToolTip("去掉已指定的射线叠加")
        btn_ray_clear.clicked.connect(self._clear_rays)
        self.lbl_ray = QLabel("")
        self.lbl_ray.setStyleSheet("color:#555;font-size:11px;")
        self.lbl_ray.setMaximumWidth(180)
        self._refresh_ray_label()
        btn_save = QPushButton("保存图像…")
        btn_save.clicked.connect(self._save)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        bar.addWidget(btn_open)
        bar.addWidget(btn_reload)
        bar.addWidget(btn_paste)
        bar.addWidget(btn_iface)
        bar.addWidget(btn_clear)
        bar.addWidget(self.ck_contours)
        bar.addWidget(self.cmap_combo)
        bar.addWidget(self.ck_dws)
        bar.addWidget(btn_dws)
        bar.addWidget(btn_dws_clear)
        bar.addWidget(self.lbl_dws)
        bar.addWidget(self.ck_rays)
        bar.addWidget(btn_ray)
        bar.addWidget(btn_ray_clear)
        bar.addWidget(self.lbl_ray)
        bar.addStretch(1)
        bar.addWidget(btn_save)
        bar.addWidget(btn_close)
        self._lay.addLayout(bar)
        hint = (
            "拖入 .smesh / v.in / .grd 绘图；拖入界面/refl 叠加；拖入 dws.dat 指定遮罩；"
            "拖入 .ray 或 rays 目录叠加抽样射线 · "
            "主按钮按当前命令页签上的 smesh / 界面字段从磁盘重读（没有则打开空图窗） · "
            "已打开的图可点「刷新」 · "
            "资源管理器复制文件后到本窗 Ctrl+V /「粘贴」 · "
            "DWS 遮罩须手动指定文件（不自动查找） · "
            "叠加射线须指定 .ray / rays 目录（与监视窗同样抽样着色，不自动查找） · "
            "滚轮缩放 · 左拖平移 · 右拖缩放 · 双击复位 · "
            "右键「写入表单」把当前 smesh / 反射面写回参数；"
            "「添加到对比模型」：第一图为 A，第二图为 B，自动画 B−A；"
            "「打开所在目录」打开当前文件所在文件夹"
        )
        if running_in_wsl():
            hint += (
                " · WSLg 不能从 Windows 资源管理器拖入，请用复制+粘贴或「打开…」"
            )
        self._hint = QLabel(hint)
        self._hint.setStyleSheet("color:#666;font-size:11px;")
        self._hint.setWordWrap(True)
        self._lay.addWidget(self._hint)
        self._empty = QLabel(
            "将 smesh、v.in 或 .grd/.nc 拖到这里\n"
            "或点「打开…」/「粘贴」（资源管理器 Ctrl+C 后 Ctrl+V）"
        )
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setStyleSheet("color:#667;font-size:16px;")
        self._lay.addWidget(self._empty, stretch=1)
        self._wire_drop_targets()
        paste_sc = QShortcut(QKeySequence.StandardKey.Paste, self)
        paste_sc.setContext(Qt.ShortcutContext.WindowShortcut)
        paste_sc.activated.connect(self._paste_from_clipboard)
        self._paste_sc = paste_sc

    def _wire_drop_targets(self) -> None:
        """子控件（尤其 matplotlib canvas）会把拖放拦掉，需转发到本窗。"""
        self.setAcceptDrops(True)
        for w in self.findChildren(QWidget):
            if w is self:
                continue
            w.setAcceptDrops(True)
            w.installEventFilter(self)

    def eventFilter(self, watched, event):  # noqa: N802
        t = event.type()
        if t == QEvent.Type.DragEnter:
            self.dragEnterEvent(event)
            return True
        if t == QEvent.Type.DragMove:
            self.dragMoveEvent(event)
            return True
        if t == QEvent.Type.Drop:
            self.dropEvent(event)
            return True
        if t == QEvent.Type.KeyPress and event.matches(QKeySequence.StandardKey.Paste):
            self._paste_from_clipboard()
            return True
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if drop_local_paths(event) or _drop_mime_maybe_files(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802
        if drop_local_paths(event) or _drop_mime_maybe_files(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        dws, rest = peel_dws_paths(drop_local_paths(event))
        rays, rest = peel_ray_paths(rest)
        smesh, ifaces = classify_plot_paths(rest, loose_iface=True)
        if smesh is None and not ifaces and not dws and not rays:
            event.ignore()
            return
        event.acceptProposedAction()
        self._apply_plot_files(smesh, ifaces, dws=dws, rays=rays)

    def _apply_plot_files(
        self,
        smesh: Path | None,
        ifaces: list[Path],
        dws: list[Path] | None = None,
        rays: list[Path] | None = None,
    ) -> None:
        if dws:
            self._set_dws_file(dws[0], redraw=smesh is None and not ifaces and not rays)
        if rays:
            self._set_ray_hint(
                rays[0],
                redraw=smesh is None and not ifaces,
            )
        if smesh is not None:
            self.load_smesh(smesh, overlay_ifaces=ifaces or None)
        elif ifaces:
            self.add_interfaces(ifaces)

    def _work_dir(self) -> Path:
        try:
            return resolve_work_dir(self.state.get_str("work_dir"))
        except Exception:
            return Path.cwd()

    def _refresh_dws_label(self) -> None:
        from ..services.dws_plot import DWS_FILE_PREF_KEY

        s = str(self.state.get_str(DWS_FILE_PREF_KEY) or "").strip()
        if s:
            self.lbl_dws.setText(Path(s).name)
            self.lbl_dws.setToolTip(s)
        else:
            self.lbl_dws.setText("未指定 DWS")
            self.lbl_dws.setToolTip(
                "本窗须手动指定 DWS 文件，不会自动查找运行包。"
            )

    def _refresh_ray_label(self) -> None:
        from ..services.ray_sample import RAYS_ROOT_PREF_KEY

        s = str(self.state.get_str(RAYS_ROOT_PREF_KEY) or "").strip()
        if s:
            self.lbl_ray.setText(Path(s).name)
            self.lbl_ray.setToolTip(s)
        else:
            self.lbl_ray.setText("未指定射线")
            self.lbl_ray.setToolTip(
                "本窗须手动指定 .ray 或 rays 目录，不会自动查找运行包。"
            )

    def _redraw_keeping_view(self) -> None:
        if self._path is None:
            return
        views = None
        home = None
        if self._host is not None:
            views = self._host._nav.capture_views()
            home = self._host._nav._home_views
        elif self._fig is not None and self._fig.axes:
            ax0 = self._fig.axes[0]
            views = [(tuple(ax0.get_xlim()), tuple(ax0.get_ylim()))]
        self.load_smesh(self._path, keep_explicit=True)
        if not views or self._fig is None or self._host is None:
            return
        nav = self._host._nav
        ax0 = self._fig.axes[0]
        ax0.set_xlim(views[0][0])
        ax0.set_ylim(views[0][1])
        if home:
            nav._home_views = [home[0]]
        nav.remember_current_views()
        self._host.canvas.draw_idle()

    def _on_contours_toggled(self, on: bool) -> None:
        from ..plots.velocity_contours import set_contours_enabled

        set_contours_enabled(self.state, bool(on))
        self._redraw_keeping_view()

    def _on_dws_toggled(self, on: bool) -> None:
        from ..services.dws_plot import (
            get_explicit_dws_path,
            set_dws_mask_enabled,
        )

        set_dws_mask_enabled(self.state, bool(on))
        if on and self._dws_abs is None and get_explicit_dws_path(
            self.state, self._work_dir()
        ) is None:
            picked = self._browse_dws_path()
            if picked is None:
                self.ck_dws.blockSignals(True)
                self.ck_dws.setChecked(False)
                self.ck_dws.blockSignals(False)
                return
            self._set_dws_file(picked, redraw=False)
        self._redraw_keeping_view()

    def _dws_browse_start(self) -> str:
        work = self._work_dir()
        from ..services.dws_plot import get_explicit_dws_path

        hit = get_explicit_dws_path(self.state, work)
        if hit is not None:
            return str(hit.parent)
        if self._path is not None:
            for parent in (
                self._path.parent,
                self._path.parent.parent,
                self._path.parent.parent.parent,
            ):
                ddir = parent / "dws"
                if ddir.is_dir():
                    return str(ddir)
                if (parent / "dws.dat").is_file():
                    return str(parent)
        return str(work)

    def _browse_dws_path(self) -> Path | None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 DWS 文件",
            self._dws_browse_start(),
            DWS_OPEN_FILTERS,
            options=file_dialog_options(),
        )
        if not path:
            return None
        from ..services.dws_plot import resolve_readable_dws_path
        from ..services.smesh_plot_core import normalize_dropped_path

        hit = resolve_readable_dws_path(path)
        if hit is not None:
            return hit
        return normalize_dropped_path(path)

    def _browse_dws(self) -> None:
        picked = self._browse_dws_path()
        if picked is None:
            return
        self._set_dws_file(picked)

    def _set_dws_file(self, path: Path, *, redraw: bool = True) -> None:
        from ..services.dws_plot import (
            resolve_readable_dws_path,
            set_dws_mask_enabled,
            set_explicit_dws_path,
        )

        set_explicit_dws_path(self.state, path, self._work_dir())
        hit = resolve_readable_dws_path(path)
        try:
            self._dws_abs = hit if hit is not None else Path(path).resolve()
        except OSError:
            self._dws_abs = hit if hit is not None else Path(path)
        self._refresh_dws_label()
        if not self.ck_dws.isChecked():
            self.ck_dws.blockSignals(True)
            self.ck_dws.setChecked(True)
            self.ck_dws.blockSignals(False)
            set_dws_mask_enabled(self.state, True)
        if redraw:
            self._redraw_keeping_view()

    def _clear_dws(self) -> None:
        from ..services.dws_plot import set_dws_mask_enabled, set_explicit_dws_path

        set_explicit_dws_path(self.state, None)
        self._dws_abs = None
        self._refresh_dws_label()
        if self.ck_dws.isChecked():
            self.ck_dws.blockSignals(True)
            self.ck_dws.setChecked(False)
            self.ck_dws.blockSignals(False)
            set_dws_mask_enabled(self.state, False)
        self._redraw_keeping_view()

    def _on_rays_toggled(self, on: bool) -> None:
        from ..services.ray_sample import (
            get_explicit_ray_hint,
            set_rays_overlay_enabled,
        )

        set_rays_overlay_enabled(self.state, bool(on))
        if on and self._ray_hint is None and get_explicit_ray_hint(
            self.state, self._work_dir()
        ) is None:
            picked = self._browse_ray_path()
            if picked is None:
                self.ck_rays.blockSignals(True)
                self.ck_rays.setChecked(False)
                self.ck_rays.blockSignals(False)
                return
            self._set_ray_hint(picked, redraw=False)
        self._redraw_keeping_view()

    def _ray_browse_start(self) -> str:
        work = self._work_dir()
        from ..services.ray_sample import get_explicit_ray_hint

        hit = get_explicit_ray_hint(self.state, work)
        if hit is not None:
            return str(hit.parent if hit.is_file() else hit)
        if self._path is not None:
            for parent in (
                self._path.parent,
                self._path.parent.parent,
                self._path.parent.parent.parent,
            ):
                rdir = parent / "rays"
                if rdir.is_dir():
                    return str(rdir)
        return str(work)

    def _browse_ray_path(self) -> Path | None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择射线文件",
            self._ray_browse_start(),
            RAY_OPEN_FILTERS,
            options=file_dialog_options(),
        )
        if not path:
            return None
        from ..services.ray_sample import resolve_ray_hint_path
        from ..services.smesh_plot_core import normalize_dropped_path

        hit = resolve_ray_hint_path(path)
        if hit is not None:
            return hit
        return normalize_dropped_path(path)

    def _browse_rays(self) -> None:
        picked = self._browse_ray_path()
        if picked is None:
            return
        self._set_ray_hint(picked)

    def _set_ray_hint(self, path: Path, *, redraw: bool = True) -> None:
        from ..services.ray_sample import (
            resolve_ray_hint_path,
            set_explicit_ray_hint,
            set_rays_overlay_enabled,
        )

        set_explicit_ray_hint(self.state, path, self._work_dir())
        hit = resolve_ray_hint_path(path)
        try:
            self._ray_hint = hit if hit is not None else Path(path).resolve()
        except OSError:
            self._ray_hint = hit if hit is not None else Path(path)
        self._refresh_ray_label()
        if not self.ck_rays.isChecked():
            self.ck_rays.blockSignals(True)
            self.ck_rays.setChecked(True)
            self.ck_rays.blockSignals(False)
            set_rays_overlay_enabled(self.state, True)
        if redraw:
            self._redraw_keeping_view()

    def _clear_rays(self) -> None:
        from ..services.ray_sample import (
            set_explicit_ray_hint,
            set_rays_overlay_enabled,
        )

        set_explicit_ray_hint(self.state, None)
        self._ray_hint = None
        self._refresh_ray_label()
        if self.ck_rays.isChecked():
            self.ck_rays.blockSignals(True)
            self.ck_rays.setChecked(False)
            self.ck_rays.blockSignals(False)
            set_rays_overlay_enabled(self.state, False)
        self._redraw_keeping_view()

    def _paste_from_clipboard(self) -> None:
        dws, rest = peel_dws_paths(clipboard_plot_paths())
        rays, rest = peel_ray_paths(rest)
        smesh, ifaces = classify_plot_paths(rest, loose_iface=True)
        if smesh is None and not ifaces and not dws and not rays:
            show_modeless_message(
                "粘贴",
                "剪贴板里没有 smesh / 界面 / DWS / 射线文件。\n"
                "请在 Windows 资源管理器中选中文件后 Ctrl+C，"
                "或 Shift+右键「复制为路径」，再回到本窗 Ctrl+V 或点「粘贴」。",
                icon=QMessageBox.Icon.Information,
            )
            return
        self._apply_plot_files(smesh, ifaces, dws=dws, rays=rays)

    def _browse(self) -> None:
        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
        except Exception:
            work = Path.cwd()
        if self._path is not None:
            start = str(self._path.parent)
        elif self._browse_start:
            start = self._browse_start
        else:
            start = str(work)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型（smesh / v.in / grd）",
            start,
            MODEL_OPEN_FILTERS,
            options=file_dialog_options(),
        )
        if path:
            self.load_smesh(Path(path))

    def _browse_iface(self) -> None:
        start = str(self._path.parent) if self._path else ""
        if not start:
            try:
                start = str(resolve_work_dir(self.state.get_str("work_dir")))
            except Exception:
                start = ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择界面文件（每行 x z）",
            start,
            REFL_OPEN_FILTERS,
            options=file_dialog_options(),
        )
        if path:
            self.add_interfaces([Path(path)])

    def _clear_ifaces(self) -> None:
        self._overlay_ifaces = []
        if self._path is not None:
            self.load_smesh(self._path, overlay_ifaces=[], keep_explicit=True)

    def add_interfaces(self, paths: list[Path]) -> None:
        files = [p for p in paths if p.is_file()]
        if not files:
            return
        if self._overlay_ifaces is None:
            self._overlay_ifaces = []
        for p in files:
            rp = p.resolve()
            if rp not in self._overlay_ifaces:
                self._overlay_ifaces.append(rp)
        if self._path is None:
            show_modeless_message(
                "叠加界面",
                "已记下界面文件，请再拖入或打开 smesh。\n"
                + "\n".join(p.name for p in files),
                icon=QMessageBox.Icon.Information,
            )
            return
        self.load_smesh(
            self._path, overlay_ifaces=self._overlay_ifaces, keep_explicit=True
        )

    def load_smesh(
        self,
        path: str | Path,
        *,
        overlay_ifaces: list[Path] | None = None,
        keep_explicit: bool = False,
    ) -> None:
        p = Path(path)
        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
        except Exception as e:
            show_modeless_message("绘制 smesh", str(e), icon=QMessageBox.Icon.Warning)
            return
        if overlay_ifaces is not None:
            self._overlay_ifaces = [Path(x).resolve() for x in overlay_ifaces]
        elif not keep_explicit:
            self._overlay_ifaces = None
        try:
            is_smesh = looks_like_smesh_name(p)
            if self._overlay_ifaces is None:
                auto = (
                    resolve_plot_refl_for_smesh(p, self.state, work)
                    if is_smesh
                    else None
                )
                iface_paths = [Path(auto)] if auto else []
            else:
                iface_paths = list(self._overlay_ifaces)
            mesh, ds, extra_model = load_model_plot_data(p, None)
            extra = list(extra_model or [])
            extra.extend(load_interface_overlays(iface_paths))

            def _warn_plot(msg: str) -> None:
                show_modeless_message("smesh 绘图", msg, icon=QMessageBox.Icon.Warning)

            cmap_use = resolve_plot_smesh_cmap(
                self.state, work, on_missing_cpt=_warn_plot
            )
            from ..plots.velocity_contours import contour_specs_for_state
            from ..services.dws_plot import read_dws_xyz
            from ..services.ray_sample import load_rays_for_smesh_plot
            from ..services.smesh_ops import parse_inverse_smesh_name

            dws_xyz = None
            dws_path = None
            dws_err = ""
            if self.ck_dws.isChecked():
                dws_path = self._dws_abs
                if dws_path is None:
                    from ..services.dws_plot import get_explicit_dws_path

                    dws_path = get_explicit_dws_path(self.state, work)
                    self._dws_abs = dws_path
                dws_xyz, dws_err = read_dws_xyz(dws_path)
                if dws_xyz is None:
                    show_modeless_message(
                        "DWS 遮罩",
                        (dws_err or "读不到 DWS 文件。")
                        + "\n\n「指定 DWS…」可选任意文件名，不必叫 dws.dat；"
                        "内容须为 tt_inverse -K 的每行 x z 覆盖权重。",
                        icon=QMessageBox.Icon.Warning,
                    )
            ray_groups = None
            ray_note = ""
            if self.ck_rays.isChecked() and is_smesh:
                ray_hint = self._ray_hint
                if ray_hint is None:
                    from ..services.ray_sample import get_explicit_ray_hint

                    ray_hint = get_explicit_ray_hint(self.state, work)
                    self._ray_hint = ray_hint
                smesh_key = parse_inverse_smesh_name(p)
                ray_iter = smesh_key[0] if smesh_key else None
                ray_groups, ray_note = load_rays_for_smesh_plot(
                    ray_hint, iter_prefer=ray_iter
                )
                if not ray_groups:
                    show_modeless_message(
                        "叠加射线",
                        (ray_note or "读不到射线。")
                        + "\n\n请指定任一 stem.ray.<iter>.<isrc>（同套会抽样），"
                        "或正演 -R 单文件；反演需 out_level(-o)≥2。",
                        icon=QMessageBox.Icon.Warning,
                    )
                    ray_groups = None
            title = p.name
            if dws_xyz is not None and dws_path is not None:
                title = f"{title}  ·  DWS {Path(dws_path).name}"
            if ray_note and ray_groups:
                title = f"{title}  ·  {ray_note}"
            fig = draw_smesh_velocity_figure(
                ds,
                mesh,
                extra,
                cmap_use,
                title,
                contour_specs=contour_specs_for_state(self.state),
                dws_xyz=dws_xyz,
                ray_groups=ray_groups,
            )
        except Exception as e:
            show_modeless_message(
                "绘制 smesh 失败", str(e), icon=QMessageBox.Icon.Critical
            )
            return
        self._set_figure(fig, p, iface_paths)

    def _set_figure(
        self, fig, path: Path, iface_paths: list[Path] | None = None
    ) -> None:
        old = self._fig
        if self._host is not None:
            self._lay.removeWidget(self._host)
            self._host.deleteLater()
            self._host = None
        if self._empty is not None:
            self._empty.hide()
        self._fig = fig
        self._host = MplNavCanvas(fig, self, on_right_click=self._on_model_right_click)
        self._lay.addWidget(self._host, stretch=1)
        self._wire_drop_targets()
        self._path = path
        title = f"smesh 速度模型 — {path.name}"
        if self.ck_dws.isChecked() and self._dws_abs is not None:
            title = f"{title}  ·  DWS {self._dws_abs.name}"
        if self.ck_rays.isChecked() and self._ray_hint is not None:
            title = f"{title}  ·  射线 {self._ray_hint.name}"
        if iface_paths:
            title = f"{title}  ·  " + "、".join(p.name for p in iface_paths[:3])
            if len(iface_paths) > 3:
                title = f"{title}…"
        self.setWindowTitle(title)
        if old is not None and old is not fig:
            try:
                import matplotlib.pyplot as plt

                plt.close(old)
            except Exception:
                pass

    def _on_model_right_click(self, _event) -> None:
        from ..services.model_compare import popup_model_context_menu

        iface = None
        if self._overlay_ifaces:
            iface = self._overlay_ifaces[0]
        popup_model_context_menu(
            self, path=self._path, state=self.state, refl_path=iface
        )

    def _save(self) -> None:
        if self._fig is None:
            show_modeless_message(
                "保存图像", "请先打开或拖入模型", icon=QMessageBox.Icon.Warning
            )
            return
        from ..plots.export_figure import save_mpl_figure

        save_mpl_figure(
            self,
            self._fig,
            start_dir=str(self._path.parent) if self._path else "",
            default_name="smesh_velocity.png",
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        global _smesh_win
        if _smesh_win is self:
            _smesh_win = None
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._fig)
            except Exception:
                pass
        super().closeEvent(event)


_smesh_win: SmeshPlotWindow | None = None


def plot_smesh_velocity_qt(
    parent: QWidget | None,
    state: FormState,
    *,
    path: str | Path | None = None,
    overlay_ifaces: list[Path] | None = None,
    skip_guess: bool = False,
    browse_start: str | Path | None = None,
) -> SmeshPlotWindow | None:
    """打开（或前置）smesh 图窗；``path`` 有则直接绘制。

    主按钮应传入当前页签解析出的 ``path`` / ``overlay_ifaces``；
    找不到时 ``skip_guess=True``，打开空图窗，不要弹出选文件框，也不要跨页签猜 ``model.smesh``。
    **每次从磁盘重读**（单例窗不再沿用上次内存里的图）。
    """
    global _smesh_win
    try:
        work = resolve_work_dir(state.get_str("work_dir"))
        if not work.exists():
            raise ValueError(f"work_dir 不存在: {work}")
    except Exception as e:
        show_modeless_message("绘制 smesh", str(e), icon=QMessageBox.Icon.Warning)
        return None

    win = _smesh_win
    if win is None:
        win = SmeshPlotWindow(state)
        _smesh_win = win
        show_modeless_dialog(win, activate=True)
    else:
        win.state = state
        win.setAcceptDrops(True)
        win._wire_drop_targets()
        win.show()
        win.raise_()
        win.activateWindow()
    if browse_start:
        win._browse_start = str(browse_start)

    target = Path(path) if path else None

    same = False
    if target is not None and win._path is not None:
        try:
            same = Path(target).resolve() == win._path.resolve()
        except OSError:
            same = False

    if overlay_ifaces is not None:
        if target is None and win._path is not None:
            win.add_interfaces(overlay_ifaces)
        elif target is not None:
            win.load_smesh(target, overlay_ifaces=overlay_ifaces)
        else:
            win.add_interfaces(overlay_ifaces)
    elif target is not None:
        if same:
            win._redraw_keeping_view()
        else:
            win.load_smesh(target)
    elif not skip_guess and win._path is not None:
        win._redraw_keeping_view()

    return win
