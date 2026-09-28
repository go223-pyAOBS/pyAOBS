"""速度图窗：把当前模型加入 A/B 对比；凑齐两个不同 smesh 后自动画 B−A。"""

from __future__ import annotations

import os
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PySide6.QtWidgets import QMenu, QWidget

from ..dialog_utils import open_containing_directory, show_modeless_message
from ..state.form_state import FormState
from .paths import resolve_work_dir
from .result_nav import (
    compute_smesh_velocity_diff,
    diff_vlim_half_range,
    diff_vlim_is_auto,
    format_diff_vlim_caption,
    format_sigma_vlim_caption,
    resample_vgrid_field_to_xarray,
    resample_vgrid_fields_to_xarray,
    resolve_diff_colorbar_limits,
    resolve_sigma_cmap_and_limits,
    sigma_data_max,
)
from .smesh_plot_core import (
    load_smesh_plot_data,
    looks_like_smesh_name,
    resolve_plot_refl_for_smesh,
    resolve_plot_smesh_cmap,
)


def _same_file(a: Path, b: Path) -> bool:
    """路径是否同一文件。用 abspath，避免 Path.resolve() 在 Windows 上反复打盘。"""
    try:
        sa, sb = os.fspath(a), os.fspath(b)
        if sa == sb:
            return True
        return os.path.normcase(os.path.abspath(sa)) == os.path.normcase(
            os.path.abspath(sb)
        )
    except OSError:
        return a == b


def _as_existing_file(raw: Path | str) -> Path | None:
    q = Path(raw)
    try:
        q = Path(os.path.normpath(os.path.abspath(os.fspath(q.expanduser()))))
    except OSError:
        pass
    return q if q.is_file() else None


@dataclass
class CompareTray:
    path_a: Path | None = None
    path_b: Path | None = None
    ensemble: list[Path] = field(default_factory=list)

    def clear(self) -> None:
        self.path_a = None
        self.path_b = None
        self.ensemble = []

    def _append_ensemble(self, path: Path) -> None:
        if any(_same_file(path, e) for e in self.ensemble):
            return
        self.ensemble.append(path)

    def set_paths(
        self, path_a: Path | str | None, path_b: Path | str | None
    ) -> None:
        def _one(p: Path | str | None) -> Path | None:
            if p is None or not str(p).strip():
                return None
            return _as_existing_file(p)

        self.path_a = _one(path_a)
        self.path_b = _one(path_b)
        for p in (self.path_a, self.path_b):
            if p is not None:
                self._append_ensemble(p)

    def set_ensemble(self, paths: list[Path | str]) -> None:
        seen: list[Path] = []
        for raw in paths:
            q = _as_existing_file(raw)
            if q is not None and not any(_same_file(q, e) for e in seen):
                seen.append(q)
        self.ensemble = seen
        if self.path_a is not None and not any(
            _same_file(self.path_a, e) for e in seen
        ):
            self.path_a = None
        if self.path_b is not None and not any(
            _same_file(self.path_b, e) for e in seen
        ):
            self.path_b = None
        if self.path_a is None and len(seen) >= 1:
            self.path_a = seen[0]
        if self.path_b is None and len(seen) >= 2:
            self.path_b = seen[1]

    def status(self) -> str:
        n = len(self.ensemble)
        ens = f"集合 {n} 个" if n else "集合空"
        if self.path_a is None:
            return f"{ens}  ·  差值：在速度图上右键「添加到对比模型」记为 A"
        if self.path_b is None:
            return f"{ens}  ·  差值 A: {self.path_a.name}  ·  再右键添加 B"
        return (
            f"{ens}  ·  差值 B−A: {self.path_b.name} − {self.path_a.name}"
        )

    def extend_ensemble(self, paths: list[Path | str]) -> int:
        """追加到集合；A/B 若空则用集合前两个不同文件填。返回新加入个数。"""
        n0 = len(self.ensemble)
        for raw in paths:
            q = _as_existing_file(raw)
            if q is not None:
                self._append_ensemble(q)
        if self.path_a is None and self.ensemble:
            self.path_a = self.ensemble[0]
        if self.path_b is None:
            for p in self.ensemble:
                if self.path_a is None or not _same_file(p, self.path_a):
                    self.path_b = p
                    break
        return len(self.ensemble) - n0

    def add(self, path: Path | str) -> tuple[str, Path | None, Path | None]:
        """
        记入 A 或 B。

        返回 ``(状态说明, plot_a, plot_b)``：仅当应立即绘制差值时 ``plot_a/b`` 非空。
        """
        p = Path(path)
        try:
            p = Path(os.path.normpath(os.path.abspath(os.fspath(p.expanduser()))))
        except OSError:
            p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"找不到模型: {p}")
        self._append_ensemble(p)

        if self.path_a is None:
            self.path_a = p
            return f"已记为对比 A：{p.name}。请在另一模型窗口右键再添加。", None, None

        if self.path_b is None:
            if _same_file(p, self.path_a):
                return "当前已是对比 A，请添加另一个不同的模型。", None, None
            self.path_b = p
            return (
                f"已凑齐，绘制 {self.path_b.name} − {self.path_a.name}",
                self.path_a,
                self.path_b,
            )

        if _same_file(p, self.path_a) or _same_file(p, self.path_b):
            return "该模型已在对比中。", None, None
        self.path_a = self.path_b
        self.path_b = p
        return (
            f"已更新对比，绘制 {self.path_b.name} − {self.path_a.name}",
            self.path_a,
            self.path_b,
        )


_TRAY = CompareTray()
_status_labels: list[weakref.ref] = []

# 与模型统计 / 挑选助手同一套写回目标
MESH_FORM_FIELDS = (
    ("inv.mesh", "tt_inverse 初始网格 (inv.mesh)"),
    ("fwd.smesh", "tt_forward 网格 (fwd.smesh)"),
    ("edit.smesh_file", "edit_smesh 文件 (edit.smesh_file)"),
    ("cb.bg_smesh", "棋盘格背景 (cb.bg_smesh)"),
    ("mc.base_mesh", "蒙特卡洛基础模型 (mc.base_mesh)"),
)
REFL_FORM_FIELDS = (
    ("inv.refl_file", "tt_inverse 反射面 (inv.refl_file)"),
    ("fwd.refl_file", "tt_forward 反射面 (fwd.refl_file)"),
    ("inv.seafloor_file", "tt_inverse 海底 (inv.seafloor_file, -Y)"),
    ("fwd.seafloor_file", "tt_forward 海底 (fwd.seafloor_file, -B)"),
)

_form_apply_hooks: list[weakref.WeakMethod] = []


def register_form_apply_hook(fn: Callable[[str, str], None]) -> None:
    """主窗注册：写回字段后刷新控件。后一次注册覆盖前一次。"""
    _form_apply_hooks.clear()
    try:
        _form_apply_hooks.append(weakref.WeakMethod(fn))
    except TypeError:
        pass


def _notify_form_applied(field_key: str, rel: str) -> None:
    alive: list[weakref.WeakMethod] = []
    for hook in _form_apply_hooks:
        fn = hook()
        if fn is None:
            continue
        alive.append(hook)
        try:
            fn(field_key, rel)
        except Exception:
            pass
    _form_apply_hooks[:] = alive


def write_path_to_form(state: FormState, field_key: str, path: Path | str) -> str:
    """把已有文件写成相对工区路径并写入 FormState，再通知主窗刷新。"""
    from .paths import to_workdir_relative

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"找不到文件: {p}")
    work = resolve_work_dir(state.get_str("work_dir"))
    rel = to_workdir_relative(str(p.resolve()), work).value
    state.set(field_key, rel)
    _notify_form_applied(field_key, rel)
    return rel


def _companion_refl_path(
    path: Path | str | None,
    state: FormState | None,
    explicit: Path | str | None = None,
) -> Path | None:
    if explicit is not None:
        p = Path(explicit)
        if p.is_file():
            return p
    if path is None or state is None:
        return None
    try:
        from .smesh_plot_core import resolve_plot_refl_for_smesh

        work = resolve_work_dir(state.get_str("work_dir"))
        hit = resolve_plot_refl_for_smesh(path, state, work)
        if hit:
            rp = Path(hit)
            if rp.is_file():
                return rp
    except Exception:
        pass
    return None


def add_writeback_submenus(
    menu: QMenu,
    state: FormState | None,
    mesh_path: Path | str | None,
    *,
    refl_path: Path | str | None = None,
) -> bool:
    """速度图右键：写入表单（速度 / 反射面）。无 smesh 时不加。"""
    if mesh_path is None:
        return False
    hit = Path(mesh_path)
    if not looks_like_smesh_name(hit):
        return False
    ok_mesh = hit.is_file() and state is not None
    sub_m = menu.addMenu("写入表单（速度）")
    sub_m.setEnabled(ok_mesh)
    for key, label in MESH_FORM_FIELDS:
        act = sub_m.addAction(label)
        act.setEnabled(ok_mesh)

        def _mesh(_c=False, k=key, p=hit) -> None:
            _writeback_clicked(state, k, p)

        act.triggered.connect(_mesh)
    refl = _companion_refl_path(hit, state, refl_path)
    ok_refl = refl is not None and refl.is_file() and state is not None
    sub_r = menu.addMenu("写入表单（反射面）")
    sub_r.setEnabled(ok_refl)
    for key, label in REFL_FORM_FIELDS:
        act = sub_r.addAction(label)
        act.setEnabled(ok_refl)
        if not ok_refl:
            continue

        def _refl(_c=False, k=key, p=refl) -> None:
            _writeback_clicked(state, k, p)

        act.triggered.connect(_refl)
    return True


def _writeback_clicked(
    state: FormState | None, field_key: str, path: Path | None
) -> None:
    if state is None or path is None:
        return
    try:
        rel = write_path_to_form(state, field_key, path)
    except Exception as e:
        show_modeless_message("写回失败", str(e))
        return
    show_modeless_message("已写回表单", f"{field_key} ← {rel}")


def compare_tray() -> CompareTray:
    return _TRAY


def compare_status() -> str:
    return _TRAY.status()


def clear_compare_tray() -> None:
    _TRAY.clear()
    _refresh_status_labels()


def _refresh_status_labels(note: str | None = None) -> None:
    text = note if note is not None else _TRAY.status()
    alive: list[weakref.ref] = []
    for ref in _status_labels:
        w = ref()
        if w is None:
            continue
        try:
            w.setText(text)
            w.setToolTip(_TRAY.status())
            alive.append(ref)
        except RuntimeError:
            pass
    _status_labels[:] = alive


def apply_smesh_diff(
    widget,
    state: FormState,
    path_a: Path | str,
    path_b: Path | str,
    *,
    mode: str = "abs",
    dws_xyz=None,
    auto_vlim: bool | None = None,
    half_range: float | None = None,
    reset_home: bool = True,
) -> Any:
    """把 B−A 差值铺到 ``MonitorModelWidget``（上 ΔV、中 B、下 A）。"""
    from ..plots.velocity_contours import contour_specs_for_state, contours_enabled

    pa, pb = Path(path_a), Path(path_b)
    work = resolve_work_dir(state.get_str("work_dir"))
    if dws_xyz is None:
        from .dws_plot import intersect_dws_xyz_for_smeshes

        dws_xyz, _n_dws = intersect_dws_xyz_for_smeshes(state, work, [pa, pb])
    diff = compute_smesh_velocity_diff(pa, pb, mode=str(mode))
    refl = resolve_plot_refl_for_smesh(pb, state, work)
    mesh_b, _ds, extra = load_smesh_plot_data(pb, refl, with_xarray=False)
    mesh_a, _, extra_a = load_smesh_plot_data(pa, refl, with_xarray=False)
    if dws_xyz is not None:
        from .smesh_ops import _dws_xyz_to_vgrid

        cov = _dws_xyz_to_vgrid(mesh_b, dws_xyz)
        if cov is not None:
            bad = ~np.isfinite(cov) | (cov <= 0.0)
            diff.dv = np.where(bad, np.nan, diff.dv)
            finite = diff.dv[np.isfinite(diff.dv)]
            if finite.size:
                peak = float(np.nanmax(np.abs(finite)))
                diff.vmin, diff.vmax = -peak, peak
                diff.mean = float(np.mean(finite))
                diff.std = float(np.std(finite))
    ds_diff = resample_vgrid_field_to_xarray(mesh_b, diff.dv)
    ds_b = mesh_b.to_xarray()
    ds_a = mesh_a.to_xarray()
    cmap = resolve_plot_smesh_cmap(state, work)
    # 开启时 specs 为 None（公用表）；不能用 bool(specs) 判断是否绘制。
    draw_contours = contours_enabled(state)
    contours = contour_specs_for_state(state)
    unit = "ΔV (%)" if str(mode) == "percent" else "ΔV (km/s)"
    if auto_vlim is None:
        auto_vlim = diff_vlim_is_auto(state)
    if half_range is None:
        half_range = diff_vlim_half_range(state, mode=str(mode))
    v_lo, v_hi = resolve_diff_colorbar_limits(
        diff.vmax, auto=bool(auto_vlim), half_range=float(half_range)
    )
    title = f"{pb.name} − {pa.name}"
    extra_b = extra if extra is not None else extra_a
    widget.set_save_dir(work)
    widget.set_velocity_stack(
        [
            {
                "ds": ds_diff,
                "mesh": mesh_b,
                "extra": extra_b,
                "cmap_spec": "seismic_r",
                "title": f"ΔV  {title}",
                "vlim": (v_lo, v_hi),
                "cb_label": unit,
                "draw_contours": False,
            },
            {
                "ds": ds_b,
                "mesh": mesh_b,
                "extra": extra_b,
                "cmap_spec": cmap,
                "title": f"B  {pb.name}",
                "cb_label": "km/s",
                "draw_contours": draw_contours,
                "contour_specs": contours,
            },
            {
                "ds": ds_a,
                "mesh": mesh_a,
                "extra": extra_b,
                "cmap_spec": cmap,
                "title": f"A  {pa.name}",
                "cb_label": "km/s",
                "draw_contours": draw_contours,
                "contour_specs": contours,
            },
        ],
        dws_xyz=dws_xyz,
        reset_home=reset_home,
    )
    extras = [
        f"mean={diff.mean:.4g}",
        f"std={diff.std:.4g}",
        f"|max|={diff.vmax:.4g}",
        format_diff_vlim_caption(diff.vmax, v_lo, v_hi, auto=bool(auto_vlim)),
        unit,
    ]
    widget.setWindowTitle(f"差值 · {title}")
    widget.set_model_source(pb, state, path_a=pa, path_b=pb)
    setattr(widget, "_pyaobs_diff_extras", extras)
    return diff


@dataclass
class EnsembleStatResult:
    n: int
    paths: list[Path]
    template_path: Path
    mean_v: Any
    std_v: Any
    refl_x: Any = None
    refl_mean: Any = None
    refl_std: Any = None
    n_refl: int = 0
    n_dws: int = 0
    mean_smesh: Path | None = None
    std_smesh: Path | None = None
    mean_refl: Path | None = None
    mesh: Any = None
    dws_xyz: Any = None
    sigma_vlim: tuple[float, float] | None = None


def _refl_overlay(stat: EnsembleStatResult) -> list[dict[str, Any]] | None:
    if stat.refl_x is None or stat.refl_mean is None:
        return None
    extra = [
        {
            "x": stat.refl_x,
            "z": stat.refl_mean,
            "label": "mean refl",
            "color": "#dc143c",
            "linewidth": 1.8,
            "linestyle": "-",
        }
    ]
    if stat.refl_std is not None:
        extra.append(
            {
                "x": stat.refl_x,
                "z": stat.refl_mean - stat.refl_std,
                "label": "mean−σ",
                "color": "#dc143c",
                "linewidth": 1.0,
                "linestyle": ":",
            }
        )
        extra.append(
            {
                "x": stat.refl_x,
                "z": stat.refl_mean + stat.refl_std,
                "label": "mean+σ",
                "color": "#dc143c",
                "linewidth": 1.0,
                "linestyle": ":",
            }
        )
    return extra


def apply_smesh_ensemble_stats(
    widget,
    state: FormState,
    paths: list[Path | str],
    *,
    auto_vlim: bool | None = None,
    half_range: float | None = None,
) -> EnsembleStatResult:
    """上均值 Vp、下误差 σ；均值图叠平均反射面 ±σ。"""
    from .smesh_ops import (
        companion_inverse_refl,
        stack_mean_std,
        stack_reflector_mean_std,
    )

    uniq: list[Path] = []
    for raw in paths:
        p = Path(raw)
        try:
            p = p.resolve()
        except OSError:
            pass
        if p.is_file() and not any(_same_file(p, e) for e in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        raise ValueError("统计至少需要 2 个不同的 smesh")
    work = resolve_work_dir(state.get_str("work_dir"))
    from .dws_plot import dws_xyz_for_plot, mean_dws_xyz_from_arrays

    # 有 DWS 文件则各点只平均有覆盖的成员（与勾选遮罩无关）
    dws_each = [
        dws_xyz_for_plot(state, work, p, enabled=True) for p in uniq
    ]
    template, mean_v, std_v = stack_mean_std(uniq, dws_xyz_list=dws_each)
    dws_xyz, n_dws = mean_dws_xyz_from_arrays(dws_each)
    stat = EnsembleStatResult(
        n=len(uniq),
        paths=uniq,
        template_path=uniq[0],
        mean_v=mean_v,
        std_v=std_v,
        n_dws=n_dws,
        mesh=template,
        dws_xyz=dws_xyz,
    )
    refl_files: list[Path] = []
    for p in uniq:
        hit = companion_inverse_refl(p)
        if hit is not None:
            refl_files.append(hit)
    stat.n_refl = len(refl_files)
    if len(refl_files) >= 2:
        rx, rz, rs = stack_reflector_mean_std(refl_files)
        stat.refl_x, stat.refl_mean, stat.refl_std = rx, rz, rs
    paint_smesh_ensemble_stats(
        widget,
        state,
        stat,
        auto_vlim=auto_vlim,
        half_range=half_range,
        reset_home=True,
    )
    return stat


def paint_smesh_ensemble_stats(
    widget,
    state: FormState,
    stat: EnsembleStatResult,
    *,
    auto_vlim: bool | None = None,
    half_range: float | None = None,
    reset_home: bool = False,
) -> EnsembleStatResult:
    """用已算好的均值/σ 重绘。改色标时走这里，不再读 smesh。"""
    from ..plots.velocity_contours import (
        auto_contour_specs,
        contour_specs_for_state,
        contours_enabled,
    )
    from .dws_plot import dws_mask_enabled

    mesh = stat.mesh
    if mesh is None:
        mesh, _, _ = load_smesh_plot_data(stat.template_path, None, with_xarray=False)
        stat.mesh = mesh
    work = resolve_work_dir(state.get_str("work_dir"))
    cmap = resolve_plot_smesh_cmap(state, work)
    draw_c = contours_enabled(state)
    contours = contour_specs_for_state(state)
    extra = _refl_overlay(stat)
    ds_mean, ds_std = resample_vgrid_fields_to_xarray(
        mesh, [stat.mean_v, stat.std_v]
    )
    if auto_vlim is None:
        auto_vlim = True
    if half_range is None:
        half_range = diff_vlim_half_range(state, mode="abs")
    data_max = sigma_data_max(stat.std_v)
    sigma_cmap, s_lo, s_hi, sigma_gamma = resolve_sigma_cmap_and_limits(
        stat.std_v, auto=bool(auto_vlim), half_range=float(half_range)
    )
    stat.sigma_vlim = (s_lo, s_hi)
    std_contours = auto_contour_specs(0.0, max(s_hi, 1e-6)) if draw_c else []
    dws_xyz = stat.dws_xyz if dws_mask_enabled(state) else None
    widget.set_save_dir(work)
    widget.set_velocity_stack(
        [
            {
                "ds": ds_mean,
                "mesh": mesh,
                "extra": extra,
                "cmap_spec": cmap,
                "title": f"均值 Vp  n={stat.n}",
                "cb_label": "km/s",
                "draw_contours": draw_c,
                "contour_specs": contours,
            },
            {
                "ds": ds_std,
                "mesh": mesh,
                "extra": extra,
                "cmap_spec": sigma_cmap,
                "title": "误差 σ (km/s)",
                "cb_label": "km/s",
                "draw_contours": draw_c,
                "contour_specs": std_contours,
                "vlim": (s_lo, s_hi),
                "norm_gamma": sigma_gamma,
            },
        ],
        dws_xyz=dws_xyz,
        reset_home=reset_home,
    )
    widget.setWindowTitle(f"集合统计 · n={stat.n}")
    widget.set_model_source(None, state)
    setattr(
        widget,
        "_pyaobs_stats_vlim_note",
        format_sigma_vlim_caption(
            s_hi,
            data_max,
            auto=bool(auto_vlim),
            scale="cpt" if str(sigma_cmap).lower().endswith(".cpt") else "robust",
        ),
    )
    return stat


def save_ensemble_velocity(
    stat: EnsembleStatResult, dest: Path | str, *, kind: str = "mean"
) -> Path:
    """写出均值或 σ 网格为 smesh。"""
    from .smesh_ops import write_velocity_grid_as_smesh

    dest_p = Path(dest)
    kind_l = str(kind).lower()
    if kind_l == "mean":
        return write_velocity_grid_as_smesh(stat.template_path, stat.mean_v, dest_p)
    if kind_l in ("std", "sigma", "error"):
        return write_velocity_grid_as_smesh(
            stat.template_path,
            stat.std_v,
            dest_p,
            allow_nonpositive=True,
        )
    raise ValueError(f"未知速度场: {kind}")


def save_ensemble_reflector(
    stat: EnsembleStatResult, dest: Path | str, *, which: str = "mean"
) -> Path:
    """写出平均反射面，或 mean±σ 界面（tomo2d -F 用的 x z）。"""
    from .smesh_ops import write_interface_xz

    if stat.refl_x is None or stat.refl_mean is None:
        raise ValueError("集合里配套反射面不足，没有平均界面")
    which_l = str(which).lower()
    z = stat.refl_mean
    header = f"ensemble mean reflector n={stat.n_refl}"
    if which_l in ("minus", "m1sigma", "-"):
        if stat.refl_std is None:
            raise ValueError("没有反射面误差")
        z = stat.refl_mean - stat.refl_std
        header = f"ensemble mean−σ reflector n={stat.n_refl}"
    elif which_l in ("plus", "p1sigma", "+"):
        if stat.refl_std is None:
            raise ValueError("没有反射面误差")
        z = stat.refl_mean + stat.refl_std
        header = f"ensemble mean+σ reflector n={stat.n_refl}"
    elif which_l not in ("mean", ""):
        raise ValueError(f"未知界面: {which}")
    return write_interface_xz(stat.refl_x, z, dest, header=header)


def write_ensemble_stat_files(
    stat: EnsembleStatResult,
    work: Path,
) -> EnsembleStatResult:
    """把均值/标准差网格与平均反射面写到 outputs/。"""
    out = Path(work) / "outputs"
    stat.mean_smesh = save_ensemble_velocity(stat, out / "ensemble_mean.smesh", kind="mean")
    stat.std_smesh = save_ensemble_velocity(stat, out / "ensemble_std.smesh", kind="std")
    if stat.refl_x is not None and stat.refl_mean is not None:
        stat.mean_refl = save_ensemble_reflector(
            stat, out / "ensemble_mean.refl", which="mean"
        )
    return stat


def show_smesh_diff_window(
    state: FormState,
    path_a: Path | str,
    path_b: Path | str,
    *,
    dws_xyz=None,
) -> None:
    """打开模型对比窗并绘制 B−A。"""
    _TRAY.set_paths(path_a, path_b)
    _refresh_status_labels()
    from ..dialogs.model_compare_dialog import open_model_compare_dialog

    dlg = open_model_compare_dialog(state)
    dlg.sync_from_tray(plot=True, dws_xyz=dws_xyz)


def add_smesh_to_compare(
    path: Path | str,
    *,
    state: FormState,
    dws_xyz=None,
) -> str:
    """加入托盘；两个不同模型齐了则自动绘制 B−A。"""
    msg, pa, pb = _TRAY.add(path)
    _refresh_status_labels(msg)
    from ..dialogs.model_compare_dialog import open_model_compare_dialog

    dlg = open_model_compare_dialog(state)
    dlg.sync_from_tray(plot=pa is not None and pb is not None, dws_xyz=dws_xyz)
    return msg


def add_smeshes_to_compare(
    paths: list[Path | str],
    *,
    state: FormState,
) -> str:
    """批量写入集合并打开对比窗；不自动画差值或统计。"""
    n = _TRAY.extend_ensemble(list(paths))
    _refresh_status_labels()
    from ..dialogs.model_compare_dialog import open_model_compare_dialog

    dlg = open_model_compare_dialog(state)
    dlg.sync_from_tray(plot=False)
    if n <= 0:
        return "所选模型均已在集合中。"
    return f"已加入集合 {n} 个（共 {len(_TRAY.ensemble)}）。请在对比窗查看；统计/差值需手动点。"


def popup_model_context_menu(
    parent: QWidget | None,
    *,
    path: Path | str | None,
    state: FormState | None,
    extra_actions: Callable[[QMenu], None] | None = None,
    allow_clear: bool = False,
    refl_path: Path | str | None = None,
) -> None:
    """速度图右键：写入表单 + 添加到对比 + 打开所在目录；「清除对比」仅对比图。

    右键拖仍是缩放；仅位移很小的点击才弹出（见 ``PyqtgraphStyleNav``）。
    统计图经 ``extra_actions`` 另加保存/写回均值；本函数在已加载 smesh 时
    再加「写入表单（速度/反射面）」。
    """
    from PySide6.QtGui import QCursor

    menu = QMenu(parent)
    if extra_actions is not None:
        extra_actions(menu)
    add_writeback_submenus(menu, state, path, refl_path=refl_path)
    if menu.actions():
        menu.addSeparator()
    act_add = menu.addAction("添加到对比模型")
    act_add.setToolTip(
        "把本图当前 smesh 记入对比集合。"
        "第一个为 A，第二个不同模型为 B，自动绘制 B−A。"
        "再添加则 A←原 B、B←新模型。"
        "v.in / grd 只用于绘制，不能加入对比。"
    )
    hit = Path(path) if path is not None else None
    act_add.setEnabled(
        hit is not None and state is not None and looks_like_smesh_name(hit)
    )
    act_clear = None
    if allow_clear:
        act_clear = menu.addAction("清除对比")
        act_clear.setToolTip("清空已记的 A/B 与集合（已打开的差值窗不关）")
    act_dir = menu.addAction("打开所在目录")
    act_dir.setToolTip("在资源管理器中打开当前 smesh 所在文件夹")
    act_dir.setEnabled(hit is not None)

    def _add() -> None:
        if hit is None or state is None:
            return
        try:
            add_smesh_to_compare(hit, state=state)
        except Exception as e:
            show_modeless_message("对比模型", str(e))

    act_add.triggered.connect(_add)
    if act_clear is not None:
        act_clear.triggered.connect(clear_compare_tray)
    act_dir.triggered.connect(lambda: open_containing_directory(hit))
    menu.exec(QCursor.pos())
