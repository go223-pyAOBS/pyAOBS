"""蒙特卡洛起始模型：smesh（默认分段 1D）或 v.in 命名界面分层扰动。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .smesh_ops import (
    _load_mesh,
    air_water_node_mask,
    apply_random_init_to_file,
    random_init_velocity_fields,
)

INIT_MODE_SMESH = "smesh"
INIT_MODE_VIN = "v.in"
# 旧文案，仅作 resolve 别名
INIT_MODE_PERTURB = "扰动已有 smesh"
INIT_MODE_LAYERS = "分段随机 1D"
INIT_MODE_OFF = "不随机"
MC_INIT_MODE_CHOICES = (
    INIT_MODE_SMESH,
    INIT_MODE_VIN,
)

_LAYER_EPS = 0.01  # km/s，相邻结点至少递增这么多，避免持平被当成反转


def _cmap_hex(cmap, t: float) -> str:
    """colormap 返回 np.float64 RGBA，叠线绘图处会 ``str(color)``，须先收成 hex。"""
    r, g, b, a = (float(x) for x in cmap(float(t)))
    ri, gi, bi = (max(0, min(255, int(round(c * 255)))) for c in (r, g, b))
    if a >= 1.0 - 1e-9:
        return f"#{ri:02x}{gi:02x}{bi:02x}"
    ai = max(0, min(255, int(round(a * 255))))
    return f"#{ri:02x}{gi:02x}{bi:02x}{ai:02x}"


@dataclass(frozen=True)
class Layered1dSample:
    """一次分段 1D（沉积 + 上地壳 + 下地壳 + 地幔）。地幔接到网格底。"""

    z_bsf: np.ndarray
    v: np.ndarray
    h_sed: float
    h_uc: float
    h_lc: float
    h_mantle: float

    @property
    def h_crust(self) -> float:
        return float(self.h_uc) + float(self.h_lc)

    @property
    def uc_bot_bsf(self) -> float:
        return float(self.h_sed) + float(self.h_uc)

    @property
    def moho_bsf(self) -> float:
        return float(self.h_sed) + float(self.h_uc) + float(self.h_lc)


@dataclass(frozen=True)
class Layer1dBounds:
    """沉积 / 上地壳 / 下地壳 / 地幔：各层厚度（地幔接到网格底）+ 顶底速度区间。"""

    sed_h: tuple[float, float] = (0.2, 2.5)
    sed_v: tuple[float, float] = (1.7, 3.6)
    uc_h: tuple[float, float] = (6.0, 11.0)
    uc_v: tuple[float, float] = (4.0, 6.5)
    lc_h: tuple[float, float] = (10.0, 25.0)
    lc_v: tuple[float, float] = (6.6, 7.5)
    mantle_v: tuple[float, float] = (7.6, 8.2)


def parse_lo_hi(raw: str, default: tuple[float, float]) -> tuple[float, float]:
    """解析 ``\"min max\"`` / ``\"min ~ max\"``；单数则 min=max。"""
    s = (raw or "").strip().replace("～", "~").replace("—", " ").replace(",", " ")
    if "~" in s:
        parts = [p.strip() for p in s.split("~") if p.strip()]
    else:
        parts = s.split()
    if not parts:
        lo, hi = default
        return float(lo), float(hi)
    try:
        nums = [float(p) for p in parts[:2]]
    except ValueError:
        lo, hi = default
        return float(lo), float(hi)
    if len(nums) == 1:
        return nums[0], nums[0]
    lo, hi = nums[0], nums[1]
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def layer1d_bounds_from_state(state) -> Layer1dBounds:
    d = Layer1dBounds()
    get = state.get_str if hasattr(state, "get_str") else lambda _k: ""
    uc_h = parse_lo_hi(get("mc.uc_h"), d.uc_h)
    lc_h = parse_lo_hi(get("mc.lc_h"), d.lc_h)
    old_crust = (get("mc.crust_h") or "").strip()
    if not (get("mc.uc_h") or "").strip() and old_crust:
        lo, hi = parse_lo_hi(old_crust, (6.0, 31.0))
        uc_h = (lo * 0.4, hi * 0.4)
        lc_h = (lo * 0.6, hi * 0.6)
    uc_v_raw = (get("mc.uc_v") or "").strip()
    if uc_v_raw:
        uc_v = parse_lo_hi(uc_v_raw, d.uc_v)
    elif (get("mc.uc_vt") or "").strip() or (get("mc.uc_vb") or "").strip():
        lo, _hi = parse_lo_hi(get("mc.uc_vt"), (4.0, 5.5))
        _lo, hi = parse_lo_hi(get("mc.uc_vb"), (5.6, 6.5))
        uc_v = (min(lo, hi), max(lo, hi))
    else:
        uc_v = parse_lo_hi(get("mc.crust_v"), d.uc_v)
    lc_v_raw = (get("mc.lc_v") or "").strip()
    lc_v = parse_lo_hi(lc_v_raw or get("mc.lc_vb"), d.lc_v)
    return Layer1dBounds(
        sed_h=parse_lo_hi(get("mc.sed_h"), d.sed_h),
        sed_v=parse_lo_hi(get("mc.sed_v"), d.sed_v),
        uc_h=uc_h,
        uc_v=uc_v,
        lc_h=lc_h,
        lc_v=lc_v,
        mantle_v=parse_lo_hi(get("mc.mantle_v"), d.mantle_v),
    )


def resolve_mc_init_mode(state) -> str:
    """``vinlayers``（v.in）或 ``layers1d``（smesh，默认分段 1D）。"""
    raw = ""
    if hasattr(state, "get_str"):
        raw = (state.get_str("mc.init_mode") or "").strip()
    key = raw.lower().replace(" ", "")
    if key in {
        "vinlayers",
        "vin",
        "v.in",
        "扰动v.in分层",
        "扰动v.in",
        INIT_MODE_VIN.lower().replace(" ", ""),
    }:
        return "vinlayers"
    return "layers1d"


def mc_init_mode_choice(state) -> str:
    """下拉显示值：``smesh`` 或 ``v.in``。"""
    return INIT_MODE_VIN if resolve_mc_init_mode(state) == "vinlayers" else INIT_MODE_SMESH


def _sample_span(rng: np.random.Generator, lo: float, hi: float) -> float:
    if hi <= lo:
        return float(lo)
    return float(rng.uniform(lo, hi))


def _sample_node_v(
    rng: np.random.Generator, lo: float, hi: float, prev: float
) -> float:
    """在 ``[lo, hi]`` 抽一个速度，且 ≥ prev+eps（区间空则顶到 prev 之上）。"""
    floor = max(float(lo), float(prev) + _LAYER_EPS)
    ceiling = float(hi)
    if floor >= ceiling:
        return floor
    return float(rng.uniform(floor, ceiling))


def sample_layered_1d_profile(
    rng: np.random.Generator,
    *,
    bounds: Layer1dBounds,
    v_water: float,
    z_max: float,
) -> Layered1dSample:
    """随机一条海底以下 1D：各层抽顶底速度；基底 / 上·下地壳交界 / 莫霍共用结点（速度连续）。"""
    prev = max(float(v_water), 0.3)
    z_cap = max(float(z_max), 1.0)

    h_sed = _sample_span(rng, *bounds.sed_h)
    h_uc = _sample_span(rng, *bounds.uc_h)
    h_lc = _sample_span(rng, *bounds.lc_h)
    total = h_sed + h_uc + h_lc
    if total > z_cap * 0.95 and z_cap > 0:
        scale = (z_cap * 0.85) / max(total, 1e-6)
        h_sed *= scale
        h_uc *= scale
        h_lc *= scale
    h_mantle = max(z_cap - (h_sed + h_uc + h_lc), _LAYER_EPS)

    zs: list[float] = []
    vs: list[float] = []

    def _push(zz: float, vv: float) -> None:
        if zs and zz < zs[-1]:
            zz = zs[-1]
        if zs and abs(zz - zs[-1]) < 1e-9:
            return
        if vs:
            vv = max(float(vv), vs[-1] + _LAYER_EPS)
        zs.append(float(zz))
        vs.append(float(vv))

    z = 0.0
    if h_sed > 1e-6:
        v0 = _sample_node_v(rng, *bounds.sed_v, prev)
        v1 = _sample_node_v(rng, *bounds.sed_v, v0)
        _push(0.0, v0)
        _push(h_sed, v1)
        z = h_sed
        v_uc_top = v1
    else:
        v_uc_top = _sample_node_v(rng, *bounds.uc_v, prev)
        _push(0.0, v_uc_top)
    v_uc_bot = _sample_node_v(rng, *bounds.uc_v, v_uc_top)
    z_mid = z + max(h_uc, _LAYER_EPS)
    _push(z_mid, v_uc_bot)
    v_lc_bot = _sample_node_v(rng, *bounds.lc_v, vs[-1] if vs else v_uc_bot)
    z_moho = z_mid + max(h_lc, _LAYER_EPS)
    _push(z_moho, v_lc_bot)
    v_m_bot = _sample_node_v(rng, *bounds.mantle_v, vs[-1] if vs else v_lc_bot)
    _push(z_cap, v_m_bot)

    z_arr = np.asarray(zs, dtype=float)
    v_arr = np.maximum.accumulate(np.asarray(vs, dtype=float))
    return Layered1dSample(
        z_bsf=z_arr,
        v=v_arr,
        h_sed=float(h_sed),
        h_uc=float(h_uc),
        h_lc=float(h_lc),
        h_mantle=float(h_mantle),
    )


def collect_layered_1d_profiles(
    src: str | Path,
    *,
    n: int,
    seed0: int,
    bounds: Layer1dBounds | None = None,
) -> tuple[list[Layered1dSample], float, float]:
    """按运行约定抽 N 条 1D（第 i 次种子 ``seed0+i``）。返回 ``(samples, z_max, v_water)``。"""
    mesh = _load_mesh(src)
    zpos = np.asarray(mesh.zpos, dtype=float)
    z_max = float(np.nanmax(zpos)) if zpos.size else 20.0
    vw = float(getattr(mesh, "v_water", 1.5) or 1.5)
    b = bounds or Layer1dBounds()
    n_use = max(int(n), 1)
    out: list[Layered1dSample] = []
    for i in range(n_use):
        rng = np.random.default_rng(int(seed0) + i)
        out.append(
            sample_layered_1d_profile(rng, bounds=b, v_water=vw, z_max=z_max)
        )
    return out, z_max, vw


def draw_layered_1d_ensemble(
    ax,
    profiles: list[Layered1dSample],
    *,
    z_max: float,
    title: str,
    highlight: int = 0,
) -> None:
    """把全部 1D 画在同一张图：横轴 Vp，纵轴海底以下深度（向下为正）。

    ``highlight`` 为第几次实现（0-based），与左侧「第 1 次实现」对应，加粗高亮。
    """
    ax.clear()
    ax.set_title(title, color="black")
    if not profiles:
        ax.text(
            0.5,
            0.5,
            "无 1D 实现",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#64748b",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return
    try:
        from matplotlib import colormaps

        cmap = colormaps["viridis"]
    except Exception:
        from matplotlib import cm

        cmap = cm.get_cmap("viridis")
    n = len(profiles)
    z_hi = max(
        float(z_max),
        max(float(np.nanmax(s.z_bsf)) for s in profiles if len(s.z_bsf)),
    )
    z_grid = np.linspace(0.0, z_hi, 240)
    stack: list[np.ndarray] = []
    mohos: list[float] = []
    hi = int(highlight) if highlight is not None else -1
    for i, s in enumerate(profiles):
        is_hi = i == hi
        color = (
            "#e11d48"
            if is_hi
            else _cmap_hex(cmap, 0.08 + 0.84 * (i / max(n - 1, 1)))
        )
        ax.plot(
            np.asarray(s.v, dtype=float),
            np.asarray(s.z_bsf, dtype=float),
            color=color,
            lw=2.4 if is_hi else 0.9,
            alpha=1.0 if is_hi else 0.45,
            zorder=6 if is_hi else 2,
            label="第 1 次实现" if is_hi else None,
        )
        ax.axhline(
            s.moho_bsf,
            color=color,
            lw=1.4 if is_hi else 0.5,
            alpha=0.9 if is_hi else 0.28,
            ls="--",
            zorder=5 if is_hi else 1,
            label="第 1 次 Moho" if is_hi else None,
        )
        mohos.append(s.moho_bsf)
        stack.append(
            np.interp(
                z_grid,
                np.asarray(s.z_bsf, dtype=float),
                np.asarray(s.v, dtype=float),
            )
        )
    mean = np.mean(np.vstack(stack), axis=0)
    ax.plot(mean, z_grid, color="black", lw=2.0, label="Vp 均值", zorder=5)
    if mohos:
        ax.axhline(
            float(np.mean(mohos)),
            color="black",
            lw=1.3,
            ls=":",
            label="Moho 均值",
            zorder=4,
        )
    ax.set_xlabel("Vp (km/s)", color="black")
    ax.set_ylabel("海底以下深度 (km)", color="black")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.tick_params(colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")


def layered_1d_init_velocity_fields(
    src: str | Path,
    *,
    seed: int,
    bounds: Layer1dBounds | None = None,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, Layered1dSample]:
    """把随机 1D（随海底）铺到整张剖面。水/气结点保持基础速度。

    返回 ``(mesh, v_基础, v_1d, ΔV, sample)``。``mesh.vgrid`` 仍为基础。
    """
    mesh = _load_mesh(src)
    v_bg = np.asarray(mesh.vgrid, dtype=float).copy()
    keep = ~air_water_node_mask(mesh)
    zpos = np.asarray(mesh.zpos, dtype=float)
    z_max = float(np.nanmax(zpos)) if zpos.size else 20.0
    vw = float(getattr(mesh, "v_water", 1.5) or 1.5)
    rng = np.random.default_rng(int(seed))
    sample = sample_layered_1d_profile(
        rng, bounds=bounds or Layer1dBounds(), v_water=vw, z_max=z_max
    )
    if not np.any(keep):
        return mesh, v_bg, v_bg.copy(), np.zeros_like(v_bg), sample
    z_bsf = np.broadcast_to(zpos[np.newaxis, :], v_bg.shape)
    painted = np.interp(z_bsf, sample.z_bsf, sample.v)
    v_new = np.where(keep, painted, v_bg)
    return mesh, v_bg, v_new, v_new - v_bg, sample


def moho_interface_xz(mesh, moho_bsf: float) -> tuple[np.ndarray, np.ndarray]:
    """界面绝对深度：``z = topo + bsf``，随海底。"""
    x = np.asarray(mesh.xpos, dtype=float).ravel()
    topo = np.asarray(mesh.topo, dtype=float).ravel()
    if x.size < 2:
        x = np.array([0.0, 1.0], dtype=float)
        t0 = float(topo[0]) if topo.size else 0.0
        topo = np.array([t0, t0], dtype=float)
    elif topo.size != x.size:
        if topo.size == 1:
            topo = np.full(x.size, float(topo[0]))
        elif topo.size >= 2:
            xt = np.linspace(float(x[0]), float(x[-1]), topo.size)
            topo = np.interp(x, xt, topo)
        else:
            topo = np.zeros_like(x)
    return x, topo + float(moho_bsf)


def write_moho_interface(mesh, moho_bsf: float, dst: str | Path) -> Path:
    from .smesh_ops import write_interface_xz

    x, z = moho_interface_xz(mesh, moho_bsf)
    return write_interface_xz(
        x,
        z,
        dst,
        header=f"MC Moho  z=topo+{float(moho_bsf):.4f} km (h_sed+h_uc+h_lc)",
    )


def bsf_overlays_from_samples(
    mesh,
    samples: list[Layered1dSample],
    *,
    bsf_attr: str = "moho_bsf",
    max_n: int = 80,
    linestyle: str = "--",
    label_prefix: str = "Moho",
    linewidth0: float = 1.6,
    linewidth: float = 0.9,
) -> list[dict]:
    """预览用：每条实现一条等厚界面（随海底）。``max_n`` 限制 2D 叠线数量。"""
    if max_n is not None and max_n > 0:
        samples = samples[: int(max_n)]
    try:
        from matplotlib import colormaps

        cmap = colormaps["viridis"]
    except Exception:
        from matplotlib import cm

        cmap = cm.get_cmap("viridis")
    n = max(len(samples), 1)
    extra: list[dict] = []
    for i, s in enumerate(samples):
        x, z = moho_interface_xz(mesh, float(getattr(s, bsf_attr)))
        color = _cmap_hex(cmap, 0.08 + 0.84 * (i / max(n - 1, 1)))
        extra.append(
            {
                "x": x,
                "z": z,
                "label": f"{label_prefix} {i + 1}" if n <= 8 else None,
                "color": color,
                "linewidth": linewidth0 if i == 0 else linewidth,
                "linestyle": linestyle,
            }
        )
    return extra


def moho_overlays_from_samples(
    mesh,
    samples: list[Layered1dSample],
    *,
    max_n: int = 80,
) -> list[dict]:
    """预览用：每条实现一条 Moho（随海底）。"""
    return bsf_overlays_from_samples(
        mesh, samples, bsf_attr="moho_bsf", max_n=max_n
    )


def layered_1d_preview_overlays(
    mesh,
    samples: list[Layered1dSample],
    *,
    max_n: int = 80,
) -> list[dict]:
    """预览用：各次 Moho（随海底）。"""
    return moho_overlays_from_samples(mesh, samples, max_n=max_n)


def apply_layered_1d_init_to_file(
    src: str | Path,
    dst: str | Path,
    *,
    seed: int,
    bounds: Layer1dBounds | None = None,
    refl_dst: str | Path | None = None,
) -> tuple[Path, Path | None]:
    mesh, _bg, v_new, _dv, sample = layered_1d_init_velocity_fields(
        src, seed=seed, bounds=bounds
    )
    mesh.vgrid = v_new
    mesh.pgrid = 1.0 / np.maximum(mesh.vgrid, 1e-9)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    mesh.to_file(str(dst_p))
    refl_p = None
    if refl_dst is not None:
        refl_p = write_moho_interface(mesh, sample.moho_bsf, refl_dst)
    return dst_p, refl_p


def mc_init_velocity_fields(
    src: str | Path,
    *,
    mode: str,
    seed: int,
    amp_percent: float = 2.0,
    bounds: Layer1dBounds | None = None,
    vin_path: str | Path | None = None,
    vin_spec=None,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    """按方式生成第 i 次初始场（不写盘）。"""
    if mode == "layers1d":
        mesh, bg, pert, dv, _sample = layered_1d_init_velocity_fields(
            src, seed=seed, bounds=bounds
        )
        return mesh, bg, pert, dv
    if mode == "vinlayers":
        if not vin_path:
            raise ValueError("扰动 v.in 分层须指定 v.in")
        if vin_spec is None:
            raise ValueError("扰动 v.in 分层须选定海底/莫霍等界面")
        from .mc_vin_layers import vin_layers_init_velocity_fields

        mesh, bg, pert, dv, _zelt = vin_layers_init_velocity_fields(
            src,
            vin_path,
            seed=seed,
            spec=vin_spec,
            bounds=bounds,
            amp_percent=amp_percent,
        )
        return mesh, bg, pert, dv
    if mode == "perturb":
        return random_init_velocity_fields(
            src, amp_percent=amp_percent, seed=seed, apply=True
        )
    mesh = _load_mesh(src)
    v = np.asarray(mesh.vgrid, dtype=float).copy()
    return mesh, v, v.copy(), np.zeros_like(v)


def apply_mc_init_to_file(
    src: str | Path,
    dst: str | Path,
    *,
    mode: str,
    seed: int,
    amp_percent: float = 2.0,
    bounds: Layer1dBounds | None = None,
    refl_dst: str | Path | None = None,
    vin_path: str | Path | None = None,
    vin_spec=None,
) -> tuple[Path, Path | None]:
    if mode == "layers1d":
        return apply_layered_1d_init_to_file(
            src, dst, seed=seed, bounds=bounds, refl_dst=refl_dst
        )
    if mode == "vinlayers":
        if not vin_path:
            raise ValueError("扰动 v.in 分层须指定 v.in")
        if vin_spec is None:
            raise ValueError("扰动 v.in 分层须选定海底/莫霍等界面")
        from .mc_vin_layers import apply_vin_layers_init_to_file

        return apply_vin_layers_init_to_file(
            src,
            dst,
            vin_path,
            seed=seed,
            spec=vin_spec,
            bounds=bounds,
            amp_percent=amp_percent,
            refl_dst=refl_dst,
        )
    if mode == "perturb":
        return (
            apply_random_init_to_file(
                src, dst, amp_percent=amp_percent, seed=seed
            ),
            None,
        )
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    if Path(src).resolve() != dst_p.resolve():
        import shutil

        shutil.copy2(src, dst_p)
    return dst_p, None
