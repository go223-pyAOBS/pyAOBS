"""蒙特卡洛：按 v.in 命名界面（海底/基底/Conrad/莫霍）划分地质层再扰动。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyAOBS.modeling.vedit.core.geo_ifaces import (
    SEAFLOOR_IFACE,
    GeoIfaceMarks,
    format_iface_label,
    parse_iface_label,
)

from .smesh_ops import _load_mesh, air_water_node_mask, write_interface_xz

UNIT_SED = "sed"
UNIT_UC = "uc"
UNIT_LC = "lc"
UNIT_MANTLE = "mantle"
UNIT_ORDER = (UNIT_SED, UNIT_UC, UNIT_LC, UNIT_MANTLE)
UNIT_LABEL = {
    UNIT_SED: "沉积",
    UNIT_UC: "上地壳",
    UNIT_LC: "下地壳",
    UNIT_MANTLE: "地幔",
}


@dataclass(frozen=True)
class VinPerturbSpec:
    """v.in 命名界面 + 要扰动的地质层。"""

    marks: GeoIfaceMarks
    units: tuple[str, ...]
    n_ifaces: int


_ZELT_CACHE: dict[tuple[str, int, int], object] = {}
_ZELT_CACHE_BY_PATH: dict[str, tuple[str, int, int]] = {}


def _zelt_cache_key(path: str | Path) -> tuple[str, int, int]:
    p = Path(path).expanduser().resolve()
    st = p.stat()
    return str(p), int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), int(st.st_size)


def load_zelt(path: str | Path, *, clone: bool = True):
    """读 v.in。同一文件（路径+mtime+大小）只解析一次；默认返回深拷贝以免改到缓存。"""
    key = _zelt_cache_key(path)
    hit = _ZELT_CACHE.get(key)
    if hit is None:
        resolved = key[0]
        old = _ZELT_CACHE_BY_PATH.pop(resolved, None)
        if old is not None:
            _ZELT_CACHE.pop(old, None)
        from pyAOBS.modeling.rayinvr.vin_io import load_zelt_model

        hit = load_zelt_model(path)
        _ZELT_CACHE[key] = hit
        _ZELT_CACHE_BY_PATH[resolved] = key
    return copy.deepcopy(hit) if clone else hit


def vin_n_ifaces(path: str | Path) -> int:
    zelt = load_zelt(path, clone=False)
    return len(getattr(zelt, "depth_nodes", []) or [])


def default_vin_marks(n_ifaces: int) -> GeoIfaceMarks:
    """与 vedit 相同：海底默认界面 2；莫霍取底界面之上一档。"""
    n = max(0, int(n_ifaces))
    sf = SEAFLOOR_IFACE if n > SEAFLOOR_IFACE else (1 if n > 1 else (0 if n else None))
    moho = n - 2 if n >= 3 else (n - 1 if n >= 2 else None)
    return GeoIfaceMarks(seafloor=sf, basement=None, conrad=None, moho=moho).clamped(n)


def parse_stored_iface(raw: str) -> int | None:
    s = (raw or "").strip()
    if not s:
        return None
    return parse_iface_label(s if s.startswith("界面") else f"界面{s}")


def format_stored_iface(iface: int | None) -> str:
    if iface is None:
        return ""
    return str(int(iface) + 1)


def marks_from_state(state, n_ifaces: int) -> GeoIfaceMarks:
    get = state.get_str if hasattr(state, "get_str") else lambda _k: ""
    raw_sf = (get("mc.vin_seafloor") or "").strip()
    marks = GeoIfaceMarks(
        seafloor=parse_stored_iface(raw_sf) if raw_sf else default_vin_marks(n_ifaces).seafloor,
        basement=parse_stored_iface(get("mc.vin_basement") or ""),
        conrad=parse_stored_iface(get("mc.vin_conrad") or ""),
        moho=parse_stored_iface(get("mc.vin_moho") or "")
        if (get("mc.vin_moho") or "").strip()
        else default_vin_marks(n_ifaces).moho,
    )
    return marks.clamped(n_ifaces)


def available_units(marks: GeoIfaceMarks) -> list[str]:
    """当前命名界面能划分出的地质层（水层不出现）。"""
    out: list[str] = []
    sf, bm, cn, mh = marks.seafloor, marks.basement, marks.conrad, marks.moho
    if sf is not None and bm is not None and bm > sf:
        out.append(UNIT_SED)
    if sf is not None and mh is not None:
        out.append(UNIT_UC)
    if cn is not None and mh is not None and cn < mh:
        out.append(UNIT_LC)
    if mh is not None:
        out.append(UNIT_MANTLE)
    return out


def unit_span_label(marks: GeoIfaceMarks, unit: str) -> str:
    name = UNIT_LABEL.get(unit, unit)
    sf, bm, cn, mh = marks.seafloor, marks.basement, marks.conrad, marks.moho
    if unit == UNIT_SED and sf is not None and bm is not None:
        return f"{name}（{format_iface_label(sf)}–{format_iface_label(bm)}）"
    if unit == UNIT_UC:
        top = bm if bm is not None else sf
        bot = cn if cn is not None else mh
        if top is not None and bot is not None:
            return f"{name}（{format_iface_label(top)}–{format_iface_label(bot)}）"
    if unit == UNIT_LC and cn is not None and mh is not None:
        return f"{name}（{format_iface_label(cn)}–{format_iface_label(mh)}）"
    if unit == UNIT_MANTLE and mh is not None:
        return f"{name}（{format_iface_label(mh)} 以下）"
    return name


def unit_iface_span(
    marks: GeoIfaceMarks, n_ifaces: int, unit: str
) -> tuple[int, int] | None:
    sf, bm, cn, mh = marks.seafloor, marks.basement, marks.conrad, marks.moho
    last = max(int(n_ifaces) - 1, 0)
    if unit == UNIT_SED:
        if sf is None or bm is None:
            return None
        return int(sf), int(bm)
    if unit == UNIT_UC:
        top = bm if bm is not None else sf
        bot = cn if cn is not None else mh
        if top is None or bot is None:
            return None
        return int(top), int(bot)
    if unit == UNIT_LC:
        if cn is None or mh is None:
            return None
        return int(cn), int(mh)
    if unit == UNIT_MANTLE:
        if mh is None:
            return None
        return int(mh), last
    return None


def parse_vin_units(raw: str, allowed: list[str] | None = None) -> list[str]:
    s = (raw or "").strip().lower().replace(",", " ").replace(";", " ")
    aliases = {
        "sed": UNIT_SED,
        "沉积": UNIT_SED,
        "uc": UNIT_UC,
        "上地壳": UNIT_UC,
        "lc": UNIT_LC,
        "下地壳": UNIT_LC,
        "mantle": UNIT_MANTLE,
        "地幔": UNIT_MANTLE,
    }
    if s in {"none", "无"}:
        return []
    if not s:
        return list(allowed or UNIT_ORDER)
    out: list[str] = []
    for part in s.split():
        key = aliases.get(part, part)
        if key in UNIT_ORDER and key not in out:
            out.append(key)
    if allowed is not None:
        allow = set(allowed)
        out = [u for u in out if u in allow]
    return out


def format_vin_units(units: list[str] | tuple[str, ...] | None) -> str:
    chosen = [u for u in (units or []) if u in UNIT_ORDER]
    return " ".join(chosen) if chosen else "none"


def vin_spec_from_state(state, path: str | Path) -> VinPerturbSpec:
    n = vin_n_ifaces(path)
    marks = marks_from_state(state, n)
    allowed = available_units(marks)
    get = state.get_str if hasattr(state, "get_str") else lambda _k: ""
    units = parse_vin_units(get("mc.vin_units") or "", allowed)
    return VinPerturbSpec(marks=marks, units=tuple(units), n_ifaces=n)


def resolve_mc_vin_path(state, work: Path | str | None) -> Path | None:
    """``mc.v_in``，空则回退 ``gen.v_in``。"""
    raw = ""
    if hasattr(state, "get_str"):
        raw = (state.get_str("mc.v_in") or state.get_str("gen.v_in") or "").strip()
    if not raw:
        return None
    from .paths import resolve_existing_file, resolve_work_dir

    wd = resolve_work_dir(str(work) if work is not None else None)
    try:
        return resolve_existing_file(raw, wd)
    except FileNotFoundError:
        p = Path(raw).expanduser()
        if p.is_file():
            return p
        cand = wd / raw
        return cand if cand.is_file() else None


def _column_profile(zelt, x: float, scales: dict[int, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    n = len(zelt.vupper_nodes)
    scales = scales or {}
    zs: list[float] = []
    vs: list[float] = []
    for i in range(n):
        z0 = float(zelt.depth_nodes[i].get_value_at(x))
        z1 = float(zelt.depth_nodes[i + 1].get_value_at(x))
        s = float(scales.get(i + 1, 1.0))
        vu = float(zelt.vupper_nodes[i].get_value_at(x)) * s
        vl = float(zelt.vlower_nodes[i].get_value_at(x)) * s
        if not zs or abs(zs[-1] - z0) > 1e-9:
            zs.append(z0)
            vs.append(vu)
        else:
            vs[-1] = vu
        zs.append(z1)
        vs.append(vl)
    return np.asarray(zs, dtype=float), np.asarray(vs, dtype=float)


def paint_vin_on_smesh(mesh, zelt, scales: dict[int, float] | None = None) -> np.ndarray:
    """把 v.in 铺到 smesh 结点。水/气结点保持原值。"""
    v_bg = np.asarray(mesh.vgrid, dtype=float).copy()
    keep = ~air_water_node_mask(mesh)
    if not np.any(keep):
        return v_bg
    xpos = np.asarray(mesh.xpos, dtype=float)
    zpos = np.asarray(mesh.zpos, dtype=float)
    topo = np.asarray(mesh.topo, dtype=float)
    if topo.size != xpos.size and topo.size:
        topo = np.interp(
            xpos,
            np.linspace(float(xpos[0]), float(xpos[-1]), topo.size),
            topo,
        )
    v_new = v_bg.copy()
    for i, x in enumerate(xpos):
        zs, vs = _column_profile(zelt, float(x), scales)
        if zs.size < 2:
            continue
        z_abs = float(topo[i] if topo.size else 0.0) + zpos
        painted = np.interp(z_abs, zs, vs, left=vs[0], right=vs[-1])
        col = v_new[i]
        mask = keep[i]
        col[mask] = painted[mask]
        v_new[i] = col
    return v_new


def _fill_band(
    vgrid: np.ndarray,
    mesh,
    zelt,
    iface_top: int,
    iface_bot: int,
    v_top: float,
    v_bot: float,
) -> None:
    keep = ~air_water_node_mask(mesh)
    xpos = np.asarray(mesh.xpos, dtype=float)
    zpos = np.asarray(mesh.zpos, dtype=float)
    topo = np.asarray(mesh.topo, dtype=float)
    n_iface = len(zelt.depth_nodes)
    top = min(max(int(iface_top), 0), n_iface - 1)
    bot = min(max(int(iface_bot), 0), n_iface - 1)
    if bot <= top:
        return
    vt, vb = float(v_top), float(v_bot)
    if vb < vt:
        vb = vt
    for i, x in enumerate(xpos):
        z0 = float(zelt.depth_nodes[top].get_value_at(float(x)))
        z1 = float(zelt.depth_nodes[bot].get_value_at(float(x)))
        span = z1 - z0
        if abs(span) < 1e-9:
            continue
        z_abs = float(topo[i] if topo.size else 0.0) + zpos
        t = (z_abs - z0) / span
        inside = (t >= -1e-9) & (t <= 1.0 + 1e-9)
        inside &= keep[i]
        if not np.any(inside):
            continue
        t_clip = np.clip(t[inside], 0.0, 1.0)
        vgrid[i, inside] = vt + t_clip * (vb - vt)


def _unit_v_bounds(bounds, unit: str) -> tuple[float, float]:
    if unit == UNIT_SED:
        return bounds.sed_v
    if unit == UNIT_UC:
        return bounds.uc_v
    if unit == UNIT_LC:
        return bounds.lc_v
    return bounds.mantle_v


def _unit_h_bounds(bounds, unit: str) -> tuple[float, float] | None:
    if unit == UNIT_SED:
        return bounds.sed_h
    if unit == UNIT_UC:
        return bounds.uc_h
    if unit == UNIT_LC:
        return bounds.lc_h
    return None


def _prev_unit_sharing_iface(
    marks: GeoIfaceMarks,
    n_ifaces: int,
    unit: str,
    selected: set[str] | list[str] | tuple[str, ...],
) -> str | None:
    """比 ``unit`` 浅、且底界面与其顶界面重合的已选层。"""
    span = unit_iface_span(marks, n_ifaces, unit)
    if span is None:
        return None
    top = int(span[0])
    chosen = set(selected)
    for prev in UNIT_ORDER:
        if prev == unit:
            break
        if prev not in chosen:
            continue
        ps = unit_iface_span(marks, n_ifaces, prev)
        if ps is not None and int(ps[1]) == top:
            return prev
    return None


def sample_unit_velocities(
    rng,
    bounds,
    units: list[str],
    *,
    v_water: float,
    marks: GeoIfaceMarks | None = None,
    n_ifaces: int | None = None,
) -> dict[str, tuple[float, float]]:
    """各层抽顶底速度；共用界面（基底/Conrad/莫霍）上下只留一个结点，速度连续。"""
    from .mc_init_models import _sample_node_v

    selected = [u for u in UNIT_ORDER if u in units]
    prev = max(float(v_water), 0.3)
    out: dict[str, tuple[float, float]] = {}
    n = int(n_ifaces) if n_ifaces is not None else 0
    for unit in selected:
        lo, hi = _unit_v_bounds(bounds, unit)
        share = (
            _prev_unit_sharing_iface(marks, n, unit, selected)
            if marks is not None and n > 0
            else None
        )
        if share is not None and share in out:
            v0 = out[share][1]
        else:
            v0 = _sample_node_v(rng, lo, hi, prev)
        v1 = _sample_node_v(rng, lo, hi, v0)
        out[unit] = (v0, v1)
        prev = v1
    return out


def sample_unit_thicknesses(rng, bounds, units: list[str]) -> dict[str, float]:
    from .mc_init_models import _sample_span

    out: dict[str, float] = {}
    for unit in UNIT_ORDER:
        if unit not in units:
            continue
        hb = _unit_h_bounds(bounds, unit)
        if hb is None:
            continue
        out[unit] = _sample_span(rng, *hb)
    return out


def _set_iface_thickness(zelt, iface_top: int, iface_bot: int, h: float) -> None:
    n = len(zelt.depth_nodes)
    top = min(max(int(iface_top), 0), n - 1)
    bot = min(max(int(iface_bot), 0), n - 1)
    if bot <= top:
        return
    node_top = zelt.depth_nodes[top]
    node_bot = zelt.depth_nodes[bot]
    hh = max(float(h), 0.01)
    for i, x in enumerate(list(node_bot.x)):
        node_bot.val[i] = float(node_top.get_value_at(float(x))) + hh
    for j in range(bot + 1, n):
        prev = zelt.depth_nodes[j - 1]
        node = zelt.depth_nodes[j]
        for i, x in enumerate(list(node.x)):
            z_prev = float(prev.get_value_at(float(x)))
            if float(node.val[i]) < z_prev + 0.01:
                node.val[i] = z_prev + 0.01


def apply_unit_thicknesses(zelt, marks: GeoIfaceMarks, n_ifaces: int, thicknesses: dict[str, float]) -> None:
    """按抽样厚度移动选中层的底界面（顶界面跟上；更深的界面必要时下推）。"""
    for unit in UNIT_ORDER:
        h = thicknesses.get(unit)
        if h is None:
            continue
        span = unit_iface_span(marks, n_ifaces, unit)
        if span is None or unit == UNIT_MANTLE:
            continue
        _set_iface_thickness(zelt, span[0], span[1], float(h))


def vin_moho_xz(zelt, marks: GeoIfaceMarks) -> tuple[np.ndarray, np.ndarray]:
    n = len(getattr(zelt, "depth_nodes", []) or [])
    idx = marks.moho
    if idx is None:
        idx = n - 2 if n >= 2 else 0
    idx = min(max(int(idx), 0), max(n - 1, 0))
    xs, zs = zelt.get_layer_geometry(idx)
    return np.asarray(xs, dtype=float), np.asarray(zs, dtype=float)


def write_vin_moho_interface(zelt, marks: GeoIfaceMarks, dst: str | Path) -> Path:
    x, z = vin_moho_xz(zelt, marks)
    iface = marks.moho
    lab = format_iface_label(iface) if iface is not None else "—"
    return write_interface_xz(x, z, dst, header=f"MC Moho from v.in {lab}")


NAMED_IFACE_STYLE = {
    "seafloor": ("#0ea5e9", "海底", "-"),
    "basement": ("#166534", "基底", "--"),
    "conrad": ("#7c3aed", "Conrad", ":"),
    "moho": ("#e11d48", "莫霍", "-"),
}
UNIT_MASK_STYLE = {
    UNIT_SED: ("#f59e0b", 0.32),
    UNIT_UC: ("#3b82f6", 0.28),
    UNIT_LC: ("#8b5cf6", 0.28),
    UNIT_MANTLE: ("#64748b", 0.22),
}


def vin_named_iface_overlays(zelt, marks: GeoIfaceMarks) -> list[dict]:
    """预览叠命名界面；未选 Conrad 不画。"""
    extra: list[dict] = []
    n = len(getattr(zelt, "depth_nodes", []) or [])
    for role, iface in (
        ("seafloor", marks.seafloor),
        ("basement", marks.basement),
        ("conrad", marks.conrad),
        ("moho", marks.moho),
    ):
        if iface is None or iface < 0 or iface >= n:
            continue
        try:
            rx, rz = zelt.get_layer_geometry(int(iface))
        except Exception:
            continue
        color, label, ls = NAMED_IFACE_STYLE[role]
        extra.append(
            {
                "x": rx,
                "z": rz,
                "label": label,
                "text": label,
                "color": color,
                "linewidth": 2.0,
                "linestyle": ls,
            }
        )
    return extra


def vin_all_iface_overlays(zelt, marks: GeoIfaceMarks | None = None) -> list[dict]:
    """未命名的 v.in 界面，浅线 +「界面N」，方便对照下拉。"""
    named: set[int] = set()
    if marks is not None:
        for i in (marks.seafloor, marks.basement, marks.conrad, marks.moho):
            if i is not None:
                named.add(int(i))
    extra: list[dict] = []
    n = len(getattr(zelt, "depth_nodes", []) or [])
    for i in range(n):
        if i in named:
            continue
        try:
            rx, rz = zelt.get_layer_geometry(i)
        except Exception:
            continue
        extra.append(
            {
                "x": rx,
                "z": rz,
                "text": format_iface_label(i),
                "color": "#94a3b8",
                "linewidth": 0.8,
                "linestyle": ":",
            }
        )
    return extra


def vin_unit_mask_overlays(zelt, marks: GeoIfaceMarks, units: list[str] | tuple[str, ...]) -> list[dict]:
    """勾选层位的半透明蒙版。"""
    extra: list[dict] = []
    n = len(getattr(zelt, "depth_nodes", []) or [])
    for unit in UNIT_ORDER:
        if unit not in units:
            continue
        span = unit_iface_span(marks, n, unit)
        if span is None:
            continue
        top, bot = span
        try:
            x0, z0 = zelt.get_layer_geometry(top)
            x1, z1 = zelt.get_layer_geometry(bot)
        except Exception:
            continue
        xx = np.asarray(x0, dtype=float)
        z_top = np.asarray(z0, dtype=float)
        z_bot = np.interp(xx, np.asarray(x1, dtype=float), np.asarray(z1, dtype=float))
        color, alpha = UNIT_MASK_STYLE[unit]
        extra.append(
            {
                "x": xx,
                "z": z_top,
                "z_lo": z_top,
                "z_hi": z_bot,
                "color": color,
                "fill_alpha": alpha,
                "draw_line": False,
                "text": UNIT_LABEL[unit],
            }
        )
    return extra


def vin_pick_overlays(
    zelt,
    marks: GeoIfaceMarks,
    units: list[str] | tuple[str, ...],
) -> list[dict]:
    extra: list[dict] = []
    extra.extend(vin_unit_mask_overlays(zelt, marks, units))
    extra.extend(vin_all_iface_overlays(zelt, marks))
    extra.extend(vin_named_iface_overlays(zelt, marks))
    return extra


DATUM_ROLE_LABEL = {
    "seafloor": "海底",
    "basement": "基底",
    "conrad": "Conrad",
    "moho": "莫霍",
}
DATUM_YLABEL = {
    "seafloor": "海底以下深度 (km)",
    "basement": "基底以下深度 (km)",
    "conrad": "Conrad 以下深度 (km)",
    "moho": "莫霍以下深度 (km)",
}


@dataclass(frozen=True)
class Vin1dSample:
    """v.in 一次实现：相对选层 datum 的 1D（Vp–深度）。"""

    z_rel: np.ndarray
    v: np.ndarray
    datum_role: str
    iface_rel: dict[str, float]


def vin_profile_x(zelt) -> float:
    """取剖面中点，作 1D 代表柱。"""
    nodes = getattr(zelt, "depth_nodes", None) or []
    if not nodes:
        return 0.0
    xs = list(getattr(nodes[0], "x", []) or [])
    if len(xs) >= 2:
        return 0.5 * (float(xs[0]) + float(xs[-1]))
    return float(xs[0]) if xs else 0.0


def _iface_role(marks: GeoIfaceMarks, idx: int) -> str:
    i = int(idx)
    for role, val in (
        ("seafloor", marks.seafloor),
        ("basement", marks.basement),
        ("conrad", marks.conrad),
        ("moho", marks.moho),
    ):
        if val is not None and int(val) == i:
            return role
    return "seafloor"


def vin_1d_datum(marks: GeoIfaceMarks, n_ifaces: int, units: list[str] | tuple[str, ...]) -> tuple[str, int] | None:
    """1D 原点：最浅勾选层的顶界面。未勾选则从海底。"""
    n = int(n_ifaces)
    chosen = [u for u in UNIT_ORDER if u in (units or ())]
    if not chosen:
        if marks.seafloor is not None:
            return "seafloor", int(marks.seafloor)
        if n > 0:
            return "seafloor", 0
        return None
    span = unit_iface_span(marks, n, chosen[0])
    if span is None:
        return None
    top = int(span[0])
    return _iface_role(marks, top), top


def _iface_z_at(zelt, iface: int, x: float) -> float:
    n = len(getattr(zelt, "depth_nodes", []) or [])
    i = min(max(int(iface), 0), max(n - 1, 0))
    return float(zelt.depth_nodes[i].get_value_at(float(x)))


def _unit_owning_layer(marks: GeoIfaceMarks, n_ifaces: int, layer_k: int) -> str | None:
    for unit in UNIT_ORDER:
        span = unit_iface_span(marks, n_ifaces, unit)
        if span is not None and span[0] <= int(layer_k) < span[1]:
            return unit
    return None


def build_vin_1d_sample(
    zelt,
    spec: VinPerturbSpec,
    sampled_v: dict[str, tuple[float, float]],
    *,
    x: float,
) -> Vin1dSample | None:
    """厚度已写进 ``zelt`` 后，在 ``x`` 处切一条相对 datum 的 1D。"""
    n = int(spec.n_ifaces)
    if n < 2:
        return None
    datum = vin_1d_datum(spec.marks, n, spec.units)
    if datum is None:
        return None
    role, i_datum = datum
    z_datum = _iface_z_at(zelt, i_datum, x)
    selected = set(spec.units)
    zs: list[float] = []
    vs: list[float] = []

    def _push(zz: float, vv: float) -> None:
        if zz < -1e-9:
            return
        zz = max(float(zz), 0.0)
        if zs and abs(zz - zs[-1]) < 1e-9:
            return
        zs.append(zz)
        vs.append(float(vv))

    k = int(i_datum)
    last_layer = n - 2
    emitted: set[str] = set()
    while k <= last_layer:
        owner = _unit_owning_layer(spec.marks, n, k)
        if owner in selected and owner in sampled_v and owner not in emitted:
            span = unit_iface_span(spec.marks, n, owner)
            if span is None:
                k += 1
                continue
            z0 = _iface_z_at(zelt, max(span[0], i_datum), x) - z_datum
            z1 = _iface_z_at(zelt, span[1], x) - z_datum
            v0, v1 = sampled_v[owner]
            if span[0] < i_datum and z1 > 1e-9:
                top_z = _iface_z_at(zelt, span[0], x)
                bot_z = _iface_z_at(zelt, span[1], x)
                span_h = bot_z - top_z
                if abs(span_h) > 1e-9:
                    t = (z_datum - top_z) / span_h
                    v0 = float(v0) + t * (float(v1) - float(v0))
            if z1 > z0 + 1e-9:
                _push(z0, v0)
                _push(z1, v1)
            emitted.add(owner)
            k = max(int(span[1]), k + 1)
            continue
        z0_abs = _iface_z_at(zelt, k, x)
        z1_abs = _iface_z_at(zelt, k + 1, x)
        z0 = z0_abs - z_datum
        z1 = z1_abs - z_datum
        vu = float(zelt.vupper_nodes[k].get_value_at(float(x)))
        vl = float(zelt.vlower_nodes[k].get_value_at(float(x)))
        if z1 <= 1e-9:
            k += 1
            continue
        if z0 < -1e-9:
            t = (z_datum - z0_abs) / max(z1_abs - z0_abs, 1e-9)
            vu = vu + t * (vl - vu)
            z0 = 0.0
        _push(z0, vu)
        _push(z1, vl)
        k += 1

    iface_rel: dict[str, float] = {}
    for r, idx in (
        ("seafloor", spec.marks.seafloor),
        ("basement", spec.marks.basement),
        ("conrad", spec.marks.conrad),
        ("moho", spec.marks.moho),
    ):
        if idx is None:
            continue
        zr = _iface_z_at(zelt, int(idx), x) - z_datum
        if zr >= -1e-6:
            iface_rel[r] = max(float(zr), 0.0)
    if not zs:
        return None
    return Vin1dSample(
        z_rel=np.asarray(zs, dtype=float),
        v=np.asarray(vs, dtype=float),
        datum_role=role,
        iface_rel=iface_rel,
    )


def sample_vin_1d_profile(
    rng,
    zelt,
    spec: VinPerturbSpec,
    bounds,
    *,
    v_water: float,
    x: float | None = None,
) -> Vin1dSample | None:
    """与 ``vin_layers_init_velocity_fields`` 同一套抽厚度→抽速度顺序。"""
    xx = float(x) if x is not None else vin_profile_x(zelt)
    apply_unit_thicknesses(
        zelt, spec.marks, spec.n_ifaces, sample_unit_thicknesses(rng, bounds, list(spec.units))
    )
    sampled = sample_unit_velocities(
        rng,
        bounds,
        list(spec.units),
        v_water=v_water,
        marks=spec.marks,
        n_ifaces=spec.n_ifaces,
    )
    return build_vin_1d_sample(zelt, spec, sampled, x=xx)


def collect_vin_1d_profiles(
    vin_path: str | Path,
    *,
    spec: VinPerturbSpec,
    n: int,
    seed0: int,
    bounds=None,
    v_water: float = 1.5,
    x: float | None = None,
) -> tuple[list[Vin1dSample], float, str]:
    """按运行约定抽 N 条 v.in 1D。返回 ``(samples, z_max, datum_role)``。"""
    from .mc_init_models import Layer1dBounds

    b = bounds or Layer1dBounds()
    n_use = max(int(n), 1)
    out: list[Vin1dSample] = []
    datum_role = "seafloor"
    z_hi = 0.0
    pristine = load_zelt(vin_path, clone=False)
    xx = float(x) if x is not None else vin_profile_x(pristine)
    for i in range(n_use):
        zelt = copy.deepcopy(pristine)
        rng = np.random.default_rng(int(seed0) + i)
        sample = sample_vin_1d_profile(rng, zelt, spec, b, v_water=v_water, x=xx)
        if sample is None:
            continue
        datum_role = sample.datum_role
        if len(sample.z_rel):
            z_hi = max(z_hi, float(np.nanmax(sample.z_rel)))
        out.append(sample)
    return out, z_hi, datum_role


def draw_vin_1d_ensemble(
    ax,
    profiles: list[Vin1dSample],
    *,
    z_max: float,
    title: str,
    highlight: int = 0,
    ylabel: str | None = None,
) -> None:
    """横轴 Vp，纵轴相对 datum 深度（向下为正）。界面用选层方案里已命名的线。"""
    from .mc_init_models import _cmap_hex

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
        max(float(np.nanmax(s.z_rel)) for s in profiles if len(s.z_rel)),
    )
    z_grid = np.linspace(0.0, max(z_hi, 0.01), 240)
    stack: list[np.ndarray] = []
    roles = []
    for r in ("basement", "conrad", "moho"):
        if any(r in s.iface_rel and s.iface_rel[r] > 1e-6 for s in profiles):
            roles.append(r)
    role_depths: dict[str, list[float]] = {r: [] for r in roles}
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
            np.asarray(s.z_rel, dtype=float),
            color=color,
            lw=2.4 if is_hi else 0.9,
            alpha=1.0 if is_hi else 0.45,
            zorder=6 if is_hi else 2,
            label="第 1 次实现" if is_hi else None,
        )
        for r in roles:
            zr = s.iface_rel.get(r)
            if zr is None or zr <= 1e-6:
                continue
            _c, lab, ls = NAMED_IFACE_STYLE[r]
            ax.axhline(
                zr,
                color=color,
                lw=1.4 if is_hi else 0.5,
                alpha=0.9 if is_hi else 0.28,
                ls=ls,
                zorder=5 if is_hi else 1,
                label=f"第 1 次 {lab}" if is_hi else None,
            )
            role_depths[r].append(float(zr))
        stack.append(
            np.interp(
                z_grid,
                np.asarray(s.z_rel, dtype=float),
                np.asarray(s.v, dtype=float),
            )
        )
    mean = np.mean(np.vstack(stack), axis=0)
    ax.plot(mean, z_grid, color="black", lw=2.0, label="Vp 均值", zorder=5)
    for r in roles:
        depths = role_depths.get(r) or []
        if not depths:
            continue
        _c, lab, ls = NAMED_IFACE_STYLE[r]
        ax.axhline(
            float(np.mean(depths)),
            color="black",
            lw=1.3,
            ls=":",
            label=f"{lab} 均值",
            zorder=4,
        )
    ax.set_xlabel("Vp (km/s)", color="black")
    ax.set_ylabel(ylabel or DATUM_YLABEL.get(profiles[0].datum_role, "深度 (km)"), color="black")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.tick_params(colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")


def vin_selection_summary(marks: GeoIfaceMarks, units: list[str] | tuple[str, ...]) -> str:
    roles = []
    if marks.seafloor is not None:
        roles.append(f"海底 {format_iface_label(marks.seafloor)}")
    if marks.basement is not None:
        roles.append(f"基底 {format_iface_label(marks.basement)}")
    if marks.conrad is not None:
        roles.append(f"Conrad {format_iface_label(marks.conrad)}")
    if marks.moho is not None:
        roles.append(f"莫霍 {format_iface_label(marks.moho)}")
    unit_txt = "、".join(UNIT_LABEL[u] for u in UNIT_ORDER if u in units) or "无"
    return ("；".join(roles) or "未指定界面") + "  ·  扰动：" + unit_txt


def load_vin_pick_background(vin_path: str | Path, mesh_path: str | Path | None):
    """选层窗速度底图：有 smesh 则把 v.in 铺上去，否则用 v.in 自己的网格。"""
    zelt = load_zelt(vin_path)
    if mesh_path and Path(mesh_path).is_file():
        mesh = _load_mesh(mesh_path)
        v = paint_vin_on_smesh(mesh, zelt)
        mesh.vgrid = v
        mesh.pgrid = 1.0 / np.maximum(v, 1e-9)
        return mesh, mesh.to_xarray(), zelt
    from .smesh_plot_core import normalize_velocity_plot_dataset

    return None, normalize_velocity_plot_dataset(zelt.to_xarray(dx=1.0, dz=0.25)), zelt


def vin_layers_init_velocity_fields(
    src: str | Path,
    vin_path: str | Path,
    *,
    seed: int,
    spec: VinPerturbSpec,
    bounds=None,
    amp_percent: float = 2.0,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, object]:
    """返回 ``(mesh, v_基础, v_扰动, ΔV, zelt)``。未选层保持 v.in；选中层改厚度并填速度。"""
    from .mc_init_models import Layer1dBounds

    del amp_percent
    mesh = _load_mesh(src)
    v_bg = np.asarray(mesh.vgrid, dtype=float).copy()
    zelt = load_zelt(vin_path)
    v_new = paint_vin_on_smesh(mesh, zelt)
    b = bounds or Layer1dBounds()
    rng = np.random.default_rng(int(seed))
    vw = float(getattr(mesh, "v_water", 1.5) or 1.5)
    apply_unit_thicknesses(
        zelt, spec.marks, spec.n_ifaces, sample_unit_thicknesses(rng, b, list(spec.units))
    )
    sampled = sample_unit_velocities(
        rng, b, list(spec.units), v_water=vw, marks=spec.marks, n_ifaces=spec.n_ifaces
    )
    for unit, (v0, v1) in sampled.items():
        span = unit_iface_span(spec.marks, spec.n_ifaces, unit)
        if span is None:
            continue
        _fill_band(v_new, mesh, zelt, span[0], span[1], v0, v1)
    return mesh, v_bg, v_new, v_new - v_bg, zelt


def apply_vin_layers_init_to_file(
    src: str | Path,
    dst: str | Path,
    vin_path: str | Path,
    *,
    seed: int,
    spec: VinPerturbSpec,
    bounds=None,
    amp_percent: float = 2.0,
    refl_dst: str | Path | None = None,
) -> tuple[Path, Path | None]:
    mesh, _bg, v_new, _dv, zelt = vin_layers_init_velocity_fields(
        src,
        vin_path,
        seed=seed,
        spec=spec,
        bounds=bounds,
        amp_percent=amp_percent,
    )
    mesh.vgrid = v_new
    mesh.pgrid = 1.0 / np.maximum(mesh.vgrid, 1e-9)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    mesh.to_file(str(dst_p))
    refl_p = None
    if refl_dst is not None:
        refl_p = write_vin_moho_interface(zelt, spec.marks, refl_dst)
    return dst_p, refl_p
