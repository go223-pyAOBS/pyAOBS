# -*- coding: utf-8 -*-
"""把 wave2d OBS 道集接到 tomo2d GUI（调用 run_obs_gather，不另写正演）。"""

from __future__ import annotations

from pathlib import Path

from ..state.form_state import FormState
from .workflow import PreparedRun


def _num(state: FormState, key: str, default: float) -> float:
    raw = state.get_str(key, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} 不是数字: {raw}") from exc


def _path(work: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = work / p
    return p


def collect_wave2d(state: FormState, work: Path) -> dict:
    vp = _path(work, state.get_str("wave.vp_smesh") or "true_vp.smesh")
    vs = _path(work, state.get_str("wave.vs_smesh") or "true_vs.smesh")
    sea = _path(work, state.get_str("wave.seafloor") or "seafloor.refl")
    out = _path(work, state.get_str("wave.out") or "wave_fwd")
    syn_raw = state.get_str("wave.syn")
    syn = _path(work, syn_raw) if syn_raw else None
    return {
        "work": work,
        "out": out,
        "vp": vp,
        "vs": vs,
        "seafloor": sea,
        "obs": _num(state, "wave.obs", 50.0),
        "obs_z": _num(state, "wave.obs_z", 2.0),
        "src_z": _num(state, "wave.src_z", 0.01),
        "offset": _num(state, "wave.offset", 80.0),
        "drec": _num(state, "wave.drec", 0.2),
        "dx": _num(state, "wave.dx", 0.1),
        "tmax": _num(state, "wave.tmax", 22.0),
        "f0": _num(state, "wave.f0", 3.0),
        "vred": _num(state, "wave.vred", 8.0),
        "pclip": _num(state, "wave.pclip", 98.0),
        "water_h": _num(state, "wave.water_h", 2.0),
        "water_v": _num(state, "wave.water_v", 1.5),
        "tred_max": _num(state, "wave.tred_max", 12.0),
        "absorb": state.get_str("wave.absorb") or "pml",
        "layout": state.get_str("wave.layout") or "reciprocal",
        "src_kind": state.get_str("wave.src_kind") or "expl",
        "skip_ray": state.get_bool("wave.skip_ray", True),
        "quick": state.get_bool("wave.quick", False),
        "syn": syn,
    }


def preview_wave2d(state: FormState, work: Path) -> str:
    kw = collect_wave2d(state, work)
    lines = [
        "run_obs_gather(  # OBS 为源，水中记压力",
        f"  vp={kw['vp']}",
        f"  vs={kw['vs']}",
        f"  seafloor={kw['seafloor']}",
        f"  out={kw['out']}",
        f"  obs={kw['obs']:g}  obs_z={kw['obs_z']:g}  layout={kw['layout']}",
        f"  offset=±{kw['offset']:g} km  drec={kw['drec']:g}  dx={kw['dx']:g}",
        f"  tmax={kw['tmax']:g}  f0={kw['f0']:g}  absorb={kw['absorb']}",
        f"  折合 {kw['vred']:g} km/s  显示 0–{kw['tred_max']:g} s",
        f"  水柱 t=√(x²+(nH)²)/v  H={kw['water_h']:g} v={kw['water_v']:g}  n=1,3,5",
        f"  skip_ray={kw['skip_ray']}  quick={kw['quick']}  syn={kw['syn']}",
        ")",
        "",
        "默认 layout=reciprocal：源在 OBS 海底，检波在浅水。",
        "折合图纵轴从 0 到 tred_max。",
    ]
    return "\n".join(lines)


def prepare_wave2d(state: FormState, work: Path) -> PreparedRun:
    kw = collect_wave2d(state, work)
    for label in ("vp", "vs", "seafloor"):
        path = kw[label]
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {label}: {path}")
    shown: list[Path] = []

    def job() -> str:
        from pyAOBS.modeling.wave2d.run_gather_017 import run_obs_gather

        res = run_obs_gather(**kw)
        png = res.png_reduced or res.png
        shown.append(png)
        return res.log + f"\n图: {png}"

    prep = PreparedRun(
        title="wave2d",
        job=job,
        preview_text=preview_wave2d(state, work),
        notes=["wave2d：OBS 为源，输出在 wave.out（默认 work_dir/wave_fwd）"],
    )
    prep.shown_png = shown  # type: ignore[attr-defined]
    return prep
