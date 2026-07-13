"""Compare feasible (Tp, chi) regions across lithology presets."""

from __future__ import annotations

from dataclasses import dataclass

from petrology.melting.hvp_scan import HvpScanResult, ProgressCallback, _count_grid, scan_hvp_lip
from petrology.fc.delta_vp import R3_DELTA_VP_WL_KW
from petrology.melting.lip_scan import _best_pareto_from_result
from petrology.melting.pymelt_lithology_adapter import (
    LITHOLOGY_PRESETS,
    list_lithology_presets,
    resolve_lithology_col_kwargs,
)


@dataclass(frozen=True)
class PresetCompareRow:
    preset: str
    description: str
    peridotite: str
    pyroxenite: str
    n_grid: int
    n_feasible: int
    n_h_match: int
    n_in_bounds: int
    tp_min: float | None
    tp_max: float | None
    chi_min: float | None
    chi_max: float | None
    best_tp: float | None
    best_chi: float | None
    best_h: float | None
    best_dh: float | None
    best_vp: float | None
    best_in_bounds: bool | None


def _scan_counts(res: HvpScanResult, *, h_tolerance_km: float) -> tuple[int, int, int]:
    n_grid = len(res.points)
    n_h = sum(1 for p in res.points if abs(p.h_match_km) <= h_tolerance_km)
    n_in_b = sum(1 for p in res.points if p.bulk_in_bounds)
    return n_grid, n_h, n_in_b


def _row_from_scan(
    *,
    name: str,
    desc: str,
    per_key: str,
    pyr_key: str,
    res: HvpScanResult,
    h_obs_km: float,
    h_tolerance_km: float,
    v_lc_obs_km_s: float,
) -> PresetCompareRow:
    n_grid, n_h, n_in_b = _scan_counts(res, h_tolerance_km=h_tolerance_km)
    rng = res.tp_chi_ranges()

    if res.feasible:
        best = min(res.feasible, key=lambda p: abs(p.h_match_km))
    else:
        best = _best_pareto_from_result(
            res,
            v_lc_obs=v_lc_obs_km_s,
            h_obs=h_obs_km,
        )

    best_tp = best_chi = best_h = best_dh = best_vp = None
    best_in_bounds: bool | None = None
    if best is not None:
        best_tp, best_chi = best.tp_c, best.chi
        best_h, best_vp = best.h_km, best.vp_bulk_km_s
        best_dh = best.h_match_km
        best_in_bounds = best.bulk_in_bounds

    tp_rng = rng.get("tp_c")
    chi_rng = rng.get("chi")
    return PresetCompareRow(
        preset=name,
        description=desc,
        peridotite=per_key,
        pyroxenite=pyr_key,
        n_grid=n_grid,
        n_feasible=res.n_feasible,
        n_h_match=n_h,
        n_in_bounds=n_in_b,
        tp_min=tp_rng[0] if tp_rng else None,
        tp_max=tp_rng[1] if tp_rng else None,
        chi_min=chi_rng[0] if chi_rng else None,
        chi_max=chi_rng[1] if chi_rng else None,
        best_tp=best_tp,
        best_chi=best_chi,
        best_h=best_h,
        best_dh=best_dh,
        best_vp=best_vp,
        best_in_bounds=best_in_bounds,
    )


def compare_lithology_presets(
    *,
    v_lc_obs_km_s: float = 7.0,
    h_obs_km: float = 30.0,
    b_km: float = 0.0,
    pyroxenite_frac: float = 0.10,
    presets: list[str] | None = None,
    tp_range_c: tuple[float, float] = (1200.0, 1400.0),
    tp_step_c: float = 25.0,
    chi_values: tuple[float, ...] = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0),
    h_tolerance_km: float = 5.0,
    melting_engine: str = "reebox",
    n_isentropic_steps: int = 32,
    refine_norm_vp: int = 8,
    verbose: bool = False,
    include_native: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> list[PresetCompareRow]:
    """
    Run ``scan_hvp_lip`` for each lithology preset (and optional native baseline).

    Returns summary rows sorted by strict feasibility, then H-only matches, then |ΔH|.
    """
    names = list(presets) if presets else list_lithology_presets()
    rows: list[PresetCompareRow] = []

    if include_native:
        names = ["__native__"] + [n for n in names if n != "__native__"]

    n_grid = _count_grid(tp_range_c, tp_step_c, list(chi_values))
    refine_steps = refine_norm_vp if (refine_norm_vp > 0 and melting_engine == "reebox") else 0
    per_preset = n_grid + refine_steps
    total_steps = len(names) * per_preset
    if progress_callback:
        progress_callback(0, total_steps, f"预设对比 0/{len(names)}")

    for i, name in enumerate(names):
        if name == "__native__":
            lith_kw = resolve_lithology_col_kwargs(lithology_backend="native")
            desc = "Native Katz + G2 (analytic)"
            per_key = "native"
            pyr_key = "native"
        else:
            if name not in LITHOLOGY_PRESETS:
                continue
            p = LITHOLOGY_PRESETS[name]
            lith_kw = resolve_lithology_col_kwargs(lithology_preset=name)
            desc = p.description
            per_key = p.peridotite_key
            pyr_key = p.pyroxenite_key

        if verbose:
            print(f"Preset scan: {name} ({per_key} + {pyr_key})", flush=True)

        base = i * per_preset

        def _preset_progress(done: int, _total: int, msg: str, *, base=base, preset=name) -> None:
            if progress_callback:
                progress_callback(
                    min(base + done, total_steps),
                    total_steps,
                    f"[{preset}] {msg}",
                )

        res = scan_hvp_lip(
            v_lc_obs_km_s=v_lc_obs_km_s,
            h_obs_km=h_obs_km,
            b_km=b_km,
            pyroxenite_frac=pyroxenite_frac,
            tp_range_c=tp_range_c,
            tp_step_c=tp_step_c,
            chi_values=list(chi_values),
            h_tolerance_km=h_tolerance_km,
            melting_engine=melting_engine,
            delta_vp_engine="wl1990",
            delta_vp_wl_kw=dict(R3_DELTA_VP_WL_KW),
            require_bulk_in_bounds=False,
            vp_bias_km_s=0.0,
            n_isentropic_steps=n_isentropic_steps,
            refine_norm_vp=refine_norm_vp,
            verbose=verbose,
            progress_callback=_preset_progress if progress_callback else None,
            **lith_kw,
        )
        rows.append(
            _row_from_scan(
                name=name,
                desc=desc,
                per_key=per_key,
                pyr_key=pyr_key,
                res=res,
                h_obs_km=h_obs_km,
                h_tolerance_km=h_tolerance_km,
                v_lc_obs_km_s=v_lc_obs_km_s,
            )
        )

    if progress_callback:
        progress_callback(total_steps, total_steps, "预设对比完成")

    rows.sort(
        key=lambda r: (
            -r.n_feasible,
            -r.n_h_match,
            abs(r.best_dh) if r.best_dh is not None else 999.0,
            r.preset,
        )
    )
    return rows


def format_preset_compare_table(rows: list[PresetCompareRow]) -> str:
    lines = [
        f"{'Preset':<20} {'n':>3} {'n_H':>4} {'n_B':>4} {'grid':>4} "
        f"{'Tp range':>14} {'χ range':>10} "
        f"{'best Tp':>7} {'χ':>4} {'H':>5} {'dH':>5} {'Vp':>6} {'B':>2}",
        "-" * 102,
        "  n=严格可行(H+Vp界)  n_H=仅|ΔH|≤tol  n_B=Vp界内  B=最近点是否在Vp界",
    ]
    for r in rows:
        tp_rng = f"{r.tp_min:.0f}-{r.tp_max:.0f}" if r.tp_min is not None else "—"
        chi_rng = f"{r.chi_min:g}-{r.chi_max:g}" if r.chi_min is not None else "—"
        bt = f"{r.best_tp:.0f}" if r.best_tp is not None else "—"
        bc = f"{r.best_chi:g}" if r.best_chi is not None else "—"
        bh = f"{r.best_h:.1f}" if r.best_h is not None else "—"
        bdh = f"{r.best_dh:+.1f}" if r.best_dh is not None else "—"
        bv = f"{r.best_vp:.3f}" if r.best_vp is not None else "—"
        bb = "Y" if r.best_in_bounds else ("N" if r.best_in_bounds is not None else "—")
        lines.append(
            f"{r.preset:<20} {r.n_feasible:3d} {r.n_h_match:4d} {r.n_in_bounds:4d} {r.n_grid:4d} "
            f"{tp_rng:>14} {chi_rng:>10} "
            f"{bt:>7} {bc:>4} {bh:>5} {bdh:>5} {bv:>6} {bb:>2}"
        )
    return "\n".join(lines)


def preset_compare_hints(
    rows: list[PresetCompareRow],
    *,
    pyroxenite_frac: float,
    h_tolerance_km: float,
    refine_norm_vp: int,
    fast_scan: bool,
) -> str:
    if not rows:
        return ""

    hints: list[str] = []
    all_strict_zero = all(r.n_feasible == 0 for r in rows)
    any_h = any(r.n_h_match > 0 for r in rows)
    any_bounds = any(r.n_in_bounds > 0 for r in rows)

    if all_strict_zero and any_h and not any_bounds and fast_scan:
        hints.append(
            "快速模式 (refine=0) 使用 melt_proxy 估算 Vp，网格点通常全部落在 KKHS02 bulk 界外；"
            "取消「快速扫描」启用 BurnMan 精修后，n 才可能 >0。"
        )
    elif all_strict_zero and any_h and not any_bounds:
        hints.append("网格内有 H 匹配点 (n_H>0)，但 Vp 均不在 bulk 界内 — 可扩大 Tp/χ 范围或检查 V_LC。")

    if all_strict_zero and not any_h:
        hints.append(
            f"连 H 匹配也为 0：当前 Φ={pyroxenite_frac:.2f}、|ΔH|≤{h_tolerance_km:g} km、"
            "Tp/χ 网格可能未覆盖观测；Tp–χ 文献扫描常固定 Φ=0，Φ>0 请用「Φ 扫描」。"
        )
    elif all_strict_zero and any_h:
        hints.append(
            f"严格可行 n=0 但存在 n_H>0：岩性差异请比较 n_H、best 行的 dH/Vp/B；"
            f"Φ={pyroxenite_frac:.2f}。"
        )

    if fast_scan and refine_norm_vp == 0 and not hints:
        hints.append("快速模式仅统计网格 melt_proxy；精修后 Vp 可能进入界内。")

    return "\n".join(f"  · {h}" for h in hints)
