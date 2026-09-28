"""Vp/Vs-Vp 图：含水参考点、DEM 纵横比/孔隙度曲线与磁盘缓存 —— 对齐 imodel_gui/rock_scatter.py。"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[misc, assignment]

from pyAOBS.utils import (
    calculate_dem_velocity,
    calculate_serpentinization_from_water_content,
    calculate_vp_from_serpentinization,
    calculate_vs_from_serpentinization,
)


def utils_package_parent() -> Path:
    """与 Tk rock_scatter._get_dem_cache_filename 一致的 pyAOBS 包根路径下的 utils。"""
    # imodel_gui/rock_scatter.py parents[2] == 包根（含 visualization 与 utils）
    here = Path(__file__).resolve()
    pkg_root = here.parents[2]
    return pkg_root / "utils"


def _format_number(num: Any) -> str:
    if isinstance(num, (int, float)):
        if abs(float(num) - int(num)) < 1e-10:
            return str(int(num))
        formatted = f"{float(num):.6f}".rstrip("0").rstrip(".")
        return formatted
    return str(num)


def dem_cache_filepath(
    host_vp: float,
    host_vs: float,
    host_density: float,
    critical_porosity: float,
    *,
    rock_name: str = "dunite",
) -> Path:
    vp_str = _format_number(host_vp)
    vs_str = _format_number(host_vs)
    density_str = _format_number(host_density)
    phic_str = _format_number(critical_porosity)
    name = f"dem_curves_cache_{rock_name}_{vp_str}_{vs_str}_{density_str}_{phic_str}.pkl"
    return utils_package_parent() / name


def load_dem_curves_cache(cache_file: Path) -> Optional[Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    try:
        if cache_file.is_file():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        warnings.warn(f"Failed to load DEM cache: {e}")
    return None


def save_dem_curves_cache(
    cache_file: Path,
    curves_data: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(curves_data, f)
    except Exception as e:
        warnings.warn(f"Failed to save DEM cache: {e}")


def lookup_aspect_curve(
    curves_data: Dict[Any, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    aspect_ratio: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """兼容 pickle 中非 float key（如 numpy 标量）。"""
    af = float(aspect_ratio)
    if af in curves_data:
        return curves_data[af]
    best_k = None
    best_d = np.inf
    for k in curves_data:
        diff = abs(float(k) - af)
        if diff < best_d:
            best_d = diff
            best_k = k
    if best_k is not None and best_d < 1e-7 * max(1.0, abs(af)):
        return curves_data[best_k]
    return None


def _validate_dem_cache(
    curves_data: Optional[Dict[Any, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    aspect_ratios: List[float],
) -> bool:
    if not isinstance(curves_data, dict) or not curves_data:
        return False
    for ar in aspect_ratios:
        row = lookup_aspect_curve(curves_data, float(ar))
        if row is None:
            return False
        vp_array, vp_vs_ratio_array, _por = row
        vp_array = np.asarray(vp_array)
        vp_vs_ratio_array = np.asarray(vp_vs_ratio_array)
        if np.sum(~np.isnan(vp_array)) == 0 or np.sum(~np.isnan(vp_vs_ratio_array)) == 0:
            return False
    return True


def calculate_dem_curves_grid(
    aspect_ratios: List[float],
    porosity_range: np.ndarray,
    *,
    host_vp_dunite: float,
    host_vs_dunite: float,
    host_density_dunite: float,
    inclusion_k: float,
    inclusion_mu: float,
    inclusion_density: float,
    critical_porosity: float,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    curves_data: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    total_tasks = len(aspect_ratios) * len(porosity_range)
    current_task = 0

    for i, aspect_ratio in enumerate(aspect_ratios):
        vp_list: List[float] = []
        ratio_list: List[float] = []
        if progress_callback:
            progress_callback(
                current_task,
                total_tasks,
                f"纵横比 {aspect_ratio:.6f} ({i + 1}/{len(aspect_ratios)})",
            )
        for j, porosity in enumerate(porosity_range):
            try:
                vp_eff, vs_eff, vp_vs_ratio = calculate_dem_velocity(
                    porosity=float(porosity),
                    aspect_ratio=float(aspect_ratio),
                    host_vp=host_vp_dunite,
                    host_vs=host_vs_dunite,
                    host_density=host_density_dunite,
                    inclusion_k=inclusion_k,
                    inclusion_mu=inclusion_mu,
                    inclusion_density=inclusion_density,
                    n_steps=200,
                    critical_porosity=critical_porosity,
                )
                if (
                    vp_eff > 0
                    and vs_eff > 0
                    and vp_vs_ratio > 0
                    and not np.isnan(vp_eff)
                    and not np.isnan(vs_eff)
                    and not np.isnan(vp_vs_ratio)
                    and vp_eff < 20.0
                    and vs_eff < 20.0
                    and vp_vs_ratio < 10.0
                ):
                    vp_list.append(float(vp_eff))
                    ratio_list.append(float(vp_vs_ratio))
                else:
                    vp_list.append(float("nan"))
                    ratio_list.append(float("nan"))
            except Exception:
                vp_list.append(float("nan"))
                ratio_list.append(float("nan"))

            current_task += 1
            if progress_callback and (j % 10 == 0 or j == len(porosity_range) - 1):
                progress_callback(
                    current_task,
                    total_tasks,
                    f"{aspect_ratio:.4f}, φ {porosity:.4f} ({j + 1}/{len(porosity_range)})",
                )

        curves_data[float(aspect_ratio)] = (
            np.asarray(vp_list, dtype=float),
            np.asarray(ratio_list, dtype=float),
            porosity_range,
        )

    if progress_callback:
        progress_callback(total_tasks, total_tasks, "完成")
    return curves_data


DEFAULT_DEM_PHYSICAL = dict(
    host_vp_dunite=8.299,
    host_vs_dunite=4.731,
    host_density_dunite=3.310,
    inclusion_k=2.2,
    inclusion_mu=0.0,
    inclusion_density=1.03,
    critical_porosity=0.65,
)


def dem_aspect_ratios_and_labels() -> Tuple[List[float], List[str]]:
    ars = [0.05, 0.03, 0.02, 0.013, 0.01, 0.0067, 0.005, 0.002, 0.001, 0.0001]
    labels = ["0.05", "0.03", "0.02", "0.013", "0.01", "0.0067", "0.005", "0.002", "0.001", "0.0001"]
    return ars, labels


def dem_porosity_params() -> Tuple[np.ndarray, List[float]]:
    porosity_max = 0.4
    n_points = 500
    porosity_range = np.linspace(0, porosity_max, n_points)
    porosities_percent = [12.51, 8.08, 5.69, 3.37, 2.22, 0.62, 0.25, 0.02]
    return porosity_range, porosities_percent


def ensure_dem_curves(
    *,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]], Dict[str, Any]]:
    """加载或计算 DEM 曲线；返回 (curves_data, dem_meta)。"""
    phys = DEFAULT_DEM_PHYSICAL
    host_vp_dunite = float(phys["host_vp_dunite"])
    host_vs_dunite = float(phys["host_vs_dunite"])
    host_density_dunite = float(phys["host_density_dunite"])
    inclusion_k = float(phys["inclusion_k"])
    inclusion_mu = float(phys["inclusion_mu"])
    inclusion_density = float(phys["inclusion_density"])
    critical_porosity = float(phys["critical_porosity"])

    aspect_ratios, aspect_ratio_labels = dem_aspect_ratios_and_labels()
    porosity_range, porosities_percent = dem_porosity_params()

    cache_file = dem_cache_filepath(
        host_vp_dunite,
        host_vs_dunite,
        host_density_dunite,
        critical_porosity,
        rock_name="dunite",
    )

    curves = load_dem_curves_cache(cache_file)
    keys_ok = curves is not None and _validate_dem_cache(curves, aspect_ratios)

    if not keys_ok:
        if log_fn:
            log_fn("Computing DEM curves (may take tens of seconds, cached under pyAOBS/utils/)…")
        curves = calculate_dem_curves_grid(
            aspect_ratios,
            porosity_range,
            host_vp_dunite=host_vp_dunite,
            host_vs_dunite=host_vs_dunite,
            host_density_dunite=host_density_dunite,
            inclusion_k=inclusion_k,
            inclusion_mu=inclusion_mu,
            inclusion_density=inclusion_density,
            critical_porosity=critical_porosity,
            progress_callback=progress_callback,
        )
        save_dem_curves_cache(cache_file, curves)

    meta = {
        "aspect_ratios": aspect_ratios,
        "aspect_ratio_labels": aspect_ratio_labels,
        "porosities_percent": porosities_percent,
        "porosity_range": porosity_range,
        "host_vp_dunite": host_vp_dunite,
        "host_vs_dunite": host_vs_dunite,
    }
    return curves, meta


def try_load_dem_curves_default_cache() -> Optional[Tuple[Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]], Dict[str, Any]]]:
    """若默认 dunite 物理参数下缓存有效则返回 (curves, meta)，否则 None（与 Tk 缓存命中路径一致）。"""
    phys = DEFAULT_DEM_PHYSICAL
    host_vp_dunite = float(phys["host_vp_dunite"])
    host_vs_dunite = float(phys["host_vs_dunite"])
    host_density_dunite = float(phys["host_density_dunite"])
    critical_porosity = float(phys["critical_porosity"])
    aspect_ratios, aspect_ratio_labels = dem_aspect_ratios_and_labels()
    porosity_range, porosities_percent = dem_porosity_params()
    cache_file = dem_cache_filepath(
        host_vp_dunite,
        host_vs_dunite,
        host_density_dunite,
        critical_porosity,
        rock_name="dunite",
    )
    curves = load_dem_curves_cache(cache_file)
    if curves is None or not _validate_dem_cache(curves, aspect_ratios):
        return None
    meta = {
        "aspect_ratios": aspect_ratios,
        "aspect_ratio_labels": aspect_ratio_labels,
        "porosities_percent": porosities_percent,
        "porosity_range": porosity_range,
        "host_vp_dunite": host_vp_dunite,
        "host_vs_dunite": host_vs_dunite,
    }
    return curves, meta


def build_water_reference() -> Dict[str, Any]:
    water_contents = [2, 4, 6, 8, 10, 12, 13]
    water_vp_list: List[float] = []
    water_ratio_list: List[float] = []
    water_labels: List[str] = []
    for w in water_contents:
        serp = float(calculate_serpentinization_from_water_content(w))
        vp_mps = float(calculate_vp_from_serpentinization(serp))
        vs_mps = float(calculate_vs_from_serpentinization(serp))
        vp_kms = vp_mps / 1000.0
        vs_kms = vs_mps / 1000.0
        water_vp_list.append(vp_kms)
        water_ratio_list.append(vp_kms / vs_kms)
        water_labels.append(f"{w}% H₂O\n({serp:.1f}% β)")
    return {"vp": water_vp_list, "ratio": water_ratio_list, "labels": water_labels}


def draw_rock_database_points_ratio(ax, db_data: pd.DataFrame) -> None:
    if db_data is None or db_data.empty:
        return

    highlighted_rocks_lower = {
        "basalt": "blue",
        "serpentinite": "green",
        "gabbro": "orange",
        "dunite": "purple",
        "granite": "brown",
    }
    highlighted_display = {
        "basalt": "Basalt",
        "serpentinite": "Serpentinite",
        "gabbro": "Gabbro",
        "dunite": "Dunite",
        "granite": "Granite",
    }

    if "rock_type" in db_data.columns:
        other_mask = pd.Series([False] * len(db_data), index=db_data.index)
        for rock_type in db_data["rock_type"].unique():
            rtl = str(rock_type).lower().strip()
            if rtl not in highlighted_rocks_lower:
                other_mask |= db_data["rock_type"] == rock_type
        if other_mask.any():
            ax.scatter(
                db_data.loc[other_mask, "vp"],
                db_data.loc[other_mask, "vp_vs_ratio"],
                c="lightgray",
                alpha=0.4,
                s=25,
                edgecolors="gray",
                linewidths=0.3,
                zorder=5,
                label="_nolegend_",
            )
        for rock_lower, color in highlighted_rocks_lower.items():
            matched = None
            for db_rock_type in db_data["rock_type"].unique():
                if str(db_rock_type).lower().strip() == rock_lower:
                    matched = db_rock_type
                    break
            if matched is None:
                continue
            mask = db_data["rock_type"] == matched
            if mask.any():
                disp = highlighted_display.get(rock_lower, str(matched))
                ax.scatter(
                    db_data.loc[mask, "vp"],
                    db_data.loc[mask, "vp_vs_ratio"],
                    c=color,
                    s=50,
                    alpha=0.8,
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=6,
                    label=disp,
                )
    else:
        ax.scatter(
            db_data["vp"],
            db_data["vp_vs_ratio"],
            c="lightgray",
            alpha=0.5,
            s=30,
            edgecolors="gray",
            linewidths=0.3,
            label="Database",
            zorder=5,
        )


def draw_water_content_points(ax, water_data: Dict[str, Any]) -> None:
    if not water_data:
        return
    vp_list = water_data["vp"]
    ratio_list = water_data["ratio"]
    labels = water_data["labels"]
    if vp_list:
        ax.scatter(
            vp_list,
            ratio_list,
            c="cyan",
            marker="D",
            s=150,
            edgecolors="darkblue",
            linewidths=1.5,
            alpha=0.8,
            zorder=8,
            label="Water Content Reference",
        )
        for vp, ratio, lbl in zip(vp_list, ratio_list, labels):
            ax.annotate(
                lbl,
                (vp, ratio),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="darkblue",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="darkblue",
                    alpha=0.7,
                ),
                zorder=9,
                ha="left",
                va="bottom",
            )


def draw_aspect_ratio_curves(
    ax,
    curves_data: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    dem_params: Dict[str, Any],
) -> None:
    aspect_ratios: List[float] = dem_params["aspect_ratios"]
    aspect_ratio_labels: List[str] = dem_params["aspect_ratio_labels"]

    curve_color = "black"
    for i, aspect_ratio in enumerate(aspect_ratios):
        row = lookup_aspect_curve(curves_data, float(aspect_ratio))
        if row is None:
            continue
        vp_array, vp_vs_ratio_array, _por = row
        mask = (
            (vp_array >= 1.0)
            & (vp_array <= 12.0)
            & (vp_vs_ratio_array >= 1.0)
            & (vp_vs_ratio_array <= 3.0)
            & (~np.isnan(vp_array))
            & (~np.isnan(vp_vs_ratio_array))
        )
        vp_filtered = vp_array[mask]
        rv_filtered = vp_vs_ratio_array[mask]
        if len(vp_filtered) <= 1:
            continue
        display_mask = (vp_filtered >= 2.5) & (vp_filtered <= 9.0) & (rv_filtered >= 1.5) & (rv_filtered <= 2.3)
        label = f"Aspect Ratio = {aspect_ratio_labels[i]}"
        if np.sum(display_mask) > 1:
            ax.plot(
                vp_filtered[display_mask],
                rv_filtered[display_mask],
                color=curve_color,
                lw=1.5,
                alpha=0.7,
                label=label,
                zorder=6,
            )
        else:
            ax.plot(
                vp_filtered,
                rv_filtered,
                color=curve_color,
                lw=1.5,
                alpha=0.7,
                label=label,
                zorder=6,
            )


def _linear_interp_sorted(x_sorted: np.ndarray, y_sorted: np.ndarray, xi: float) -> Optional[float]:
    """单调升序 x 上的分段线性插值。"""
    if x_sorted.size < 2:
        return None
    xf = float(xi)
    if xf < float(x_sorted[0]) or xf > float(x_sorted[-1]):
        return None
    j = int(np.searchsorted(x_sorted, xf))
    j0 = max(1, min(j, x_sorted.size - 1))
    x_lo, x_hi = float(x_sorted[j0 - 1]), float(x_sorted[j0])
    if x_hi <= x_lo:
        return float(y_sorted[j0 - 1])
    t = (xf - x_lo) / (x_hi - x_lo)
    return float(y_sorted[j0 - 1]) + t * (float(y_sorted[j0]) - float(y_sorted[j0 - 1]))


def draw_porosity_curves(
    ax,
    curves_data: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    dem_params: Dict[str, Any],
) -> None:
    if plt is None:
        return
    aspect_ratios: List[float] = dem_params["aspect_ratios"]
    porosities_percent: List[float] = dem_params["porosities_percent"]
    porosities = [p / 100.0 for p in porosities_percent]
    porosity_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(porosities)))

    try:
        from scipy.interpolate import interp1d as scipy_interp1d  # noqa: WPS433
    except ImportError:
        scipy_interp1d = None  # type: ignore[misc, assignment]

    for i, target_porosity in enumerate(porosities):
        vp_curve: List[float] = []
        rr_curve: List[float] = []
        tp = float(target_porosity)
        for aspect_ratio in aspect_ratios:
            tup = lookup_aspect_curve(curves_data, float(aspect_ratio))
            if tup is None:
                continue
            vp_array, vp_vs_ratio_array, por_array = tup
            valid_mask = (
                (~np.isnan(vp_array))
                & (~np.isnan(vp_vs_ratio_array))
                & (vp_array > 0)
                & (vp_vs_ratio_array > 0)
            )
            if np.sum(valid_mask) <= 1:
                continue
            por_valid_raw = por_array[valid_mask]
            vp_valid_raw = vp_array[valid_mask]
            rr_valid_raw = vp_vs_ratio_array[valid_mask]
            sort_idx = np.argsort(por_valid_raw)
            por_valid = por_valid_raw[sort_idx].astype(float)
            vp_valid = vp_valid_raw[sort_idx].astype(float)
            rr_valid = rr_valid_raw[sort_idx].astype(float)
            if por_valid.size < 2 or not np.all(np.diff(por_valid) > 0):
                continue

            vp_at = float("nan")
            rr_at = float("nan")

            if scipy_interp1d is not None:
                try:
                    f_vp = scipy_interp1d(
                        por_valid,
                        vp_valid,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    f_rr = scipy_interp1d(
                        por_valid,
                        rr_valid,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    if por_valid[0] <= tp <= por_valid[-1]:
                        vp_at = float(f_vp(tp))
                        rr_at = float(f_rr(tp))
                    elif tp < por_valid[0] and por_valid[0] - tp < 0.02:
                        vp_at = float(vp_valid[0])
                        rr_at = float(rr_valid[0])
                    elif tp > por_valid[-1] and tp - por_valid[-1] < 0.02:
                        vp_at = float(vp_valid[-1])
                        rr_at = float(rr_valid[-1])
                except Exception:
                    pass
            else:
                if por_valid[0] <= tp <= por_valid[-1]:
                    vp_i = _linear_interp_sorted(por_valid, vp_valid, tp)
                    rr_i = _linear_interp_sorted(por_valid, rr_valid, tp)
                    if vp_i is not None and rr_i is not None:
                        vp_at = vp_i
                        rr_at = rr_i
                elif tp < por_valid[0] and por_valid[0] - tp < 0.02:
                    vp_at = float(vp_valid[0])
                    rr_at = float(rr_valid[0])
                elif tp > por_valid[-1] and tp - por_valid[-1] < 0.02:
                    vp_at = float(vp_valid[-1])
                    rr_at = float(rr_valid[-1])

            if (
                vp_at > 0
                and rr_at > 0
                and vp_at < 20.0
                and rr_at < 10.0
                and not np.isnan(vp_at)
                and not np.isnan(rr_at)
            ):
                vp_curve.append(vp_at)
                rr_curve.append(rr_at)

        if len(vp_curve) > 1:
            vp_arr = np.asarray(vp_curve)
            rr_arr = np.asarray(rr_curve)
            sort_ix = np.argsort(vp_arr)
            vp_sorted = vp_arr[sort_ix]
            rr_sorted = rr_arr[sort_ix]
            dm = (
                (vp_sorted >= 2.5)
                & (vp_sorted <= 9.0)
                & (rr_sorted >= 1.5)
                & (rr_sorted <= 2.3)
            )
            kwargs = dict(
                color=porosity_colors[i],
                lw=2,
                alpha=0.8,
                linestyle="--",
                label=f"Porosity = {porosities_percent[i]:.2f}%",
                zorder=7,
            )
            if np.sum(dm) > 1:
                ax.plot(vp_sorted[dm], rr_sorted[dm], **kwargs)
            else:
                ax.plot(vp_sorted, rr_sorted, **kwargs)


def setup_vp_vs_ratio_axes(ax, *, host_vp_dunite: float, host_vs_dunite: float) -> None:
    ax.set_xlabel("P-wave Velocity (km/s)", fontsize=12)
    ax.set_ylabel("Vp/Vs Ratio", fontsize=12)
    ax.set_title("Vp/Vs vs Vp Plot - Rock Database", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_xlim(3.0, 8.5)
    ax.set_ylim(1.6, 2.15)
    ax.set_yticks(np.arange(1.6, 2.16, 0.05))

    ratio = host_vp_dunite / host_vs_dunite
    ax.scatter(
        [host_vp_dunite],
        [ratio],
        c="yellow",
        marker="o",
        s=200,
        edgecolors="black",
        linewidths=2,
        label="Dunite",
        zorder=10,
    )
    ax.annotate(
        "Dunite",
        (host_vp_dunite, ratio),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", edgecolor="black", alpha=0.8),
        zorder=11,
    )
