"""
Kinzler (1997) isobaric batch melting of depleted spinel lherzolite (HZ-Dep1).

Port of the lherzolite melting logic in Kinzler & Grove (1992–1993) and
Kinzler (1997), following the structure of MAGMARS_v1.m (Collinet et al. 2015)
but using Earth lherzolite experimental regressions from ``lherzolite_reg.txt``.

Generates KKHS02 supplementary grid: P = 1–3 GPa, F = 0.02–0.20.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

_OXIDE_ORDER = (
    "SiO2",
    "TiO2",
    "Al2O3",
    "Cr2O3",
    "FeO",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "P2O5",
)

# Kinzler & Grove (1992b) Table 1a — HZ-Dep1 depleted MORB mantle (wt%).
HZ_DEP1_WT = {
    "SiO2": 45.18,
    "TiO2": 0.18,
    "Al2O3": 3.98,
    "Cr2O3": 0.38,
    "FeO": 7.88,
    "MgO": 38.33,
    "CaO": 3.58,
    "Na2O": 0.32,
    "K2O": 0.03,
    "P2O5": 0.02,
}

MM_VEC = np.array([60.084, 79.878, 50.981, 75.995, 71.8464, 70.937, 40.3045, 56.079, 30.990, 47.098, 70.972])

STOICHIO_CF = np.array(
    [
        [0.47, 0.8, 0.08, -0.35],
        [-0.9, 1.6, 0.08, 0.22],
        [-1.2, 1.9, 0.08, 0.22],
        [-1.8, 2.4, 0.08, 0.32],
    ]
)

START_MODE = np.array(
    [
        [46 - 12, 12, 2.5, 100 - 46 - 2.5],
        [46 - 27.2, 27.2, 2.5, 100 - 46 - 2.5],
        [46 - 34.2, 34.2, 2.5, 100 - 46 - 2.5],
        [46 - 38.4, 38.4, 2.5, 100 - 46 - 2.5],
    ],
    dtype=float,
)
START_MODE = START_MODE / START_MODE.sum(axis=1, keepdims=True) * 100.0

PRESSURE_KNOTS = np.array([1.0, 1.5, 2.0, 2.5])

GRID_P_GPA = (1.0, 1.5, 2.0, 2.5, 3.0)
GRID_F = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20)

_DATA_DIR = Path(__file__).resolve().parent / "data"
_REG_PATH = _DATA_DIR / "lherzolite_reg.txt"


@dataclass(frozen=True)
class RegressionCoeffs:
    t: np.ndarray
    sio2: np.ndarray
    cao: np.ndarray
    feo: np.ndarray
    mgo: np.ndarray
    cr2o3: np.ndarray


def _normalize_bulk(values: dict[str, float]) -> np.ndarray:
    vec = np.array([values[k] for k in _OXIDE_ORDER], dtype=float)
    return vec / vec.sum() * 100.0


def _weigh_obs(data: np.ndarray) -> np.ndarray:
    weights = data[:, 0]
    weights = np.round(weights / weights.min()).astype(int)
    rows = []
    for i, w in enumerate(weights):
        rows.extend([data[i]] * max(w, 1))
    return np.asarray(rows)


def _load_regression(path: Path = _REG_PATH) -> np.ndarray:
    raw = np.loadtxt(path)
    weighted = _weigh_obs(raw)
    ox = weighted[:, 3:14]
    ox = ox / ox.sum(axis=1, keepdims=True) * 100.0
    out = weighted.copy()
    out[:, 3:14] = ox
    return out


def _fit_regression(data: np.ndarray) -> RegressionCoeffs:
    p = data[:, 1]
    t = data[:, 2]
    sio2, tio2, al2o3, cr2o3, feo, mno, mgo, cao, na2o, k2o, p2o5 = data[:, 3:14].T
    mgn = (mgo / 40.3045) / (mgo / 40.3045 + feo / 71.8464) * 100.0

    t_pred = np.column_stack([np.ones_like(p), p, mgn, al2o3, na2o + k2o, p2o5])
    t_coeff = np.linalg.lstsq(t_pred, t, rcond=None)[0]

    femg_pred = np.column_stack([np.ones_like(p), p, p**0.3, mgn, al2o3, na2o, k2o, p2o5])
    feo_coeff = np.linalg.lstsq(femg_pred, feo, rcond=None)[0]
    mgo_coeff = np.linalg.lstsq(femg_pred, mgo, rcond=None)[0]

    sica_pred = np.column_stack(
        [
            np.ones_like(p),
            p,
            p**0.3,
            mgn,
            al2o3,
            na2o,
            k2o,
            p2o5,
            (na2o + k2o) / p,
            p2o5 * p,
        ]
    )
    sio2_coeff = np.linalg.lstsq(sica_pred, sio2, rcond=None)[0]
    cao_coeff = np.linalg.lstsq(sica_pred, cao, rcond=None)[0]

    cr_pred = np.column_stack([np.ones_like(p), t, p, al2o3, na2o + k2o, p2o5])
    cr2o3_coeff = np.linalg.lstsq(cr_pred, cr2o3, rcond=None)[0]

    return RegressionCoeffs(t_coeff, sio2_coeff, cao_coeff, feo_coeff, mgo_coeff, cr2o3_coeff)


def _interp_rows(table: np.ndarray, p_gpa: float) -> np.ndarray:
    if p_gpa <= PRESSURE_KNOTS[0]:
        return table[0].copy()
    if p_gpa >= PRESSURE_KNOTS[-1]:
        return table[-1].copy()
    for i in range(len(PRESSURE_KNOTS) - 1):
        p0, p1 = PRESSURE_KNOTS[i], PRESSURE_KNOTS[i + 1]
        if p0 <= p_gpa <= p1:
            w = (p_gpa - p0) / (p1 - p0)
            return (1.0 - w) * table[i] + w * table[i + 1]
    return table[-1].copy()


def _stoichio_at_p(p_gpa: float) -> np.ndarray:
    sto = _interp_rows(STOICHIO_CF, p_gpa)
    return sto / sto.sum()


def _start_mode_at_p(p_gpa: float) -> np.ndarray:
    return _interp_rows(START_MODE, p_gpa)


def _partition_matrix(p_gpa: float) -> np.ndarray:
    """Rows: Na, K, Al, P, Ti; cols: opx, cpx, spinel, oliv."""
    part = np.array(
        [
            [np.nan, np.nan, 1.1, 0.01],
            [0.003, 0.003, 0.8, 0.001],
            [np.nan, np.nan, np.nan, np.nan],
            [0.01, 0.01, 0.1, 0.08],
            [0.1, np.nan, 1.5, 0.03],
        ],
        dtype=float,
    )
    part[0, 0] = 0.0091 * p_gpa**2 + 0.0054 * p_gpa + 0.0168
    part[0, 1] = 0.0158 + 0.0746 * p_gpa
    part[2, 0] = 0.1614 * p_gpa + 0.0849
    part[2, 1] = 0.1877 * p_gpa + 0.0932
    part[2, 3] = 0.0037 * p_gpa + 0.0024
    part[4, 1] = -0.0533 * p_gpa + 0.1952
    if 1.0 <= p_gpa <= 2.5:
        part[2, 2] = p_gpa**2 - 3.1 * p_gpa + 3.2
    elif p_gpa < 1.0:
        part[2, 2] = -4.8 * p_gpa + 5.9
    else:
        part[2, 2] = 1.72 * p_gpa - 2.6
    return part


def _cpx_fraction(p_gpa: float, tot_px: float) -> float:
    if p_gpa <= 1.0:
        return -26.62 + 0.8395 * tot_px
    if p_gpa < 1.5:
        return (
            -560.8
            + 854 * p_gpa
            - 137.5 * p_gpa**2
            - 184.9 * p_gpa**3
            + 19.19 * tot_px
            - 36.21 * p_gpa * tot_px
            + 17.93 * p_gpa**2 * tot_px
        )
    if p_gpa < 2.0:
        return 191.2 - 267.4 * p_gpa - 7.638 * p_gpa**2 - 4.521 * tot_px + 6.696 * p_gpa * tot_px
    if p_gpa < 2.5:
        return 620.5 - 482.3 * p_gpa - 7.822 * p_gpa**2 - 13.96 * tot_px + 11.43 * p_gpa * tot_px
    return -645.0 + 14.86 * tot_px


def _mode_at_f(p_gpa: float, f_melt: float) -> np.ndarray:
    mode = _start_mode_at_p(p_gpa).copy()
    if f_melt <= 0.001:
        return mode
    tot_px = mode[0] + mode[1]
    cpx = _cpx_fraction(p_gpa, tot_px)
    mode[1] = max(cpx, 0.0)
    mode[0] = max(tot_px - cpx, 0.0)
    sto = _stoichio_at_p(p_gpa)
    mode = mode - sto * 100.0 * f_melt / max(1.0 - f_melt, 1e-9)
    mode = np.clip(mode, 0.0, None)
    s = mode.sum()
    return mode / s * 100.0 if s > 0 else mode


def _toplis_kd(p_gpa: float, t_c: float, wf: np.ndarray, mgn_sol: float) -> float:
    r_cst = 8.314
    t_k = t_c + 273.15
    p_bar = p_gpa * 1e4
    mol = wf / MM_VEC
    mol_pc = mol / mol.sum() * 100.0
    psy = (0.46 * (100.0 / (100.0 - mol_pc[0])) - 0.93) * (mol_pc[8] + mol_pc[9]) + (
        -5.33 * (100.0 / (100.0 - mol_pc[0])) + 9.69
    )
    sio2_pc_a = mol_pc[0] + psy * (mol_pc[8] + mol_pc[9])
    return float(
        np.exp(
            (-6766.0 / (r_cst * t_k) - 7.34 / r_cst)
            + np.log(max(0.036 * sio2_pc_a - 0.22, 1e-6))
            + (3000.0 * (1.0 - 2.0 * mgn_sol)) / (r_cst * t_k)
            + (0.035 * (p_bar - 1.0)) / (r_cst * t_k)
        )
    )


def _batch_incompatible(c0: float, d_bulk: float, p_b: float, f_melt: float) -> float:
    if f_melt >= 1.0 - 1e-9:
        return c0 / max(d_bulk, 1e-12)
    denom = d_bulk + (f_melt / (1.0 - f_melt)) * (1.0 - p_b)
    return c0 / max(denom, 1e-12)


def _melt_once(
    p_gpa: float,
    f_melt: float,
    bulk: np.ndarray,
    coeffs: RegressionCoeffs,
    kd_init: float = 0.35,
    max_iter: int = 8,
) -> np.ndarray:
    """Return melt oxides (wt%, 11-vector incl. MnO) at pressure and F."""
    f_melt = float(np.clip(f_melt, 1e-4, 0.999))
    mode = _mode_at_f(p_gpa, f_melt)
    sto = _stoichio_at_p(p_gpa)
    part = _partition_matrix(p_gpa)

    d_na = np.nansum(part[0] * mode / 100.0)
    d_k = np.nansum(part[1] * mode / 100.0)
    d_al = np.nansum(part[2] * mode / 100.0)
    d_p = np.nansum(part[3] * mode / 100.0)
    d_ti = np.nansum(part[4] * mode / 100.0)
    p_b_na = np.nansum(part[0] * sto)
    p_b_k = np.nansum(part[1] * sto)
    p_b_al = np.nansum(part[2] * sto)
    p_b_p = np.nansum(part[3] * sto)
    p_b_ti = np.nansum(part[4] * sto)

    c_na = _batch_incompatible(bulk[7], d_na, p_b_na, f_melt)
    c_k = _batch_incompatible(bulk[8], d_k, p_b_k, f_melt)
    c_al = _batch_incompatible(bulk[2], d_al, p_b_al, f_melt)
    c_p = _batch_incompatible(bulk[9], d_p, p_b_p, f_melt)
    c_ti = _batch_incompatible(bulk[1], d_ti, p_b_ti, f_melt)

    mgn_sol = (bulk[5] / 40.3045) / (bulk[5] / 40.3045 + bulk[4] / 71.8464)
    kd = kd_init
    mgn_liq = 1.0 / ((1.0 / mgn_sol - 1.0) / kd + 1.0) * 100.0

    wf = np.zeros(11)
    for _ in range(max_iter):
        t1 = float(coeffs.t @ np.array([1.0, p_gpa, mgn_liq, c_al, c_na + c_k, c_p]))

        femg = np.array([1.0, p_gpa, p_gpa**0.3, mgn_liq, c_al, c_na, c_k, c_p])
        sica = np.array(
            [1.0, p_gpa, p_gpa**0.3, mgn_liq, c_al, c_na, c_k, c_p, (c_na + c_k) / p_gpa, c_p * p_gpa]
        )

        c_sio2 = float(sica @ coeffs.sio2)
        c_cao = float(sica @ coeffs.cao)
        c_mgo = float(femg @ coeffs.mgo)
        c_feo = c_mgo * (bulk[4] / bulk[5]) / kd
        c_cr = float(np.array([1.0, t1, p_gpa, c_al, c_na + c_k, c_p]) @ coeffs.cr2o3)

        # MAGMARS assembly: normalize regression majors + batch Na, then mix minors.
        majors = np.array([c_sio2, c_al, c_feo, c_mgo, c_cao, c_na])
        majors = majors / majors.sum()
        wf = np.array(
            [
                majors[0],
                c_ti / 100.0,
                c_al / 100.0,
                max(c_cr, 0.0) / 100.0,
                majors[2],
                majors[2] / 38.7,
                majors[3],
                majors[4],
                c_na / 100.0,
                c_k / 100.0,
                c_p / 100.0,
            ]
        )
        wf = np.clip(wf, 0.0, None)
        wf = wf / wf.sum() * 100.0

        t2 = float(coeffs.t @ np.array([1.0, p_gpa, mgn_liq, wf[2], wf[8] + wf[9], wf[10]]))
        kd_new = _toplis_kd(p_gpa, t2, wf, mgn_sol)
        mgn_liq_new = 1.0 / ((1.0 / mgn_sol - 1.0) / kd_new + 1.0) * 100.0
        if abs(kd_new - kd) < 1e-4 and abs(mgn_liq_new - mgn_liq) < 0.05:
            break
        kd, mgn_liq = kd_new, mgn_liq_new

    return wf


_COEFFS: RegressionCoeffs | None = None


def _get_coeffs() -> RegressionCoeffs:
    global _COEFFS
    if _COEFFS is None:
        _COEFFS = _fit_regression(_load_regression())
    return _COEFFS


def _wf_to_oxides(wf: np.ndarray) -> dict[str, float]:
    """Map 11-vector (with MnO at index 5) to catalog oxide dict."""
    return {
        "SiO2": float(wf[0]),
        "TiO2": float(wf[1]),
        "Al2O3": float(wf[2]),
        "Cr2O3": float(wf[3]),
        "FeO": float(wf[4]),
        "MgO": float(wf[6]),
        "CaO": float(wf[7]),
        "Na2O": float(wf[8]),
        "K2O": float(wf[9]),
        "P2O5": float(wf[10]),
    }


def batch_melt_oxides(
    p_gpa: float,
    f_melt: float,
    source: dict[str, float] | None = None,
) -> dict[str, float]:
    """Isobaric batch melt major-element oxides (wt%) for depleted lherzolite."""
    bulk = _normalize_bulk(source or HZ_DEP1_WT)
    wf = _melt_once(p_gpa, f_melt, bulk, _get_coeffs())
    out = _wf_to_oxides(wf)
    out["H2O"] = 0.0
    return out


def generate_grid_rows(
    pressures_gpa: Iterable[float] = GRID_P_GPA,
    melt_fractions: Iterable[float] = GRID_F,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for p in pressures_gpa:
        for f in melt_fractions:
            ox = batch_melt_oxides(p, f)
            pid = f"{p:.1f}".rstrip("0").rstrip(".")
            fid = f"{f:.2f}".rstrip("0").rstrip(".")
            rows.append(
                {
                    "id": f"k97_grid_P{pid}_F{fid}",
                    "source": "Kinzler_1997_calculated",
                    "source_type": "pyrolite_depleted",
                    "P_melt_GPa": p,
                    "F_melt": f,
                    **{k: round(ox[k], 2) for k in _OXIDE_ORDER},
                    "H2O": 0,
                    "melt_style": "calculated",
                    "include_in_regression": "TRUE",
                    "notes": "Kinzler (1997) isobaric batch melt HZ-Dep1; kinzler1997_batch.py",
                }
            )
    return rows


def write_grid_csv(
    path: Path | str,
    pressures_gpa: Iterable[float] = GRID_P_GPA,
    melt_fractions: Iterable[float] = GRID_F,
) -> None:
    path = Path(path)
    fieldnames = [
        "id",
        "source",
        "source_type",
        "P_melt_GPa",
        "F_melt",
        *_OXIDE_ORDER,
        "H2O",
        "melt_style",
        "include_in_regression",
        "notes",
    ]
    rows = generate_grid_rows(pressures_gpa, melt_fractions)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "mantle_melts" / "kinzler1997_grid.csv"
    write_grid_csv(out)
    print(f"Wrote {len(GRID_P_GPA) * len(GRID_F)} rows to {out}")
