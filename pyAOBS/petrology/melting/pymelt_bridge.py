"""
Optional pyMelt backend (user installs: ``pip install pyMelt``).

Maps pyAOBS Modern-track parameters (Tp, Φ, optional active λ) to pyMelt
``mantle.adiabaticMelt`` + ``geosettings`` crust thickness.

Note: pyMelt 2.x may break on SciPy >= 1.11 / 1.14 (removed
``scipy.misc.derivative``, ``scipy.integrate.trapz``). Call
``ensure_pymelt_scipy_compat()`` before ``import pyMelt`` (done automatically).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_pymelt_compat_applied = False


def ensure_pymelt_scipy_compat() -> None:
    """
    Patch pyMelt imports for newer SciPy / NumPy.

    - ``scipy.misc.derivative`` → finite-difference shim
    - ``scipy.integrate.trapz`` → ``numpy.trapezoid`` / ``numpy.trapz``
    """
    global _pymelt_compat_applied
    if _pymelt_compat_applied:
        return

    import numpy as np

    def _derivative(func, x0, dx=1.0, n=1, order=3, *args, **kwargs):
        if n != 1:
            raise NotImplementedError("n != 1 not supported in pymelt shim")
        if order not in (3, 5):
            order = 3
        # scipy.misc.derivative(..., args=(a, b)) — pull into positional for func
        extra = kwargs.pop("args", None)
        if extra is not None:
            args = tuple(extra) + tuple(args)

        def _call(x):
            return func(x, *args)

        if order == 3:
            return (_call(x0 + dx) - _call(x0 - dx)) / (2.0 * dx)
        xs = np.array([-2, -1, 0, 1, 2], dtype=float) * dx
        ys = np.array([_call(x0 + h) for h in xs])
        return float(np.dot(ys, np.array([1, -8, 0, 8, -1])) / (12.0 * dx))

    try:
        import scipy.misc as sm

        if not hasattr(sm, "derivative"):
            sm.derivative = _derivative  # type: ignore[attr-defined]
    except ImportError:
        pass

    try:
        import scipy.integrate as si

        if not hasattr(si, "trapz"):
            si.trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # type: ignore[attr-defined]
        # pyMelt / older code may also expect simps (removed in SciPy 1.12+)
        if not hasattr(si, "simps") and hasattr(np, "trapezoid"):
            si.simps = np.trapezoid  # type: ignore[attr-defined]
    except ImportError:
        pass

    _pymelt_compat_applied = True


def _import_pymelt():
    ensure_pymelt_scipy_compat()
    import pyMelt
    from pyMelt import geosettings, lithologies
    from pyMelt.mantle_class import mantle

    return pyMelt, geosettings, lithologies, mantle


def _instantiate_lithology(lith_module, *names: str):
    """
    Resolve lithology across pyMelt API versions.

    v1-style: ``katz.katz()``, ``pertermann.pertermann()``
    v2-style: ``katz.lherzolite()``, ``pertermann.g2()``
    """
    last_err: Exception | None = None
    for name in names:
        obj = getattr(lith_module, name, None)
        if obj is None:
            continue
        try:
            if isinstance(obj, type):
                return obj()
            if callable(obj):
                return obj()
            return obj
        except Exception as exc:  # pragma: no cover - version-specific ctor args
            last_err = exc
            continue
    msg = f"none of {names!r} found on {lith_module!r}"
    if last_err is not None:
        raise AttributeError(msg) from last_err
    raise AttributeError(msg)


def _km_to_lithosphere_pressure_gpa(b_km: float, *, rho_kg_m3: float = 3300.0) -> float:
    """Approximate lithospheric base pressure (GPa) from thickness."""
    h_m = float(max(b_km, 0.0)) * 1000.0
    return rho_kg_m3 * 9.81 * h_m / 1e9


def _pymelt_tc_km(tc: float | None) -> float:
    """Normalize pyMelt crust thickness to km across unit conventions."""
    if tc is None:
        return float("nan")
    val = float(tc)
    if val != val:  # NaN
        return val
    if val > 1.0e4:
        return val / 1.0e6
    if val > 500.0:
        return val / 1000.0
    return val


def _crust_from_spreading_centre(geosettings, col, *, p_lith_gpa: float):
    spreading_cls = getattr(geosettings, "spreadingCentre", None)
    if spreading_cls is None:
        return float("nan"), "missing_spreadingCentre"

    sc = spreading_cls(col, P_lithosphere=float(p_lith_gpa))
    h_km = _pymelt_tc_km(getattr(sc, "tc", None))
    return h_km, "geosettings.spreadingCentre"


def build_mantle(
    *,
    pyroxenite_frac: float = 0.0,
    peridotite_key: str = "katz_lherzolite",
    pyroxenite_key: str = "pertermann_g2",
    peridotite_h2o_wt: float = 0.0,
    pyroxenite_h2o_wt: float = 0.0,
):
    """
    Peridotite + optional pyroxenite via ``pymelt_lithology_adapter`` registry keys.

    Returns (mantle_object, lithology_names).
    """
    from .pymelt_lithology_adapter import instantiate_pymelt_lithology

    _, _, _, mantle_cls = _import_pymelt()
    phi = float(max(0.0, min(pyroxenite_frac, 0.99)))

    per = instantiate_pymelt_lithology(peridotite_key, h2o_wt=peridotite_h2o_wt)
    per_name = peridotite_key

    if phi <= 1e-9:
        man = mantle_cls([per], [1.0], names=[per_name])
        return man, [per_name]

    pyr = instantiate_pymelt_lithology(pyroxenite_key, h2o_wt=pyroxenite_h2o_wt)
    man = mantle_cls(
        [per, pyr],
        [1.0 - phi, phi],
        names=[per_name, pyroxenite_key],
    )
    return man, [per_name, pyroxenite_key]


@dataclass(frozen=True)
class PyMeltColumnResult:
    tp_c: float
    pyroxenite_frac: float
    active_lambda: float
    p0_gpa: float
    pf_gpa: float
    fbar: float
    f_max: float
    h_km: float
    f_per_lithology: dict[str, float]
    pooled_melt_wt: dict[str, float] | None
    pymelt_version: str
    crust_method: str
    column: Any  # pyMelt.meltingColumn


def _estimate_majors_from_column(column, man) -> dict[str, float] | None:
    """
    RMC-style pooled majors along the pyMelt column using Kinzler (1997) batch
    (same chemistry backend as REEBOX-core lithologies).
    """
    from petrology.melting.kinzler1997_batch import HZ_DEP1_WT
    from petrology.melting.lithology import G2_BULK_WT
    from petrology.melting.melt_chemistry import ChemistryBackend, melt_oxides

    _OX = ("SiO2", "TiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O", "K2O")
    source_map: dict[str, dict] = {}
    chem_map: dict[str, ChemistryBackend] = {}
    for name in man.names:
        key = name.lower()
        if "pyrox" in key or "g2" in key:
            source_map[name] = dict(G2_BULK_WT)
            chem_map[name] = "kinzler1997"
        else:
            source_map[name] = dict(HZ_DEP1_WT)
            chem_map[name] = "pmelts_klb1"

    lith = getattr(column, "lithologies", None)
    if not lith:
        return None

    p_arr = column.P
    n = len(p_arr)
    if n < 2:
        return None

    acc = {k: 0.0 for k in _OX}
    f_pool = 0.0
    for i in range(1, n):
        p = float(p_arr.iloc[i] if hasattr(p_arr, "iloc") else p_arr[i])
        step_mass = 0.0
        step_acc = {k: 0.0 for k in _OX}
        for j, name in enumerate(man.names):
            f_series = lith[name].F
            f0 = float(f_series.iloc[i - 1] if hasattr(f_series, "iloc") else f_series[i - 1])
            f1 = float(f_series.iloc[i] if hasattr(f_series, "iloc") else f_series[i])
            dfi = max(f1 - f0, 0.0) * float(man.proportions[j])
            if dfi <= 0.0:
                continue
            ox = melt_oxides(
                p,
                max(0.5 * (f0 + f1), 0.02),
                chemistry_backend=chem_map[name],
                source=source_map[name],
            )
            for k in _OX:
                step_acc[k] += dfi * float(ox.get(k, 0.0))
            step_mass += dfi
        if step_mass <= 0.0:
            continue
        for k in _OX:
            acc[k] += step_acc[k]
        f_pool += step_mass

    if f_pool <= 1e-9:
        return None
    total = sum(acc.values())
    if total <= 0.0:
        return None
    return {k: 100.0 * acc[k] / total for k in _OX}


def forward_pymelt_column(
    *,
    tp_c: float,
    pyroxenite_frac: float = 0.0,
    active_lambda: float = 0.0,
    chi: float | None = None,
    pf_gpa: float | None = None,
    p0_gpa: float | None = None,
    d_p_gpa: float = -0.004,
    b_km: float = 0.0,
    align_geometry: bool = True,
) -> PyMeltColumnResult:
    """
    Adiabatic decompression melt column via pyMelt.

    Parameters
    ----------
    tp_c
        Mantle potential temperature (°C).
    pyroxenite_frac
        Mass fraction of pyroxenite in source (Φ).
    active_lambda
        Legacy active weight (χ−1). Ignored when ``chi`` is set and geometry is aligned.
    chi
        REEBOX active-upwelling factor (≥1). When set with ``align_geometry=True``,
        uses ``build_reebox_column`` for P0/Pf (same as REEBOX-core default).
    pf_gpa, p0_gpa
        Manual melt-column bounds (GPa). Overridden when ``chi`` alignment is active.
    b_km
        Pre-existing lithosphere thickness (km); feeds REEBOX geometry and pyMelt
        ``P_lithosphere``.
    align_geometry
        If True and ``chi`` is given, match REEBOX-core ``build_reebox_column`` P0/Pf.
    """
    pymelt, geosettings, _, _ = _import_pymelt()
    man, names = build_mantle(pyroxenite_frac=pyroxenite_frac)

    h_geom_km: float | None = None
    crust_method = "integration_fallback"
    pstart: float | None = p0_gpa
    pend: float | None = pf_gpa

    if chi is not None and float(chi) >= 1.0 and align_geometry:
        from .lithology import heterogeneous_source
        from .reebox_geometry import build_reebox_column

        liths = heterogeneous_source(pyroxenite_frac=pyroxenite_frac)
        rg = build_reebox_column(
            liths,
            tp_c=float(tp_c),
            b_km=float(b_km),
            chi=float(chi),
            n_isentropic_steps=48,
        )
        pstart = rg.p0_gpa
        pend = rg.pf_gpa
        h_geom_km = rg.h_km

    if pstart is None and pend is None:
        pend = 0.01
    elif pend is None:
        pend = max(float(pstart) - 0.5, 0.01) if pstart is not None else 0.01

    melt_kwargs: dict[str, Any] = {
        "Pend": float(pend),
        "dP": float(d_p_gpa),
        "ReportSSS": False,
    }
    if pstart is not None:
        melt_kwargs["Pstart"] = float(pstart)
        melt_kwargs["adjust_pressure"] = True

    col = man.adiabaticMelt(float(tp_c), **melt_kwargs)

    import numpy as np

    p_series = col.P
    p0 = float(p_series.iloc[0] if hasattr(p_series, "iloc") else p_series[0])
    pf = float(p_series.iloc[-1] if hasattr(p_series, "iloc") else p_series[-1])

    f_cols: dict[str, float] = {}
    calc = getattr(col, "calculation_results", None)
    for n in names:
        if calc is not None and n in calc.columns:
            f_cols[n] = float(calc[n].iloc[-1])
        elif n in getattr(col, "lithologies", {}):
            f_cols[n] = float(col.lithologies[n].F.iloc[-1])

    f_arr = np.asarray(col.F)
    f_total = float(f_arr[-1])
    f_max = float(np.nanmax(f_arr))
    fbar = float(np.nanmean(f_arr))

    p_lith_gpa = _km_to_lithosphere_pressure_gpa(b_km)

    # Crust thickness: spreadingCentre melt integration when possible; else REEBOX H if aligned.
    h_km = float("nan")
    try:
        h_km, crust_method = _crust_from_spreading_centre(
            geosettings,
            col,
            p_lith_gpa=p_lith_gpa,
        )
        if h_geom_km is not None:
            crust_method = f"{crust_method}+aligned_P"
    except Exception:
        pass

    if not (h_km == h_km):
        if h_geom_km is not None:
            h_km = float(h_geom_km)
            crust_method = "reebox_triangular_ref"
        elif float(active_lambda) > 0 and chi is None:
            try:
                h_pass, _ = _crust_from_spreading_centre(
                    geosettings,
                    col,
                    p_lith_gpa=p_lith_gpa,
                )
                h_km = h_pass * (1.0 + float(active_lambda))
                crust_method = "spreadingCentre_x_(1+lambda)"
            except Exception:
                pass
        if not (h_km == h_km):
            h_km = 30.0 * f_max * max(p0 - pf, 0.0) / max(p0, 0.1)
            crust_method = "melt_integral_proxy"
    majors = _estimate_majors_from_column(col, man)

    return PyMeltColumnResult(
        tp_c=float(tp_c),
        pyroxenite_frac=float(pyroxenite_frac),
        active_lambda=float(active_lambda),
        p0_gpa=p0,
        pf_gpa=pf,
        fbar=fbar,
        f_max=f_max,
        h_km=float(h_km),
        f_per_lithology=f_cols,
        pooled_melt_wt=majors,
        pymelt_version=str(getattr(pymelt, "__version__", "?")),
        crust_method=crust_method,
        column=col,
    )
