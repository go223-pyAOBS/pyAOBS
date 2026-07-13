# Changelog

All notable changes to **pyAOBS** are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [SemVer](https://semver.org/) (major bumps may skip intermediate public majors when the product milestone warrants it).

## [3.0.0rc1] — 2026-07-13

First **3.0** release candidate. Scope is a usable install of the main OBS / workbench stack plus KKHS02 petrology GUI; research dumps and multi‑GB test data are **not** shipped in the sdist.

### Included

- **Workbench** — plugin shell for imodel / zplot / iphase / tomo2d / LIP petrology
- **Visualization** — imodel Qt, zplotpy (runtime), iphase
- **Model building** — Zelt `v.in` tools
- **Modeling** — TOMO2D / RAYINVR entry points (as previously in-tree)
- **Petrology (KKHS02)** — GUI + reproduction track; Step‑3 ΔVp defaults to **raw** W&L + Langmuir (`fig5_dvp_calibrate=False`); Fig.5 plot layout matches paper panels (a–d)
- **Packaging** — `setup.py` / `MANIFEST.in` prune venvs, article trees, Perple_X, ScienceDirect caches, zplotpy test data, OBEM bundles, etc.

### Not in 3.0.0rc1 (explicit)

- Pixel-level Fig.5 1σ matching the printed paper (raw catalog scatter is wider; optional empirical lift exists but is off by default)
- Full Modern melting track as a frozen API
- Perple_X binaries and paper-reproduction `article/` trees inside the installable package
- PyPI upload (this tag is for GitHub / local `pip install` first)

### Packaging notes

- Prefer: `pip install -e ".[gui-qt]"` from a clean clone, then `python -m pyAOBS.workbench.app`
- `import petrology` (bare) is supported via a path bootstrap when `pyAOBS` is imported; editable installs remain the recommended path for GUI work
- Vendored BurnMan / pyrolite trees remain available for offline petrology; prefer `pip install burnman` when convenient

### Upgrade from 0.2.0

- Version jump **0.2.0 → 3.0.0rc1** marks the workbench + petrology milestone (no public 1.x / 2.x series)
- Petrology H–Vp / Step‑2 scans use **uncalibrated** ΔVp by default (physical consistency)

## [0.2.0] — prior

- Early PyPI-oriented packaging of Zelt / visualization core (`setup.py` version 0.2.0)
