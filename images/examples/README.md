# Example figures for the root README

Curated copies of in-repo demo outputs. Each figure is embedded in the **matching feature section** of the root `README.md` (not a single gallery).

| File | README section | Source |
|------|----------------|--------|
| `tomo2d_inv_models.png` | 正演 / 反演 → TOMO2D | `modeling/tomo2d/example_water/crust_inv/check_inv_models.png` |
| `tomo2d_rays.png` | 正演 / 反演 → TOMO2D | `.../converse_fwd/check_rays.png` |
| `tomo2d_ttimes.png` | 正演 / 反演 → TOMO2D | `.../converse_fwd/check_ttimes.png` |
| `rayinvr_multiple_rays.png` | 正演 / 反演 → RAYINVR | `modeling/rayinvr/multiple_rays.png` |
| `imodel_vpvs_rocks_water_porosity.png` | 可视化 GUI → imodel | `vp_vs_ratio_dem` + 岩石库离线生成 |
| `imodel_vp_vs_lithology.png` | 可视化 GUI → imodel | 同上（Vp–Vs 岩性散点） |
| `imodel_vpvs_rocks_aspect.png` | 可视化 GUI → imodel | 同上（DEM aspect-ratio） |
| `imodel_dem_clip_voigt_hill.png` | 可视化 GUI → imodel | `utils/dem_clip_voigt_vs_hill.png` |
| `petrology_melting_schematic.png` | 岩石学 / LIP | `petrology/figures/` |
| `petrology_fig12_hvp.png` | 快速开始示例 5 + 岩石学 / LIP | `petrology/figures/` |
| `petrology_fig2_fc.png` | 岩石学 / LIP | `petrology/figures/` |
| `petrology_fig5_dvp.png` | 岩石学 / LIP | `petrology/figures/` |
| `petrology_fig15c.png` | 岩石学 / LIP | `petrology/figures/` |
| `field_deploy.png` | 野外站位工具 | `field/tests/` |
| `field_recovery.png` | 野外站位工具 | `field/tests/` |

Regenerate petrology panels with `pyAOBS/petrology/validation/reproduce_*.py` if needed.  
Regenerate imodel panels with Rocks GUI helpers (`visualization/imodel/gui/vp_vs_ratio_dem.py`).
