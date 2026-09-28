# PPP / PPS / PSS / PSP 正演

几何与 `../converse_fwd` 同一套速度：`H=2`、`Zc=5`、OBS `x=50`。偏移 `(0,8,…,28,32,…,60)` km（最大 60）；网格 `x=0–125`、`z=0–16`。

**PSP 与 converse 用同一份混合网格** `mixed.smesh`（盖层 Vp，面下 Vs=`3.40+0.12(z-5)`，转换面结点划给 Vs）。不要对 PSP 用壳幔 Vp 界面：插值会把快 P 漏进盖层底。

**PPP 用真 Vp** `vp.smesh`（界面是壳幔 Vp，无 `-U`）。不要把 Vs0 写进 PPP 的 P 网格，否则转换面变成慢层，中短偏移初至会贴水走。

PPS / PSS 走 `vp_psx.smesh` + `vs.smesh`（`-U`）：界面结点划给 Vs，盖层底 P 插值与 converse 相同。真双场不必 `-k`。

| raytype | 网格 | 含义 |
|---------|------|------|
| **0** | `vp.smesh` | 初至（零偏移是水柱） |
| **1** | `vp.smesh` `-F conv` | PPP：转换面反射（台下垂直往返盖层 P） |
| **6** | `mixed.smesh` | PSP：与 converse 相同 |
| **7** | `vp_psx`+`vs` `-U` | PPS：与双场 PSP 同一套整条弯曲；一星；台侧盖层 S，其余 P |
| **8** | `vp_psx`+`vs` `-U` | PSS：与双场 PSP 同一套整条弯曲；两星；台侧盖层 S，面下 S，炮侧 P |

```bash
python make_ps_fwd_case.py

tt_forward -Mvp.smesh -Ggeom_ppp.dat \
  -N8/8/0.8/8/1e-4/1e-5 -Rrays_ppp.dat > syn_ppp.dat

tt_forward -Mvp.smesh -Ggeom_ppr.dat -Fconv.refl \
  -N8/8/0.8/8/1e-4/1e-5 -Rrays_ppr.dat > syn_ppr.dat

tt_forward -Mmixed.smesh -Ggeom_psp.dat -Xconv.refl \
  -N8/8/0.8/8/1e-4/1e-5 -Rrays_psp.dat > syn_psp.dat

tt_forward -Mvp_psx.smesh -Uvs.smesh -Ggeom_psx.dat -Xconv.refl -Bseafloor.refl \
  -N8/8/0.8/8/1e-4/1e-5 -Rrays_psx.dat > syn_psx.dat

python make_ps_fwd_case.py --merge
python check_ps_fwd.py --no-show
```

`check_rays.png` 底图是 `mixed.smesh`，CPT 与 converse 相同。P **蓝**、S **粉**。

折合 PSP 工区见 `../converse_fwd`。第二步 PPS+PSS 反 Vs 见 `../ps_inv`。
