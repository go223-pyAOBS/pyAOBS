# PSP 反 Vs — 冻真 Vp，只反面下

与 `../pss_inv` / `../pps_inv` 同一套真模型。数据只用 raytype **6**。不跑 PPP。两步工区仍见 `../ps_inv`。

PSP：盖层 P、面下 S。只 6 时整层冻盖层，只反面下 Vs。

| 步 | 数据 | `-M` | 反什么 | 冻 |
|----|------|------|--------|----|
| ① | `geom_inv.dat` **6** | **真 Vp** | 面下 Vs | 真 Vp、水、**盖层** |

```bash
python make_psp_inv_case.py
# WSL: bash run_wsl.sh
python check_psp_inv.py --no-show
```
