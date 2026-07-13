# Katz (2003) Fig.7 — 实验 F(P,T) 与固相线对比

Fig.7：**P–T 平面**上全部无水橄榄岩熔融实验，散点颜色 = 熔融分数 **F**；叠加 **Katz (2003) 选用固相线**（Table 2 式 4）与 **Hirschmann (2000)** 固相线。

## 固相线关系

| 曲线 | A1 (°C) | A2 | A3 | 说明 |
|------|---------|----|----|------|
| Hirschmann (2000) | 1120.7 | 132.9 | −5.1 | 回归固相线 |
| Katz (2003) eq. (4) | **1085.7** | 132.9 | −5.1 | **A1 低 35°C**，更好拟合低 F 实验 |

## 文件

| 文件 | 用途 |
|------|------|
| `experiments.csv` | Fig.7 用 (P, T, F, study_id) |
| `../katz2003_fig6/` | Fig.6 子集 + Jaques Table 4/5 源表 |

## 构建与复现

```bash
py -3.11 petrology/validation/import_katz2003_fig6_esi.py --extract-jaques
py -3.11 petrology/validation/import_katz2003_fig7_data.py --build
py -3.11 petrology/validation/import_katz2003_fig7_data.py --stats
py -3.11 petrology/validation/reproduce_katz2003_fig7.py
```

## 数据状态

当前 `experiments.csv` 由 **Fig.6 种子点 + Jaques 全压点** 合并去重（约 50+ 点），**尚未**与 Katz ESI 完整无水实验库逐点核对。原文 caption：Hirschmann 固相线 ±10°C 内 **29** 点、平均 **F≈10 wt%**、**5** 点 F=0 — 需 ESI Table S2 补全后 `--stats` 才可与原文一致。
