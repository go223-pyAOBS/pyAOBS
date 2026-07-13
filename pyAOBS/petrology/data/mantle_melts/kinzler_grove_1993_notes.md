# Kinzler & Grove (1993) — catalog 录入说明

## 文献（KKHS02 Fig.3 灰色圆点来源之一）

| 项目 | 内容 |
|------|------|
| **题名** | Corrections and further discussion of the primary magmas of mid-ocean ridge basalts, 1 and 2 |
| **期刊** | *J. Geophys. Res.* **98**, **22339–22347**（非 98(B4):6907–6926；后者为 **1992b**） |
| **DOI** | [10.1029/93JB02164](https://doi.org/10.1029/93JB02164) |

**性质**：对 **1992a/1992b** 熔融参数化、分配系数与聚合熔体计算的 **勘误与补充讨论**；KKHS02 Fig.3 说明中的 *「Gray circles with dots … calculated by Kinzler and Grove [1993]」* 指 **修正后重新计算的多压近分离聚合熔体**，通常 **无独立高压实验表**。

## 与 1992b 的分工

| 来源 | catalog id | 角色 |
|------|------------|------|
| **K&G 1992b** Table 10 | `kg92_nf_*` | 1992 版三角区聚合熔体（待核对） |
| **K&G 1993** 重算 | `kg93_nf_*` | **1993 修正方程**下的代表聚合熔体；P/F 与 kg92 **错开** |

## 已写入 `catalog.csv` 的 2 点（草稿）

| id | P_mean (GPa) | F_melt | 路径 / 角色 |
|----|--------------|--------|-------------|
| `kg93_nf_08_12` | **1.2** | **0.08** | 三角 NF **1.3→0.9 GPa**；**~1 GPa 附近**、中低 F（区别于 kg92 的 5%@1.0） |
| `kg93_nf_10_25` | **2.5** | **0.10** | 三角 NF **2.6→1.8 GPa**；**>2 GPa**、中 F（区别于 kg92 的 15%@2.0） |

### 氧化物（wt%）— 录入草稿

**`kg93_nf_08_12`**

| SiO₂ | TiO₂ | Al₂O₃ | Cr₂O₃ | FeO | MgO | CaO | Na₂O | K₂O |
|------|------|-------|-------|-----|-----|-----|------|-----|
| 48.90 | 0.58 | 17.00 | 0.11 | 7.30 | 11.50 | 10.60 | 2.40 | 0.11 |

**`kg93_nf_10_25`**

| SiO₂ | TiO₂ | Al₂O₃ | Cr₂O₃ | FeO | MgO | CaO | Na₂O | K₂O |
|------|------|-------|-------|-----|-----|-----|------|-----|
| 47.40 | 0.52 | 14.50 | 0.14 | 8.90 | 13.60 | 11.40 | 1.55 | 0.22 |

- `source_type` = **pyrolite_depleted**（HZ-Dep1 / DMM 型，与 1992b 一致）  
- `melt_style` = **aggregated_fractional**（1993 修正模型重算，非新实验玻璃）  
- `H₂O` = 0  

## 请你用原文核对

1. 打开 **1993** PDF → 文中 **recalculated aggregated / polybaric near-fractional** 表或图（表号以 PDF 为准；可能在修正 1992b Table 10 的段落）。  
2. 按 **HZ-Dep1**（或文内等价 depleted 源区）找与上表 **压力路径 + 总 F** 最接近的两行，替换 `catalog.csv`。  
3. 若 1993 **仅公式勘误、无独立成分表**：可保留本 2 点作 **1993-corrected model** 占位，或在 notes 中注明「与 kg92 共用 1992b 路径、系数用 1993」。

## 抄表步骤

1. 确认源区 = **depleted pyrolite / HZ-Dep1**（与 1992b 一致）。  
2. 选 **triangular / near-fractional aggregated** 行（非等压批熔；批熔留给 `kinzler1997_grid.csv`）。  
3. **F_melt** = 总聚合熔融分数；**P_melt_GPa** = 路径平均压力。

## 下一步

- [ ] 核对 1993 PDF → 更新 `catalog.csv`  
- [ ] `kinzler1997_grid.csv`（程序生成）  
- [ ] `kinzler_1997_experiment_notes.md` 实验表核对
