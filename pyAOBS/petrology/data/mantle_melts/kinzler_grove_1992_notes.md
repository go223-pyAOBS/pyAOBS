# Kinzler & Grove (1992) — catalog 录入说明

## 文献（KKHS02 熔体库来源之一）

KKHS02 引用 **Kinzler & Grove (1992)** 为橄榄岩熔融与 **多压近分离（near-fractional）聚合熔体**；实际为 **JGR 同年两篇**（非 *Am. Mineral.* 789–804，清单旧引用有误）：

| 篇 | 题名 | JGR | DOI |
|----|------|-----|-----|
| **1992a** | Primary magmas of mid-ocean ridge basalts, **1. Experiments and methods** | 97(B5), 6885–6906 | [10.1029/91JB02840](https://doi.org/10.1029/91JB02840) |
| **1992b** | Primary magmas of mid-ocean ridge basalts, **2. Applications** | 97(B5), 6907–6926 | [10.1029/91JB02841](https://doi.org/10.1029/91JB02841) |

**catalog 用的聚合熔体成分在 1992b**（三角熔融区、HZ-Dep1 等源区），**不是** 1992a 的 MORB 玄武岩实验玻璃表。

## 已写入 `catalog.csv` 的 2 点

| id | P_mean (GPa) | F_melt | 角色 |
|----|--------------|--------|------|
| `kg92_nf_05_10` | **1.0** | **0.05** | **低 F、~1 GPa**（低压端） |
| `kg92_nf_15_20` | **2.0** | **0.15** | **较高 F、~2 GPa**（中高压端） |

### 熔融路径（1992b Table 10 类）

| id | 压力路径 | 总 F | 说明 |
|----|----------|------|------|
| `kg92_nf_05_10` | **1.1 → 0.9 GPa** | 5% | 三角区近分离聚合；`P_melt` 取路径平均 **1.0 GPa** |
| `kg92_nf_15_20` | **2.4 → 1.4 GPa** | 15% | 较高熔融程度；`P_melt` 取 **2.0 GPa** |

### 氧化物（wt%）— 录入草稿

**`kg92_nf_05_10`**

| SiO₂ | TiO₂ | Al₂O₃ | Cr₂O₃ | FeO | MgO | CaO | Na₂O | K₂O |
|------|------|-------|-------|-----|-----|-----|------|-----|
| 49.85 | 0.48 | 17.55 | 0.11 | 7.15 | 10.45 | 10.35 | 2.55 | 0.12 |

**`kg92_nf_15_20`**

| SiO₂ | TiO₂ | Al₂O₃ | Cr₂O₃ | FeO | MgO | CaO | Na₂O | K₂O |
|------|------|-------|-------|-----|-----|-----|------|-----|
| 46.95 | 0.55 | 14.85 | 0.14 | 9.05 | 14.20 | 11.55 | 1.65 | 0.25 |

- `source_type` = **pyrolite_depleted**（1992b **HZ-Dep1** / DMM 型亏损源区）  
- `melt_style` = **aggregated_fractional**  
- `H₂O` = 0  

## 请你用原文核对

1. 打开 **1992b** PDF → **Table 10**（或文中 “triangular melting regime” 聚合熔体表；表号以 PDF 为准）。  
2. 按 **HZ-Dep1**（depleted MORB mantle）行，找到与上表 **压力路径 + 总 F** 最接近的两行，替换 `catalog.csv` 中主量。  
3. **不要与 `kinzler1997_morb_primary_fig2` 混淆**：后者是 Kinzler (1997) 的 MORB 原始熔体（P≈1.5 GPa, F≈9%），成分接近 1992b 中 **另一条** 路径，但 **id 与文献应分开**。

## 与 Kinzler (1997) 网格的关系

| 来源 | 条数 | 作用 |
|------|------|------|
| **K&G 1992**（本文件） | 2 | 发表实验/模型 **代表点** |
| **Kinzler (1997) 网格** | 30 | 程序生成，覆盖 1–3 GPa × F |
| **kinzler1997_morb_primary** | 1 | Fig.2 起点，非 Fig.3 回归必需 |

## 抄表步骤（1992b）

1. **Table 1a** → 确认源区为 **HZ-Dep1**（或文内 DMM 等价组成）。  
2. **Table 10**（聚合熔体）→ 选 **near-fractional / triangular** 行 → 抄 **SiO₂…K₂O**。  
3. **F_melt** = 表中 **total melt fraction**（如 5% → 0.05）。  
4. **P_melt_GPa** = 路径 **平均压力**（文内或表注；非路径上限 alone）。

## 下一步

- [ ] 核对 Table 10 → 更新 `catalog.csv`  
- [x] Kinzler & Grove **(1993)** ×2 → [`kinzler_grove_1993_notes.md`](kinzler_grove_1993_notes.md)  
- [ ] `kinzler1997_grid.csv`（程序生成）
