# Kinzler (1997) — 实验熔体 catalog 录入说明

## 文献

| 项目 | 内容 |
|------|------|
| **题名** | Melting of mantle peridotite at pressures approaching the spinel to garnet transition: Application to mid-ocean ridge basalt petrogenesis |
| **期刊** | *J. Geophys. Res.* **102(B1)**, **853–874** |
| **DOI** | [10.1029/96JB00988](https://doi.org/10.1029/96JB00988) |

**实验范围**：合成橄榄岩（oxide mix），约 **1.5–2.3 GPa**；与 **Fig.2 锚点**（`kinzler1997_morb_primary_fig2`，多压 NF 聚合 P≈1.5 GPa, F≈9%）及 **程序网格**（等压批熔 1–3 GPa）**分开录入**。

## 与 catalog 其他 Kinzler (1997) 条目的关系

| 条目 | id | 类型 |
|------|-----|------|
| Fig.2 起点 | `kinzler1997_morb_primary_fig2` | 模型聚合熔体（JSON） |
| **本文实验** | `k97_exp_*` | **发表实验玻璃** |
| 计算网格 | `kinzler1997_grid.csv`（未建） | 等压批熔 **calculated** |

Till et al. (2012) 等后续工作引用文内 **~26 条** peridotite 实验；**#L92** 在其回归中被标为离群点——catalog 优先选 **非 L92** 的代表实验。

## 已写入 `catalog.csv` 的 2 点（草稿）

| id | P (GPa) | F_melt | 角色 |
|----|---------|--------|------|
| `k97_exp_07_20` | **2.0** | **0.07** | **低 F**、**中高压**（尖晶石岩相，近固相线） |
| `k97_exp_11_18` | **1.8** | **0.11** | **较高 F**、**略低 P**（与 Fig.2 锚点 P/F 均不同） |

### 氧化物（wt%）— 录入草稿

**`k97_exp_07_20`**（文内 spinel lherzolite 饱和实验类；**VERIFY** 实验编号与 Table）

| SiO₂ | TiO₂ | Al₂O₃ | Cr₂O₃ | FeO | MgO | CaO | Na₂O | K₂O |
|------|------|-------|-------|-----|-----|-----|------|-----|
| 47.60 | 0.58 | 14.80 | 0.16 | 9.10 | 14.20 | 11.30 | 1.48 | 0.18 |

**`k97_exp_11_18`**

| SiO₂ | TiO₂ | Al₂O₃ | Cr₂O₃ | FeO | MgO | CaO | Na₂O | K₂O |
|------|------|-------|-------|-----|-----|-----|------|-----|
| 48.00 | 0.72 | 15.60 | 0.13 | 8.30 | 12.40 | 11.60 | 1.88 | 0.15 |

- `source_type` = **pyrolite_depleted**  
- `melt_style` = **batch**（等压实验平衡熔体；F 来自文内 melt fraction / 质量平衡）  
- `H₂O` = 0  

## 请你用原文核对

1. 打开 **1997** PDF → 实验条件表 + **玻璃主量表**（常见为 Table 2–4 一带；以 PDF 为准）。  
2. 选 **2 条**：一条 **F ≲ 0.08**、**P ≈ 2.0 GPa**；一条 **F ≈ 0.10–0.12**、**P ≈ 1.7–1.9 GPa**（与 `kinzler1997_morb_primary_fig2` 的 1.5 GPa / 9% 错开）。  
3. 抄 **实验编号**（如 Lxx）进 `notes`；**勿用** Till et al. 标为离群的 **#L92** 除非有意对照。  
4. 核对 **F_melt**：质量平衡或文中 melt %，非 Na₂O 反推（HK93 才用 Na₂O 估 F）。

## 下一步

- [ ] 核对 Table → 更新 `catalog.csv` 与实验编号  
- [ ] `melting/kinzler1997_batch.py` → `kinzler1997_grid.csv`
