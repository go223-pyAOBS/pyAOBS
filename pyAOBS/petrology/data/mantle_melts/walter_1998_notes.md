# Walter (1998) — catalog 录入说明

## 文献（KKHS02 汇编；实际数据源）

- **Walter, M. J. (1998).** Melting of garnet peridotite and the origin of komatiite and depleted lithosphere. *Journal of Petrology*, **39**(1), 29–60.  
  DOI: [10.1093/petroj/39.1.29](https://doi.org/10.1093/petroj/39.1.29)

> KKHS02 参考文献常写作 *Contrib. Mineral. Petrol.* **132**, 396–410；**熔体主量与 P–F 关系以本篇 J. Petrology 实验为准**（3–7 GPa，KR4003 橄榄岩）。

## 已写入 `catalog.csv` 的 2 点（最小清单）

| id | Run | P (GPa) | F_melt | T (°C) | 在 Fig.3 中的角色 |
|----|-----|---------|--------|--------|------------------|
| `w98_run50_01` | **50.01** | **5.0** | **0.10** | 1680 | 高压、**低熔融程度** |
| `w98_run70_02` | **70.02** | **7.0** | **0.22** | ~1810 | **最高压** 端元 |

### 氧化物来源

- **Table 2**：压力、温度、相组合  
- **Table 3**：淬冷 **熔体（melt）** 主量（wt%）  
- **F_melt**：Table 3 脚注 **Melt %**（质量平衡），除以 100

### Run 50.01（`w98_run50_01`）

| 氧化物 | wt% |
|--------|-----|
| SiO₂ | 44.78 |
| TiO₂ | 1.26 |
| Al₂O₃ | 7.15 |
| Cr₂O₃ | 0.31 |
| FeO | 11.88 |
| MgO | 22.28 |
| CaO | 9.54 |
| Na₂O | 0.86 |
| K₂O | 0.60 |
| MnO | 0.20 |

### Run 70.02（`w98_run70_02`）

| 氧化物 | wt% |
|--------|-----|
| SiO₂ | 46.11 |
| TiO₂ | 0.66 |
| Al₂O₃ | 5.43 |
| Cr₂O₃ | 0.37 |
| FeO | 11.61 |
| MgO | 25.29 |
| CaO | 8.49 |
| Na₂O | 0.63 |
| K₂O | 0.30 |
| MnO | 0.20 |

- `melt_style` = **batch**（等压批熔实验）  
- `source_type` = **pyrolite_fertile**（KR4003）  
- `H₂O` = 0（名义干实验）

## 可选第 3 点（一般不必）

| Run | P | F | 说明 |
|-----|---|---|------|
| **30.10** | 3 GPa | 0.37 | 较低压；文献曾更正该点 SiO₂（见 Walter 1998 后续讨论 / SP-6 225–240） |
| **60.05** | 6 GPa | 0.41 | 中高压、较高 F |

仅当 Fig.3 在 **3 GPa** 角缺数据时再补 **一条**。

## 抄表时对照（原文）

1. **Table 2** → 选含 **melt** 的 run → 记 **P (GPa)**、**T (°C)**  
2. **Table 3** → 同 run 号 → 抄 **SiO₂…K₂O** 与 **Melt %**  
3. **不要抄** Table 4–7 的 cpx/opx/garnet 行（那是矿物，不是熔体）

## 与 Baker 的分工

| | Baker & Stolper (1994) | Walter (1998) |
|--|------------------------|---------------|
| **P** | 全在 **1 GPa** | **3–7 GPa** |
| **作用** | 低压 MORB 型成分 | **高压角**（KKHS02 库必需） |
