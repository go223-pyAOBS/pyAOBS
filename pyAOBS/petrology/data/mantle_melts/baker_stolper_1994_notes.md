# Baker & Stolper (1994) — catalog 录入说明

## 文献（KKHS02 汇编用的实际论文）

- **Baker, M. B., & Stolper, E. M. (1994).** Determining the composition of high-pressure mantle melts using diamond aggregates. *Geochimica et Cosmochimica Acta*, **58**(13), 2811–2827.  
  DOI: [10.1016/0016-7037(94)90116-3](https://doi.org/10.1016/0016-7037(94)90116-3)

> 注：KKHS02 参考文献中另有一条 *J. Geology* 102:223「reverse modeling」标题；**熔体成分以 GCA 本文 Table 2 为准**（Falloon et al., 1999 亦引用此表 runs **55T, 24, 26**）。

## 已写入 `catalog.csv` 的 3 点

| id | Run | T (°C) | P (GPa) | 角色 |
|----|-----|--------|---------|------|
| `bs94_run55T` | 55T | 1270 | 1.0 | 较低温端 / 近固相线区（**F 待核对**） |
| `bs94_run24` | 24 | 1330 | 1.0 | 中间温度 |
| `bs94_run26` | 26 | 1390 | 1.0 | 较高温；Falloon (1999) 认为更接近 **MM-3 平衡熔体** |

### 氧化物来源

- **Falloon, D. H., et al. (1999).** Peridotite melting at 1.0 and 1.5 GPa: an evaluation of techniques using diamond aggregates and mineral mixes for mantle melt fractionation studies. *J. Petrology*, **40**(9), 1343–1375. **Table 1**（标明 *Glass compositions from runs 55T, 24 and 26 are from Baker & Stolper (1994)*）。

### 请你用原文核对的两项

1. **`F_melt`**：打开 Baker & Stolper (1994) **Table 2**（或文中 wt% melt / melt fraction 列），替换 catalog 中暂定值（55T: 0.06, 24: 0.10, 26: 0.12）。
2. **`K2O`**：原文 Table 2 若给出 K₂O，替换当前占位 **0.1 wt%**，并重新归一化主量至 100%。

## 抄表时对照列（原文 Table 2）

```text
Run | T(°C) | P(10 kbar=1 GPa) | wt% melt 或 F | SiO2 TiO2 Al2O3 FeO* MgO CaO Na2O K2O ...
```

- 压力：**10 kbar = 1.0 GPa**（全部 Baker 1 GPa 实验）
- `melt_style` = `diamond_aggregate_trap`
- `source_type` = `pyrolite_fertile`（MM-3 富集地幔橄榄岩）
- `H2O` = 0（名义干实验；Falloon 讨论低温点可能有微量水）

## 第三点（可选，不进主回归）

若需要覆盖 **近固相线高 SiO₂**（~57 wt% @ F≈0.02），来自后续 **Baker et al. (1995) Nature** / Hirschmann 等压实验，**另开 id**（非本次 GCA Table 2 三 run）。

## 下一步（catalog 清单）

- [ ] 核对 Baker Table 2 → 更新 `F_melt`、K₂O  
- [x] Kinzler & Grove (1992) ×2 → 见 [`kinzler_grove_1992_notes.md`](kinzler_grove_1992_notes.md)（**核对 Table 10**）  
- [ ] Kinzler & Grove (1993) ×2  
- [ ] Kinzler (1997) 实验点 + **kinzler1997_grid.csv**（程序生成）
