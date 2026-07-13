# 最小 melt catalog 清单（R2 Fig.3 / R3 Fig.5）

目标：**用最少人工录入** 支撑 Reproduction 首版验收，不必复现 KKHS02 每一个散点。

**规模目标**

| 类别 | 条数 | 录入方式 |
|------|------|----------|
| 程序生成网格 | **30** | `kinzler1997_grid.csv`（Kinzler 1997 等压批熔） |
| 发表实验（手工） | **12–18** | 本清单各文献代表点 |
| 已有 JSON | **1** | `kinzler1997_morb_primary.json`（Fig.2 锚点，可并入 catalog） |
| 可选对照（不进回归） | **1–2** | Hirose & Kushiro (1993) |
| **合计** | **~43–49** | MVP 足够；后补至 80+ 再抠 1σ |

**参考态（计算 Vp 时）**：600 MPa, 400°C（与 eq.(1)、Fig.5 一致）。

---

## A. 程序生成（优先，零手工抄表）

### Kinzler (1997) depleted pyrolite 等压批熔网格 — **已生成**

| 参数 | 最小网格 |
|------|----------|
| **P (GPa)** | 1.0, 1.5, 2.0, 2.5, 3.0（5 档） |
| **F** | 0.02, 0.05, 0.08, 0.10, 0.15, 0.20（6 档） |
| **点数** | 5 × 6 = **30** |

- 文件：[`kinzler1997_grid.csv`](kinzler1997_grid.csv)
- 实现：[`melting/kinzler1997_batch.py`](../../melting/kinzler1997_batch.py)
- 字段：`melt_style=calculated`, `source_type=pyrolite_depleted`, `include_in_regression=TRUE`

**验收**：MORB 附近 (P≈1 GPa, F≈0.1) norm Vp ≈ **7.1 km/s** — 运行 `validation/reproduce_fig3.py`（当前 `auto+auto` 实测约 **7.29 km/s**）。

---

## A2. R2 norm Vp + Fig.3（本步）

| 模块 | 路径 | 状态 |
|------|------|------|
| CIPW norm | `cipw.py` | pyrolite（vendored/pip）；fallback 内置 |
| HS 混合 | `mixing.py` | ✅ |
| 矿物 K,G,Vp | `minerals.py` | BurnMan SLB_2022；empirical 回退 |
| norm Vp | `norm_velocity.py` | ✅ @ 600 MPa, 400°C |
| eq.(1) | `vp_regression.py` | ✅ 论文系数 + 双 tanh 窗 |
| 合并 catalog | `data/load_catalog.py` | catalog + grid → 40 点 |
| Fig.3 | `validation/reproduce_fig3.py` | ✅ MVP 散点图 |
| Fig.5 | `validation/reproduce_fig5.py` | ✅ MVP ΔVp 包络图 |

```bash
# WSL：先验收后端
python -m petrology.validation.validate_backends

# Fig.3（默认 auto → pyrolite + burnman）
python petrology/validation/reproduce_fig3.py

# 强制只用 vendored 副本
python -m petrology.validation.validate_backends --vendored-only

# 对照 catalog 线性 surrogate（RMS 更低）：
python petrology/validation/reproduce_fig3.py --linear-surrogate

# Fig.5（MVP 参数化 FC 包络）
python petrology/validation/reproduce_fig5.py

# Step2 反演 MVP（V_LC + f_lower -> V_bulk 区间 -> 可行 (P,F)）
python -m petrology.validation.run_invert_mvp --v-lc 7.0 --f-lower 0.5
# 等效显式参数：
# python -m petrology.validation.run_invert_mvp --v-lc 7.0 --f-lower 0.5 --vp-bias -0.10 --vp-tol 0.05

# Step2 批处理（CSV 多观测点）
python -m petrology.validation.run_invert_batch petrology/data/mantle_melts/invert_batch_template.csv

# Step4 H–Vp 正演（Fig.12 MVP）
python -m petrology.validation.reproduce_fig12 --b-km 0 --chi 1,2,4,8,12

# MVP 回退（无 pyrolite/burnman 时）
python petrology/validation/reproduce_fig3.py --cipw-backend fallback --mineral-backend empirical
```

---

## B. 发表实验（手工，按优先级）

原则：每文献 **1–3 点**，覆盖 **P–F 角落**（低 F 高压、高 F 低压、中间 MORB），避免同一论文重复相似点。

### Kinzler & Grove (1993) — **已录入草稿**

- 文件：[`catalog.csv`](catalog.csv) → `kg93_nf_08_12`, `kg93_nf_10_25`
- 核对：[`kinzler_grove_1993_notes.md`](kinzler_grove_1993_notes.md)（**JGR 98:22339–22347**；1993 修正模型重算聚合熔体，非 1992b 页码）

### Kinzler (1997) 实验 — **已录入草稿**

- 文件：[`catalog.csv`](catalog.csv) → `k97_exp_07_20`, `k97_exp_11_18`
- 核对：[`kinzler_1997_experiment_notes.md`](kinzler_1997_experiment_notes.md)（**JGR 102(B1):853–874** 实验玻璃表；与 Fig.2 锚点 / 网格分开）

### Kinzler & Grove (1992) — **已录入草稿**

- 文件：[`catalog.csv`](catalog.csv) → `kg92_nf_05_10`, `kg92_nf_15_20`
- 核对：[`kinzler_grove_1992_notes.md`](kinzler_grove_1992_notes.md)（**JGR 97:6885–6906 / 6907–6926**，1992b Table 10 聚合熔体）

### Walter (1998) — **已录入草稿**

- 文件：[`catalog.csv`](catalog.csv) → `w98_run50_01`, `w98_run70_02`
- 核对：[`walter_1998_notes.md`](walter_1998_notes.md)（**J. Petrology 39:29–60**, Table 2–3）

### Baker & Stolper (1994) — **已录入草稿**

- 文件：[`catalog.csv`](catalog.csv) → `bs94_run55T`, `bs94_run24`, `bs94_run26`
- 核对：[`baker_stolper_1994_notes.md`](baker_stolper_1994_notes.md)（**F_melt、K₂O** 须对照 GCA Table 2）

### 必填（8–12 点即可启动 R2）

| # | 文献 | 建议取点 | 数 | 去何处找 | 回归 |
|---|------|----------|-----|----------|------|
| 1 | **Kinzler & Grove (1992)** | 多压 **aggregated / near-fractional** 熔体；含 **低 F、~1 GPa** 与 **较高 F、~2 GPa** 各 1 | **2** | **JGR 97:6907–6926 (1992b)** Table 10；HZ-Dep1 三角熔融区 | ✅ |
| 2 | **Kinzler & Grove (1993)** | 补充 **1 GPa 附近** 与 **>2 GPa** 各 1 代表 batch/aggregated melt | **2** | **JGR 98:22339–22347**；1993 修正重算 | ✅ **草稿** |
| 3 | **Baker & Stolper (1994)** | 地幔橄榄岩熔体：**低 F 高压** + **MORB 型** 各 1 | **2** | *J. Geology* 102: 223–239，实验熔体主量表 | ✅ |
| 4 | **Kinzler (1997)** | 文内 **实验点**（非模型网格）1–2 个；与 Fig.2 primary **不同** P/F 的 melt | **1–2** | JGR 102(B1): 853–874 | ✅ **草稿 ×2** |
| 5 | **Walter (1998)** | **高压** (≥2.5 GPa)、**低–中 F** peridotite melt | **1–2** | *Contrib. Mineral. Petrol.* 132: 396–410 | ✅ |

小计：**8–10 点**

### 建议补全（Fig.3 散点形态更稳，+4–6 点）

| # | 文献 | 建议取点 | 数 | 备注 |
|---|------|----------|-----|------|
| 6 | Kinzler & Grove (1992/93) | 再补 **F≈0.15–0.20** 或 **P≈3 GPa** 各 1 | **+2** | 填 Fig.3 高 F / 高压角 |
| 7 | Baker & Stolper (1994) | **最高 F** 或 **最硅质** 1 点 | **+1** | 成分空间上角 |
| 8 | Walter (1998) | **最高 P** 另 1 点 | **+1** | 高压端 |
| 9 | Kinzler (1997) | 实验表再 1 点 | **+1** | 与网格交叉验证 |

小计：**+4–5 点** → 发表实验 **12–15 点**

### 可选对照（不进主回归）

| 文献 | 取点 | 数 | 字段 |
|------|------|-----|------|
| **Hirose & Kushiro (1993)** | HK66 或辉石岩相关 melt 1 点 | **1** | `include_in_regression=false` |
| 同上 | pyrolite 对照 melt 1 点（若有） | **+1** | 同上，仅作 Fig.3 灰色 overlay |

---

## C. 已有文件（直接并入 catalog）

| 文件 | id 建议 | 用途 |
|------|---------|------|
| `kinzler1997_morb_primary.json` | `kinzler1997_morb_primary_fig2` | Fig.2/R3 锚点；`P=1.5`, `F=0.09`；**可 include_in_regression=true** |

---

## D. 每条记录必填字段

```text
id, source, source_type, P_melt_GPa, F_melt,
SiO2, TiO2, Al2O3, Fe2O3, FeO, MgO, CaO, Na2O, K2O, Cr2O3, P2O5, H2O,
melt_style, include_in_regression, notes
```

| 字段 | 规则 |
|------|------|
| `P_melt_GPa`, `F_melt` | 文献给定熔融 **压力/程度**；aggregated melt 用论文 **平均 P、F** |
| `melt_style` | `batch` \| `aggregated_fractional` \| `calculated` |
| `source_type` | 最小 catalog 统一 `pyrolite_depleted`（除 Hirose 标 `pyroxenite`） |
| 主量 | wt%，**归一化至 100%**；FeO/Fe₂O₃ 按原文，缺则 FeO 合并 |
| `H2O` | 干实验填 **0** |

模板文件：`catalog_template.csv`（仅表头）。

---

## E. 录入顺序（推荐）

```text
1. kinzler1997_morb_primary.json → catalog 第 1 行
2. 运行 Kinzler1997 网格脚本 → 30 行
3. Baker & Stolper (1994) 2 点（表最清晰，练手）
4. Kinzler & Grove 1992 2 点
5. Kinzler & Grove 1993 2 点
6. Kinzler (1997) 实验 1–2 点
7. Walter (1998) 1–2 点
8. 跑 reproduce_fig3.py；缺角再补点
9. 同一 catalog 跑 R3 / Fig.5
```

**不必等 catalog 满再写代码**：步骤 1–2 + norm_velocity 即可并行开发。

---

## F. 最小验收（catalog 完成后）

- [x] 总点数 ≥ **40**（当前 40）
- [x] P 覆盖 **1–3 GPa**；F 覆盖 **0.02–0.20**
- [x] norm Vp @ 600/400 散布 **约 6.8–7.8 km/s**（当前约 7.02–7.89，量级达标）
- [ ] eq.(1) 实验点子集 misfit **~0.05 km/s** 量级（当前约 0.27，待扩充实验点与后端一致化）
- [ ] Hirose 点 **未** 进入回归拟合
- [x] Fig.5：F_xl=0.7–0.8 时 ΔVp **~0.15 ± 0.02 km/s**（当前 0.145 ± 0.019）

补充说明：
- `auto + auto`（fallback CIPW + mixed mineral）下，`k97_grid_P1_F0.1` 为 **7.288 km/s**，与 eq.(1) 的 **7.262 km/s** 接近。
- 若终端出现 `pyrolite.mplstyle` 的 `legend.bbox_to_anchor` 警告，这是用户 matplotlib 样式兼容问题，不影响数值计算。
- `reproduce_fig5.py`（auto+auto）当前输出：**ΔVp(F=0.7–0.8) = 0.145 ± 0.019 km/s**，`V_UC` 范围约 **6.25–7.40 km/s**，与 Fig.5 目标量级一致。

---

## G. 暂不做（catalog_v2 / 突破阶段）

- Kogiso et al. (2004)、Davis et al. (2011)、Tuff & Green (2013) 等 **post-2002** 库
- 俯冲/含水/碳酸盐体系
- 每篇论文全实验 exhaustive 录入

见 README **Extended melt catalog (post-2002)** 与 **M2** 分源区扩展。

---

## H. 文献完整引用（便于检索）

1. Kinzler, R. J., & Grove, T. L. (1992a). Primary magmas of MORB, 1. *J. Geophys. Res.*, 97(B5), 6885–6906.  
   Kinzler, R. J., & Grove, T. L. (1992b). Primary magmas of MORB, 2. *J. Geophys. Res.*, 97(B5), 6907–6926.  
2. Kinzler, R. J., & Grove, T. L. (1993). Corrections and further discussion of the primary magmas of mid-ocean ridge basalts, 1 and 2. *J. Geophys. Res.*, **98**, 22339–22347. [doi:10.1029/93JB02164](https://doi.org/10.1029/93JB02164)  
3. Baker, M. B., & Stolper, E. M. (1994). *J. Geology*, 102(2), 223–239.（熔体成分以 GCA 为准）  
4. Kinzler, R. J. (1997). *J. Geophys. Res.*, **102(B1)**, 853–874. [doi:10.1029/96JB00988](https://doi.org/10.1029/96JB00988)  
5. Walter, M. J. (1998). *J. Petrology*, 39(1), 29–60.（KKHS02 常误引 CMP 132:396）  
6. Hirose, I., & Kushiro, I. (1993). *Contrib. Mineral. Petrol.*, 114(4), 511–522.（**exclude**）  
7. Korenaga et al. (2002) KKHS02 — Fig.3 对照用，**不必从正文反抄点**（以 1–5 + 网格为准）。
