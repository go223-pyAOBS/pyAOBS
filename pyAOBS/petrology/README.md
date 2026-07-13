# pyAOBS.petrology — LIP 地壳地震学 petrology 工作流

本包用于复现与扩展 Korenaga et al. (2002) 中「由地壳地震结构约束 LIP 地幔熔融」的框架，并与 `pyAOBS.visualization.imodel` 的观测速度、厚度结果衔接。实现上采用 **双轨并行**：**Reproduction** 对齐 KKHS02 图件与数值；**Modern** 在保留同一反演接口的前提下升级物性与正演引擎。

**目标文献**

- Korenaga, J., Kelemen, P. B., Holbrook, W. S., & Sobolev, S. V. (2002). *Methods for resolving the origin of large igneous provinces from crustal seismology.* Journal of Geophysical Research: Solid Earth, 107(B9), ECV 1-1–ECV 1-27. [doi:10.1029/2001JB001030](https://doi.org/10.1029/2001JB001030)

下文简称 **KKHS02**。

---

## 双轨路线：Reproduction track vs Modern track

两条路线 **共用同一观测量与反演接口**（`H`、`V_LC`、`f_lower` → bulk 可行域 → 熔融参数），**分歧在中间正演引擎**。开发时可共用 `minerals.py`、`mixing.py`、`invert.py` 外壳，通过 `track` / `backend` 配置切换内核。

### 共用层（两轨不变）

KKHS02 的 **问题结构** 在 Modern 路线中仍保留：

| 共用概念 | 说明 |
|----------|------|
| **V_LC 为观测量** | 下地壳速度；不依赖全壳平均 |
| **Bulk Vp 为假想端元** | 地震不能直接反演 bulk |
| **边界反演** | `V_bulk ∈ [V_LC − ΔVp_max, V_LC]`；不假设单一就位模式 (Fig.1) |
| **ΔVp = V_cumulate − V_bulk** | FC 与 EQ/bulk 端元之差 |
| **H–Vp 解释** | 厚度 +（bulk 或 V_LC 上界）→ Tp、χ、b |
| **imodel 接口** | 导出 H、V(z)、V_LC、不确定性 |

```text
                    ┌─────────────────────────────────────┐
                    │  共用：imodel → H, V_LC, f_lower    │
                    │  共用：invert → bulk 可行域          │
                    │  共用：H–Vp 图 / 可行域带           │
                    └──────────────────┬──────────────────┘
                                       │
              ┌────────────────────────┴────────────────────────┐
              ▼                                                 ▼
    ┌─────────────────────┐                         ┌─────────────────────┐
    │  Reproduction       │                         │  Modern             │
    │  (对齐 KKHS02)      │                         │  (更自洽正演)       │
    └─────────────────────┘                         └─────────────────────┘
```

---

### Reproduction track（复现轨）

**目标**：数值与图件 **可对照 KKHS02**（Fig.2–5、11–12、15c）。

| 模块 | 方法 | 对应原文 |
|------|------|----------|
| **Vp 物性** | S&B (1994) 或 **BurnMan + Fig.2 锚点校准** | §2.1 |
| **熔体 → 固相 Vp** | **CIPW norm** (pyrolite) → HS | W&M (1989); Philpotts (1990) |
| **Vp(P,F) 快查** | **方程 (1)** 窗函数回归 | §2.2 |
| **熔体成分库** | Kinzler 系实验 + Kinzler (1997) 网格 | §2.2 |
| **分离结晶** | **W&L (1990)** + Langmuir (1992) 高压 | §2.1, §2.3 |
| **熔融 / H** | TK (1983) 固相线 + 线性 F(P) + **χ** | §4 |
| **验收** | ΔVp≈0.15 km/s @ F=0.7–0.8；格陵兰 χ>8 | Fig.5, §5 |

**计划配置**：`pipeline.run(..., track="reproduction")`

**优先阶段**：见下文 [建议实现阶段 — Reproduction](#建议实现阶段) R0–R4。

---

### Modern track（现代化轨）

**目标**：在 **不丢弃边界框架** 的前提下，用更现代 EOS、相平衡与熔融量学做正演；输出 **可行域 / 误差带**，而非单条标准曲线。

| 模块 | 升级相对 KKHS02 | 工具倾向 |
|------|-----------------|----------|
| **Vp 物性** | SLB_2022 等 **热力学自洽 EOS**；默认 BurnMan，无 S&B 锚点硬调 | BurnMan |
| **矿物组合** | **平衡矿物** 替代纯 CIPW 名义矿物（bulk / 熔体端元） | `equilibrate` / Perple_X；协议见 **[M2 对照协议](#modern-m2热力学-bulk-vp-与-cipw-对照协议)** |
| **分离结晶** | **热力学 FC 路径**（成分–压力依赖液相线） | 自研 W&L (`fc/wl1990.py`) + BurnMan / Pyrolite |
| **熔体成分** | (P,F,源区) → 成分 **正演**，方程 (1) 降为快速近似或 lookup | Katz / pMELTS / pyrolite |
| **熔融 / H** | **Katz** 或改进参数化；χ 保留；可选源区 heterogeneity (pyrolite) | `melting/katz.py` |
| **壳体** | **孔隙–速度** 校正（Mavko）；Vp–ρ 与 imodel 重力联合 | `porosity.py` |
| **反演** | **蒙特卡洛 / 可行域带**（H、V_LC 误差传播） | `invert.py` v2 |

**计划配置**：`pipeline.run(..., track="modern")`

**优先阶段**：见下文 M1–M5；**可与 Reproduction 交叉验证**（ΔVp 量级、χ–Tp 定性结论是否一致）。**M2 实施细节**见 **[Modern M2：热力学 bulk Vp 与 CIPW 对照协议](#modern-m2热力学-bulk-vp-与-cipw-对照协议)**。

---

### 两轨对照总表

| 环节 | Reproduction | Modern | 现代轨优势 |
|------|--------------|--------|------------|
| 单矿物 K,G,ρ | S&B 1994 / 校准 BurnMan | BurnMan SLB | 自洽、可外推 P–T |
| Bulk 固相组合 | CIPW norm | 热力学平衡 | 矿物比例更物理 |
| 堆晶 V_LC | W&L FC | 热力学 FC | 路径与 Fo/An 演化更细 |
| Vp(P,F) | 方程 (1) | 全正演 或 eq.(1) 作 surrogate | 少经验窗函数 |
| 1D 熔融 | 线性 F(P), TK 固相线 | Katz / 多阶段 F(P) | H、F̄ 更可靠 |
| 反演输出 | 标准 H–Vp 等值线 | **H–Vp 带** + 后验可行域 | 显式不确定性 |
| 主要风险 | 锚点依赖、CIPW 局限 | 参数多、可识别性下降 | 需约束源区与路径 |

---

### BurnMan vs S&B (1994)（Modern 默认 / Reproduction 可选）

| | S&B (1994) | BurnMan (SLB) |
|--|------------|---------------|
| **在 KKHS02 中** | 原文弹性汇编 | 未使用 |
| **Reproduction** | 优先或 BurnMan+校准 | MVP 实现 |
| **Modern** | 不推荐为主 | **默认** |
| **先进性** | 与 2002 论文数值一致 | 理论、固溶体、维护、深部扩展 |
| **ΔVp** | 论文标定 | 相对关系通常仍合理；需对比 W&L 路径 |

Modern 轨 **不追求** 复现 Fig.3 每一点，而追求 **forward model 自洽 + 不确定性带**。

---

### 共用 imodel 观测接口

两轨从 imodel 读取的字段一致（规划）：

| 字段 | 用途 |
|------|------|
| `H_igneous` | 全火成地壳厚度 → H–Vp 横轴 |
| `V_LC` | 下地壳谐和平均 Vp（校正到 petrology 参考态） |
| `f_lower` | 下地壳体积占比 → ΔVp 与 bulk 区间权重 |
| `sigma_H`, `sigma_VLC` | Modern：误差传播；Reproduction：可选 |

**温压参考态**：petrology 链使用 **600 MPa, 400°C**（或图件指定态），**不用** imodel `empirical_formulas` 的 200 MPa/25°C 岩性校正。

---

### 代码组织（双轨）

```text
petrology/
  config.py                 # track="reproduction"|"modern", backend 注册
  pipeline.py               # 统一入口；内部分派 track
  tracks/
    reproduction/           # eq1, wl1990, active_upwelling_classic
    modern/                 # forward_melt, equilibrate_fc, katz_melting, porosity
  # 共用
  minerals.py               # backend: sb1994 | burnman_slb
  mixing.py
  invert.py                 # bounds 共用；modern 扩展 uncertainty
  ...
```

**切换示例**：

```python
from pyAOBS.petrology.pipeline import run_lip_interpretation

# 复现轨：对齐 KKHS02
run_lip_interpretation(H_km=30, v_lc=7.0, track="reproduction")

# 现代轨：全正演 + 可行域
run_lip_interpretation(
    H_km=30, v_lc=7.0, sigma_h=2, sigma_vlc=0.03, track="modern"
)
```

---

### 何时选哪条轨

| 场景 | 建议 |
|------|------|
| 论文方法学对照、审稿、教学 | **Reproduction** |
| 新区 LIP、imodel 实际项目、发新方法 | **Modern**（或 Modern 为主、Reproduction 作 sanity check） |
| 仅实现 MVP | **Reproduction R0–R1**，再 fork **Modern M1** |
| 绝对 Vp 与实验室对比 | Modern + BurnMan |
| 证明“升级没跑飞” | Modern 与 Reproduction **ΔVp、χ 定性对比** |

---

## Paper mapping（摘要四步 ↔ 章节 ↔ 图 ↔ 代码）

原文摘要叙述的四步与实现模块的对应关系如下。**摘要顺序**为 1→2→3→4；**建议实现顺序**为 1→3→2（闭合边界）→4，与公式依赖一致。

### Step 1 — 实验熔体 → bulk Vp 与熔融参数 (P, F)

> *First, a quantitative relation between bulk crustal velocity and mantle melting parameters is established on the basis of data from mantle melting experiments.*

| 项目 | 内容 |
|------|------|
| **论文章节** | §2.2 Primary Mantle Melts and Melting Systematics |
| **核心输出** | **方程 (1)**：`Vp_bulk = f(P, F)`，参考态 600 MPa、400°C |
| **数据** | Kinzler & Grove (1992, 1993), Baker & Stolper (1994), Kinzler (1997), Walter (1998) 等；补充 Kinzler (1997) 等压批熔网格；**不用** Hirose & Kushiro (1993) 做主回归 |
| **方法** | 熔体主量 → **CIPW norm** → 矿物弹性性质 → **Hashin–Shtrikman** 混合 → norm-based Vp（详见 **[Norm-based Vp 计算链](#norm-based-vp-计算链21-矿物物理)**） |
| **物性** | Sobolev & Babeyko (1994) 矿物 Vp（实现默认 **BurnMan** Ol/Pl/Cpx + HS，必要时锚点校准） |
| **主要图件** | Fig.3（实验点 vs 回归）, Fig.4（与 KH95 对比） |
| **计划模块** | `data/mantle_melts/`, `cipw.py`, `minerals.py`, `mixing.py`, `norm_velocity.py`, `vp_regression.py` |
| **验收** | MORB 条件 (P≈1 GPa, F≈0.1) → Vp≈7.1 km/s；回归 1σ≈0.05 km/s |
| **详细说明** | **[Figure 3 / §2.2 熔体库与方程 (1)](#figure-3--22-熔体库与方程-1bulk-vp--p-f)** |

**本步不包含**：分离结晶、主动上涌、Tp/χ。

---

## Figure 3 / §2.2 熔体库与方程 (1)（bulk Vp ↔ P, F）

原文 §2.2 [13]–[16] 在 **bounding 框架** 内建立 **norm-based bulk Vp** 与 **地幔熔融参数 (P, F)** 的普遍关系，供 Step 4 **H–Vp 正演** 与反演使用。与 Fig.2 不同：此处 **参考态固定为 600 MPa、400°C**（全文 **reference state**），且对象是 **地幔熔体实验库**，不是单一 MORB 结晶路径。

### 目的（与 Step 2 的衔接）

> *To establish a general relation between the norm-based bulk crustal velocity and lower crustal velocity of igneous crust using the bounding approach … a reasonably wide range of mantle melt compositions needs to be considered.*

| 层次 | 内容 |
|------|------|
| **输入** | 多种 **地幔熔体主量成分** + 各自 **熔融压力 P**、**熔融程度 F** |
| **中间量** | 每条熔体 → CIPW → **norm-based Vp** @ reference state |
| **输出** | **方程 (1)**：`Vp_norm(P, F)` 经验回归（Fig.3） |
| **用途** | Step 4：`Vp_bulk_pred = f(P̄, F̄)`；**不**替代 Fig.2 的 FC 算 V_LC |

**bulk Vp 在此 ≡ 熔体 norm-based Vp**：假想该熔体 **100% 平衡固结** 的全壳速度（§2.1 Terminology）。

### 参考态（全局，与 Fig.2 区分）

| 参数 | 值 | 备注 |
|------|-----|------|
| **P_ref** | **600 MPa** | 0.6 GPa |
| **T_ref** | **400°C** | 673.15 K |
| **Vp 散布** | **6.8 – 7.8 km/s** | 编译实验点 norm Vp（Fig.3 纵轴） |

Fig.2 使用 **100 MPa, 100°C** 仅为图例展示；**方程 (1) 与 Fig.5 主体** 用 **600 MPa, 400°C**。

### 编译实验数据源

| 文献 | 内容 | 回归 |
|------|------|------|
| **Kinzler & Grove (1992, 1993)** | 橄榄岩熔融实验；含多压近分离聚合熔体 | ✅ 主回归 |
| **Baker & Stolper (1994)** | 地幔熔体 | ✅ |
| **Kinzler (1997)** | 熔融模型与实验 | ✅ |
| **Walter (1998)** | 高压熔融 | ✅ |
| **Hirose & Kushiro (1993)** | 部分辉石岩 HK66 等 | ❌ **排除**（Na₂O 估 F 可能偏高；源区非 pyrolite） |

计划表结构 `data/mantle_melts/catalog.csv`：

```text
source, P_GPa, F, SiO2, TiO2, Al2O3, FeO, MgO, CaO, Na2O, K2O, ...
include_in_regression, melt_style   # batch | aggregated_fractional | calculated
```

**最小录入清单**（~43–49 点、录入顺序、文献表位置）：[`data/mantle_melts/catalog_minimal_checklist.md`](data/mantle_melts/catalog_minimal_checklist.md)；空表头模板 [`catalog_template.csv`](data/mantle_melts/catalog_template.csv)。

### Kinzler [1997] 补充网格（数量占主导）

> *… calculated the composition of isobaric batch melts of depleted pyrolite mantle for pressures of 1–3 GPa and melt fraction ranging from 0.02 to 0.2, using the method of Kinzler [1997] … supplement composition-velocity systematics.*

| 参数 | 值 |
|------|-----|
| **源区** | depleted **pyrolite** |
| **模式** | **等压批熔**（isobaric batch） |
| **P** | **1 – 3 GPa** |
| **F** | **0.02 – 0.20** |
| **实现** | 移植 Kinzler (1997) 公式 **或** 预计算 CSV 网格 |

原文：补充点数量 **多于** 发表实验一个量级，但回归时 **实验点与补充点等权**（按 weighted misfit），以保证两者 misfit 均小。

### 回归方法 — 方程 (1)（KH95 思路的扩展）

> *… following the approach of Kelemen and Holbrook [1995] (KH95), the compressional wave velocities for mantle melts are related to their pressures and degrees of melting using multiple linear regression (Figure 3)*

- **因变量**：norm-based Vp @ **600 MPa, 400°C**  
- **自变量**：熔融 **压力 P (GPa)**、熔融程度 **F**  
- **形式**：分 **低 P/F** 与 **高 P/F** 两段 **双曲正切窗** `W_L`, `W_H`，避免高阶多项式振荡（§2.2 [13] 末）  
- **系数**：见 `data/korenaga2002_eq1.json`（生产用 **b1=−0.55**；正文印刷表 +0.55 为符号勘误，见 JSON `print_errata`）

**典型预测（验收）**

| 条件 | 方程 (1) Vp | 说明 |
|------|-------------|------|
| **MORB** P≈1 GPa, F≈0.1 | **~7.1 km/s** | §2.2 [17]；修正 b1 后 raw eq.(1) 即达此值 |
| 全库散布 | 6.8 – 7.8 km/s | 与直接 norm 计算对比 Fig.3 |

**精度**：补充网格 1σ≈**0.01 km/s**；发表实验 1σ≈**0.05 km/s**（§2.2 [14]）。

### 数据处理流程（Reproduction R2）

```text
catalog.csv + kinzler1997_grid.csv
        │  每条记录: (P, F, oxides, source)
        ▼
norm_velocity @ 600 MPa, 400°C  →  Vp_obs_i
        ▼
vp_regression.fit()  →  方程 (1) 系数（或直接使用 korenaga2002_eq1.json）
        ▼
validation/reproduce_fig3.py  →  观测 Vp vs 预测 Vp 散点图
        ▼
vp_regression.predict(P, F)  →  Step 4 / invert 快查
```

### 与 Fig.2、Fig.4 的关系

| 图 | 关系 |
|----|------|
| **Fig.2** | **单一样本**（Kinzler MORB primary）+ **结晶**；100 MPa, 100°C |
| **Fig.3** | **全熔体库** + **方程 (1)**；600 MPa, 400°C |
| **Fig.4** | 方程 (1) vs **KH95** 实验室岩石回归路径；说明 KKHS02 norm 路线更紧 |

### 与双轨的关系

| | Reproduction | Modern |
|--|--------------|--------|
| **Fig.3 / eq.(1)** | **必做** R2；严格对照 Fig.3 | 可选作 **surrogate**；主链可 `(P,F)→成分→equilibrate→BurnMan Vp` |
| **熔体库** | Kinzler 系 + 排除 Hirose | 可 + Katz/pMELTS 成分，对比 eq.(1) |
| **参考态** | 600 MPa, 400°C | 同左（与 imodel 校正链分离） |

### 实现检查清单（`reproduce_fig3.py`）

- [ ] `catalog.csv` 含各文献代表点；Hirose/HK66 **标记 exclude**  
- [ ] Kinzler (1997) 网格 P∈[1,3] GPa, F∈[0.02,0.2]  
- [ ] 每条 norm Vp @ **600 MPa, 400°C**；范围 **6.8–7.8 km/s**  
- [ ] 加载 `korenaga2002_eq1.json`；MORB (1 GPa, 0.1) → **~7.1 km/s**  
- [ ] 回归 misfit：实验 ~0.05、补充 ~0.01 km/s 量级  
- [ ] 可选：与 KH95 方程 (2)(3) 对比 Fig.4b 趋势  

---

## Terminology（§2.1 核心术语）

KKHS02 §2.1 引入若干**计算端元速度**；它们与地震反演直接输出的层速度不是同一概念。实现与反演时须严格区分符号，避免把 **下地壳观测 Vp** 误当作 **bulk Vp**。

### Norm-based velocity（标准矿物速度）

- **定义**：对给定全岩或熔体主量成分做 **CIPW norm**（Ol、Pl、Cpx、Qz 等名义矿物比例），将质量分数转为体积分数，再用单矿物弹性性质经 **Hashin–Shtrikman** 混合得到的 **Vp**（算法细节见 **[Norm-based Vp 计算链](#norm-based-vp-计算链21-矿物物理)**）。
- **假想性**：假设 **pristine、均匀混合、无孔隙** 的固相组合；不考虑蚀变、裂隙、定向组构。
- **用途**：为任意成分提供 **可重复、可对比** 的速度标尺；地幔熔体实验点（Step 1）与结晶路径上的熔体/固相（Step 3）均用同一套 norm + HS 链计算。

### Bulk crustal velocity（全壳 bulk 速度）

- **定义**：若 **原始地幔熔体以 100% 平衡结晶** 固结（无熔体分离、无上地壳孔隙主导效应），整个火成地壳对应的 **假想平均 Vp**。
- **与熔融参数的联系**：Step 1 的 **方程 (1)** 将 bulk Vp 与熔融 **压力 P、程度 F** 定量关联；Step 4 用 **P̄, F̄** 预测 **Vp_bulk_pred**。
- **假想性**：真实洋壳上地壳高度多孔、蚀变，**无法在地震模型中直接观测** “全熔体平衡结晶” 的整壳平均速度；不应简单用全球平均 **~6.9 km/s** 当作 bulk 的观测参考（§3 讨论裂隙孔隙后更明确）。

### 可观测 vs 不可观测

| 符号 / 名称 | 是否可直接由地震层析给出 | 说明 |
|-------------|--------------------------|------|
| **V_LC**（下地壳平均 Vp） | **是**（厚壳、深部相对可靠） | 反演与 imodel 导出的主要观测量 |
| **Norm-based Vp** | **否**（需全岩/熔体化学 + 计算） | 成分 → Vp 的桥梁 |
| **Bulk Vp** | **否**（假想端元） | 由 V_LC + FC 边界推断 **可行区间** |
| **全壳算术平均 Vp** | 部分可算 | KH95 做法；KKHS02 认为上壳孔隙使此法不可靠 |

### Cumulate vs equilibrium crystallization（堆晶 vs 平衡结晶）

原文：*The uncertainty in the estimation of parental melt composition from lower crustal velocity thus depends on the magnitude of the difference between the velocity of possible **cumulates** and the velocity of rocks produced by **equilibrium crystallization** of primary melts.*

| 端元 | 结晶模式 | 物理含义 | 速度角色 |
|------|----------|----------|----------|
| **Cumulate（堆晶）** | **分离结晶 FC**（W&L）：结晶固相从体系中移除 | 下地壳堆晶；通常更富 MgO | **V 最高** → 理论 **V_LC**；观测 bulk 的 **上界** |
| **Equilibrium bulk（平衡全晶）** | **平衡结晶 EQ** 或 primary melt 的 norm 全固相 | “全壳平衡”端元 | 接近 **bulk / Vp(P,F)**；**较低** |
| **Residual melt（残余熔体）** | FC 后剩余熔体成分 | 上地壳熔体演化端元 | Fig.5c–d；**不**用于 thick-LIP 主反演 |

**Cumulate** 在原文中的定义：*a rock that forms via partial crystallization of a melt, after which the remaining melt is removed from the system of interest*（§2.1）。

### 不确定性：ΔVp 与 bulk 可行区间

母熔体成分（经 bulk Vp → P, F）的不确定性，主要取决于 **堆晶与平衡端元 Vp 之差**：

```text
ΔVp(F_xl, P_fc) = V_LC_theory − V_bulk_theory
                = V_cumulate(F)  − V_equilibrium_bulk(F)
```

- **F_xl**：下地壳固相分数（结晶程度；Fig.5 常用 0.5–0.8）
- **P_fc**：结晶压力（100 / 400 / 800 MPa）
- 典型量级：F_xl ≈ 0.7–0.8 时 **ΔVp ≈ 0.15 km/s**（±0.02）；f_lower ≈ 0.5 时 bulk 可比 V_LC **最多低 ~0.20 km/s**

由 **观测下地壳速度** 约束 **bulk Vp 区间**（Step 2；就位过程不确定时不假设 Fig.1 某一模式）：

```text
V_bulk_upper = V_LC_obs
V_bulk_lower = V_LC_obs − ΔVp_max(F_xl, P_fc, composition)
```

- 纯 FC 给出下地壳速度的 **上限** 情景 → bulk 只能 **≤ V_LC**
- 不能从单一 V_LC **唯一** 确定母熔体；只能得到 bulk 的 **bounded range**
- 该区间再经 **方程 (1)** 映射到 **(P, F)**，并与 Step 4 的 **H–Vp** 正演对比

### 符号速查

| 符号 | 含义 |
|------|------|
| `V_norm` | Norm-based velocity（任意成分） |
| `V_bulk` | Bulk crustal velocity（平衡结晶端元 / 方程 (1) 输出） |
| `V_LC` | 理论或观测的下地壳速度；FC 累积堆晶 assemblage 的理论值亦记作 V_LC |
| `V_UC` | 理论 upper crust（残余熔体 norm Vp；Fig.5c–d） |
| `ΔVp` | V_LC − V_bulk（相对 bulk 的正偏差） |
| `f_lower` | 下地壳占全火成地壳的体积比例 |
| `F_xl` | 下地壳结晶固相分数 |

### 与 Paper mapping 的对应

| 术语 / 关系 | Step |
|-------------|------|
| Norm-based Vp、方程 (1)、V_bulk(P,F) | 1 |
| V_LC_obs → bulk 区间（概念） | 2 |
| FC → V_cumulate；EQ → V_bulk；ΔVp(F) | 3 |
| V_bulk_pred = f(P̄, F̄) 与 H 一起画 H–Vp 图 | 4 |

---

## Norm-based Vp 计算链（§2.1 矿物物理）

KKHS02 §2.1 [11] 段说明 **所有** norm-based / bulk / 结晶组合 Vp 的共用算法：Step 1（熔体实验回归）、Step 3（W&L 结晶固相）与 primary/residual 熔体假固相速度 **同一套流程**。

### 原文要点

> *Major crystallizing minerals from low-H₂O, primitive basaltic liquids at ≤1 GPa are olivine, plagioclase, and clinopyroxene, and the predicted crystallizing assemblages range from dunite to gabbro.*

| 要点 | 含义 |
|------|------|
| **矿物相** | 橄榄石 (Ol)、斜长石 (Pl)、单斜辉石 (Cpx)；低压低 H₂O 原始玄武质熔体 |
| **组合范围** | 从 **纯橄岩 dunite** 到 **辉长岩 gabbro**（随结晶程度 / W&L 路径变化） |
| **压力语境** | ≤1 GPa 为主；高压 Cpx 稳定性由 Langmuir et al. (1992) 在 FC 模块中处理 |

> *The elastic properties of minerals depend on temperature, pressure, and composition.*

- 单矿物 Vp 是 **(P, T, 成分)** 的函数：如 Fo（Ol）、An（Pl）、En/Di（Cpx）。
- **结晶组合**（W&L 输出 Fo/An/比例）与 **CIPW norm 名义矿物** 均需带成分调用物性库。

> *We use the compilation of Sobolev and Babeyko [1994] to calculate effective isotropic moduli and density … Hashin-Shtrikman bounds … For mafic mineral assemblages, the bounds are typically tighter than 0.03 km s⁻¹, so that the average of the upper and lower bounds is sufficient.*

| 步骤 | 原文 | 实现策略 |
|------|------|----------|
| 单矿物 K, G, ρ | Sobolev & Babeyko (1994) | **首选**：移植 S&B 参数化；**MVP**：BurnMan `olivine` / `plagioclase` / `clinopyroxene` Solution + Fig.2 锚点校准 |
| 多相混合 | Hashin–Shtrikman 上下界 | BurnMan `HashinShtrikmanUpper/Lower/Average` |
| 报告 Vp | HS 上下界 **算术平均** | 基性组合界差 &lt;0.03 km/s 时可只输出 `HS_average` |

> *We calculate hypothetical solid velocities for primary and residual liquids in a manner similar to that of White and McKenzie [1989]; the weight proportions of minerals and their compositions are calculated based on the CIPW norm … weight proportions are converted to the volume proportions using mineral densities … Hereafter we will refer to this velocity based on the CIPW norm as the ‘norm-based velocity’.*

**White & McKenzie (1989) 式流程**（熔体 / 全岩成分 → norm-based Vp）：

```text
主量 wt% (或氧化物)
    → CIPW norm (Philpotts 1990, Ch.6) → 名义矿物质量分数 + 成分 (Fo, An, …)
    → 各单矿物 ρ(P,T,成分) → 质量分数 → 体积分数
    → 各单矿物 K,G(P,T,成分) → Hashin–Shtrikman → Vp_norm
```

- **Primary / residual liquids**：对 **熔体成分** 做 CIPW，得到 **假想全固相** norm 组合的速度（并非熔体本身的 Vp）。
- **Crystallizing assemblages**（Step 3）：对 W&L 给出的 **实际结晶矿物比例与 Fo/An/Di** 走 **同一 HS 链**，不再经过 CIPW。

> *Although in principle the use of the CIPW norm may not be as accurate as thermodynamic calculations … norm-based velocity for bulk crustal composition can serve as a useful, easily reproduced reference to evaluate the influence of fractionation on lower crustal velocity.*

- **CIPW 的局限**：名义矿物 ≠ 热力学平衡组合（S&B 1994 可做 Gibbs 平衡，但 KKHS02 刻意选 **易复现** 的 norm 路线）。
- **设计意图**：重点是比较 **分离结晶前后 Vp 差（ΔVp）**，不是精确模拟每一矿物格子。

### 两条输入路径（同一 Vp 引擎）

| 输入来源 | 矿物比例与成分从哪来 | 用于 |
|----------|----------------------|------|
| **路径 A — CIPW** | pyrolite `CIPW_norm()` → 名义 Ol/Pl/Cpx/Qz | 熔体实验点、primary/residual melt、bulk Vp(P,F) 回归 |
| **路径 B — W&L** | FC/EQ 每步结晶相比例 + Fo/An/Di | 累积堆晶 V_LC、incremental 组合（Fig.2, Fig.5） |

```text
                    ┌── 路径 A: CIPW norm
  成分 ─────────────┤
                    └── 路径 B: W&L 结晶组合
                              │
                              ▼
              minerals.py   单矿物 K, G, ρ @ (P, T, Fo/An/…)
                              │
                              ▼
              mixing.py     HS_average → Vp (km/s)
                              │
                              ▼
              norm_velocity.py / assemblage_vp()
```

### 参考态（勿与 imodel 校正混淆）

| 用途 | P, T | 文献位置 |
|------|------|----------|
| 方程 (1) 回归、Fig.3/5 主体 | **600 MPa, 400°C** | §2.2 参考态 |
| Fig.2 示例曲线 | 100 MPa, 100°C | 图注单独说明 |
| Fig.6 实验室对比 | **1 GPa, 25°C** | §2.4 |
| imodel 岩性分类 | 200 MPa, 25°C 线性校正 | **不用于** 本 petrology 链 |

### 计划 API（`norm_velocity.py` / `minerals.py` / `mixing.py`）

```python
# 路径 A
vp_norm = norm_velocity_from_bulk_wt(
    oxides_wt, P_pa=600e6, T_k=673.15, cipw_backend="pyrolite"
)

# 路径 B
vp_asm = assemblage_vp_hs(
    phases={"olivine": (0.4, 0.85), "plagioclase": (0.35, 0.60), ...},
    P_pa=600e6, T_k=673.15,
    averaging="hashin_shtrikman_average",
)
```

### 验收（与 §2.1 一致）

- 基性三矿物组合：HS upper − lower **&lt; 0.03 km/s**
- Fig.2 MORB 起点：bulk norm Vp ~**7.17** km/s；FC 累积组合 **&gt;7.3** km/s
- 同一 `(P,T)` 下 dunite 端 Vp **&gt;** gabbro 端 Vp（结晶程度增加方向）

---

### Step 2 — 下地壳 Vp → bulk Vp 可行区间（就位过程不确定）

> 术语定义见上文 **Terminology（§2.1 核心术语）** 一节。
> *Second, we show how lower crustal velocity can be used to place bounds on the expected range of bulk crustal velocity, despite ambiguity in crustal emplacement processes.*

| 项目 | 内容 |
|------|------|
| **论文章节** | §2.1, §2.3；概念在 §2.1，数值在 §2.3 |
| **核心思想** | 不假设 Fig.1 中某一种壳体就位模式（sheeted sill、gabbro glacier、intrusion 等） |
| **观测量** | **下地壳平均 Vp**（`V_LC`），不是全壳平均 |
| **边界逻辑** | 纯分离结晶下堆晶 Vp 最高 → `V_bulk ≤ V_LC`；FC 给出相对 bulk 的最大负偏差 |
| **主要图件** | Fig.5（ΔVp vs V_LC）；反演时 Fig.15c |
| **计划模块** | `invert.py`（`bulk_vp_bounds(v_lc, f_lower, d_vp_fc)`） |
| **与 imodel** | 自模型取下地壳谐和平均 Vp、全壳厚度 H、下地壳占比 `f_lower` |
| **详细说明** | **[Figure 5 / §2.3](#figure-5--23-理论上下地壳速度与-bounding)** |

**闭合公式（与 Step 3 联立）**

```text
V_bulk_upper = V_LC_obs
V_bulk_lower = V_LC_obs − ΔVp(F_xl, P_fc)    # ΔVp 来自 FC 模型，Fig.5
```

---

### Step 3 — 多压分离结晶 → ΔVp(f_lower)

> *By modeling fractional crystallization processes at a range of crustal pressures, these bounds are derived as a function of the proportion of lower versus upper crust.*

| 项目 | 内容 |
|------|------|
| **论文章节** | §2.1（FC 定义 cumulate）, §2.3 Theoretical Upper and Lower Crustal Velocities |
| **结晶模型** | **Weaver & Langmuir (1990)**，**Langmuir et al. (1992)** 高压扩展 |
| **压力** | 100 / 400 / 800 MPa；可选 polybaric 800→100 MPa（Fig.2） |
| **模式** | **分离结晶 (FC)**：累积结晶组合 → `V_LC`（理论下地壳速度上限） |
| **对比** | 平衡结晶 (EQ) → 残余熔体 norm Vp（上地壳端元，Fig.5c–d） |
| **主要图件** | Fig.2（单起点 MORB 示例）, Fig.5（全熔体库 FC 扫掠） |
| **计划模块** | `fc/wl1990.py` + `fc/wl_partition.py` |
| **验收** | F_xl≈0.7–0.8：ΔVp≈0.15 km/s（±0.02）；f_lower≈0.5 时 bulk 最多低 ~0.20 km/s |
| **详细说明** | **[Figure 5 / §2.3](#figure-5--23-理论上下地壳速度与-bounding)** |

**与 CIPW 的分工**：W&L 驱动**结晶路径与堆晶比例**；CIPW 用于**熔体/全岩 norm-based Vp**（Step 1 同一套 HS 链）。

| **详细说明** | **[Figure 5 / §2.3 理论上下地壳速度与 bounding](#figure-5--23-理论上下地壳速度与-bounding)** |

---

## Figure 5 / §2.3 理论上下地壳速度与 bounding

原文 §2.3 [18]–[19] 把 **Step 1 熔体库** 与 **Step 2 边界反演** 连接起来：对**每条已发表地幔熔体**做 **完美分离结晶 (FC)**，得到 **理论下地壳 Vp** 与 **理论上地壳 Vp**，相对 **norm-based bulk Vp** 的偏差汇总为 **Fig.5**。这是 **摘要 Step 2 + Step 3 的数值核心**。

### 在整体流程中的位置

```text
Step 1 熔体库 → 每条 melt 的 norm-based bulk Vp（100% 固结假想）
        │
        ▼
Step 3 对同一库 @ P_fc = 100 / 400 / 800 MPa 做 perfect FC
        │
        ├─ 累积堆晶 assemblage Vp  →  V_LC_theory（理论下地壳）
        └─ 残余熔体 norm Vp        →  V_UC_theory（理论上地壳）
        │
        ▼
Step 2  ΔVp = V_LC_theory − V_bulk  （Fig.5）
        │
        ▼
反演   V_bulk_lower = V_LC_obs − ΔVp(F_xl, f_lower)
       V_bulk_upper = V_LC_obs
```

**与 Fig.2 的分工**：Fig.2 是 **单一样本**（Kinzler MORB primary）的 FC/EQ **教学曲线**；Fig.5 是对 **Step 1 全熔体库** 的 FC **统计包络**，供反演取 **ΔVp 典型值与散布**。

### FC 建模设置

| 项目 | 内容 |
|------|------|
| **输入** | Step 1 编译的 **全部发表地幔熔体**（与 Fig.3 同源，不含 Hirose 主回归时可仍用于 FC 扫掠 — 以论文为准） |
| **模式** | **Perfect fractional crystallization**（W&L + Langmuir 1992 高压规则） |
| **结晶压力** | **100、400、800 MPa**（三档等压 FC；Fig.5a–b 显示 **F≈0.7–0.8 时 ΔVp 对 P_fc 近不敏感**） |
| **报告态** | norm-based Vp @ **600 MPa, 400°C**（与 bulk 端、方程 (1) 一致；≠ Fig.2 的 100 MPa/100°C） |

### 理论下地壳速度（cumulate / FC assemblage）

> *We denote the velocity of the cumulative fractionated assemblage as **theoretical lower crustal velocity**.*

| 符号 | 含义 |
|------|------|
| **V_LC_theory** | FC 至固相分数 **F_xl** 时，**累积结晶组合**的 norm-based Vp |
| **V_bulk** | 同一起点熔体的 **norm-based bulk Vp**（未分离、100% 平衡固结假想） |
| **ΔVp** | **V_LC_theory − V_bulk**（Fig.5 纵轴；纯 FC 下堆晶 **快于** bulk → ΔVp **> 0**） |

**bounding 逻辑（Step 2）**

> *Subtracting this deviation from the observed lower crustal velocity provides a **lower bound** on the possible range of bulk crustal velocity …*

```text
V_bulk ∈ [ V_LC_obs − ΔVp_max ,  V_LC_obs ]
```

- **上界** `V_bulk = V_LC_obs`：极端情况下壳体几乎无熔体抽离、无孔隙，观测下地壳即 bulk。  
- **下界** `V_bulk = V_LC_obs − ΔVp`：最大 FC 抽离熔体后，bulk 最多比观测 **V_LC 低 ΔVp**。

**f_lower 的作用**：ΔVp 是 **单一路径** 上的 bulk vs 堆晶差；全壳平均还依赖 **下地壳占全壳比例** `f_lower`。原文例：**f_lower = 0.5** 时，bulk 最多可比 **V_LC_obs 低 0.20 km/s**。

### Fig.5 关键数值（验收）

| 条件 | ΔVp / 边界 | 说明 |
|------|------------|------|
| **F_xl = 0.7–0.8** | **~0.15 km/s**，1σ **±0.02 km/s** | 典型洋壳速度范围；**对 100/400/800 MPa 近独立**（Fig.5a–b） |
| **f_lower = 0.5** | bulk 最多低 **~0.20 km/s** | 相对 **V_LC_obs** 的下界 |
| **F_xl → 0**（低固相分数） | ΔVp **散布大** | 强依赖 **起点熔体成分** 与结晶路径 |
| **F_xl → 1**（完全固结） | ΔVp → **0** | 无残余熔体，累积组合趋近 bulk |

**物理直觉**：低 F 时仅橄榄石等早期相 → ΔVp 大且成分敏感（Fig.2 中 ~20% 固结后急降）；高 F 时堆晶成分趋同 → **ΔVp 收敛、散布减小** — 故反演在 **F_xl ≈ 0.5–0.8** 标定主边界（README Fig.2 节亦述）。

### 理论上地壳速度（residual melt / EQ 端元）

> *Similarly, the norm-based velocity for a **residual liquid composition** is denoted as the **theoretical upper crustal velocity** …*

| 项目 | 内容 |
|------|------|
| **V_UC_theory** | FC 至 **F_xl** 时 **残余熔体** 的 norm-based Vp |
| **可能范围** | **6.3 – 7.3 km/s**（Fig.5c–d） |
| **为何不用作反演主观测量** | 实际上地壳 **高孔隙、蚀变** → 测不到；且 **F 增大时偏差散布更大** |
| **策略含义** | 用 **演化熔体（如上地壳 lava 成分）** 反推 bulk **不可靠** → **下地壳 Vp 是最实用的 compositional 约束**（§2.3 [19] 末） |

### 与 Step 1、Step 4 的边界

| 步骤 | 与 Fig.5 关系 |
|------|----------------|
| **Step 1** | 提供 **同库** 的 **V_bulk** 与 melt 成分；Fig.5 的 bulk 端 **不是** 方程 (1)，而是 **各 melt 直接 norm 计算** |
| **Step 2** | Fig.5 的 ΔVp **直接喂给** `invert.py` |
| **Step 3** | Fig.5 **就是** Step 3 对全库的产出 |
| **Step 4** | **不经过 Fig.5**；H–Vp 用方程 (1) 的 **V_bulk(P̄,F̄)**，再与 Step 2 边界在 Fig.15c 等处对照 |

### 实现检查清单（`reproduce_fig5.py`）

- [ ] 对 catalog 每条 melt：W&L FC @ **100 / 400 / 800 MPa**  
- [ ] 累积堆晶 Vp 与 bulk Vp @ **600 MPa, 400°C**  
- [ ] Fig.5a–b：ΔVp vs **V_LC_theory** 或 vs F；F=0.7–0.8 带 **~0.15 ± 0.02 km/s**  
- [ ] **f_lower=0.5** 场景：bulk 下界 **~0.20 km/s** 低于 V_LC_obs  
- [ ] Fig.5c–d：残余熔体 V_UC **6.3–7.3 km/s**；高 F 散布增大  
- [ ] 三压曲线在 F=0.7–0.8 **近重合**  

---

## Figure 6–7 / §2.4 V–ρ 体系与温压导数（FC 建模的副产品）

原文 §2.4 [20]–[21] 说明：**Step 3 结晶建模** 除给出 ΔVp（Fig.5）外，还可导出 **pristine 火成岩的 Vp–ρ 体系** 及 **温压导数**，用于与实验室经验关系对比、把观测速度校正到标准态，以及 **Vp → 密度** 约束（重力异常；Korenaga et al., 2001）。

### 在整体流程中的位置

```text
Step 3 FC（同 Fig.5 库）
        │
        ├─ 主链 → ΔVp → Step 2 bounding → H–Vp（Step 4）
        │
        └─ 副链 → Fig.6 V–ρ @ 1 GPa, 25°C
                  Fig.7 ∂V/∂T, ∂V/∂P（有限差分）
                  → 可选：重力 / 密度反演
```

| 性质 | 说明 |
|------|------|
| **是否 H–Vp 反演必需** | **否** — 主链 Step 1–4 不依赖 Fig.6–7 |
| **依赖** | 与 Fig.2/5 相同的 **W&L FC + S&B/BurnMan 单矿物 + HS 混合** |
| **输入** | 全库 **parental melt** + 各 **FC 累积组合（fractionated assemblage）** |
| **输出** | 标准态 **(Vp, ρ)** 散点；**dVp/dT、dVp/dP**；火成岩 **V–ρ 容许域** |

### Fig.6 — 速度–密度散点（实验室对比态）

> *The compressional wave velocities and densities of fractionated assemblages, at a pressure of **1 GPa** and a temperature of **25°C**, are plotted in Figure 6, together with those for parental melt compositions.*

| 项目 | 内容 |
|------|------|
| **参考态** | **1 GPa（1000 MPa）, 25°C** — 便于与 **Christensen (1979)** 等实验室 **Vp–ρ 经验关系** 对比 |
| **纵轴/横轴** | **Vp** vs **ρ**（或 Fig.6 双 panel：Vp、ρ 随成分/ F 变化 — 以论文图为准） |
| **曲线族** | 各起点 **parental melt** + 沿 FC 路径的 **累积结晶组合** |
| **物理含义** | 地幔熔融成因的 **未蚀变、无孔隙** 火成岩在 **V–ρ 平面** 上可能占据的区域 |

**与主链参考态区分（重要）**

| 用途 | P, T |
|------|------|
| 方程 (1)、Fig.3/5 bulk/ΔVp | **600 MPa, 400°C** |
| Fig.2 示例 | **100 MPa, 100°C** |
| **Fig.6 实验室对比** | **1 GPa, 25°C** |
| imodel 岩性经验校正 | **200 MPa, 25°C**（**本 petrology 链不用**） |

Fig.6 的 **1 GPa/25°C** 是为 **对照文献**，不是 KKHS02 反演 H–Vp 时的报告态。

### Fig.7 — 组合矿物的温压导数

> *For each crystal assemblage, we can calculate its velocity and density at any given temperature and pressure, using the temperature and pressure derivatives of constituent minerals … using **finite difference approximation** (Figure 7).*

| 项目 | 内容 |
|------|------|
| **单矿物** | S&B (1994) 等给出的 **∂K/∂T, ∂G/∂T, ∂ρ/∂T** 及压力导数 |
| **组合** | 固定成分 → 在 **(P,T)** 上微扰 → 重算 HS **Vp, ρ** → **有限差分** 得 **dVp/dT, dVp/dP** |
| **非线性** | 混合效应对导数 **非线性**（Fig.7 可见）；总体可用 **线性关系近似** 做温压校正 |
| **与实验室** | 趋势与 **Christensen (1979); Kern & Tubia (1993)** 等 **基性–超基性全岩** 测量 **定性一致** |
| **理论优势** | 实验室导数受 **残余孔隙、裂隙闭合滞后** 影响（地质时间尺度上可忽略）；理论链基于 **单晶数据 + 混合理论**，更适合 **pristine** 端元 |

**实用用途**：把不同 **(P,T)** 下算得或观测的 Vp **校正到统一参考态**（如 600 MPa/400°C 或 1 GPa/25°C），便于与 Fig.3/5 或实验室标度对比。

### V–ρ 容许空间与重力

> *… comprehensively explore possible regions in the **V–ρ space** for pristine igneous rocks originating from mantle melting … particularly useful when velocity structure is used to infer possible **density structure** to study **gravity anomalies** [Korenaga et al., 2001].*

| 应用 | 与 pyAOBS 关系 |
|------|----------------|
| **Vp → ρ 转换带** | 在 FC 导出的 **V–ρ 云** 内取合理 ρ，约束壳体密度 |
| **重力联合反演** | 超出 KKHS02 地震主链；可与 imodel **Vp–ρ 或重力** 模块对接（Modern **M4/M5** 方向） |
| **Modern track** | BurnMan SLB 自带 **P–T 依赖**；可替代有限差分，但仍需 HS 组合步骤 |

### 与 Step 1–4 对照

| 步骤 | 与 Fig.6–7 |
|------|-------------|
| **Step 1** | parental melt 点落在 Fig.6 |
| **Step 3** | FC 组合点沿 Fig.6 扫出 **V–ρ 条带** |
| **Step 2** | **无直接依赖** |
| **Step 4** | **无直接依赖**（H–Vp 用方程 (1)，不用 Fig.6 导数） |

### 计划模块与验收

| 模块 | 职责 |
|------|------|
| `minerals.py` | 单矿物 **K,G,ρ** 及 **∂/∂T, ∂/∂P** |
| `mixing.py` | HS 混合；**`vp_rho_at(P,T)`** |
| `vrho_systematics.py` | FC 库 → Fig.6 散点；有限差分 → Fig.7 |
| `reference_state.py` | 多参考态校正（600/400、1 GPa/25°C 等） |

**验收（`reproduce_fig6.py`, `reproduce_fig7.py`）**

- [ ] 全库 parental + FC assemblage @ **1 GPa, 25°C**  
- [ ] V–ρ 点云与 **Christensen (1979)** 经验带 **定性一致**  
- [ ] dVp/dT、dVp/dP 与文献 **趋势一致**；混合非线性在 Fig.7 可见  
- [ ] 线性导数近似可回推参考态，误差在论文接受范围内  
- [ ] **不**与 imodel `empirical_formulas`（200 MPa/25°C）混用  

**实现优先级**：**R0 延伸 / 可选 R3+** — 在 Fig.5 FC 跑通后几乎零额外成分数据，主要是 **minerals 导数 + 有限差分**。

---

## Figure 2 示例（§2.1 结晶建模验收基准）

原文 §2.1 [12] 以 **正常 MORB 原始熔体** 为起点，展示 **三条结晶路径** 下 Vp、ρ 随 **固相分数 F** 的演化。这是 **Reproduction track 的首个端到端验收**（W&L + HS + 可选 B&W 熔体密度），也是理解 **ΔVp 为何随 F 急降** 的关键图。

### 起点熔体 — Kinzler [1997]

| 项目 | 值 |
|------|-----|
| **来源** | 多压 **近分离熔融** 聚合熔体；“**triangular shaped melting regime**” |
| **平均熔融程度** | **F_melt = 9%**（0.09） |
| **平均熔融压力** | **P_melt = 1.5 GPa** |
| **主量 (wt%)** | SiO₂ 48.2, TiO₂ 0.94, Al₂O₃ 16.4, Cr₂O₃ 0.12, FeO 7.96, MgO 12.5, CaO 11.4, K₂O 0.07, Na₂O 2.27 |

计划写入 `data/mantle_melts/kinzler1997_morb_primary.json` 作为 **Fig.2 / validation 默认起点**。

**Bulk 参考**：该起点 primary melt 的 norm-based Vp（Fig.2a 灰线 / bulk 端元）约 **7.17 km/s**（**100 MPa, 100°C**，见下）。

### 三条结晶路径

| # | 路径 | P_fc 条件 | 曲线（Fig.2a） |
|---|------|-----------|----------------|
| **(1)** | **分离结晶 FC** | **等压 100 MPa** | 实线：累积 / 增量堆晶；残余熔体 norm Vp |
| **(2)** | **多压分离结晶** | **800 → 100 MPa** 降压 FC | 点线：高压下 Cpx 更稳 |
| **(3)** | **平衡结晶 EQ** | **等压 100 MPa** | 虚线：与 FC 对比 |

**实现要点（Reproduction）**

- **(1)(3)**：W&L @ 100 MPa 至目标 F_solid。
- **(2)**：FC 步进中 **同步降低 P**（800→100 MPa）；每步用 Langmuir (1992) 高压规则更新液相线 / Cpx 稳定性。
- 所有 **结晶固相 Vp** 在 **报告态** 统一换算到 **100 MPa, 100°C**（与 Fig.2a 一致，**≠** 方程 (1) 的 600 MPa/400°C）。

### Fig.2a — Vp vs 固相分数 F

| 输出曲线 | 含义 | 用途 |
|----------|------|------|
| **Cumulative FC assemblage**（黑） | 至当前 F **累积移除** 的堆晶平均组合 Vp | **V_LC 理论值**；下地壳主用 |
| **Incremental FC assemblage**（细灰） | **本步** 新结晶固相 Vp | 辅助；反映早期纯 Ol 极高速 |
| **Norm-based（残余熔体）**（灰） | 当前 **熔体成分** CIPW → 假想固相 Vp | 上地壳 / 残余液演化 |
| **Bulk / primary** | 起点熔体 norm Vp | **~7.17 km/s** @ 100 MPa, 100°C |

**关键物理（原文）**

1. **早期仅 Ol 结晶**（Fig.2c–e）→ 堆晶 Vp 极高 → **ΔVp = V_LC − V_bulk 很大**。
2. **F_solid ≳ 0.2（~20% 固结）** 后 Pl/Cpx 加入 → **ΔVp 急降**。
3. 若下地壳只占全壳 **很小比例**，用大 ΔVp 约束 bulk **不可靠**；洋壳 **f_lower 通常 >50%**（Mutter & Mutter, 1993）→ Fig.5 在 **F_xl = 0.5–0.8** 标定主边界。
4. **高压结晶 (2)**：Cpx 更稳定（Bender, Presnall, Grove 等）→ 组合 Vp **略低于** 100 MPa 等压 FC（Fig.2a、d）。

### Fig.2b — 密度

| 相 | 方法 |
|----|------|
| **结晶固相** | 与 Vp 相同 HS 链 → ρ |
| **残余熔体** | **Bottinga & Weill [1970]** |

**浮力结论**：残余熔体随结晶变 **更密**，但 **始终轻于** 同期固相 → 下地壳 **不作** 密度过滤器（Sparks et al., 1980; Stolper & Walker, 1980）。**Modern track 可选**验证，**非 Vp 反演必需**。

### Fig.2c–2e — 相比例

- **(c,f)**：100 MPa FC — Ol → Pl → Cpx 出现顺序与比例 vs F  
- **(d,g)**：800→100 MPa polybaric FC  
- **(e,h)**：100 MPa EQ  

**验收**：早期 **单 Ol**；~20% 固结后 ΔVp 显著减小；FC 累积 Vp **>7.3 km/s**；bulk **~7.17 km/s**。

### 与 Paper mapping / 双轨的关系

| 项目 | Reproduction | Modern |
|------|--------------|--------|
| 起点成分 | Kinzler (1997) 上表 | 同左或 Katz 源区熔体 |
| FC/EQ | W&L + Langmuir (1992) | 热力学 FC（可对照 W&L） |
| Vp @ Fig.2 | HS + S&B/BurnMan @ **100 MPa, 100°C** | BurnMan SLB；报告态可同 |
| 熔体 ρ | B&W (1970) | B&W 或 BurnMan melt（若可用） |
| 验收脚本 | `validation/reproduction/reproduce_fig2.py` | `validation/modern/fig2_compare_tracks.py` |

### 实现检查清单（`reproduce_fig2.py`）

- [ ] 加载 Kinzler (1997) MORB primary 成分  
- [ ] 跑路径 (1)(2)(3)；输出 F、相比例、Fo/An  
- [ ] 算 cumulative / incremental / residual norm Vp @ 100 MPa, 100°C  
- [ ] 标出 bulk ≈ 7.17 km/s；FC 累积 >7.3 km/s @ 高 F  
- [ ] 确认 F &lt; 0.2 时 ΔVp 过大；F ≥ 0.2 后 ΔVp 急降  
- [ ] 可选：残余熔体 ρ（B&W）> 固相 ρ 的排序  

---

### Step 4 — 简单 1D 熔融 + 主动上涌 → H–Vp

> *Finally, a simple mantle melting model is constructed to illustrate the effects of potential temperature, active [mantle upwelling]…*

| 项目 | 内容 |
|------|------|
| **论文章节** | §4 Mantle Melting Model With Active Upwelling；案例 §5 |
| **参数** | **Tp**（潜在温度）, **b**（先存岩石圈盖层, km）, **χ**（主动上涌比 Vm/Vs, ≥1） |
| **子模型** | Takahashi & Kushiro (1983) 固相线；McKenzie & Bickle (1988) 绝热线 → P₀；线性 F(P)，默认 (∂F/∂P)_S=12%/GPa |
| **自洽** | P_f=(b+H)/30 GPa；**H=χ·30·(P₀−P_f)·F̄**（eq.10）；F̄=0.5·F(P_f)（eq.8）；P̄=0.5(P₀+P_f)（eq.9） |
| **Vp** | **Step 1 方程 (1)**：`Vp_bulk = f(P̄, F̄)` — 不经过 FC |
| **主要图件** | Fig.10（示意）, Fig.11（Tp/χ/b 扫描）, Fig.12（**H–Vp 图**）, Fig.15c（格陵兰） |
| **计划模块** | `melting/solidus.py`, `active_upwelling.py`, `melting_functions.py`, `hvp_diagram.py` |
| **验收** | 格陵兰 transect 2：H~30 km, V_LC~7.0 → χ>8, Tp~1300°C |

#### Fig.11 d/h 已知差异（V_bulk 形态 vs 印刷图）

§4 熔融链（eq.6–11、12%/GPa）经 **`check_fig11.py`** 与 panel **a–c / e–g** 对照：**P̄、F̄、H 与原文一致**。仅 **d/h（V_bulk vs Tp）** 在用正文 **eq.(1)(P̄,F̄)** 时与 digitized 原图系统性不符；问题隔离在 **V 映射**，不是 ΔP/H 自洽。

| 现象 | digitized 原图 (d/h) | 本仓库 eq.(1)(P̄,F̄) | 同一 (P̄,F̄) 上 Holbrook 线性 |
|------|----------------------|---------------------|------------------------------|
| 低温端 V | ~**6.90–6.92** km/s | ~**7.05–7.12**（偏高 ~0.12–0.17） | ~**6.99–7.00**（更接近） |
| Tp≈1400°C 曲线簇 spread（χ 或 b） | **~0.004** km/s（极紧） | **~0.07** km/s（约 17×） | ~**0.023** km/s（仍宽于原图） |
| χ↑ 时 V 趋势 | **降低**（χ=1 > χ=8） | **升高**（与原文相反） | **降低**（与原文同向） |
| Tp≈1400°C 交汇水平 | ~**7.19** km/s（非精确 7.2） | ~**7.32–7.39** | ~**7.12–7.15** |

**正文依据**：§2.2 [16]–[17] 选定 **eq.(1)**（MORB：P=1 GPa、F=0.1 → **7.1 km/s**），并说明 KH95 式 (2) 偏低（~6.95 km/s）。Fig.11 caption 只写 “predicted bulk crustal velocity”，**未写明 equation (1)**。

**较稳妥解释**（无法在无作者原始脚本时 100% 定罪）：

1. **框架/反演**（Step 1、Fig.3、Fig.15c、invert）应继续用 **eq.(1)**。  
2. **Fig.11 印刷曲线**的绝对速度与紧簇形态，在 **同一已验证 (P̄,F̄)** 上更接近 **Holbrook 型线性 V(P,F)**（`predict_v_bulk_fig11_km_s`），而非窗函数 eq.(1) 沿 §4 轨线的表现 —— 可能存在 **正文公式与 Fig.11 作图不同步**，或 **纵轴压缩/参考线** 造成的视觉交汇（仓库 `article/复现korenaga/test.py` 对 d/h 使用 ylim 7.0–7.35 并标 1420°C/7.2 辅助线）。  
3. **常数 bias**（如 Fig.3 目录平均偏差）只能平移 eq.(1) 曲线，**不能**恢复 1400°C 处的紧簇与 χ 排序。  
4. eq.(1) 拟合域为 **P=1–3 GPa、F=0.02–0.2**；低温端 P̄&lt;1 GPa 为 **外推**，会加剧偏高，但不足以单独解释 1400°C 处的 spread 差异。

**相关脚本与数据**：

| 脚本 | 用途 |
|------|------|
| `validation/reproduce_fig11.py` | 完整 4×2 Fig.11；**d/h 默认 eq.(1)**（与正文 Step 4 表述一致） |
| `validation/reproduce_fig11_dh_compare.py` | **仅 d/h**；实线 eq.(1)、虚线 Holbrook、叠 digitized 原图 |
| `validation/analyze_fig11_digitized.py` | digitized 与三种 V 模型的 RMSE / spread 定量报告 |
| `validation/check_fig11.py` | §4 熔融 **PASS/FAIL**；末尾 **informational** 打印 d/h 差异摘要 |
| `data/ScreenShot_*_124209_264.txt` 等 | GetData digitized panel (d)/(h) |

**实践建议**：复现 **a–c** 用 `check_fig11.py`；对比 **d/h** 用 `reproduce_fig11_dh_compare.py`；论文/报告里分开写 “§4 熔融已复现” 与 “d/h 与 eq.(1) 形态 known discrepancy”。

#### Fig.15 / SIGMA transect → Fig.12a 应用（观测到投图）

§5 格陵兰 **SIGMA transect 2** 的完整流程（原文 Fig.15a–c）：

```text
地震反射/拾取 → 原地 Vp(x, z)
        │
        ▼  每点：P(z) 静岩压，T(z)=20°C/km 传导地温（Fig.7 导数）
   V_600/400 = V_obs + 0.0002(600−P) − 0.0004(400−T)
        │
        ▼  下地壳内谐和平均
   V_LC(x)  （Fig.15a）
        │
        ▼  横向 20 km 窗 @ 10 km 步长 + 100 MC
   (H, V_LC ± σ)  （Fig.15b/c）
        │
        ▼  直接投 Fig.12a 背景（模型线为 V_bulk）
   Fig.15c  open circles
        │
        ▼  仅厚壳 H > 15 km 读 Tp、χ
   V_bulk,model ≤ V_LC；允许低至 V_LC − ΔVp_max（Fig.5，例 7.0→6.85 km/s）
```

| 环节 | 原文 | 本仓库 |
|------|------|--------|
| P–T 校正 | Fig.7：0.2×10⁻³ km/s/MPa，−0.4×10⁻³ km/s/°C → 600 MPa、400°C | `petrology/seismic/reference_state.py` |
| 谐和平均 + 滑窗 + MC | 20 km 窗，100 ensembles | `petrology/seismic/transect.py` |
| 投 Fig.12a | **(H, V_LC) 直接叠加**，不先减 ΔVp | `validation/reproduce_fig15_transect.py` |
| Step 2 读图 | **V_LC 为 V_bulk 上界**；ΔVp 为解释带宽（非作图平移） | 图上半透明竖段 + 本文 |
| 薄壳排除 | 向海 H 小、孔隙压低 V_LC → 上界失效 | `H ≤ 15 km` 灰色空心点 |
| 演示数据 | SIGMA transect 2 | `data/greenland_transect2_demo.csv`（`--write-demo-csv`） |

**运行**：

```bash
py -3.11 petrology/validation/reproduce_fig15_transect.py --write-demo-csv
py -3.11 petrology/validation/reproduce_fig15_transect.py --show
```

输入 CSV 列：`distance_km`, `depth_km`, `vp_insitu_km_s`, `h_whole_km`（每行一个深度样点）。

**与 `reproduce_fig15c.py` 区别**：后者为 **单点 anchor**（H=30、V_LC=7.0）+ Tp–χ 扫描；`reproduce_fig15_transect.py` 为 **沿迹多窗** 观测→投图链。

**与 Step 2/3 的衔接**：H–Vp 正演用 Step 4 + Step 1；**Fig.15c 作图** 为 **(H, V_LC)** 直接叠 Fig.12a；Step 2 ΔVp 用于 **读图**（V_bulk ≤ V_LC），非坐标平移。

**灵敏度（可选）**：Fig.13 — (∂F/∂P)_S=16%/GPa；Asimow et al. (1997) 三阶段熔融函数。

---

## 端到端数据流

```text
Step 1   实验/计算熔体 ──→ 方程 (1) Vp_bulk(P, F)
                              │
Step 3   W&L FC @ 多P ──→ ΔVp(F), V_LC_theory  （Fig.5）
                              │
Step 4   Tp, b, χ ──→ H, P̄, F̄ ──→ Vp_bulk_pred  → Fig.12a 背景
                              │
Fig.15   地震 V_obs ──→ P–T 校正 ──→ V_LC ──→ (H, V_LC) 直接投 Fig.12a
                              │
         读图（厚壳）:  V_bulk,model ≤ V_LC ；可低至 V_LC − ΔVp_max（Step 2，非作图平移）
                              │
         imodel:  匹配 Tp 等值线 + χ/b 轨迹族
```

---

## 图件 — 模块 — 验收 速查表

| 图 | 对应 Step | 计划脚本 | 关键验收 |
|----|-----------|----------|----------|
| Fig.2 | 1+3 | `validation/reproduce_fig2.py` | FC vs EQ vs polybaric；V_cum > V_bulk |
| Fig.3 | 1 | `validation/reproduce_fig3.py` | 实验点 vs 方程 (1) |
| Fig.5 | 2+3 | `validation/reproduce_fig5.py` | ΔVp(F)≈0.15 km/s @ F=0.7–0.8 |
| Fig.6–7 | 3（副链） | `validation/reproduce_fig6.py`, `reproduce_fig7.py` | V–ρ @ 1 GPa/25°C；温压导数 vs Christensen |
| Fig.11 | 4 | `validation/reproduce_fig11.py`, `check_fig11.py`, `reproduce_fig11_dh_compare.py` | a–c/e–g：§4 PASS；d/h：eq.(1) vs 原图见上节 |
| Fig.12 | 4 | `validation/reproduce_fig12.py` | H–Vp 等值线形态 |
| Fig.15c | 2+4+obs | `validation/reproduce_fig15_transect.py`, `reproduce_fig15c.py` | 沿迹 (H,V_LC) 投 12a；anchor 扫描 |

---

## 建议实现阶段

### Reproduction track（R0–R4）

| 阶段 | 内容 | 验收 |
|------|------|------|
| **R0** | `seed_compositions.csv`, `norm_velocity`, BurnMan HS + 可选 S&B | Fig.2 锚点量级 |
| **R1** | 方程 (1) + `active_upwelling` + Fig.12 | H–Vp 形态 |
| **R2** | Kinzler 网格 + Fig.3 | 1σ≈0.05 km/s |
| **R3** | W&L + Langmuir FC + Fig.5（默认裸算 ΔVp；可选 Fig.5 经验抬升） | ΔVp≈0.13–0.15 km/s @ F=0.7–0.8 |
| **R4** | `invert` + imodel + Fig.15c | 格陵兰 χ>8 |

### Modern track（M1–M5，可与 R 阶段交错）

| 阶段 | 内容 | 相对 R 轨升级点 |
|------|------|-----------------|
| **M1** | BurnMan SLB 默认 + 保留 bounding `invert` | 物性；接口同 R |
| **M2** | CIPW → **`equilibrate` 平衡组合** 算 bulk Vp | 矿物组合；详见 **[M2 对照协议](#modern-m2热力学-bulk-vp-与-cipw-对照协议)** |
| **M3** | **Katz** 熔融 + H–Vp **误差带** | §4 替代 |
| **M4** | 热力学 **FC** + 孔隙校正 | §2.3 + §3 |
| **M5** | pyrolite **源区** + 可选 2D/异质 | §6.3 |

**推荐路径**：先 **R0→R1** 打通 imodel 与 H–Vp；并行或随后 **M1**；**R3 与 M2/M4** 可共享 `fc/` 外壳。

---

## Modern M2：热力学 bulk Vp 与 CIPW 对照协议

Modern **M2** 阶段的核心任务：**在保留 KKHS02 bounding 框架的前提下**，用 **热力学相平衡 + BurnMan SLB** 建立 **成分 → Vp** 正演，并与 Reproduction 的 **CIPW norm 路线** 做 **系统对照**。目标不是单纯替换引擎，而是 **量化偏差** 并判断 **ΔVp、χ/Tp 反演** 是否敏感。

### 目标与产出

| 项目 | 内容 |
|------|------|
| **科学问题** | CIPW 名义矿物 vs (P,T) 平衡组合：\|ΔVp\| 多大？是否改变 Fig.5 带（~0.15 km/s）与 bulk 下界（~0.20 km/s）？ |
| **代码产出** | `tracks/modern/equilibrate_vp.py`、`compare_cipw_thermo.py`、离线 lookup 表 |
| **图件产出** | `validation/modern/m2_bias_fig3.py`（bulk Vp 散点）；`m2_delta_vp_envelope.py`（ΔVp 包络 vs R3） |
| **数据产出** | `data/mantle_melts/catalog_with_vp.parquet`（双轨 Vp 列） |
| **下游** | M3 全正演 `(P,F)→Vp`；M4 热力学 FC；`invert.py` v2 **thermo bounds** |

### 计算链（双轨并行）

```text
                    同一 melt 主量 + 元数据 (P_melt, F, source_type)
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
    Track A — Reproduction                    Track B — Modern M2
    pyrolite CIPW_norm()                       BurnMan equilibrate()
    → 名义 Ol/Pl/Cpx/Qz 质量分数               → 平衡固相 modes + 端元成分
    → minerals (S&B 或 BurnMan+锚点)           → minerals (BurnMan SLB)
    → mixing.hs_average()                      → mixing.hs_average()
              │                                           │
              └─────────────────────┬─────────────────────┘
                                    ▼
              报告态统一 @ P_ref=600 MPa, T_ref=400°C
              输出: Vp_cipw, Vp_thermo, ΔV = Vp_thermo − Vp_cipw
```

**Layer 3（HS 混合）两轨共用** `mixing.py`，保证差异主要来自 **矿物组合**，而非混合算法。

### Bulk 端元定义（与 KKHS02 norm-based 对齐）

KKHS02 **norm-based bulk Vp**：给定 **熔体主量**，假想 **100% 固相、均匀混合、无孔隙** 的全壳速度。M2 须在热力学路线中 **显式选定** 与之可比的平衡约束：

| 方案 | 平衡约束 | 推荐 | 说明 |
|------|----------|------|------|
| **M2-A（默认）** | 熔体成分在 **(P_ref, T_ref)** 下 **subsolidus 全固相平衡**（ melt fraction = 0） | ✅ 首选 | 最接近“若该成分在参考态完全固结” |
| **M2-B** | 成分在 **P_ref** 下沿 **固相线温度** 平衡（liquidus 附近，微量熔体→0） | 备选 | 更贴 Melting P；与 M2-A 差作灵敏度 |
| **M2-C** | 对 **全岩固相成分**（非熔体）直接 subsolidus 平衡 | FC 堆晶用 | **不**用于 primary melt bulk；用于 M4 累积组合 |

**术语**：Modern 轨输出仍记 **`Vp_norm_modern`** 或 **`Vp_bulk_thermo`**，README/论文中与 KKHS02 **`norm-based velocity`** 对照时须 **注明平衡方案（M2-A/B）**。

**不采用**：对 **熔体相** 本身做 HS（除非 M4 残余熔体分支）；M2 bulk 端 **仅固相组合**。

### 输入字段（`catalog.csv` / API）

每条熔体记录 **至少** 含：

```text
id, source, source_type, P_melt_GPa, F_melt,
SiO2, TiO2, Al2O3, FeO, MgO, CaO, Na2O, K2O,   # wt%, 归一化至 100%
Cr2O3, P2O5, H2O, CO2,                           # 可选；默认干/无 CO2
include_in_regression, melt_style,
equilibrate_status, m2_scheme                           # 输出回填
```

| 字段 | 要求 |
|------|------|
| **`source_type`** | `pyrolite_depleted` \| `pyrolite_fertile` \| `pyroxenite` \| `enriched` \| `hydrous` \| `carbonated` \| `subduction` \| `unknown` |
| **主量氧化物** | wt%，求和前 **归一化**；缺失相按 0 并 `quality_flag` |
| **`P_melt_GPa`, `F_melt`** | Step 1 回归用；M2 bulk 计算 **不依赖** 二者，但用于着色与 `(P,F)` 图 |
| **挥发分** | 默认 **干体系**（H₂O=0, CO₂=0）；含水/碳酸盐 **单独批次**，须标注 `source_type` |

**API 草案**（`tracks/modern/equilibrate_vp.py`）：

```python
def bulk_vp_thermo(
    oxides_wt: dict[str, float],
    *,
    P_pa: float = 600e6,
    T_k: float = 673.15,
    scheme: str = "M2-A",
    backend: str = "burnman_equilibrate",
    mineral_eos: str = "slb_2022",
) -> EquilibrateVpResult:
    """返回 Vp_km_s, rho_kg_m3, phases, converged, fallback_used."""
```

### 相平衡约束（BurnMan `equilibrate` 默认）

| 参数 | 默认值 | 备注 |
|------|--------|------|
| **体系** | Na₂O–CaO–FeO–MgO–Al₂O₃–SiO₂（NCFMAS）；+Ti/K/Cr 按 BurnMan 支持逐步扩展 | M2 首版可 NCFMAS |
| **fO₂** | FMQ 或 iron saturation（与数据库一致） | 敏感性测试 ±0.5 log fO₂ |
| **H₂O / CO₂** | 0（干） | `hydrous`/`carbonated` 源区另表 |
| **相集合** | olivine, orthopyroxene, clinopyroxene, plagioclase, spinel, garnet（P 依赖） | 由 P_ref 自动启用 |
| **熔体分数** | **0.0**（M2-A） | 全固相平衡 |
| **失败处理** | 记录 `converged=False` → **回退 CIPW** + `fallback_used=True` | 不静默丢弃 |

可选后端：`perple_x`（M2+ 对高压/碳酸盐）；接口与 BurnMan 输出 **统一为 phase modes 字典**。

### 参考态与温压校正

| 用途 | P, T | 说明 |
|------|------|------|
| **M2 主报告态** | **600 MPa, 400°C** | 与 eq.(1)、Fig.3/5 **一致** |
| **Fig.2 锚点对比** | 100 MPa, 100°C | 单独脚本；Kinzler MORB primary |
| **Fig.6 实验室对比** | 1 GPa, 25°C | M2 扩展；`reference_state.py` |
| **非参考态平衡** | 若在 (P_melt, T_melt) 平衡 | 须 **经 Layer 2 EOS 校正** 到 600/400 再与 CIPW 比 |

**禁止**：用 imodel `empirical_formulas`（200 MPa/25°C）做 M2 校正。

### 对照指标

对 catalog 每条记录及 R3 FC 路径上的 **bulk 端 / cumulate 端** 计算：

| 指标 | 定义 | 用途 |
|------|------|------|
| **ΔV_bulk** | `Vp_thermo − Vp_cipw` @ bulk 端 | Fig.3 替代散点偏差 |
| **ΔV_LC** | cumulate 端同理 | Fig.5 偏差 |
| **ΔΔVp** | `(V_LC_thermo − V_bulk_thermo) − (V_LC_cipw − V_bulk_cipw)` | **bounding 核心**：反演是否变 |
| **相模态差** | \|f_Ol\|, \|f_Pl\|, \|f_Cpx\| 等 CIPW vs 平衡 | 解释 ΔV 来源 |

### 验收阈值（M2 阶段通过标准）

**单点锚点**（与 R0 共用 Kinzler MORB primary）：

| 检查 | Reproduction (CIPW) | M2 期望 |
|------|---------------------|---------|
| bulk @ 600 MPa, 400°C | ~7.1 km/s（eq.(1)）或 norm 直接算 | 与 CIPW **\|ΔV_bulk\| < 0.10 km/s**（干 pyrolite 熔体） |
| bulk @ 100 MPa, 100°C（Fig.2） | ~7.17 km/s | 记录偏差；**不强制**与 KKHS02 绝对一致（Modern 用 SLB） |

**全库统计**（`catalog` + 可选 post-2002 扩展）：

| 检查 | 阈值 | 说明 |
|------|------|------|
| pyrolite 子集 \|ΔV_bulk\| 中位数 | **< 0.08 km/s** | 主链源区 |
| pyroxenite / enriched 子集 \|ΔV_bulk\| | **可 > 0.10 km/s** | 预期更大；**分源区报告** |
| \|ΔΔVp\| @ F_xl=0.7–0.8（pyrolite） | **< 0.05 km/s** | 若成立 → Fig.5 bounding **稳健** |
| \|ΔΔVp\| 导致 χ 变化 | **< 30%**（典型 H~30 km, V_LC~7.0） | 需 `invert` 灵敏度脚本；**可放宽**若仅改绝对 Vp |
| equilibrate 收敛率 | **> 90%**（pyrolite）；失败须 fallback 记录 | 碳酸盐/极端成分可更低 |

**发表级 Figure（最低集）**：

1. **Vp_cipw vs Vp_thermo** 全库散点（600/400），按 `source_type` 分色  
2. **ΔV_bulk** 直方图：pyrolite vs pyroxenite  
3. **ΔΔVp vs F_xl** 与 R3 W&L 带叠加  
4. 一例 **χ–Tp 可行域**：CIPW bounds vs thermo bounds  

### 性能与 lookup

| 策略 | 说明 |
|------|------|
| **离线批算** | 全 catalog × (600 MPa, 400°C) → `catalog_with_vp.parquet` |
| **在线反演** | 默认读 lookup；无条目时 **即时 equilibrate**（慢） |
| **Surrogate（M3 预备）** | 对 pyrolite 子集训练 **(oxides, P, T) → Vp_thermo** 快速近似；eq.(1) 仅作 cross-check |

目标：单点 equilibrate **< 2 s**（BurnMan）；全库 **< 1 h**（并行）。

### 与 post-2002 实验库

扩展 catalog（Kogiso、Davis、Tuff & Green 等）时 **同一 M2 协议**：

- `source_type` **必填**；分源区统计 ΔV、ΔΔVp  
- **突破叙事**：pyroxenite 上 \|ΔV_bulk\| 大 → CIPW **系统性偏离** → thermo bounds **缩/扩** bulk 区间  

计划路径：`data/mantle_melts/catalog_v2/`（与 R2 `catalog.csv` 并存）。

### 模块与脚本

```text
petrology/
  tracks/
    modern/
      equilibrate_vp.py       # M2-A/B 平衡 → phase modes
      compare_cipw_thermo.py  # 双轨批处理 + 指标
  reference_state.py          # 多参考态 P,T 换算
  validation/modern/
      m2_bulk_scatter.py      # Fig: Vp_cipw vs Vp_thermo
      m2_delta_vp_envelope.py # Fig: ΔΔVp vs F_xl
      m2_chi_sensitivity.py   # invert: χ 带 CIPW vs thermo
```

### 失败与回退

```text
equilibrate 失败 / 非物理相组合
    → fallback: CIPW + flag fallback_used=True
    → invert 默认仍可用 reproduction bounds；modern 模式可选 strict（丢弃该源区）
```

### M2 与后续阶段衔接

| 阶段 | M2 交付物如何使用 |
|------|-------------------|
| **M3** | `(P,F,源区)→Katz 成分→**M2 bulk_vp_thermo**` 替代 eq.(1) |
| **M4** | FC 每步 **M2-C**（固相组合）+ 残余熔体分支 |
| **M5** | 2D 反演读 lookup；源区异质 = 多 `source_type` 并行 bounds |
| **invert v2** | `bulk_vp_bounds(..., vp_engine="cipw"\|"thermo"\|"both")` |

---

## 外部依赖（可选 extras）

| 用途 | 包 / 工具 |
|------|-----------|
| CIPW | `pyrolite` |
| 矿物 Vp + HS | 仓库内 `burnman` |
| FC（优先） | 自研 W&L 1990 + Langmuir 1992（`fc/wl1990.py`，无 Petrolog3） |
| FC（备选） | 自研 `fc/wl1990.py` |
| 数值求根 | `scipy` |
| Modern 相平衡 | BurnMan `equilibrate`；可选 Perple_X |
| Modern 熔融 | Katz 参数化；可选 pMELTS |
| Modern 地球化学 | `pyrolite` |

**非必需**：MELTS、pMELTS（Modern M4+ 可选；Reproduction 作 Kinzler 对照）。

---

## 与 pyAOBS 其它模块的边界

| 模块 | 关系 |
|------|------|
| **imodel** | 提供 `H`、层速度、`V_LC`；**不**使用 imodel 内线性温压校正做本工作流（本链用 BurnMan EOS 或 KKHS02 参考态 600 MPa/400°C） |
| **zplot / workbench** | 无直接依赖；Workbench 可后续挂 `petrology` 插件 |
| **utils/empirical_formulas.py** | imodel 岩性分类用；与 petrology 参考态分离 |

---

## 目录结构（规划）

```text
petrology/
  README.md                 # 本文件（含双轨说明）
  __init__.py
  config.py                 # track / backend
  pipeline.py
  tracks/
    reproduction/           # 复现轨专用
    modern/                 # 现代轨专用
  data/
    mantle_melts/
    korenaga2002_eq1.json
  cipw.py
  minerals.py               # sb1994 | burnman_slb
  mixing.py
  norm_velocity.py
  vp_regression.py          # Reproduction：方程 (1)
  melts_database.py
  fc/
    wl_partition.py         # W&L 饱和/分配逻辑
    wl1990.py               # Reproduction FC 引擎
    equilibrate_fc.py       # Modern
  melting/
    solidus.py
    active_upwelling.py     # Reproduction §4
    katz.py                 # Modern
    melting_functions.py
    hvp_diagram.py
  porosity.py               # Modern
  invert.py
  validation/
    reproduction/           # reproduce_fig*.py
    modern/                 # forward_vs_eq1.py 等对比
```

当前状态：**规划与文档**；双轨接口已定义；实现建议从 **R0→R1** 与 **M1** 开始。

---

## 参考文献（核心）

- Korenaga, J., Kelemen, P. B., Holbrook, W. S., & Sobolev, S. V. (2002). Methods for resolving the origin of large igneous provinces from crustal seismology. *JGR*, 107(B9). [doi:10.1029/2001JB001030](https://doi.org/10.1029/2001JB001030)
- Katz, R. F., Spiegelman, M., & Langmuir, C. H. (2003). A new parameterization of hydrous mantle melting. *Geochemistry, Geophysics, Geosystems*, 4(9), 1073. [doi:10.1029/2002GC000433](https://doi.org/10.1029/2002GC000433)
- Brown, E. L., & Lesher, C. E. (2014). North Atlantic magmatism controlled by temperature, mantle composition and buoyancy. *Nature Geoscience*, 7(11), 901–905. [doi:10.1038/ngeo2264](https://doi.org/10.1038/ngeo2264) — active source buoyancy, active (Brown & Lesher) RMC.
- Brown, E. L., & Lesher, C. E. (2016). REEBOX PRO: A forward model simulating melting of thermally and lithologically variable upwelling mantle. *Geochemistry, Geophysics, Geosystems*, 17, 3929–3968. [doi:10.1002/2016GC006579](https://doi.org/10.1002/2016GC006579) — 增量等熵熔融 eq. (2)–(3)、Appendix A Katz 参数化、Table 1 热力学常数；**编译 MATLAB 应用**（非 Python）。
- Pertermann, M., & Hirschmann, M. M. (2003). Partial melting experiments on a MORB-like pyroxenite … *JGR*, 108(B4). — G2 辉石岩固相线（REEBOX 默认 pyroxenite 端元）。
- Weaver, J. S., & Langmuir, C. H. (1990). Calculation of phase equilibrium in mineral-melt systems. *Computers & Geosciences*, 16(1), 1–19.
- Langmuir, C. H., et al. (1992). How deep do common basaltic magmas form and differentiate? *JGR*.
- Sobolev, S. V., & Babeyko, A. Y. (1994). [矿物弹性与 petrological modeling 汇编]
- Kelemen, P. B., & Holbrook, W. S. (1995). Origin of thick, high-velocity crust … *JGR*.
- Takahashi, E., & Kushiro, I. (1983). Melting of dry peridotite … *Contrib. Mineral. Petrol.* — KKHS02 复现轨固相线（非 Katz）。
