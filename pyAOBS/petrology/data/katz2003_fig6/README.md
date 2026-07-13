# Katz (2003) Fig.6 实验散点对照表

Katz et al. (2003), *Geochemistry, Geophysics, Geosystems*, [10.1029/2002GC000433](https://doi.org/10.1029/2002GC000433)

Fig.6 为 **四个等压 panel** 上的 **F(T)** 实验散点 vs Table 2 模型曲线（0 / 1 / 1.5 / 3 GPa）。

## 附录 / 电子补充材料结构

Wiley 电子补充：`es2002gc000433.pdf`（与主文同 DOI 页面下载 **Supporting Information**）。

| 表 | 名称（常见称呼） | 内容 |
|----|------------------|------|
| **Table S1** | *Summary of Experimental Peridotites (Anhydrous Experiments Only)* | 文献/岩性注册：`study_id`、岩石类型、modal cpx (wt%) |
| **Table S2**（及后续） | Fig.6 用到的 **(P, T, F)** 点 | 每条实验：`P` (GPa)、`T` (°C)、`F`（质量分数 0–1）、对应 `study_id` |

**验收标准**：`experiments.csv` 应与 **Katz ESI Table S2** 逐行一致，而不是与 LEPR 或单篇原文自动对齐。

## 本目录文件

| 文件 | 用途 |
|------|------|
| `studies.csv` | 对应 ESI **Table S1** 的文献/岩性索引（`fig6_panels` 为推断，需与 ESI 核对） |
| `experiments.csv` | **(P, T, F, 文献)** 点表；`verified` 列：`yes` / `partial` / `no` |
| `../ExPetDB_download_LEPR-2007-08-17.xlsx` | LEPR 全库（**无 F 列**），仅用于核对 T、P、相组合 |
| `../Pyrolite_equilibrium_melt_compositions_*.xlsx` | **Jaques & Green (1980)** 双表：Sheet1 **Table 4**（Hawaiian Pyrolite）、Sheet2 **Table 5**（Tinaquillo lherzolite）；提取为 `jaques_green_1980_pyrolite_table4.csv` 与 `jaques_green_1980_tinaquillo_table5.csv` |

## 列说明（experiments.csv）

| 列 | 含义 |
|----|------|
| `p_gpa` | 等压压力 (GPa)；Fig.6 四 panel 为 0, 1, 1.5, 3 |
| `t_c` | 实验温度 (°C) |
| `f` | 熔融分数（**质量分数**，0–1） |
| `study_id` | 与 `studies.csv` 链接 |
| `fig6_panel` | Katz 图 panel：`a`=0 GPa, `b`=1, `c`=1.5, `d`=3 |
| `verified` | `yes` = 已与 Katz ESI 核对；`partial` = 来自原文献、待 ESI 逐点确认 |

## Fig.6 复现验收（当前范围）

**目标**：验证 `katz2003.py` Table 2 无水熔融模型 + 实验散点叠加逻辑正确，**不要求**与 Katz 原文 Fig.6 全部文献逐点一致。

| 项目 | 状态 |
|------|------|
| 四 panel 等压 **0 / 1 / 1.5 / 3 GPa** | ✓ 模型 F(T) + cpx-out |
| Panel **(b)(c)** 实验散点 | ✓ **Jaques & Green (1980)** Table 4（Pyrolite）+ Table 5（Tinaquillo），共 20 点 |
| Panel **(a)(d)** 实验散点 | 未纳入验收（可日后从 ESI Table S2 扩展） |
| 输出图 | `petrology/figures/katz2003_fig6.png` |

一键复现：

```bash
py -3.11 petrology/validation/import_katz2003_fig6_esi.py --extract-jaques
py -3.11 petrology/validation/reproduce_katz2003_fig6.py
```

或批量导出 Fig.1–6：`py -3.11 petrology/validation/katz2003_workflow.py --only katz_fig6`

## 推荐工作流（扩展用，非验收必需）

1. 下载 `es2002gc000433.pdf`，放到 `petrology/data/es2002gc000433.pdf`
2. 核对 / 导入：

   ```bash
   py -3.11 petrology/validation/import_katz2003_fig6_esi.py --list
   py -3.11 petrology/validation/import_katz2003_fig6_esi.py --check-lepr
   py -3.11 petrology/validation/import_katz2003_fig6_esi.py --extract-jaques
   ```

3. 可选：将 ESI Table S2 其余文献并入 `experiments.csv`，`reproduce_katz2003_fig6.py --all-studies`

## 与 LEPR 的关系

LEPR 存 **相组合 + 成分**，不存 Katz 用的 **F**。用法：

- 用 `Citation` + `P (GPa)` + `T (C)` 找同一实验
- 到 **原文或 Katz ESI** 读取 **F**
- 不要假设 LEPR 一行 = Fig.6 一个点

## 当前数据状态

- **验收用散点**：`JG1980_HP`、`JG1980_TQ`（Jaques & Green 1980 Table 4/5，1 与 1.5 GPa），默认绘制。
- **其余行**：Baker、Falloon、Walter 等 **种子点**（`verified=partial/no`），仅 `--all-studies` 时显示；完整对齐 Katz 原图需 ESI Table S2。
