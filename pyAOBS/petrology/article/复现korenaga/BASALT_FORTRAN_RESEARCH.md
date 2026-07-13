# BASALT.FOR (1990) 与 BASALT+langmuir.FOR 对照研究

本文档对照仓库内两份 Fortran 原文与 Python 端口（`petrology/fc/basalt1990/`、`petrology/fc/wl_state.py`），供 Korenaga Fig.2/5 复现与后续修 STATE/FC 使用。

## 1. 文件与 Python 映射

| Fortran 子程序 | BASALT.FOR (1990) | BASALT+langmuir | Python |
|----------------|-------------------|-----------------|--------|
| BLOCK DATA | 1 atm, SPARM 仅 T | + SPARM(2)=P (kbar) | `common.py` / `wl_components.py` |
| KDCALC | mode 1/2/3; `10**(A/T)+B` | + mode 4; `10**(A/T+B+C*P)` | `kd_calc.py` / `wl_kd.py` |
| STATE | Newton–Raphson | 同结构 | `solver.py` / `wl_state.py` |
| STOICH | 仅 Cpx, 阈值 1.0 | Cpx: P>4 kbar 阈值 1.1 | `solver.stoich` / `wl_state.stoich` |
| CIMPL / MATEQ | 隐式 SiO₂ | 同 | `solver` / `wl_state` |
| DRIVER | 单压降温循环 | + polybaric + **TEMPRUN** | `driver.py` / `wl_driver.py` |
| MAIN | 任务 1–10 | + 任务 11 单压/多压切换 | `basalt1990/main.py`（1990）；`wl_langmuir/main.py`（Langmuir） |

## 2. 组分与相

- **相**：PLAG(1), OL(2), CPX(3)
- **显式组分**：CAA, NAAL, MGO, FEO, CAWO, TIO2
- **隐式**：SIO2（`CIMPL` 由矿物化学计量反推）
- 输入可为 7 列 CDAT；前 6 列归一化后进入 CSJ

## 3. KDCALC 公式差异（核心）

### 1990 — `ARHENF(A,B,T) = 10**(A/T) + B`

```fortran
FKDAJ(2,3)=ARHENF(2715.,-1.158,TK)    ! Ol MgO  → @1273K ≈ 134
FKDAJ(1,1)=ARHENF(2446.,-(1.122+0.2562*AN),TK)  ! Pl CAA
```

### Langmuir 1992 — `ARHENF(A,B,CP,T) = 10**(A/T + B + CP*P)`

```fortran
FKDAJ(2,3)=ARHENF(3740.,-1.87,0.0008,TK)   ! Ol MgO  → @1273K,P=0 ≈ 11.7
FKDAJ(1,1)=10.**(2446./TK -(1.122+0.2562*AN)+0.012*P)
```

Ol MgO Kd 在 1273 K 相差约 **10×**；1990 版典型 MORB 常全程 **liquid only**。

## 4. 调用顺序差异（重要）

### 4.1 BASALT.FOR (1990) DRIVER

```
TEMP = PARMS(1) + PARMS(5)     ! PARMS(5)=273.16 为 °C 偏移
循环:
  CALL STATE(NL,LIST,NERR)     ! 不更新 SPARM(1)，不预调 KDCALC
  MODEL 2: FLR *= FL; CSJ = CLJ
  TEMP += PARMS(3)
```

**STATE 入口**：仅 `KDCALC(1)`（清零 FKDAJ），循环内 `KDCALC(3)`（Pl 成分依赖）。

**Fortran 怪癖**：`KDCALC(2)`（Ol/Cpx 温变 Kd）**从未在 STATE 内调用** → 严格照抄时 Ol/Cpx Kd 恒为 0。

**Python 默认修正**（`basalt1990`）：

- `sync_sparm_temp=True`：每步 `SPARM(1)=TEMP`
- `fill_temp_kd=True`：STATE 入口在 `KDCALC(1)` 后再 `KDCALC(2)` + 一次 `KDCALC(3)`

### 4.2 BASALT+langmuir TEMPRUN

```
SPARM(1) = TEMP
CALL KDCALC(4)          ! T+P+An 全套 Kd
CALL STATE(...)
```

**STATE 入口仍为 `KDCALC(1)`** → 会 **清零** 刚由 mode 4 填好的 Kd！

内循环仅 `KDCALC(3)` 更新 Pl → 物理上 Ol/Cpx 在严格逐行移植时也会丢失 mode 4 的温压项。

**Python `wl_state.state_solve`**：入口改为 `kd_calc(4)`（**不**先 `kd_calc(1)`），与 TEMPRUN 意图一致，属 **有意的偏离字面 STATE**。

### 4.3 Polybaric DRIVER (langmuir)

```
PHI = P_HIGH (PARMS(5))
循环 PHI 降至 P_LOW (PARMS(6))，步长 DP (PARMS(4)):
  SPARM(2) = PHI
  TEMP = PARMS(1) + PARMS(5)   ! 注意：与 1990 相同写法，但 PARMS(5) 已被复用为 P_HIGH
  CALL TEMPRUN(...)
  CSJ = CLJ
  PHI -= DP
```

**PARMS 数组冲突**：langmuir 扩展把 `PARMS(5),PARMS(6)` 用作 **P_HIGH/P_LOW**，但单压分支仍写 `TEMP=PARMS(1)+PARMS(5)`，且任务 10 仍用 `PARMS(5)` 作 °C/K 切换——与 1990 不兼容。Python 端口将 **温度偏移** 与 **压力** 分开存储。

## 5. STOICH（Cpx 饱和）

| 条件 | 1990 | langmuir |
|------|------|----------|
| 判据 | `T = 2*(Mg+Fe)_cpx + Si_cpx ≥ 1.0` | P≤4 kbar: ≥1.0；P>4 kbar: ≥1.1 |

## 6. MATEQ

Fortran 主元行对角元被 **除两次**（`A(K,K)=1/AKK` 后又 `A(K,J)*=A(K,K)`）。Python 端口与 `wl_state` 均已改为 **只除一次**。

## 7. Python 引擎选型（Fig.2 / 验证）

| 引擎 | 模块 | 对应 Fortran | 备注 |
|------|------|--------------|------|
| `basalt1990` + 1990 Kd | `basalt1990/` | BASALT.FOR | MORB 多不结晶 |
| `basalt1990` + langmuir 注入 | `basalt1990_fc.py` | 混合 | FC 高 FA 时可能 nerr=1 |
| `wl_state` + TEMPRUN | `wl_driver.py`, `wl_fc_state.py` | BASALT+langmuir | 推荐 **字面 Langmuir 路径** |
| `heuristic` | `wl_partition.py` | 无直接 Fortran | Fig.2 默认；MORB 最稳 |

## 8. 诊断与测试命令

```powershell
$env:PYTHONPATH="d:\python-learn\pyAOBS\pyAOBS"
py -3.11 petrology/validation/research_basalt_fortran.py
py -3.11 petrology/validation/compare_kd_three.py
py -3.11 petrology/validation/test_basalt1990_fortran.py
py -3.11 petrology/validation/test_wl_langmuir_kd.py
py -3.11 petrology/validation/test_wl_state_fc.py
py -3.11 -m petrology.fc.basalt1990.main --batch --sample kinzler1997_morb_primary --model 2 --ti 1200 --tf 900 --dt -10
```

## 9. 待研究 / 待实现

1. ~~**langmuir MAIN 任务 11**~~ → `petrology/fc/wl_langmuir/`（`python -m petrology.fc.wl_langmuir`）
2. **`literal_state=True`** 模式：复现 Fortran STATE 入口 `KDCALC(1)` 行为，用于与文献逐行对照
3. **FC 数值稳定**（已实现）：`wl_state_stabilize.py` — NR `max_fa_delta`、Ol→Pl→Cpx 顺序、每步 `max_solid_fraction`（默认 10%）；TEMPRUN/FC 默认 `DEFAULT_FC_STABILIZE`
4. **CDAT 样例库**（已实现）：`petrology/fc/cdat_library.py` + `petrology/data/cdat/`；`py -3.11 petrology/validation/build_cdat_library.py`
5. **与 Petrolog3 / W&L 原文实验表** 定量对比（若找到论文附录数据）

## 10. 参考文献

- Weaver J.S., Langmuir C.H. (1990) Calculation of phase equilibrium in mineral-melt systems. *Computers & Geosciences* 16(1), 1–19.
- Langmuir C.H. et al. (1992) How deep do common basaltic magmas form and differentiate? *JGR*.
- Korenaga J. et al. (2002) Methods for resolving the origin of large igneous provinces. *JGR* 107(B9).
