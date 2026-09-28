# TOMO2D GUI 快速说明

## 启动

```bash
python -m pyAOBS.modeling.tomo2d
python -m pyAOBS.modeling.tomo2d.gui
```

Workbench：插件 **tomo2d.gui**。

## 快捷键

| 键 | 作用 |
|---|---|
| **F1** | 打开本帮助（非模态；已打开则前置） |

顶栏「帮助」按钮与 F1 相同。帮助窗可切换 **GUI 快速说明** 与各程序章节（`help_docs` / TomoHelp）。

## 工区

顶栏常显按钮（无菜单栏）：

1. **新建工区**：选目录 → 自动创建 `meta/`、`inputs/`、`outputs/`、`runs/`、`cache/`，并写 `meta/tomo2d_project.json`
2. **打开工区**：选 `tomo2d_project.json`（或含该文件的目录）
3. **保存工区**：把当前表单写入工程 JSON
4. **退出**：关闭窗口（未保存时提示）

路径尽量相对工区根（如 `inputs/geom.dat`、`outputs/model.smesh`）。

示例：`examples/meta/tomo2d_project.json`。

## 主界面

- **顶栏常显**：新建/打开/保存工区、导出/导入配置、绘制 smesh、反演分析、**反演监视**、**模型挑选**、**模型对比**、**帮助**（F1）、**写 tomo2d_gui.log**、退出
- **绘制 smesh**：点按钮打开图窗（当前页有 smesh 则直接画；没有则打开空窗，再「打开…」/拖入/粘贴，**不**立刻弹选文件框）。把 `.smesh`、**v.in**、**.grd/.nc** 或界面/refl 拖到**主窗口或该图窗**即可（可一起拖；图窗内也可「叠加界面…」）。色标下拉 **vp** / **vs** / **vpvs**；勾选等值线时画对应表。Vp/Vs 一般用 grd，没有 smesh。
- **顶栏可折叠**：左侧路径与右侧并行/策略同排同高
- **左侧命令列表**：竖排、文字横向完整显示（字号约 14px）；中间参数面板较窄（≤580px）；右侧预览/日志约 380–720px
- **右侧**：命令预览 + 执行日志（勾选「写 tomo2d_gui.log」时摘要写入工作目录）
- **参数说明**：鼠标悬停**左侧命令**或表单控件查看（`param_hints`）；移开即消失。F1 打开完整帮助
- **正演/反演页**：默认只展开「常用」；射线弯曲、更多输出、反射/正则/联合重力等收在折叠组里，按需展开
- **其它命令页**：同样按「常用 / 高级」折叠（gen_*、edit/stat、pipeline、tx、棋盘格、蒙特卡洛、**wave2d**）
- **9) tx.in→tomo2d**：支持**多个** tx.in 合并为同一对 `ttimes.dat` / `geom.dat`（共用 `station.lis`；「添加多个…」或多行路径）。OBS 列表与「预览 tx.in」共用，勾选同时用于显示与转换
- **13) wave2d**：OBS 弹性道集正演（独立模块，不改射线核）。默认互易几何：OBS 竖力源、水中记压力；折合图 0–12 s，并叠水柱理论曲线 \(t=\sqrt{x^2+(nH)^2}/v\)（\(n=1,3,5\)）。详见 [`../wave2d/README.md`](../wave2d/README.md)

### 并行与策略（顶栏）

对应 `src/README_OMP_BUILD.md` 中的环境变量，由 GUI 写入子进程（覆盖同名系统变量）。

- **常显**：OMP 线程、`tt_inverse`/`tt_forward` 并行、**图论FS枚举**（默认开；不勾选=原扫 C/B），以及预设（快速 / 稳健 / 对拍 / 调试）
- **加速策略 | 开发者/对拍**（同一行，默认折叠）：复用/C2F；**LSQR 列预条件默认关**；**灵敏度加权默认关**；**线搜索默认关**；Legacy/DIAG（日常勿开）

| 界面 | 环境变量 |
|---|---|
| OMP_NUM_THREADS | `OMP_NUM_THREADS`（空则不覆盖） |
| tt_inverse / tt_forward OMP | `TOMO2D_INV_OMP` / `TOMO2D_FWD_OMP` |
| Legacy baseline | `TOMO2D_INV_LEGACY_BASELINE` |
| LSQR 列预条件 + D_j 上限 | `TOMO2D_INV_LSQR_PRECOND` / `TOMO2D_INV_LSQR_PRECOND_MAX` |
| 灵敏度加权 T/R + κ | `TOMO2D_INV_SENS_WEIGHT` / `TOMO2D_INV_SENS_KAPPA` |
| 线搜索（Armijo） | `TOMO2D_INV_LINESEARCH`（可选 `TOMO2D_INV_LS_C` / `LS_RHO` / `LS_AMIN`） |
| LM 信赖域 | `TOMO2D_INV_LM`（可选 `TOMO2D_INV_LM_LAMBDA` / `LM_UP` / `LM_DOWN` / `LM_LMAX`） |
| LSQR 迭代硬上限 | `TOMO2D_INV_LSQR_MAXITER`（≤0 不额外封顶） |
| LSQR ATOL | `TOMO2D_INV_LSQR_ATOL`（未设则用默认 `1e-3`） |
| 前向复用 + 阈值 | `TOMO2D_INV_REUSE_FORWARD` / `TOMO2D_INV_REUSE_THRESH` |
| Coarse-to-fine + C2F_* | `TOMO2D_INV_COARSE2FINE` 等 |
| 图论 FS 邻居枚举 | `TOMO2D_GRAPH_FS_ENUM`（0=原扫 C/B） |
| 诊断哈希 | `TOMO2D_INV_DIAG` |

预览 tt_forward / tt_inverse / pipeline 时会列出将下发的环境变量。开始运行时执行日志会先打一行「并行: …」。开 `tt_inverse` OMP 后不再刷逐炮 `*` `.`；较新二进制会打印 `threads=` / `nsrc=` 以及 `ray tracing k/N sources (OMP)`。

### tt_inverse 运行包输出归类

勾选「可复现运行包」时，结束后将 `runs/ttinv_*/outputs/` 扁平文件归入：

| 子目录 | 内容 |
|---|---|
| `models/` | `*.smesh.<iter>.<iset>`、`*.refl.*` |
| `residuals/` | `*.tres.*`、`*.rgrav.*` |
| `rays/` | `*.ray.*`（需较高 `-o`） |
| `dws/` | dws / grav_dws |
| `logs/` | `*.log` |
| `other/` | 未识别文件 |

**不**自动写 `final.smesh`。选用哪次模型请结合 `-L` 日志、反演参数与地质认识后自行决定。`manifest.json` 的 `output_files` 带 `kind` 字段。

### 反演监视（准实时）

运行 `tt_inverse`（运行包模式）时自动打开非模态监视窗；也可点顶栏「反演监视…」。

- 每约 2 s 尾随 `-L`：画 χ² / pred χ²、RMS 与速度粗糙度（行序）；无 `.ray` 时不扫描射线文件
- 检测新写出的 `*.smesh.<iter>.<iset>`（含 `models/`）并刷新速度图
- **子进程 stdout/stderr 按行流式**刷到执行日志；监视窗状态条做节流。开 OMP 时不再刷逐炮 `*` `.`，改为 `parallel ray tracing enabled` 与 `ray tracing k/N sources (OMP)`（需较新二进制）
- **抽样射线**：**默认开启**，**按 OBS/炮着色**（细线半透明，压在速度场之上、界面/反射面之下）。尚无 `.ray` 时只画速度场。需 `out_level (-o)≥2`。可随时关掉。
- **等值线**：勾选后随色标切换内置表（vp=`contour_p`，vs=`contour_s`，vpvs=`contour_vpvs`；A 标注、C 细虚线）；绘制 smesh / 监视 / 模型挑选共用
- **DWS 遮罩**：无覆盖留白；有覆盖按 log(DWS) 透明（越大越实）。`-K` 在反演结束才写出。色标仍是速度。查找顺序见下文「DWS 遮罩」。
- 若该轮写出了 `*.refl.<iter>.<iset>`，监视与模型挑选会叠**该轮更新后的反射面**（否则退回表单 `-F`）
- **status.jsonl**：C++ 每轮追加 NDJSON（环境变量 `TOMO2D_INV_STATUS_JSONL`，GUI 默认 `outputs/status.jsonl`）；监视窗用其刷新末态。需使用含该改动的 `tt_inverse` 二进制
- **进度条**：按 `iter / niter`（表单 `-I`）显示；曲线末点高亮
- **快捷打开**：`models/`、`-L` 日志、`status.jsonl`、输出目录
- **结束提示**：进程结束后绿色条提示末态，并引导「模型挑选…」
- **最新写出 ≠ 最优模型**；不画全量射线
- 要看到中间模型刷新，请勿勾选 `print_final_only (-l)`，并确保填写了 `-O`（运行包默认有）

### 模型挑选助手

顶栏「模型挑选…」或监视窗同名按钮。先选一次运行包（下拉为工区 **`runs/`**，默认最近一次 ★；**浏览…** 可自选工区外的包），再列出该次各轮 `*.smesh.<iter>.<iset>`，并挂上 `-L` / `status.jsonl` 的 χ²、RMS、pred χ²、粗糙度；点击预览 **上方两行走时拟合（折射 / 反射残差 vs 接收点 X，与速度场共用横轴）**、下方速度场。拟合数据来自该轮 `{out}.tres.<iter>.<isrc>`（需 `-O` 且 `out_level≥1`）。**新二进制**写出三列 `rcv_x residual raytype`（0=折射、1=反射），拟合图**首选第三列**；旧两列 `.tres` 仍按行序从运行包 `-G`（`inputs/data.dat`）补震相。开了 `-R` 时，拟合图上黑叉为本轮剔除点；清单在 `{out}.outliers.<iter>.<iset>`，末轮另有 `{out}.outliers.final`（需新二进制；旧运行包没有）。`-R` 每轮会加回全体拾取再判，故「最终」以末轮清单为准。

- 每次勾选「可复现运行包」的 tt_inverse 都会在 `runs/ttinv_…/` 新建目录；挑选按运行包隔离，不会把历次反演混在一张表里
- **浏览…**：自选运行包（当前工区 `runs/` 以外亦可）。可点包根、`outputs/` 或 `models/`；刷新时仍保留已选的自选包
- **右键写入表单 / 打开所在目录**：在预览速度图上右键，把当前 smesh / 配套反射面写入 `inv.mesh` 等字段，或在资源管理器中打开所在文件夹
- **表单路径**：仅当未建运行包时，退回绑定当前表单的 `inv.out_root`
- **不**自动判定最优；请结合指标与地质认识选用。两套模型的差值：预览图右键「添加到对比模型」，或顶栏「模型对比…」

### 模型对比

顶栏「模型对比…」、监视窗 / 输出页同名按钮。集合与 A/B 在点「统计均值…」或「绘制差值」时才读盘。网格须一致。

- **集合**：「统计均值…」画 **上均值 Vp、下误差 σ**（两套也行）。各网格点只平均该处 **DWS>0** 的成员（无覆盖不拉低均值）。均值图用 Vp 公用等值线；误差图按 σ 量程取等值线（与「叠加等值线」开关相同），色标为 **0→蓝绿（0 为白）**。配套 `*.refl.<iter>.<iset>` 不少于 2 条时，均值图叠 **平均反射面 ±σ**（红实线 / 点线）。图上 DWS 为有覆盖成员的平均。
- **保存 / 写回**：统计后在 **均值图或误差图上右键点一下**（位移很小；右键拖仍是缩放）：
  - 均值图：保存均值速度、平均反射面、mean±σ；并可 **写入表单**（如 `inv.mesh` / `inv.refl_file`）
  - 误差图：保存速度 σ，以及同样的界面项
  默认文件名在工区 `outputs/`（如 `ensemble_mean.smesh`、`ensemble_std.smesh`、`ensemble_mean.refl`）。
- **差值**：指定 **A（基准）** 与 **B（对比）**，绘制 **B−A**（上 ΔV，中 B，下 A；蓝=变快、红=变慢）。DWS 为 **A∩B**（两侧都有覆盖才显示，权利用 `min`）。ΔV 不叠速度等值线。
- **色标 ±km/s /「随数据」**：只作用于 **ΔV 与误差 σ**，**不改均值 Vp**（均值仍用顶栏速度色标）。改完立即重绘，不必再点统计/绘制差值。
  - ΔV：对称 ±；随数据 = ±本图 `|max|`
  - σ：0 到该 km/s；随数据 = 0–本图 σ 最大；不勾选则用 ±km/s 的正半幅作上界
- **DWS 列表…**：核对每个 smesh 实际用到的 DWS 文件。
- **速度图右键**：已加载 smesh 时可 **写入表单**、**添加到对比模型**、**打开所在目录**。第一个加入的为 **A**，第二个不同模型为 **B**，自动打开差值图。再添加则 A←原 B、B←新模型。拖入 / Ctrl+V 多个文件写入集合（≥3 个自动统计）。

## QC：棋盘格 / 蒙特卡洛

- **10) 棋盘格测试**：一键「背景+棋盘 → 正演 → 自背景反演」，输出真/恢复百分异常场（`runs/checkerboard_*`）。「棋盘预览图…」弹出窗口：**上**扰动 ΔV，**中**棋盘后 Vp，**下**棋盘前 Vp。可勾选叠加等值线（中/下速度图）与 DWS 遮罩（三幅共用，按背景 smesh 就近查找）。
- **11) 蒙特卡洛**：模型方式 **smesh**（默认分段随机 1D）或 **v.in**（点「选层…」在速度底图上勾选海底/基底/Conrad/莫霍并给层位蒙版；勾选层同时扰动厚度与顶底速度，-F 用莫霍）。选中一项后锁定另一模型路径。可叠加走时噪声；预览右侧叠绘全部 N 条 1D 与 Moho。「蒙特卡洛结果图…」画均值 Vp、误差 σ 与界面均值 ±σ；结果在 `runs/montecarlo_*`
- 反演控制参数均取自 **tt_inverse** 页（输出路径由 QC 流程覆盖）

## 绘图与子窗

通用鼠标（速度场 Matplotlib 与分析图 pyqtgraph 相同）：**滚轮缩放 · 左拖平移 · 右拖缩放 · 双击复位**。结果提示与帮助窗均为 **非模态**。

### DWS 遮罩

无覆盖（DWS≤0）留白；有覆盖按 log(DWS) 透明（越大越实）。水层与地形线仍画。色标仍是速度（或 ΔV / σ）。

- **绘制 smesh**：须在图窗「指定 DWS…」（或拖入/粘贴），**不**自动查找；文件名不限。
- **反演监视 / 模型挑选 / 模型对比 / 棋盘预览**：勾选后按 **每个** smesh 就近查找，**不**扫整个 `runs/`：
  1. 与 smesh **同目录**（含同目录 `dws/`，用户自选模型常见）
  2. 该次 GUI 运行包 `outputs/dws/`（以及旧式 `outputs/dws.dat`）
  3. 监视/挑选给出的运行包目录或 `-O`
  4. **最后**才用表单 `inv.dws_file`（避免集合里所有模型共用一份）
- 对比窗「DWS 列表…」可列出每个 smesh 实际命中的文件。`tt_inverse -K` 在反演**结束**才写出。

### 速度场

smesh / 监视 / 挑选 / 差值 / 集合统计：Matplotlib `imshow`。速度色标下拉 **vp** / **vs** / **vpvs**（内置 CPT）。

### 反演分析（Pareto / 叠画 / 参数影响）

原生 **pyqtgraph**。左击点/条/曲线，**各分析窗同步高亮**（再点空白取消）；**Ctrl+单击**加减选，**Shift+单击**按下标连选，**Shift+左拖**框选。叠画与参数影响不附图例。

- **右击点一下**已选（位移很小）出菜单：**绘制迭代曲线**、添加至模型对比、反演参数列表；单选才可绘制末轮模型、打开目录。**右拖**空白处仍为缩放。
- **参数影响**横轴为对数，刻度用十进制（写 **150**、**600**，不写 1.5×10² / 6×10²），并标出图上的实际参数值。
- **末步汇总表**：按 score 升序（越小越好）；Ctrl/Shift 多选、右击菜单同上。

## 日常交互

- **路径拖放**：各 `PathRow`（含顶栏 bin/work_dir）可拖入文件或目录；目录模式若拖入文件则取父目录
- **最近路径**：路径行右侧 ▾；浏览/拖放成功后记入（QSettings `pyAOBS/tomo2d`）
- **布局记忆**：主窗、反演监视、模型挑选、模型对比关闭时保存几何与分割条，下次恢复

## 默认少写盘 / 少刷屏

新建工区时 GUI 默认关闭非必要输出（不覆盖你已保存的配置）：

- **tt_inverse**：`out_level` / `verbose` 留空；**`print_final_only (-l)` 默认开**（少写中间 smesh）
- **tt_forward**：`out_ray` 及多余输出留空；`verbose` 留空
- **仍保留**：运行包 `-L`、轻量 `status.jsonl`（曲线监视）；抽样射线 / 中间模型需自行提高 `out_level` 或关掉 `-l`

## 正演页与反演的关系

- **主路径**：实测数据 →「6) tt_inverse」（内部每轮已做正演）。不必先跑 tt_forward。
- **tt_forward**：独立工具（算走时/射线）。合成数据时展开页内折叠区「合成数据（可选）」：可勾选写入反演空位，或手动「写入 inv.mesh / inv.data」（默认不自动写回）。
- **tt_inverse**：「上游填充」、监视就绪检查、跳转 gen_damp/vcorr/dcorr；「同步 -N」仅在与正演页对齐参数时用。
- **正演 → 反演写回**（合成流程）：除 `smesh` / `out_ttime` / `-N` / `-F` / 海底（正演 `-B` → 反演 `-Y`）外，还会把空位填上 **转换面**（正演 `-X` → 反演 `-B`）、**Vs 网格 `-U`** 与 **`-k`**。
- **`-A`（贴面反射）**：勾选后反射**沿界面贴面**走；不勾时远偏移反射可以**穿幔成初至**。改的是路径，不是算得更细；正演开 `-A` 时 OpenMP 会退回串行。
- **`-U`（vsmesh）**：独立 Vs，与 `-M` 同维。6/7/8 真双场时 P 段读 `-M`、S 段读 `-U`；有 `-U` 可不传 `-k`。
- **pipeline**：可选 `gen_smesh -> tt_forward -> tt_inverse`（合成链）；Pipeline 内反演暂不挂运行包监视。
- **右侧「输出」页**：浏览 `outputs/`、`runs/`，任务结束后自动刷新
- **13) wave2d**：波场对照用；不参与走时反演方程。输出目录默认 `wave_fwd/`（相对工区）。

## 13) wave2d（弹性 OBS 道集）

独立模块 `pyAOBS.modeling.wave2d`，经本页调用 `run_obs_gather`；**不改** tomo2d 射线/反演核。

| 区 | 要点 |
|----|------|
| **模型** | `Vp`/`Vs` smesh、`seafloor.refl`；输出目录默认 `wave_fwd`；可选 `syn` 叠射线到时 |
| **OBS / 水柱** | OBS 坐标与源深；偏移与道距；水深 \(H\)、水速 \(v\)（理论曲线） |
| **正演与显示** | `dx`/`tmax`/`f0`；折合速度与 **0→tred_max**（默认 12 s）；`layout=reciprocal`（默认）或 `water-obs`；吸收 `pml`/`cerjan` |

- **reciprocal**：源在 OBS 海底竖力，检波在浅水压力（走时 ≡ 浅水炮→OBS）。
- **water-obs**：浅水多炮 → 单台 OBS（慢）。
- 图上水柱曲线：\(t=\sqrt{x^2+(nH)^2}/v+\mathrm{delay}\)，\(n=1,3,5\)。
- 勾选「不重跑 tt_forward」时若已有 `syn` 仍叠点；「快速」会放粗 `dx`、缩短 `tmax`。

CLI：`python -m pyAOBS.modeling.wave2d.run_gather_017 --work <工区>`。详见 [`../wave2d/README.md`](../wave2d/README.md)。

## gen_smesh：从 v.in 做 topo=0（水+壳）

默认 zelt 仍把 `ilayer` 写入 smesh 的 **topo**（网格挂在海底，z 为海底以下）。要「海面网格、海底以上水速、以下壳幔」：

1. `vel_opt=zelt`，`ilayer` = **v.in 里的海底层**（如 `vpfd41.in` 的层 2，不是海面层 1）
2. 勾选 **挂海面 topo=0 (-S)**（默认不勾，原 zelt 行为不变）
3. `z_file` 改为 **海面起算的绝对深度**，第一点约 0；`zmax` 要比原来的「海底以下深度」大约多一个水深
4. 水速用 `-Q`（默认 1.5）；莫霍仍用 `-F`
5. 可选 **seafloor_out (-G)**：写出海底界面，给正演 `-B` / 反演 `-Y`（与 tt_inverse 的 `-G` 走时文件不是一回事）

「上游填充」会把 `gen.seafloor_out` 填到正演/反演海底空位。控件悬停见 `param_hints`；命令行细节见帮助窗 **gen_smesh** 章节。

## 关于

更完整的模块说明见上级 [`README.md`](../README.md)。程序参数细节见帮助窗中的 TomoHelp 各章节，或 `help_docs.py`。

水层 2/3、台侧多次 4/5 的正反演已接到 `tt_forward` / `tt_inverse`（正演 `-B`、反演 `-Y`；反演 `-B` 仍是转换波）。策略笔记见 [`WATER_MULTIPLES.md`](WATER_MULTIPLES.md)。
