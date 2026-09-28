# idata — 数据转换与道头编辑工区（工程化）

独立于 `raw2sac` 的 PySide6 工区（与 `relocation` 同级）。

**使用说明（几何：sx/sy/gx/gy、scalco/scalel、counit、offset）**：[`docs/HELP.md`](docs/HELP.md)  
GUI：工具栏 **帮助**（`F1`）。

## 启动

```bash
python pyAOBS/processors/idata/run.py
# 或
python -m pyAOBS.processors.idata.gui
```

Workbench `data.gui` → `processors/idata/run.py`。

## 工区布局

```
<workdir>/
  meta/idata_project.json   # 工程状态
  inputs/                   # 原始 RAW/SAC/UKOOA/config
  outputs/                  # SEGY/SU 产出
  convert/                  # 中间产物（可选）
  cache/
```

工具栏：**新建工区 | 打开工区 | 保存工区 | 帮助 | 退出** | 打开 SEGY/SU / 保存数据 / 另存为 / 导出 SU | 清空日志

## 阶段

1. **转换**（分步 + 一键流水线；目标 SEGY 或 SU）
2. **道头编辑**
3. **工区几何**

几何约定：**炮 = sx/sy，OBS = gx/gy**（旧对调数据可选 `geom=obs` 解释）。

工程 JSON 记录：转换表单、当前数据路径、geom 模式、阶段页。

## 一键 CLI（`processors/raw2sac/`）

```bash
python sac2segy.py data.sac shots.ukooa sac2y.ini out.segy
python sac2su.py   data.sac shots.ukooa sac2y.ini out.su
python raw2segy.py rawfile 1000 256 shots.ukooa sac2y.ini out_dir
python raw2su.py   rawfile 1000 256 shots.ukooa sac2y.ini out_dir
python obem2segy.py obem.ini shots.ukooa sac2y.ini out_dir
python obem2su.py   obem.ini shots.ukooa sac2y.ini out_dir
python segy2su.py in.segy out.su
```
