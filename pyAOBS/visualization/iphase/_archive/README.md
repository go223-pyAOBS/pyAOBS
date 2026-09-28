# iphase 归档

正式入口：

```bash
python -m pyAOBS.visualization.iphase.gui
```

兼容旧命令（转发到 Qt）：

```bash
python -m pyAOBS.visualization.iphase.iphase_gui
```

## 目录说明

| 路径 | 说明 |
|------|------|
| `iphase_gui_tk.py` | 原 Tk 单体主 GUI（对照用，勿作启动入口） |
| `drafts/` | 草稿脚本与历史文档（`test1d.py`、`md2pdf*`、`IMPLEMENTATION_STRATEGY.md` 等） |
| `formula_build/` | `TIME_DIFF_FORMULAS` 的 TeX/PDF 与编译残留；正文仍在包根 `TIME_DIFF_FORMULAS.md` |
| `datasets/txin/` | 大批量 OBS `tx_*.in` 本地合集 |
| `datasets/test_1d/` | 1D/正演本地试验输出 |
| `datasets/rin_gui/` | `r.in` 编辑器试验用小样例 |
| `datasets/examples_extra/` | `examples/` 中移出的 `.bak` / 衍生 CSV |

包根保留：核心库 `.py`、`gui/`、`docs/`、`examples/`（精简演示数据）、`TIME_DIFF_FORMULAS.md`。
