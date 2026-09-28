# wave2d — 2D 弹性 OBS 道集正演

独立 Virieux **P–SV** 弹性波场（水柱 `μ=0`），**不改** tomo2d 射线核。用于与射线走时对照、检查水柱多次与转换相。

## 快速开始

```bash
# CLI（默认 OBS 为源、水中记压力；折合剖面 0–12 s）
python -m pyAOBS.modeling.wave2d.run_gather_017 \
  --work <含 true_vp/true_vs/seafloor 的工区> \
  --out <输出目录，默认工区旁 wave_fwd> \
  --skip-ray

# 或在 TOMO2D GUI：左侧「13) wave2d」
python -m pyAOBS.modeling.tomo2d.gui
```

API：`run_obs_gather(...)`（`run_gather_017.py`）。

## 默认几何（互易）

| | |
|--|--|
| **源** | OBS 海底竖力 `vz`（`obs_z≈2` km） |
| **检波** | 浅水压力（约 `max(2·dx, src_z)`） |
| **含义** | 走时 ≡「浅水炮 → OBS」；源不在水柱内，但水柱多次仍应按 \(nH/v\) 出现 |

可选 `--layout water-obs`：浅水多炮 → 单台 OBS（慢，每炮一次正演）。

## 图上叠线

- **射线**（可选）：`syn_*.dat` 中 PPP(0)/PmP(1)/PPS(7) 等到时 + Ricker delay  
- **水柱理论曲线**：\(t=\sqrt{x^2+(nH)^2}/v_w+\mathrm{delay}\)，默认 \(n=1,3,5\)，\(H=2\)、\(v_w=1.5\)  
- **折合**：默认 \(v_\mathrm{red}=8\) km/s，纵轴 **0 → tred_max（12 s）**

## 主要依赖文件

工区需：`true_vp.smesh`、`true_vs.smesh`、`seafloor.refl`（或 GUI 中指定路径）。

吸收：默认 C-PML（`absorb=pml`）；可选 Cerjan。

## 包内模块

| 文件 | 作用 |
|------|------|
| `elastic2d.py` | 传播、源/检、Gather |
| `pml.py` | C-PML |
| `grid.py` | smesh → 规则网格，水柱强制 Vs=0 |
| `run_gather_017.py` | OBS 道集 CLI / `run_obs_gather` |
| `io_smesh.py` | 解析 smesh / 拾取 / 射线 |

## 说明

- 走时**不拾取**自波场；叠点来自 tomo2d `syn`（有则叠）。  
- 大体积 `wave_fwd/*.npz` 默认不进 git，可本地重跑生成。  
- GUI 表单字段见 tomo2d [`docs/HELP.md`](../tomo2d/docs/HELP.md)「13) wave2d」。
