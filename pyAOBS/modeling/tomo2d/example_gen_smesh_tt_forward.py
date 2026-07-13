#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小工作流示例：gen_smesh（均匀梯度 + 均匀网格）→ tt_forward。

用法（在含输入/输出文件的目录下，或改下面 WORK_DIR）:
  python -m pyAOBS.modeling.tomo2d.example_gen_smesh_tt_forward

或指定 tomo2d 可执行文件所在目录:
  set PYAOBS_TOMO2D_BIN=/path/to/tomo2d/Distrib   # Windows: set
  export PYAOBS_TOMO2D_BIN=/path/to/tomo2d/Distrib # Linux/macOS
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from pyAOBS.modeling.tomo2d.tomand import TomoAnd, TomoCommandError


# ---------- 按你的工区修改 ----------
WORK_DIR = Path.cwd()
BIN_DIR = os.environ.get("PYAOBS_TOMO2D_BIN") or os.environ.get("TOMO2D_BIN")

# 输出/中间文件名（相对 WORK_DIR）
SMESH_OUT = "demo.smesh"
# 若你的 tt_forward 不需要几何文件，设为 False 并令 GEOM_FILE 忽略
USE_GEOM_FILE = True
GEOM_FILE = "demo.geom"

# 均匀初始模型与网格（单位与 tomo2d 一致，一般为 km / km/s 等，以你数据为准）
V0 = 1.5
GRADIENT = 0.0
NX, NZ = 51, 31
XMAX, ZMAX = 50.0, 15.0

# 正演输出
TTIME_OUT = "demo.tt"
# --------------------------------


def main() -> int:
    work = WORK_DIR.resolve()
    work.mkdir(parents=True, exist_ok=True)

    tomo = TomoAnd(bin_path=BIN_DIR) if BIN_DIR else TomoAnd()

    smesh_path = work / SMESH_OUT
    ttime_path = work / TTIME_OUT
    geom_path = work / GEOM_FILE if USE_GEOM_FILE else None

    if USE_GEOM_FILE and geom_path is not None and not geom_path.is_file():
        print(f"未找到几何文件: {geom_path}")
        print("请放置该文件或设 USE_GEOM_FILE = False。")
        return 1

    try:
        print(">>> gen_smesh …")
        r1 = tomo.gen_smesh(
            vel_opt="uniform",
            v0=V0,
            gradient=GRADIENT,
            grid_opt="uniform",
            nx=NX,
            nz=NZ,
            xmax=XMAX,
            zmax=ZMAX,
            out_file=SMESH_OUT,
        )
        if r1:
            print(f"gen_smesh 完成，网格已写入 {SMESH_OUT}（由 out_file 参数落盘）。")

        print(">>> tt_forward …")
        r2 = tomo.tt_forward(
            smesh=str(smesh_path),
            geom=str(geom_path) if geom_path is not None else None,
            out_opts={"ttime": str(ttime_path)},
        )
        if r2 and r2.stdout:
            print(r2.stdout.strip())

    except ValueError as e:
        print(f"参数错误: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"{e}", file=sys.stderr)
        return 3
    except TomoCommandError as e:
        print(e, file=sys.stderr)
        if e.stdout:
            print("stdout:\n", e.stdout, file=sys.stderr)
        return 4

    print(f"完成。慢度网格: {smesh_path}，走时输出: {ttime_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
