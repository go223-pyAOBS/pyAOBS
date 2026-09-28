"""RAYINVR ``v.in`` 统一入口。

文本解析委托 ``pyAOBS.model_building.read``（``read_vin_model`` /
``write_vin_model``）；本模块提供：

- 文件探测（合并 imodel / zplotpy 启发式）
- 字典读写
- ``ZeltVelocityModel2d`` / vedit ``Model`` 加载
- 字典 ↔ 可编辑 ``Model`` 互转

供 vedit / zplotpy / imodel / rayinvr 共用，避免多套 v.in 解析分叉。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Type, Union

import numpy as np

from pyAOBS.model_building.read import read_vin_model, write_vin_model

PathLike = Union[str, Path]


def is_vin_path(path: PathLike) -> bool:
    """按文件名判断（``v.in`` / ``*.in``；兼容旧后缀 ``*.vin``）。

    排除 RAYINVR 其它常见 ``*.in``（``r.in`` / ``tx.in`` / ``f.in`` 等）。
    """
    name = Path(path).name.lower()
    if name == "v.in" or name.endswith(".vin"):
        return True
    if not name.endswith(".in"):
        return False
    # 非速度模型的标准 RAYINVR 输入
    if name in {
        "r.in",
        "tx.in",
        "f.in",
        "d.in",
        "p.in",
        "rec.in",
        "vm.in",
        "i1.in",
        "i2.in",
    }:
        return False
    return True


def is_vin_content(path: PathLike) -> bool:
    """轻量内容探测：前几行是否像层号 + 三行组。"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            first = f.readline().strip()
            if not first:
                return False
            parts = first.split()
            if not parts:
                return False
            try:
                layer_num = int(parts[0])
            except ValueError:
                return False
            if not (1 <= layer_num <= 200):
                return False
            second = f.readline().strip()
            if not second:
                return False
            parts2 = second.split()
            if not parts2:
                return False
            try:
                int(parts2[0])
            except ValueError:
                return False
            third = f.readline().strip()
            return bool(third)
    except OSError:
        return False


def is_vin_file(path: PathLike) -> bool:
    """文件名或内容任一命中即视为 v.in。"""
    p = Path(path)
    if not p.is_file():
        return False
    return is_vin_path(p) or is_vin_content(p)


def read_vin_dict(path: PathLike) -> dict:
    """读取为 ``read_vin_model`` 字典结构。"""
    return read_vin_model(str(path))


def write_vin_dict(path: PathLike, model_dict: dict) -> Path:
    """将字典写回 Zelt ``v.in``。"""
    p = Path(path)
    write_vin_model(str(p), model_dict)
    return p


def load_zelt_model(path: PathLike):
    """加载 ``ZeltVelocityModel2d``（可视化 / 射线用）。"""
    from pyAOBS.model_building.zeltform import ZeltVelocityModel2d

    return ZeltVelocityModel2d(model_file=str(path))


def _clean_triple(
    xs: list, ys: list, flags: list
) -> tuple[list[float], list[float], list[int]]:
    out_x: list[float] = []
    out_y: list[float] = []
    out_f: list[int] = []
    n = min(len(xs), len(ys), len(flags))
    for i in range(n):
        x, y, f = xs[i], ys[i], flags[i]
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if np.isnan(xf) or np.isnan(yf):
            continue
        try:
            fi = int(f)
        except (TypeError, ValueError):
            fi = 0
        out_x.append(xf)
        out_y.append(yf)
        out_f.append(fi)
    return out_x, out_y, out_f


def vin_dict_to_edit_model(model_dict: dict, *, model_cls: Optional[Type] = None) -> Any:
    """字典 → vedit ``Model``（可编辑节点结构）。"""
    if model_cls is None:
        from pyAOBS.modeling.vedit.model import Model as model_cls  # type: ignore

    import importlib

    mod = importlib.import_module(model_cls.__module__)
    TripleLine = mod.TripleLine
    Layer = mod.Layer
    EndLayer = mod.EndLayer

    layers = []
    n = len(model_dict["layer_boundary_x"])
    for i in range(n):
        dx, dz, df = _clean_triple(
            model_dict["layer_boundary_x"][i],
            model_dict["layer_boundary_z"][i],
            model_dict["layer_boundary_flags"][i],
        )
        ux, uv, uf = _clean_triple(
            model_dict["upper_x_velocities"][i],
            model_dict["upper_velocities"][i],
            model_dict["upper_velocity_flags"][i],
        )
        lx, lv, lf = _clean_triple(
            model_dict["lower_x_velocities"][i],
            model_dict["lower_velocities"][i],
            model_dict["lower_velocity_flags"][i],
        )
        layer = Layer(
            [
                TripleLine([dx, dz, df]),
                TripleLine([ux, uv, uf]),
                TripleLine([lx, lv, lf]),
            ]
        )
        layer.fix_depth()
        layer.fix_v_top()
        layer.fix_v_bot()
        layers.append(layer)

    bx = model_dict.get("bottom_boundary_x") or []
    bz = model_dict.get("bottom_boundary_z") or []
    bf = model_dict.get("bottom_boundary_flags") or ([0] * len(bx))
    ex, ez, ef = _clean_triple(bx, bz, bf)
    if not ex:
        # 兜底：用末层深度两端造平底
        last = layers[-1].depth
        ex = [float(last.x[0]), float(last.x[-1])]
        ez = [float(last.y[0]) + 1.0, float(last.y[-1]) + 1.0]
        ef = [0, 0]
    end = EndLayer([TripleLine([ex, ez, ef]), None, None])
    end.fix_depth()
    layers.append(end)
    return model_cls(layers)


def edit_model_to_vin_dict(model: Any) -> dict:
    """vedit ``Model`` → ``read_vin_model`` 字典。"""
    d = {
        "layer_boundary_x": [],
        "layer_boundary_z": [],
        "layer_boundary_flags": [],
        "upper_x_velocities": [],
        "upper_velocities": [],
        "upper_velocity_flags": [],
        "lower_x_velocities": [],
        "lower_velocities": [],
        "lower_velocity_flags": [],
        "bottom_boundary_x": [],
        "bottom_boundary_z": [],
        "bottom_boundary_flags": [],
    }
    body = list(model[:-1])
    end = model[-1]
    for ly in body:
        d["layer_boundary_x"].append(list(ly.depth.x))
        d["layer_boundary_z"].append(list(ly.depth.y))
        d["layer_boundary_flags"].append(list(ly.depth.vary))
        d["upper_x_velocities"].append(list(ly.v_top.x))
        d["upper_velocities"].append(list(ly.v_top.y))
        d["upper_velocity_flags"].append(list(ly.v_top.vary))
        d["lower_x_velocities"].append(list(ly.v_bot.x))
        d["lower_velocities"].append(list(ly.v_bot.y))
        d["lower_velocity_flags"].append(list(ly.v_bot.vary))
    d["bottom_boundary_x"] = list(end.depth.x)
    d["bottom_boundary_z"] = list(end.depth.y)
    d["bottom_boundary_flags"] = list(end.depth.vary)
    return d


def load_edit_model(path: PathLike, *, model_cls: Optional[Type] = None) -> Any:
    """经统一 ``read_vin_model`` 加载为可编辑 ``Model``。"""
    return vin_dict_to_edit_model(read_vin_dict(path), model_cls=model_cls)


def save_edit_model(
    model: Any, path: PathLike, *, format_wide: bool = False
) -> Path:
    """保存可编辑模型。

    ``format_wide=True`` 时仍走 ``Model.dump``（宽列格式）；
    否则走统一 ``write_vin_model``。
    """
    p = Path(path)
    if format_wide and hasattr(model, "dump"):
        model.dump(str(p), format_wide=True)
        return p
    write_vin_dict(p, edit_model_to_vin_dict(model))
    return p


__all__ = [
    "edit_model_to_vin_dict",
    "is_vin_content",
    "is_vin_file",
    "is_vin_path",
    "load_edit_model",
    "load_zelt_model",
    "read_vin_dict",
    "save_edit_model",
    "vin_dict_to_edit_model",
    "write_vin_dict",
]
