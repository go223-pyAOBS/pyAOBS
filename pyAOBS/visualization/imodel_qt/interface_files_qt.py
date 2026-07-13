"""从文本文件加载 v(z) 界面曲线（与 Tk `ModelSurfaceMixin._load_interface_file` 同款解析）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def parse_interface_file(path: str | Path) -> List[Dict[str, Any]]:
    """
    返回若干条界面，每条为 dict:
      - name: str
      - x, z: ndarray (float)
    """
    p = Path(path)
    interfaces_in_file: List[Dict[str, Any]] = []
    current_interface = {"x": [], "z": []}
    current_name: str | None = None

    with p.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            if line.startswith("#") and ("Interface" in line or "Boundary" in line):
                if len(current_interface["x"]) > 0:
                    interface_name = current_name or f"Interface {len(interfaces_in_file) + 1}"
                    interfaces_in_file.append(
                        {
                            "x": np.array(current_interface["x"], dtype=float),
                            "z": np.array(current_interface["z"], dtype=float),
                            "name": interface_name,
                        }
                    )
                current_interface = {"x": [], "z": []}
                if "Interface" in line:
                    parts = line.split("Interface")
                    current_name = parts[-1].strip() if len(parts) > 1 else line.replace("#", "").strip()
                elif "Boundary" in line:
                    current_name = line.replace("#", "").strip()
                else:
                    current_name = None
            continue
        try:
            parts = line.split()
            if len(parts) >= 2:
                x = float(parts[0])
                z = float(parts[1])
                current_interface["x"].append(x)
                current_interface["z"].append(z)
        except ValueError:
            continue

    if len(current_interface["x"]) > 0:
        interface_name = current_name or f"Interface {len(interfaces_in_file) + 1}"
        interfaces_in_file.append(
            {
                "x": np.array(current_interface["x"], dtype=float),
                "z": np.array(current_interface["z"], dtype=float),
                "name": interface_name,
            }
        )

    if not interfaces_in_file:
        data = np.loadtxt(str(p), delimiter=None)
        if data.ndim == 2 and data.shape[1] >= 2:
            interfaces_in_file.append(
                {
                    "x": np.asarray(data[:, 0], dtype=float),
                    "z": np.asarray(data[:, 1], dtype=float),
                    "name": p.name,
                }
            )
        else:
            raise ValueError("Expected 2 columns (x, z).")

    return interfaces_in_file


def interfaces_xy_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """判断两组 x/z 是否一致（用于合并 loaded 与 legacy basement 显示）。"""
    try:
        xa = np.asarray(a["x"], dtype=float).ravel()
        za = np.asarray(a["z"], dtype=float).ravel()
        xb = np.asarray(b["x"], dtype=float).ravel()
        zb = np.asarray(b["z"], dtype=float).ravel()
        return bool(np.array_equal(xa, xb) and np.array_equal(za, zb))
    except Exception:
        return False
