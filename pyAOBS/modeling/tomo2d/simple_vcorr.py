"""写出 CorrelationLength2d 的简单 2×2 顶/底相关长度文件。

对应手工 heredoc（水平/垂直相关长度只随深度变，沿 x 不变）::

    2 2
    xmin xmax
    topo topo
    zmin zmax
    Lht Lhb
    Lht Lhb
    Lvt Lvb
    Lvt Lvb

``tt_inverse -CV`` / ``CorrelationLength2d`` 按此格式读入后，在 (x,z) 上双线性插值。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


_SIMPLE_REQUIRED = ("Lht", "Lhb", "Lvt", "Lvb", "xmax", "zmax")


def _fmt_x(value: Any) -> str:
    f = float(value)
    if abs(f - round(f)) < 1e-12:
        return str(int(round(f)))
    return _fmt_len(f)


def _fmt_len(value: Any) -> str:
    f = float(value)
    tenths = round(f, 1)
    if abs(f - tenths) < 1e-9:
        return f"{tenths:.1f}"
    return f"{f:.6g}"


def format_simple_vcorr(
    *,
    Lht: Any,
    Lhb: Any,
    Lvt: Any,
    Lvb: Any,
    xmax: Any,
    zmax: Any,
    xmin: Any = 0,
    zmin: Any = 0,
    topo: Any = 0,
) -> str:
    """返回 CorrelationLength2d 文本（末行换行）。"""
    x0, x1 = _fmt_x(xmin), _fmt_x(xmax)
    t = _fmt_len(topo)
    z0, z1 = _fmt_len(zmin), _fmt_len(zmax)
    ht, hb = _fmt_len(Lht), _fmt_len(Lhb)
    vt, vb = _fmt_len(Lvt), _fmt_len(Lvb)
    return (
        "2 2\n"
        f"{x0} {x1}\n"
        f"{t} {t}\n"
        f"{z0} {z1}\n"
        f"{ht} {hb}\n"
        f"{ht} {hb}\n"
        f"{vt} {vb}\n"
        f"{vt} {vb}\n"
    )


def format_simple_vcorr_from_kwargs(kwargs: Mapping[str, Any]) -> str:
    missing = [k for k in _SIMPLE_REQUIRED if kwargs.get(k) is None]
    if missing:
        raise ValueError(f"gen_vcorr(simple_2x2) 缺少必需参数: {', '.join(missing)}")
    return format_simple_vcorr(
        Lht=kwargs["Lht"],
        Lhb=kwargs["Lhb"],
        Lvt=kwargs["Lvt"],
        Lvb=kwargs["Lvb"],
        xmax=kwargs["xmax"],
        zmax=kwargs["zmax"],
        xmin=kwargs.get("xmin", 0),
        zmin=kwargs.get("zmin", 0),
        topo=kwargs.get("topo", 0),
    )


def write_simple_vcorr(path: str | Path, **kwargs: Any) -> str:
    """写入 ``path``，返回文件内容。"""
    text = format_simple_vcorr_from_kwargs(kwargs)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8", newline="\n")
    return text
