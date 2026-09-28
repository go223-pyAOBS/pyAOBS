# -*- coding: utf-8 -*-
from __future__ import annotations

import os

# pyAOBS/visualization/obs_rtm_qt/services/paths.py → pyAOBS/
_PKG_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

_OBS_RTM_QT = os.path.join(_PKG_ROOT, "visualization", "obs_rtm_qt")

# CLI 在包内 scripts/；工区 madagascar_obs_rtm/ 仅数据
_SCRIPT_CANDIDATES = (
    os.path.join(_OBS_RTM_QT, "scripts"),
    # 兼容旧布局（若仍有人把脚本放在工区或 modeling）
    os.path.join(_OBS_RTM_QT, "madagascar_obs_rtm"),
    os.path.join(_PKG_ROOT, "modeling", "madagascar_obs_rtm"),
)


def madagascar_rtm_dir() -> str:
    """默认示例工区目录（仅数据，不含 CLI）。"""
    d = os.path.join(_OBS_RTM_QT, "madagascar_obs_rtm")
    if os.path.isdir(d):
        return d
    alt = os.path.join(_PKG_ROOT, "modeling", "madagascar_obs_rtm")
    return alt if os.path.isdir(alt) else d


def script_path(name: str) -> str:
    tried = []
    for d in _SCRIPT_CANDIDATES:
        p = os.path.join(d, name)
        tried.append(p)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "找不到脚本 %s，已试:\n  %s" % (name, "\n  ".join(tried))
    )


def zplotpy_dir() -> str:
    return os.path.join(_PKG_ROOT, "visualization", "zplotpy")
