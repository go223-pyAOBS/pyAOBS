# -*- coding: utf-8 -*-
"""定位 raw2sac 工具目录（转换脚本 / segy_trace_header），避免硬编码。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Optional

# .../processors/idata/gui/services/this.py → processors/
_PROCESSORS = Path(__file__).resolve().parents[3]
RAW2SAC_DIR = _PROCESSORS / "raw2sac"

_sth_mod: Optional[Any] = None


def ensure_raw2sac_on_path() -> Path:
    """保证 raw2sac 在 sys.path 最前（仅在需要时调整）。"""
    s = str(RAW2SAC_DIR.resolve())
    if not sys.path or sys.path[0] != s:
        while s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)
    return RAW2SAC_DIR


def import_segy_trace_header():
    """导入正确的 segy_trace_header（带缓存，热路径零开销）。"""
    global _sth_mod
    if _sth_mod is not None:
        return _sth_mod

    ensure_raw2sac_on_path()
    target = (RAW2SAC_DIR / "segy_trace_header.py").resolve()
    mod = sys.modules.get("segy_trace_header")
    if mod is not None:
        cur = Path(getattr(mod, "__file__", "") or "").resolve()
        try:
            counit_off = mod.SEGY_TRACE_FIELDS.get("counit", (None,))[0]
        except Exception:
            counit_off = None
        if cur != target or counit_off != 88:
            del sys.modules["segy_trace_header"]
            mod = None
    if mod is None:
        import segy_trace_header as mod  # type: ignore
    try:
        counit_off = mod.SEGY_TRACE_FIELDS.get("counit", (None,))[0]
    except Exception:
        counit_off = None
    if counit_off != 88:
        mod = importlib.reload(mod)
        counit_off = mod.SEGY_TRACE_FIELDS.get("counit", (None,))[0]
    if counit_off != 88:
        raise RuntimeError(
            "segy_trace_header.counit offset=%s (expect 88) from %s"
            % (counit_off, getattr(mod, "__file__", "?"))
        )
    _sth_mod = mod
    return mod
