#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单场热初值出图（复用 inv_graph6k_hot/plot_hot.py）。"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "inv_graph6k_hot"))
import plot_hot as p  # noqa: E402

p.HERE = HERE
p.START_TITLE = "start mixed  rec_vp/1.73+0.50"
p.REC_TITLE = "rec mixed（单场）"
p.TRUE_TITLE = "true Vs"
p.SUPTITLE_MODELS = "崎岖面  单场 type 6  PPP→Vs  观测=真 mixed 正演   色标 3.70–5.20"
p.SUPTITLE_RAYS = "射线  收回 vs 真值（单场 PSP=6，真模型观测）"

if __name__ == "__main__":
    raise SystemExit(p.main())
