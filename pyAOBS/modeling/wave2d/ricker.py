# -*- coding: utf-8 -*-
from __future__ import annotations

import math


def ricker(t: float, f0: float, t0: float | None = None) -> float:
    delay = 1.2 / f0 if t0 is None else t0
    x = math.pi * f0 * (t - delay)
    xx = x * x
    return (1.0 - 2.0 * xx) * math.exp(-xx)


def ricker_delay(f0: float) -> float:
    return 1.2 / f0
