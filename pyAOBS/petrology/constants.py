"""Shared physical constants for petrology (single source of truth)."""

from __future__ import annotations

# KKHS02 §4 / Fig.7: depth–pressure conversion (km per GPa)
KM_PER_GPA = 30.0

# MB88 adiabat: T(P) = T_p + 20·P (°C, GPa)
ADIABAT_GRAD_C_PER_GPA = 20.0
