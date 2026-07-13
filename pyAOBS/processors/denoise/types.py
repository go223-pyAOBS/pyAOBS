from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class DenoiseDebug:
    """Denoise debug payload for visualization (e.g., zplotpy).

    Notes:
    - These fields are placeholders for P1/P2; P0 may return None.
    - Shapes are implementation-dependent, but should be consistent per backend.
    """

    org_tf: np.ndarray
    gcv_tf: np.ndarray
    final_tf: np.ndarray
    freq: np.ndarray


@dataclass
class DenoiseResult:
    """Result container for denoise operations."""

    data: np.ndarray
    debug: Optional[DenoiseDebug] = None
    meta: Dict[str, Any] = field(default_factory=dict)

