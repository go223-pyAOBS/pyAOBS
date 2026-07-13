"""Track / backend defaults for petrology forward models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Track = Literal["reproduction", "modern"]
CipwBackend = Literal["auto", "pyrolite", "fallback"]
MineralBackend = Literal["auto", "burnman", "empirical", "fig2", "sb1994", "sb1994_fig2ol"]


@dataclass(frozen=True)
class PetrologyConfig:
    """KKHS02 reproduction defaults vs modern thermodynamic track."""

    track: Track = "reproduction"
    cipw_backend: CipwBackend = "auto"
    mineral_backend: MineralBackend = "auto"
    reference_p_pa: float = 600e6
    reference_t_k: float = 673.15  # 400 °C
    prefer_pip_vendored: bool = True

    @classmethod
    def reproduction(cls) -> PetrologyConfig:
        return cls(track="reproduction", cipw_backend="auto", mineral_backend="auto")

    @classmethod
    def modern(cls) -> PetrologyConfig:
        return cls(track="modern", cipw_backend="pyrolite", mineral_backend="burnman")


DEFAULT_CONFIG = PetrologyConfig.reproduction()
