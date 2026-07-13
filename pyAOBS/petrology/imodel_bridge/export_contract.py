"""imodel ↔ petrology data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class CrustObservation:
    """Single (H, V_LC) anchor for Fig.12a overlay."""

    h_whole_km: float
    v_lc_km_s: float
    f_lower: float = 0.7
    v_lc_sigma_km_s: float | None = None
    source: str = "manual"
    x_km: float | None = None
    n_samples: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrustObservation:
        return cls(
            h_whole_km=float(data["h_whole_km"]),
            v_lc_km_s=float(data["v_lc_km_s"]),
            f_lower=float(data.get("f_lower", 0.7)),
            v_lc_sigma_km_s=(
                float(data["v_lc_sigma_km_s"]) if data.get("v_lc_sigma_km_s") is not None else None
            ),
            source=str(data.get("source", "manual")),
            x_km=float(data["x_km"]) if data.get("x_km") is not None else None,
            n_samples=int(data["n_samples"]) if data.get("n_samples") is not None else None,
        )


@dataclass
class Fig12aOverlayRequest:
    observation: CrustObservation
    display_tracks: tuple[str, ...] = ("fig12a_digitized",)
    read_tp: bool = True
    delta_vp_max_km_s: float = 0.15
    thick_crust_h_min_km: float = 15.0


def save_observation_json(obs: CrustObservation, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obs.to_dict(), indent=2), encoding="utf-8")


def load_observation_json(path: Path | str) -> CrustObservation:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CrustObservation.from_dict(data)
