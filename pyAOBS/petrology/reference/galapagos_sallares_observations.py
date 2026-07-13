"""Representative GVP (H, V_LC) points for Sallarès et al. (2005) Fig. 10 overlay.

Values are long-wavelength Layer 3 averages corrected to 400 °C, 600 MPa
(Sallarès et al. 2003 §5.1; 2005 §6 Comparison with velocity models).
Approximate crest / thickened segments — for GUI illustration, not digitized.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GalapagosProfilePoint:
    profile_id: str
    label: str
    h_km: float
    v_lc_km_s: float
    v_sigma_km_s: float = 0.08
    thick_crest: bool = True


# Fig. 10 symbols: five ridge profiles + normal CNSC crust (NOC).
GALAPAGOS_SALLARES_OBSERVATIONS: tuple[GalapagosProfilePoint, ...] = (
    GalapagosProfilePoint("P1", "E Carnegie (thin)", 13.0, 7.12, 0.10, False),
    GalapagosProfilePoint("P2", "W Carnegie (thick)", 19.0, 6.88, 0.08, True),
    GalapagosProfilePoint("P3", "Malpelo (thick)", 19.0, 6.90, 0.08, True),
    GalapagosProfilePoint("P4", "N Cocos (thick)", 19.0, 6.85, 0.10, True),
    GalapagosProfilePoint("P5", "S Cocos (med.)", 16.5, 7.05, 0.10, True),
    GalapagosProfilePoint("NOC", "Normal CNSC crust", 7.0, 7.10, 0.06, False),
)

FIG10_PANEL_LABELS: dict[str, str] = {
    "10a_ref": "Fig.10a 参考 (d=15, w=1, Δz=50, α=0.25)",
    "10b_uniform_chi": "Fig.10b 均匀上涌 (α=1)",
    "10c_thick_damp": "Fig.10c 厚含水区 (Δz=75 km)",
    "10d_high_w": "Fig.10d 高含水熔融率 (w=2 %/GPa)",
    "10e_high_d": "Fig.10e 高干熔融率 (d=20 %/GPa)",
}
