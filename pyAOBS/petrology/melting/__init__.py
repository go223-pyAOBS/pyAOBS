"""Mantle melting models (Kinzler 1997 batch + Modern active column)."""

from .column import MeltingColumnResult, forward_melting_column
from .heterogeneous import HeterogeneousColumnResult, forward_heterogeneous_column
from .hvp_scan import HvpScanPoint, HvpScanResult, scan_hvp_lip
from .isentropic import IsentropicPath, integrate_isentropic_path
from .kinzler1997_batch import batch_melt_oxides, generate_grid_rows
from .melt_chemistry import ChemistryBackend, melt_oxides, pmelts_melt_oxides
from .lip_forward import LipForwardResult, forward_lip_column
from .lip_scan import PhiScanResult, PhiScanSlice, forward_point_diagnostics, scan_hvp_lip_phi
from .lithology import Lithology, dry_peridotite_lithology, g2_pyroxenite_lithology, heterogeneous_source
from .pymelt_lithology_adapter import (
    LITHOLOGY_PRESETS,
    heterogeneous_source_pymelt,
    list_lithology_presets,
    list_pymelt_lithology_keys,
    lithology_catalog,
    lithology_diagnostics,
    print_lithology_catalog,
    pymelt_lithology,
    resolve_lithology_col_kwargs,
)
from .pymelt_bridge import PyMeltColumnResult, build_mantle, forward_pymelt_column
from .reebox_import import ReeboxBenchmarkRow, load_reebox_csv
from .rmc import RmcMeltResult, accumulate_column_melt

__all__ = [
    "HeterogeneousColumnResult",
    "HvpScanPoint",
    "HvpScanResult",
    "IsentropicPath",
    "Lithology",
    "MeltingColumnResult",
    "LipForwardResult",
    "PhiScanResult",
    "PhiScanSlice",
    "PyMeltColumnResult",
    "ReeboxBenchmarkRow",
    "RmcMeltResult",
    "accumulate_column_melt",
    "batch_melt_oxides",
    "melt_oxides",
    "pmelts_melt_oxides",
    "dry_peridotite_lithology",
    "forward_heterogeneous_column",
    "forward_lip_column",
    "forward_point_diagnostics",
    "forward_melting_column",
    "g2_pyroxenite_lithology",
    "generate_grid_rows",
    "heterogeneous_source",
    "heterogeneous_source_pymelt",
    "LITHOLOGY_PRESETS",
    "integrate_isentropic_path",
    "list_lithology_presets",
    "list_pymelt_lithology_keys",
    "lithology_catalog",
    "lithology_diagnostics",
    "print_lithology_catalog",
    "forward_pymelt_column",
    "pymelt_lithology",
    "resolve_lithology_col_kwargs",
    "build_mantle",
    "scan_hvp_lip",
    "scan_hvp_lip_phi",
]
