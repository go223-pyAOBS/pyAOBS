"""
iphase - 拾取震相（tx.in 格式）处理与可视化

提供 tx.in 文件的读写、相位筛选、多相位组合及可视化功能，
与 txphase.f / txconv.f 行为兼容。

用法:
  from pyAOBS.visualization.iphase import read_tx, write_tx, select_phases, combine_ppp_pps_pss
  python -m pyAOBS.visualization.iphase select tx.in -o tx.out --phases 1 2 3
  python -m pyAOBS.visualization.iphase combine tx1.in tx2.in tx3.in --ip1 1 --ip2 2 --ip3 3 --ip4 10 --ip5 11
  python -m pyAOBS.visualization.iphase info tx.in
"""

from .models import Pick, Shot, PhaseDataset
from .io_tx import read_tx, write_tx
from .phase_filter import select_phases, exclude_phases
from .phase_combine import (
    combine_ppp_pps_pss,
    combine_ppp_pps_pss_from_datasets,
    compute_ppp_pps_diff_pairs,
)
from .qc_metrics import summary, phase_counts, stats_ppp_pps_diff_by_offset, DiffFitStats
from .plot_tx import (
    plot_offset_time,
    plot_phase_coverage,
    plot_difference_offset,
    plot_difference_offset_with_fit,
    plot_residuals_histogram,
)
from .theoretical_ppp_pps import (
    pps_minus_ppp_from_conversion_receiver_segment,
    pps_minus_ppp_from_ppp_slope,
    pss_minus_psp_from_pss_slope_with_profile,
    invert_local_h_vpratio,
    invert_m_field_from_paths,
    fit_ppp_time_curve_local_linear,
    pps_minus_ppp_1d_layer,
    fit_sediment_params,
    FitResult,
    LocalInversionResult,
    MFieldInversionResult,
    pps_minus_ppp_from_rayinvr,
)
from .equi2d import build_equi_dataset
from .theory2d_service import (
    ForwardRunResult,
    Theory2DBundle,
    run_rayinvr_forward,
    set_rin_tfile,
    update_rin_shots_from_receivers_and_depth,
    validate_rayinvr_inputs,
    build_theory2d_bundle_from_txout,
    collect_theory2d_delta_by_branch,
    make_delta_query,
    parse_pois_from_rin,
    parse_pois_full_from_rin,
    parse_rin_input_files,
    collect_phase_points_from_txout,
    write_pois_full_to_rin,
    write_pois_to_rin,
)
from .rin_phase_groups import (
    ReflectionEvent,
    ConversionEvent,
    PhaseGroup,
    PhaseGroupParseResult,
    RinPhaseArrays,
    describe_all_phase_groups,
    describe_phase_group_line,
    describe_phase_group_short_sentence,
    describe_ray_code_zh,
    describe_ray_wave_phrase,
    infer_phase_type_label,
    parse_phase_groups_from_arrays,
    parse_phase_groups_from_rin_text,
    parse_phase_groups_from_rin_file,
    parse_rin_phase_arrays_from_text,
    parse_rin_phase_arrays_from_file,
    compile_phase_groups_to_arrays,
    apply_phase_groups_to_rin_text,
    apply_phase_groups_to_rin_file,
)

__all__ = [
    "Pick",
    "Shot",
    "PhaseDataset",
    "read_tx",
    "write_tx",
    "select_phases",
    "exclude_phases",
    "combine_ppp_pps_pss",
    "combine_ppp_pps_pss_from_datasets",
    "compute_ppp_pps_diff_pairs",
    "summary",
    "phase_counts",
    "stats_ppp_pps_diff_by_offset",
    "DiffFitStats",
    "plot_offset_time",
    "plot_phase_coverage",
    "plot_difference_offset",
    "plot_difference_offset_with_fit",
    "plot_residuals_histogram",
    "pps_minus_ppp_from_conversion_receiver_segment",
    "pps_minus_ppp_from_ppp_slope",
    "pss_minus_psp_from_pss_slope_with_profile",
    "invert_local_h_vpratio",
    "invert_m_field_from_paths",
    "fit_ppp_time_curve_local_linear",
    "pps_minus_ppp_1d_layer",
    "fit_sediment_params",
    "FitResult",
    "LocalInversionResult",
    "MFieldInversionResult",
    "pps_minus_ppp_from_rayinvr",
    "ForwardRunResult",
    "Theory2DBundle",
    "run_rayinvr_forward",
    "set_rin_tfile",
    "update_rin_shots_from_receivers_and_depth",
    "validate_rayinvr_inputs",
    "build_theory2d_bundle_from_txout",
    "make_delta_query",
    "parse_pois_from_rin",
    "parse_pois_full_from_rin",
    "parse_rin_input_files",
    "collect_phase_points_from_txout",
    "collect_theory2d_delta_by_branch",
    "write_pois_full_to_rin",
    "write_pois_to_rin",
    "ReflectionEvent",
    "ConversionEvent",
    "PhaseGroup",
    "PhaseGroupParseResult",
    "RinPhaseArrays",
    "parse_phase_groups_from_arrays",
    "parse_phase_groups_from_rin_text",
    "parse_phase_groups_from_rin_file",
    "parse_rin_phase_arrays_from_text",
    "parse_rin_phase_arrays_from_file",
    "compile_phase_groups_to_arrays",
    "apply_phase_groups_to_rin_text",
    "apply_phase_groups_to_rin_file",
    "describe_all_phase_groups",
    "describe_phase_group_line",
    "describe_phase_group_short_sentence",
    "describe_ray_code_zh",
    "describe_ray_wave_phrase",
    "infer_phase_type_label",
    "build_equi_dataset",
]
