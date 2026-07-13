"""
iphase 可视化测试

运行方式（在项目根目录）:
  python -m pyAOBS.visualization.iphase.test_iphase_visual
  或
  python pyAOBS/visualization/iphase/test_iphase_visual.py
"""

from pathlib import Path
import numpy as np

# 定位示例数据
_IPHASE_DIR = Path(__file__).resolve().parent
TX_DEMO = _IPHASE_DIR / "examples" / "tx_demo.in"
TX_OBS17 = _IPHASE_DIR / "txin" / "tx_OBS27.in"

# PPP/PPS/PSS 相位号（OBS17）
PHASE_PPP = 5
PHASE_PPS = 14
PHASE_PSS = 24
PHASE_PPS_MINUS_PPP = 104  # 输出1
PHASE_PSP_CORR = 105       # 输出2


def _offset_range(ds, phase_id):
    """返回某相位的偏移范围 (min, max)。"""
    xs = []
    for s in ds.shots:
        for p in s.picks:
            if p.phase_id == phase_id:
                xs.append(p.x)
    return (min(xs), max(xs)) if xs else (None, None)


def _phase_true_offset_time(ds, phase_id):
    """收集某相位的 (true_offset, time) 数组。"""
    offs, ts = [], []
    for s in ds.shots:
        for p in s.picks:
            if p.phase_id == phase_id:
                offs.append(p.x - s.xshot)
                ts.append(p.t)
    if not offs:
        return np.array([]), np.array([])
    return np.array(offs, dtype=float), np.array(ts, dtype=float)


def _phase_model_distance_time(ds, phase_id):
    """收集某相位的 (model_distance=x, time) 数组。"""
    xs, ts = [], []
    for s in ds.shots:
        for p in s.picks:
            if p.phase_id == phase_id:
                xs.append(p.x)
                ts.append(p.t)
    if not xs:
        return np.array([]), np.array([])
    return np.array(xs, dtype=float), np.array(ts, dtype=float)


def main():
    from pyAOBS.visualization.iphase import (
        read_tx,
        select_phases,
        combine_ppp_pps_pss_from_datasets,
        compute_ppp_pps_diff_pairs,
        stats_ppp_pps_diff_by_offset,
        pps_minus_ppp_from_ppp_slope,
        fit_ppp_time_curve_local_linear,
        summary,
        plot_offset_time,
        plot_phase_coverage,
        plot_difference_offset,
        plot_difference_offset_with_fit,
        plot_residuals_histogram,
    )
    import matplotlib.pyplot as plt

    tx_path = TX_OBS17 if TX_OBS17.exists() else TX_DEMO
    if not tx_path.exists():
        print(f"File not found: {TX_DEMO} or {TX_OBS17}")
        return 1

    print(f"Read: {tx_path}")
    ds = read_tx(tx_path)
    s = summary(ds)
    print(f"  n_shots={s['n_shots']}, n_picks={s['n_picks']}, phases={s['phase_counts']}")

    # 筛选 PPP(5), PPS(14), PSS(24)
    needed = {PHASE_PPP, PHASE_PPS, PHASE_PSS}
    missing = needed - set(ds.phase_ids())
    if missing:
        print(f"  WARNING: Missing phases {missing} (need 5=PPP, 14=PPS, 24=PSS)")
        if TX_OBS17.exists():
            print("  Fallback: run with tx_demo.in for basic select test only.")
        tx_path = TX_DEMO
        ds = read_tx(tx_path)
        # 简单测试：只做筛选与可视化
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f"iphase - {tx_path.name} (no PPP/PPS/PSS)", fontsize=12)
        plot_offset_time(ds, ax=axes[0, 0])
        axes[0, 0].set_title("Raw: offset vs time")
        plot_phase_coverage(ds, ax=axes[0, 1])
        axes[0, 1].set_title("Raw: phase coverage")
        phase_ids = ds.phase_ids()
        if phase_ids:
            ds_sel, st = select_phases(ds, [phase_ids[0]])
            plot_offset_time(ds_sel, ax=axes[1, 0])
            axes[1, 0].set_title(f"Phase {phase_ids[0]} only")
        axes[1, 1].axis("off")
        plt.tight_layout()
        plt.show()
        return 0

    ds_ppp, _ = select_phases(ds, [PHASE_PPP])
    ds_pps, _ = select_phases(ds, [PHASE_PPS])
    ds_pss, _ = select_phases(ds, [PHASE_PSS])

    r_ppp = _offset_range(ds_ppp, PHASE_PPP)
    r_pps = _offset_range(ds_pps, PHASE_PPS)
    r_pss = _offset_range(ds_pss, PHASE_PSS)
    print(f"  PPP(5) offset range: {r_ppp[0]:.2f} - {r_ppp[1]:.2f} km")
    print(f"  PPS(14) offset range: {r_pps[0]:.2f} - {r_pps[1]:.2f} km")
    print(f"  PSS(24) offset range: {r_pss[0]:.2f} - {r_pss[1]:.2f} km")

    # 检查 PSS 是否在 PPP/PPS 范围内
    if r_pss[0] is not None and r_ppp[0] is not None:
        ppp_min, ppp_max = r_ppp
        pss_out = (r_pss[0] < ppp_min or r_pss[1] > ppp_max) or (
            r_pss[0] < r_pps[0] or r_pss[1] > r_pps[1]
        )
        if pss_out:
            print("  INFO: PSS offset range is outside or partly outside PPP/PPS range.")
            print("        Some PPP picks may have no matching PPS+PSS at same offset.")

    # PPP/PPS/PSS 组合（tol 放宽以应对浮点差异）
    res = combine_ppp_pps_pss_from_datasets(
        ds_ppp, ds_pps, ds_pss,
        ip1=PHASE_PPP, ip2=PHASE_PPS, ip3=PHASE_PSS,
        ip4=PHASE_PPS_MINUS_PPP, ip5=PHASE_PSP_CORR,
        tol=0.05,
    )
    print(f"  Combine result: npick_matched={res.npick_matched}, npick_unmatched={res.npick_unmatched}, nshot={res.nshot}")

    if res.npick_unmatched > 0:
        print("  INFO: Some PPP picks had no matching PPS+PSS (PSS offset may be outside PPP/PPS range).")

    # PPP-PPS 差值对（真实偏移距 = x - xshot，有符号）
    diff_pairs = compute_ppp_pps_diff_pairs(
        ds_ppp, ds_pps, ip1=PHASE_PPP, ip2=PHASE_PPS, tol=0.05
    )
    print(f"  PPP-PPS diff pairs: {len(diff_pairs)}")

    # 正/负偏移距统计：拟合与偏离误差
    fit_stats = stats_ppp_pps_diff_by_offset(diff_pairs, degree=2)
    for label in ["positive", "negative"]:
        s = fit_stats.get(label)
        if s is not None:
            print(f"  [{label} offset] n={s.n}, mean_residual={s.mean_residual:.5f}s, std={s.std_residual:.5f}s, rms={s.rms:.5f}s")

    # 观测 PPS-PPP 差值
    offsets = np.array([p[1] for p in diff_pairs])  # true_offset
    t_obs = np.array([p[2] for p in diff_pairs])    # PPS-PPP

    # PPP 拾取曲线（用于估计射线参数 p=dt/dx）
    ppp_offsets, ppp_times = _phase_true_offset_time(ds_ppp, PHASE_PPP)
    pps_offsets, pps_times = _phase_true_offset_time(ds_pps, PHASE_PPS)
    ppp_x, ppp_t = _phase_model_distance_time(ds_ppp, PHASE_PPP)
    print(f"  PPP picks for slope strategy: {len(ppp_offsets)}")
    print(f"  PPS picks for diagnostic slope: {len(pps_offsets)}")

    # C->R 段参数（可按已知沉积层参数调整）
    h_cr = 1.9   # km，转换点到接收点的垂向距离
    vp_cr = 4.5  # km/s
    vs_cr = 1.9  # km/s

    # 2x3 可视化
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"iphase PPP/PPS/PSS combine - {tx_path.name}", fontsize=12)

    # 1. PPP-PPS 差值-真实偏移距
    ax1 = axes[0, 0]
    plot_difference_offset(diff_pairs, ax=ax1)
    ax1.set_title("PPS-PPP vs true offset (x-xshot, km)")

    # 2. 原始 PPP/PPS/PSS
    ax2 = axes[0, 1]
    plot_offset_time(ds, phase_ids=[PHASE_PPP, PHASE_PPS, PHASE_PSS], ax=ax2)
    if len(ppp_x) >= 5:
        x_fit = np.linspace(ppp_x.min(), ppp_x.max(), 300)
        t_fit = fit_ppp_time_curve_local_linear(
            ppp_x,
            ppp_t,
            x_fit,
            window_points=13,
            split_by_sign=False,
        )
        ax2.plot(x_fit, t_fit, "k--", linewidth=2, label="PPP local-linear fit")
        ax2.legend(fontsize=8)
    ax2.set_title("Input: PPP(5), PPS(14), PSS(24)")

    # 3. 所有匹配的 PPP-PPS（差值-模型距离 + 拟合曲线）
    ax3 = axes[0, 2]
    plot_difference_offset_with_fit(diff_pairs, fit_stats, x_axis="true_offset", ax=ax3)
    ax3.set_title("PPP-PPS vs true_offset + fit")

    # 4. 拟合残差直方图
    ax4 = axes[1, 0]
    plot_residuals_histogram(fit_stats, ax=ax4)
    ax4.set_title("Fit residual (obs - fit)")

    # 5. 共享射线参数策略：主流程用 PPP 估计 p(x)，PPS 仅作诊断
    ax5 = axes[1, 1]
    if len(diff_pairs) >= 3 and len(ppp_offsets) >= 4:
        off_plot = np.linspace(offsets.min(), offsets.max(), 300)
        try:
            # 主流程：共享 p(x)，由 PPP 估计
            dt_obs_theory, p_obs = pps_minus_ppp_from_ppp_slope(
                ppp_offsets, ppp_times, offsets,
                h_cr=h_cr, vp=vp_cr, vs=vs_cr, window_points=15, split_by_sign=True
            )
            dt_plot, _ = pps_minus_ppp_from_ppp_slope(
                ppp_offsets, ppp_times, off_plot,
                h_cr=h_cr, vp=vp_cr, vs=vs_cr, window_points=15, split_by_sign=True
            )

            valid = np.isfinite(dt_obs_theory)
            if np.any(valid):
                rms = float(np.sqrt(np.mean((t_obs[valid] - dt_obs_theory[valid]) ** 2)))
                pmin, pmax = float(np.nanmin(np.abs(p_obs[valid]))), float(np.nanmax(np.abs(p_obs[valid])))
                print(f"  PPP-slope strategy: RMS={rms:.5f}s, |p| range=[{pmin:.5f}, {pmax:.5f}] s/km")
            print(f"  PPP-slope strategy: valid theory points={int(np.sum(np.isfinite(dt_obs_theory)))}/{len(dt_obs_theory)}")

            # 诊断：PPS 独立估计 p(x)，仅用于检查与 PPP 是否一致
            if len(pps_offsets) >= 4:
                dt_obs_theory_pps, p_obs_pps = pps_minus_ppp_from_ppp_slope(
                    pps_offsets, pps_times, offsets,
                    h_cr=h_cr, vp=vp_cr, vs=vs_cr, window_points=11, split_by_sign=True
                )
                dt_plot_pps, _ = pps_minus_ppp_from_ppp_slope(
                    pps_offsets, pps_times, off_plot,
                    h_cr=h_cr, vp=vp_cr, vs=vs_cr, window_points=11, split_by_sign=True
                )
                valid_both = np.isfinite(p_obs) & np.isfinite(p_obs_pps)
                if np.any(valid_both):
                    p_diff = np.abs(p_obs[valid_both] - p_obs_pps[valid_both])
                    print(
                        "  p-consistency (PPP vs PPS independent): "
                        f"median={float(np.median(p_diff)):.6f}, "
                        f"p90={float(np.percentile(p_diff, 90)):.6f} s/km"
                    )
                ax5.plot(off_plot, dt_plot_pps, color="0.65", linestyle="--", linewidth=1.5, label="Diag: theory from PPS slope")

            order = np.argsort(offsets)
            ax5.scatter(offsets[order], t_obs[order], c="k", s=15, label="Obs PPS-PPP")
            ax5.plot(off_plot, dt_plot, "m-", lw=2, label="Theory from PPP slope")
            ax5.set_title("Shared-p strategy: PPP main, PPS diagnostic")
            ax5.set_xlabel("True offset (km)")
            ax5.set_ylabel("PPS-PPP (s)")
            ax5.grid(True, alpha=0.5)
            ax5.legend(fontsize=8)
        except ValueError as e:
            ax5.text(0.1, 0.5, f"PPP-slope model invalid:\n{e}", transform=ax5.transAxes)
            ax5.set_title("PPP-slope strategy")
            ax5.grid(True, alpha=0.5)
    else:
        ax5.axis("off")

    # 6. 修正 PSP
    ax6 = axes[1, 2]
    plot_offset_time(res.tx2_out, phase_ids=[PHASE_PSP_CORR], ax=ax6)
    ax6.set_title(f"Output2: corrected PSP (phase {PHASE_PSP_CORR})")

    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
