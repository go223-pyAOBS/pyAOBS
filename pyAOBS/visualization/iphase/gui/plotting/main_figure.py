# -*- coding: utf-8 -*-
"""主窗口 2x2 走时图绘制（从 _business 拆出，行为不变）。"""

from __future__ import annotations

import re
import warnings

import numpy as np
from matplotlib.lines import Line2D

from ...io_tx import read_tx
from ...phase_combine import compute_ppp_pps_diff_pairs
from ...phase_filter import select_phases
from ...theoretical_ppp_pps import (
    fit_ppp_time_curve_local_linear,
    pps_minus_ppp_from_ppp_slope,
    pss_minus_psp_from_pss_slope_with_profile,
)
from ...theory2d_service import parse_rin_input_files
from ..services.file_result import (
    PHASE_PPP,
    PHASE_PPS,
    PHASE_PSS,
    equi_tx_path_for_result as _equi_tx_path_for_result,
    obs_model_distance_from_tx as _obs_model_distance_from_tx,
    obs_tag_from_result as _obs_tag_from_result,
    phase_model_trueoff_time as _phase_model_trueoff_time,
    phase_true_offset_time as _phase_true_offset_time,
    result_cache_key as _result_cache_key,
    result_display_name as _result_display_name,
)


class MainFigurePlotMixin:
    """主图绘制 + 标注辅助。依赖宿主提供 results/fig/canvas 与 _calc_theory* 等。"""

    def _draw_results(self) -> None:
        # 保留用户滚轮/平移后的视图，避免参数微变重绘时“缩不了”
        plot = getattr(self, "plot", None)
        if plot is not None and hasattr(plot, "snapshot_before_clear"):
            try:
                plot.snapshot_before_clear()
            except Exception:
                pass
        self.fig.clear()
        self.theory2d_notice = ""
        # 2Dequi：绘图前写 tx_2Dequiv.in；勾选「写等效 PSP」时再自动正演一遍以更新 tx.out
        if self.theory_mode.get() == "2Dequi" and self.results:
            if bool(self.equi_write_equiv_psp.get()):
                self._run_2dequi_forward_all()
            else:
                self._ensure_tx_equi_for_results()
        ax2 = self.fig.add_subplot(2, 2, 1)  # 图2
        ax3 = self.fig.add_subplot(2, 2, 2)  # 图3
        ax5 = self.fig.add_subplot(2, 2, 3)  # 图5
        ax6 = self.fig.add_subplot(2, 2, 4)  # 图6

        multi_mode = len(self.results) > 1
        x_axis = "model_distance"
        title_suffix = ""
        y3_values: list[float] = []
        y5_values: list[float] = []
        fit_error_stats: list[tuple[str, str, dict[str, float]]] = []
        theory_obs_error_stats: list[tuple[str, str, dict[str, float]]] = []
        psp_compare_stats: list[tuple[str, str, float]] = []
        qc_2d_vs_1d: list[tuple[str, int, int, float]] = []
        equi_missing_obs: list[str] = []
        # 主窗口左下/右下：按符号类型固定亮色，便于区分
        c_obs = "#1E90FF"          # observed PPS-PPP
        c_theory = "#FF7F00"       # theoretical PPS-PPP (PPP slope)
        c_theory_diag = "#32CD32"  # diagnostic theory (PPS slope)
        c_psspsp_dt = "#FF00FF"    # theoretical PSS-PSP
        c_psp_corr = "#00BFFF"     # PSP from PPS-PPP correction
        c_pss_raw = "#FF8C00"      # raw PSS
        c_psp_new = "#FF00FF"      # PSP from PSS theoretical conversion
        c_equi_psp_mean = "#00CED1"  # 2Dequi：PSS − mean(PPS−PPP) 等效 PSP
        c_theo_psp = "#006400"       # tx.out 理论 PSP 散点边色
        c_psp_tx_pp = "#8B4513"      # PSP from tx.out PPS-PPP correction

        obs_marks: list[tuple[float, str, str]] = []
        legend_ax2 = {
            "ppp": False, "pps": False, "pss": False, "psp_in": False,
            "equi_ppp": False, "equi_pps": False, "fit_ppp": False,
            "fit_pps": False, "fit_pss": False, "theo_ppp": False,
            "theo_pps": False, "theo_psp": False, "equi_psp": False,
        }
        legend_ax3 = {"picked_ppsppp": False, "fit_ppsppp": False, "picked_psspsp": False, "fit_psspsp": False}
        legend_ax5 = {
            "obs_ppsppp": False, "picked_psspsp": False, "theo_ppsppp": False,
            "theo_psspsp_1d": False, "theo_psspsp_txout": False, "theo_ppsppp_diag": False,
        }
        legend_ax6 = {
            "psp_txin": False, "psp_obs_corr": False, "psp_theory_pss": False,
            "pss_txin": False, "psp_txout_ppsppp": False, "psp_txout_theo": False,
        }

        for i, r in enumerate(self.results):
            rp = self._file_result_for_plot(r)
            obs_tag = _obs_tag_from_result(r)
            label = _result_display_name(r)
            color = f"C{i % 10}"
            psp_id_curr = int(self.psp_phase_id.get())
            inv_for_pss = None
            if self.pss_inversion_ready:
                inv_for_pss = self.pss_profile_by_file.get(_result_cache_key(r))
            x_obs = _obs_model_distance_from_tx(r)
            if x_obs is not None:
                obs_marks.append((x_obs, obs_tag, color))

            # 由原始拾取计算 PSS-PSP 差值，规则与 PPS-PPP 保持一致：
            # - 勾选严格配对：同道严格匹配
            # - 未勾选：在每个 PSS 点处，用 PSP 拟合曲线取值再做差
            pss_psp_pairs: list[tuple[float, float, float]] = []
            try:
                ds_psp_in, _ = select_phases(r.ds, [psp_id_curr])
                if bool(self.strict_diff_pair.get()):
                    # t_diff = t_PSS - t_PSP
                    pss_psp_pairs = compute_ppp_pps_diff_pairs(
                        ds_psp_in,
                        r.ds_pss,
                        ip1=psp_id_curr,
                        ip2=PHASE_PSS,
                        tol=0.05,
                    )
                else:
                    psp_x, psp_off, psp_t = _phase_model_trueoff_time(ds_psp_in, psp_id_curr)
                    pss_x, pss_off, pss_t = _phase_model_trueoff_time(r.ds_pss, PHASE_PSS)
                    for sgn in (-1.0, 1.0):
                        m_psp = (psp_off * sgn) > 0.0
                        m_pss = (pss_off * sgn) > 0.0
                        if np.count_nonzero(m_psp) < 3 or np.count_nonzero(m_pss) < 1:
                            continue
                        psp_fit = fit_ppp_time_curve_local_linear(
                            psp_x[m_psp],
                            psp_t[m_psp],
                            pss_x[m_pss],
                            window_points=self._window_points_value(),
                            split_by_sign=False,
                        )
                        for j in range(int(np.sum(m_pss))):
                            t_psp_f = float(psp_fit[j])
                            if not np.isfinite(t_psp_f):
                                continue
                            md = float(pss_x[m_pss][j])
                            to = float(pss_off[m_pss][j])
                            dt = float(pss_t[m_pss][j]) - t_psp_f  # PSS-PSP
                            pss_psp_pairs.append((md, to, dt))
                    pss_psp_pairs.sort(key=lambda t: t[0])
            except Exception:
                pss_psp_pairs = []

            # 图2：PPP/PPS/PSS/PSP 折合走时（Vred=7.0）；观测来自原始 tx.in
            vred_ax2 = 7.0
            psp_id_ax2 = int(self.psp_phase_id.get())
            for pid, cpid in (
                (PHASE_PPP, "C0"),
                (PHASE_PPS, "C1"),
                (PHASE_PSS, "C2"),
                (psp_id_ax2, "#7B68EE"),
            ):
                if pid <= 0:
                    continue
                xm2, off2, tt2 = _phase_model_trueoff_time(r.ds, pid)
                vv2 = np.isfinite(xm2) & np.isfinite(off2) & np.isfinite(tt2)
                if np.any(vv2):
                    tred2 = tt2[vv2] - np.abs(off2[vv2]) / vred_ax2
                    ax2.scatter(xm2[vv2], tred2, c=cpid, s=13, alpha=0.35)
                    if pid == PHASE_PPP:
                        legend_ax2["ppp"] = True
                    elif pid == PHASE_PPS:
                        legend_ax2["pps"] = True
                    elif pid == PHASE_PSS:
                        legend_ax2["pss"] = True
                    elif pid == psp_id_ax2:
                        legend_ax2["psp_in"] = True
            # 2Dequi（未勾选写等效 PSP）：在观测层之上叠加合成等效 PPP/PPS（空心）
            if (
                self.theory_mode.get() == "2Dequi"
                and _equi_tx_path_for_result(r).exists()
                and not bool(self.equi_write_equiv_psp.get())
            ):
                # 优先使用 rp（_file_result_for_plot 结果）；若该通道异常，回退直接读取 tx_2Dequiv.in
                ds_equi_src = rp.ds
                try:
                    n_ppp_e = int(np.sum(np.asarray([p.phase_id == PHASE_PPP for s0 in ds_equi_src.shots for p in s0.picks], dtype=bool)))
                    n_pps_e = int(np.sum(np.asarray([p.phase_id == PHASE_PPS for s0 in ds_equi_src.shots for p in s0.picks], dtype=bool)))
                except Exception:
                    n_ppp_e, n_pps_e = 0, 0
                if (n_ppp_e + n_pps_e) == 0:
                    try:
                        ds_equi_src = read_tx(_equi_tx_path_for_result(r))
                    except Exception:
                        ds_equi_src = rp.ds
                for pid, cpid, mk in ((PHASE_PPP, "C0", "s"), (PHASE_PPS, "C1", "^")):
                    xe, oe, te = _phase_model_trueoff_time(ds_equi_src, pid)
                    ve = np.isfinite(xe) & np.isfinite(oe) & np.isfinite(te)
                    if np.any(ve):
                        trede = te[ve] - np.abs(oe[ve]) / vred_ax2
                        ax2.scatter(
                            xe[ve],
                            trede,
                            s=24,
                            marker=mk,
                            facecolors="none",
                            edgecolors=cpid,
                            linewidths=1.0,
                            alpha=0.95,
                            zorder=4,
                        )
                        if pid == PHASE_PPP:
                            legend_ax2["equi_ppp"] = True
                        elif pid == PHASE_PPS:
                            legend_ax2["equi_pps"] = True
                # 记录该 OBS 是否实际画出了等效 PPP/PPS（用于状态栏提示）
                xe_ppp, oe_ppp, te_ppp = _phase_model_trueoff_time(ds_equi_src, PHASE_PPP)
                xe_pps, oe_pps, te_pps = _phase_model_trueoff_time(ds_equi_src, PHASE_PPS)
                has_ppp_e = bool(np.any(np.isfinite(xe_ppp) & np.isfinite(oe_ppp) & np.isfinite(te_ppp)))
                has_pps_e = bool(np.any(np.isfinite(xe_pps) & np.isfinite(oe_pps) & np.isfinite(te_pps)))
                if not (has_ppp_e or has_pps_e):
                    equi_missing_obs.append(str(obs_tag))
            elif self.theory_mode.get() == "2Dequi":
                # 文件不存在也计入提示
                equi_missing_obs.append(str(obs_tag))
            # 2Dequi + 勾选：等效 PSP = PSS − mean(PPS−PPP) + 折合走时拟合曲线
            if (
                self.theory_mode.get() == "2Dequi"
                and bool(self.equi_write_equiv_psp.get())
                and r.diff_pairs
            ):
                arr_dt_e = [
                    float(p[2]) for p in r.diff_pairs if len(p) >= 3 and np.isfinite(p[2])
                ]
                if arr_dt_e:
                    avg_ppp_m = float(sum(arr_dt_e) / len(arr_dt_e))
                    xm_eq, off_eq, tt_eq = _phase_model_trueoff_time(r.ds_pss, PHASE_PSS)
                    veq = np.isfinite(xm_eq) & np.isfinite(off_eq) & np.isfinite(tt_eq)
                    if np.any(veq):
                        tred_eq = tt_eq[veq] - avg_ppp_m - np.abs(off_eq[veq]) / vred_ax2
                        ax2.scatter(
                            xm_eq[veq],
                            tred_eq,
                            s=28,
                            marker="D",
                            facecolors=c_equi_psp_mean,
                            edgecolors="#008B8B",
                            linewidths=0.75,
                            alpha=0.95,
                            zorder=4,
                        )
                        legend_ax2["equi_psp"] = True
                        for sgn in (-1.0, 1.0):
                            mseg = veq & ((off_eq * sgn) > 0.0)
                            if np.count_nonzero(mseg) < 4:
                                continue
                            xb = np.asarray(xm_eq[mseg], dtype=float)
                            yb = np.asarray(
                                tt_eq[mseg]
                                - avg_ppp_m
                                - np.abs(off_eq[mseg]) / vred_ax2,
                                dtype=float,
                            )
                            so = np.argsort(xb)
                            xb, yb = xb[so], yb[so]
                            xfit = np.linspace(float(np.min(xb)), float(np.max(xb)), 120)
                            yfit = fit_ppp_time_curve_local_linear(
                                xb,
                                yb,
                                xfit,
                                window_points=self._window_points_value(),
                                split_by_sign=False,
                            )
                            vf = np.isfinite(yfit)
                            if np.any(vf):
                                ax2.plot(
                                    xfit[vf],
                                    yfit[vf],
                                    "--",
                                    color="#008B8B",
                                    linewidth=1.3,
                                    alpha=0.9,
                                    zorder=3,
                                )
            ds_psp_fit, _ = select_phases(r.ds, [psp_id_ax2])
            # 图2叠加 PPP/PPS/PSS/PSP(LocalLinear) 拟合曲线（按左右偏移距分支）
            for ds_phase, pid, fit_style in (
                (r.ds_ppp, PHASE_PPP, "--"),
                (r.ds_pps, PHASE_PPS, "-."),
                (r.ds_pss, PHASE_PSS, ":"),
                (ds_psp_fit, psp_id_ax2, "-"),
            ):
                px, poff, pt = _phase_model_trueoff_time(ds_phase, pid)
                if px.size < 5:
                    continue
                pt = pt - np.abs(poff) / vred_ax2
                for sgn in (-1.0, 1.0):
                    ms = (poff * sgn) > 0.0
                    if np.count_nonzero(ms) < 5:
                        continue
                    xb = px[ms]
                    yb = pt[ms]
                    # 同一分支内按“连续数据段”再拆分，避免跨大空白区过度拟合
                    so = np.argsort(xb)
                    xbs = xb[so]
                    ybs = yb[so]
                    if xbs.size < 5:
                        continue
                    dxb = np.diff(xbs)
                    dxb_f = dxb[np.isfinite(dxb)]
                    if dxb_f.size == 0:
                        continue
                    gap_th = 10.0 * float(np.median(dxb_f))
                    if not np.isfinite(gap_th) or gap_th <= 0:
                        gap_th = float(np.max(dxb_f)) + 1e-6
                    cut_idx = np.where(dxb > gap_th)[0]
                    seg_starts = np.concatenate(([0], cut_idx + 1))
                    seg_ends = np.concatenate((cut_idx + 1, [xbs.size]))
                    for s0, s1 in zip(seg_starts, seg_ends):
                        if (s1 - s0) < 4:
                            continue
                        xs_seg = xbs[s0:s1]
                        ys_seg = ybs[s0:s1]
                        xfit = np.linspace(float(np.min(xs_seg)), float(np.max(xs_seg)), 140)
                        tfit = fit_ppp_time_curve_local_linear(
                            xs_seg,
                            ys_seg,
                            xfit,
                            window_points=self._window_points_value(),
                            split_by_sign=False,
                        )
                        vf = np.isfinite(tfit)
                        if np.any(vf):
                            ax2.plot(
                                xfit[vf],
                                tfit[vf],
                                fit_style,
                                color="black",
                                linewidth=2.0,
                                alpha=0.95,
                            )
                            if pid == PHASE_PPP:
                                legend_ax2["fit_ppp"] = True
                            elif pid == PHASE_PPS:
                                legend_ax2["fit_pps"] = True
                            elif pid == PHASE_PSS:
                                legend_ax2["fit_pss"] = True

            # 图3：PPP-PPS 差值 + 拟合
            # 单/多文件都使用 LocalLinear 拟合；多文件横轴统一 model distance
            stat = self._plot_diff_with_locallinear_fit(
                ax3,
                r.diff_pairs,
                label,
                color,
                x_axis=x_axis,
                window_points=self._window_points_value(),
            )
            if stat is not None:
                fit_error_stats.append((obs_tag, color, stat))
                legend_ax3["picked_ppsppp"] = True
                legend_ax3["fit_ppsppp"] = True
            for _md, _to, td in r.diff_pairs:
                if np.isfinite(td):
                    y3_values.append(float(td))
            # 图2(右上)叠加：拾取 PSS-PSP 差值及其拟合曲线
            if pss_psp_pairs:
                arr_ps3 = np.asarray(pss_psp_pairs, dtype=float)
                x_ps3 = arr_ps3[:, 0] if x_axis == "model_distance" else arr_ps3[:, 1]
                off_ps3 = arr_ps3[:, 1]
                td_ps3 = arr_ps3[:, 2]
                vv_ps3 = np.isfinite(x_ps3) & np.isfinite(off_ps3) & np.isfinite(td_ps3)
                if np.any(vv_ps3):
                    ax3.scatter(
                        x_ps3[vv_ps3],
                        td_ps3[vv_ps3],
                        s=20,
                        alpha=0.45,
                        c="#FF1493",
                        marker="D",
                    )
                    legend_ax3["picked_psspsp"] = True
                    for sgn in (-1.0, 1.0):
                        ms = vv_ps3 & ((off_ps3 * sgn) > 0.0)
                        if np.count_nonzero(ms) < 4:
                            continue
                        xx = np.asarray(x_ps3[ms], dtype=float)
                        yy = np.asarray(td_ps3[ms], dtype=float)
                        so = np.argsort(xx)
                        xx = xx[so]
                        yy = yy[so]
                        y_fit = fit_ppp_time_curve_local_linear(
                            xx,
                            yy,
                            xx,
                            window_points=self._window_points_value(),
                            split_by_sign=False,
                        )
                        vf = np.isfinite(y_fit)
                        if np.any(vf):
                            xs_seg, ys_seg = self._break_line_on_large_gap(xx[vf], y_fit[vf])
                            ax3.plot(xs_seg, ys_seg, "-", color="#FF1493", linewidth=1.5, alpha=0.9)
                            legend_ax3["fit_psspsp"] = True
                for _md, _to, td in pss_psp_pairs:
                    if np.isfinite(td):
                        y3_values.append(float(td))

            # 图5：理论 vs 观测（与右上图一致，均用 r.diff_pairs）
            if r.diff_pairs:
                model_x = np.array([p[0] for p in r.diff_pairs], dtype=float)
                true_off = np.array([p[1] for p in r.diff_pairs], dtype=float)
                t_obs = np.array([p[2] for p in r.diff_pairs], dtype=float)
                x_plot = model_x

                ax5.scatter(x_plot, t_obs, s=14, alpha=0.65, c=c_obs, label=f"{label} obs")
                legend_ax5["obs_ppsppp"] = True
                y5_values.extend([float(v) for v in t_obs if np.isfinite(v)])
                # 图3(左下)叠加：拾取 PSS-PSP 差值（不画拟合）
                if pss_psp_pairs:
                    arr_ps5 = np.asarray(pss_psp_pairs, dtype=float)
                    x_ps5 = arr_ps5[:, 0] if x_axis == "model_distance" else arr_ps5[:, 1]
                    td_ps5 = arr_ps5[:, 2]
                    vv_ps5 = np.isfinite(x_ps5) & np.isfinite(td_ps5)
                    if np.any(vv_ps5):
                        ax5.scatter(
                            x_ps5[vv_ps5],
                            td_ps5[vv_ps5],
                            s=20,
                            alpha=0.45,
                            c="#FF1493",
                            marker="D",
                        )
                        legend_ax5["picked_psspsp"] = True
                        y5_values.extend([float(vv) for vv in td_ps5[vv_ps5] if np.isfinite(vv)])
                # 左下图新增：PSS-PSP 理论时差曲线（仅 1D 模式绘制）；2Dequi+写等效PSP 时用 tx.out 的 2D 曲线
                xm_pss3, off_pss3, t_pss3 = _phase_model_trueoff_time(r.ds_pss, PHASE_PSS)
                if xm_pss3.size >= 4 and self.theory_mode.get() not in ("2D", "2Dequi"):
                    try:
                        if inv_for_pss is not None:
                            hx3 = inv_for_pss.x_conv
                            hh3 = inv_for_pss.h_conv
                            hr3 = inv_for_pss.vpratio_conv
                        else:
                            hx3 = None
                            hh3 = None
                            hr3 = None
                        dt_pss3, _, _, _, _ = pss_minus_psp_from_pss_slope_with_profile(
                            xm_pss3,
                            off_pss3,
                            t_pss3,
                            vp=float(self.vp_cr.get()),
                            h_profile_x=hx3,
                            h_profile=hh3,
                            vpratio_profile=hr3,
                            h_default=float(self.h_cr.get()),
                            vpratio_default=float(self.vp_cr.get()) / max(float(self.vs_cr.get()), 1e-6),
                            window_points=self._window_points_value(),
                            split_by_sign=True,
                            n_iter=2,
                            smooth_dense_half_win=self._smooth_dense_half_win_value(),
                        )
                        x3 = xm_pss3
                        vv3 = np.isfinite(x3) & np.isfinite(dt_pss3)
                        if np.any(vv3):
                            so3 = np.argsort(x3[vv3])
                            xs3 = x3[vv3][so3]
                            ys3 = dt_pss3[vv3][so3]
                            xs3_seg, ys3_seg = self._break_line_on_large_gap(xs3, ys3)
                            ax5.plot(xs3_seg, ys3_seg, "--", color=c_psspsp_dt, linewidth=1.6, alpha=0.95)
                            legend_ax5["theo_psspsp_1d"] = True
                            y5_values.extend([float(vv) for vv in ys3 if np.isfinite(vv)])
                    except Exception:
                        pass
                elif (
                    xm_pss3.size >= 4
                    and self.theory_mode.get() == "2Dequi"
                    and bool(self.equi_write_equiv_psp.get())
                ):
                    try:
                        xm_pr, off_pr, _tpr = _phase_model_trueoff_time(r.ds_pss, PHASE_PSS)
                        dt_ps_tx = self._calc_theory_2d_pss_psp(r, xm_pr, off_pr)
                        vvpt = np.isfinite(xm_pr) & np.isfinite(dt_ps_tx)
                        if np.any(vvpt):
                            so_p = np.argsort(xm_pr[vvpt])
                            xpa = xm_pr[vvpt][so_p]
                            ypa = dt_ps_tx[vvpt][so_p]
                            xsg, ysg = self._break_line_on_large_gap(xpa, ypa)
                            ax5.plot(xsg, ysg, "-", color=c_psp_new, linewidth=1.5, alpha=0.92)
                            legend_ax5["theo_psspsp_txout"] = True
                            y5_values.extend([float(v) for v in ypa if np.isfinite(v)])
                    except Exception:
                        pass

                t_th = self._calc_theory(rp, model_x, true_off)
                v = np.isfinite(t_th)
                if np.any(v):
                    so = np.argsort(x_plot[v])
                    xs = x_plot[v][so]
                    ys = t_th[v][so]
                    # 仅在有数据约束的连续区间内连线：大间隔处断开，避免误导性直连
                    xs_seg, ys_seg = self._break_line_on_large_gap(xs, ys)
                    ax5.plot(xs_seg, ys_seg, "-", color=c_theory, linewidth=1.6, alpha=0.95, label=f"{label} theory")
                    legend_ax5["theo_ppsppp"] = True
                    y5_values.extend([float(vv) for vv in t_th[v] if np.isfinite(vv)])
                    # 左下图误差统计（theory vs obs）
                    err = (t_obs[v] - t_th[v]).astype(float)
                    aerr = np.abs(err)
                    theory_obs_error_stats.append(
                        (
                            obs_tag,
                            color,
                            {
                                "rms": float(np.sqrt(np.mean(err ** 2))),
                                "abs_median": float(np.median(aerr)),
                                "abs_max": float(np.max(aerr)),
                            },
                        )
                    )
                # 2D模式：第一张图叠加理论 PPP/PPS（左右支分侧缓存合并，改一侧只动一侧）
                if self.theory_mode.get() in ("2D", "2Dequi"):
                    try:
                        phase_pts = self._merged_theory_phase_pts(
                            r,
                            phase_ids=(PHASE_PPP, PHASE_PPS, int(psp_id_curr)),
                        )
                        if PHASE_PPP in phase_pts:
                            xh, th, sh = phase_pts[PHASE_PPP]
                            vv = np.isfinite(xh) & np.isfinite(th) & np.isfinite(sh)
                            if np.any(vv):
                                th = th[vv] - np.abs(xh[vv] - sh[vv]) / vred_ax2
                                ax2.scatter(
                                    xh[vv],
                                    th,
                                    s=12,
                                    marker="s",
                                    facecolors="none",
                                    edgecolors="red",
                                    linewidths=0.9,
                                    alpha=0.9,
                                )
                                legend_ax2["theo_ppp"] = True
                        if PHASE_PPS in phase_pts:
                            xh, th, sh = phase_pts[PHASE_PPS]
                            vv = np.isfinite(xh) & np.isfinite(th) & np.isfinite(sh)
                            if np.any(vv):
                                th = th[vv] - np.abs(xh[vv] - sh[vv]) / vred_ax2
                                ax2.scatter(
                                    xh[vv],
                                    th,
                                    s=12,
                                    marker="^",
                                    facecolors="none",
                                    edgecolors="purple",
                                    linewidths=0.9,
                                    alpha=0.9,
                                )
                                legend_ax2["theo_pps"] = True
                        if int(psp_id_curr) in phase_pts:
                            xh, th, sh = phase_pts[int(psp_id_curr)]
                            vv = np.isfinite(xh) & np.isfinite(th) & np.isfinite(sh)
                            if np.any(vv):
                                th = th[vv] - np.abs(xh[vv] - sh[vv]) / vred_ax2
                                ax2.scatter(
                                    xh[vv],
                                    th,
                                    s=15,
                                    marker="P",
                                    facecolors="none",
                                    edgecolors=c_theo_psp,
                                    linewidths=1.05,
                                    alpha=0.92,
                                    zorder=5,
                                )
                                legend_ax2["theo_psp"] = True
                    except Exception:
                        pass
                if self.theory_mode.get() in ("2D", "2Dequi"):
                    t_th_1d = self._calc_theory_1d(rp, true_off)
                    v2 = np.isfinite(t_th) & np.isfinite(t_th_1d)
                    if np.any(v2):
                        diff = (t_th[v2] - t_th_1d[v2]).astype(float)
                        rms21 = float(np.sqrt(np.mean(diff ** 2)))
                        qc_2d_vs_1d.append(
                            (
                                _obs_tag_from_result(r),
                                int(np.sum(v2)),
                                int(len(t_th)),
                                rms21,
                            )
                        )

                # 诊断对比：用 PPS 导数估计的理论曲线（仅 1D 模式，不参与主误差统计）
                pps_offsets, pps_times = _phase_true_offset_time(rp.ds_pps, PHASE_PPS)
                if pps_offsets.size >= 4 and self.theory_mode.get() not in ("2D", "2Dequi"):
                    t_th_pps, _ = pps_minus_ppp_from_ppp_slope(
                        pps_offsets,
                        pps_times,
                        true_off,
                        h_cr=float(self.h_cr.get()),
                        vp=float(self.vp_cr.get()),
                        vs=float(self.vs_cr.get()),
                        window_points=self._window_points_value(),
                        split_by_sign=True,
                        smooth_dense_half_win=self._smooth_dense_half_win_value(),
                    )
                    vp2 = np.isfinite(t_th_pps)
                    if np.any(vp2):
                        so2 = np.argsort(x_plot[vp2])
                        xs2 = x_plot[vp2][so2]
                        ys2 = t_th_pps[vp2][so2]
                        xs2_seg, ys2_seg = self._break_line_on_large_gap(xs2, ys2)
                        ax5.plot(
                            xs2_seg,
                            ys2_seg,
                            "--",
                            color=c_theory_diag,
                            linewidth=1.2,
                            alpha=0.85,
                        )
                        legend_ax5["theo_ppsppp_diag"] = True

            # 图6：修正 PSP
            # 右下图按 OBS 区分颜色，并叠加原始 PSS 作为对比
            vred = 4.0  # km/s, 折合速度
            equi_ax6 = self.theory_mode.get() == "2Dequi" and bool(self.equi_write_equiv_psp.get())
            c_psp_input = "#7B68EE"   # PSP picks that already exist in original tx.in
            # 原始 tx.in 中已存在的 PSP（按用户指定相位号）也显示出来，便于与校正/理论结果对比
            xs_psp_input, ts_psp_input_red = [], []
            for s in r.ds.shots:
                for p in s.picks:
                    if p.phase_id == int(self.psp_phase_id.get()):
                        xs_psp_input.append(p.x)
                        offp = float(p.x - s.xshot)
                        ts_psp_input_red.append(float(p.t - abs(offp) / vred))
            if xs_psp_input:
                ax6.scatter(
                    xs_psp_input,
                    ts_psp_input_red,
                    c=c_psp_input,
                    s=24,
                    alpha=0.75,
                    marker="s",
                )
                legend_ax6["psp_txin"] = True
            # PSP（输出）
            xs_psp_corr = np.asarray([], dtype=float)
            ts_psp_corr = np.asarray([], dtype=float)        # reduced domain (display)
            ts_psp_corr_real = np.asarray([], dtype=float)   # real-time domain (comparison)
            try:
                ds_psp_picked = self._build_psp_dataset(r, mode="picked")
            except Exception:
                ds_psp_picked = None
            if ds_psp_picked is not None:
                xs_psp, ts_psp_red, ts_psp_real = [], [], []
                for s in ds_psp_picked.shots:
                    for p in s.picks:
                        if p.phase_id == int(self.psp_phase_id.get()):
                            xs_psp.append(p.x)
                            offp = float(p.x - s.xshot)
                            ts_psp_red.append(float(p.t - abs(offp) / vred))
                            ts_psp_real.append(float(p.t))
                if xs_psp:
                    xs_psp_corr = np.asarray(xs_psp, dtype=float)
                    ts_psp_corr = np.asarray(ts_psp_red, dtype=float)
                    ts_psp_corr_real = np.asarray(ts_psp_real, dtype=float)
                    ax6.scatter(
                        xs_psp_corr,
                        ts_psp_corr,
                        c=c_psp_corr,
                        s=28,
                        alpha=0.85,
                        marker="o",
                        label=f"{label} PSP({int(self.psp_phase_id.get())})",
                    )
                    legend_ax6["psp_obs_corr"] = True
            # PSS（原始；2Dequi 下与 tx_2Dequiv 中原始 PSS 一致）
            xs_pss, ts_pss_red = [], []
            for s in r.ds_pss.shots:
                for p in s.picks:
                    if p.phase_id == PHASE_PSS:
                        xs_pss.append(p.x)
                        offp = float(p.x - s.xshot)
                        ts_pss_red.append(float(p.t - abs(offp) / vred))
            if xs_pss:
                ax6.scatter(
                    xs_pss,
                    ts_pss_red,
                    c=c_pss_raw,
                    s=22,
                    alpha=0.45,
                    marker="x",
                    label=f"{label} PSS({PHASE_PSS})",
                )
                legend_ax6["pss_txin"] = True
            # tx.out 理论 PSP（2D / 2Dequi 未写等效PSP）；写等效PSP 时右下不叠此列，避免与需求四条目混淆
            if self.theory_mode.get() in ("2D", "2Dequi") and not equi_ax6:
                try:
                    tx_out6_psp = r.path.parent / "tx.out"
                    pts_psp6 = collect_phase_points_from_txout(
                        tx_out6_psp, phase_ids=(int(psp_id_curr),)
                    )
                    if int(psp_id_curr) in pts_psp6:
                        xh6p, th6p, sh6p = pts_psp6[int(psp_id_curr)]
                        vp6 = np.isfinite(xh6p) & np.isfinite(th6p) & np.isfinite(sh6p)
                        if np.any(vp6):
                            tr6p = th6p[vp6] - np.abs(xh6p[vp6] - sh6p[vp6]) / vred
                            ax6.scatter(
                                xh6p[vp6],
                                tr6p,
                                s=18,
                                marker="P",
                                facecolors="none",
                                edgecolors="#228B22",
                                linewidths=1.05,
                                alpha=0.9,
                                zorder=5,
                            )
                            legend_ax6["psp_txout_theo"] = True
                except Exception:
                    pass
            xm_pss, off_pss, t_pss = _phase_model_trueoff_time(r.ds_pss, PHASE_PSS)
            # 2D / 2Dequi 未写等效PSP：用 tx.out 的 PPS-PPP 时差校正得到的 PSP
            if self.theory_mode.get() in ("2D", "2Dequi") and not equi_ax6 and xm_pss.size >= 4:
                try:
                    dt_pp_tx = self._calc_theory_2d(r, xm_pss, off_pss)
                    t_psp_tx_pp = t_pss - dt_pp_tx
                    t_psp_tx_pp_red = t_psp_tx_pp - np.abs(off_pss) / vred
                    vpp = np.isfinite(xm_pss) & np.isfinite(t_psp_tx_pp_red)
                    if np.any(vpp):
                        ax6.scatter(
                            xm_pss[vpp],
                            t_psp_tx_pp_red[vpp],
                            c=c_psp_tx_pp,
                            s=20,
                            alpha=0.8,
                            marker="v",
                            zorder=4,
                        )
                        legend_ax6["psp_txout_ppsppp"] = True
                except Exception:
                    pass
            # 推断 PSP：t_PSP ≈ t_PSS_pick - Δt_theory。2D/2Dequi 仅用 tx.out 上 PSS-PSP；
            # 插不出整条支时不回退 PPS-PPP，避免与「PSP from observed PPS-PPP」混淆。
            if xm_pss.size >= 4:
                try:
                    if self.theory_mode.get() in ("2D", "2Dequi"):
                        dt_pss = self._calc_theory_2d_pss_psp(r, xm_pss, off_pss)
                    else:
                        # 1D: 用 PSS 斜率 + (初值/反演)h-vpvs
                        if inv_for_pss is not None:
                            hx = inv_for_pss.x_conv
                            hh = inv_for_pss.h_conv
                            hr = inv_for_pss.vpratio_conv
                        else:
                            hx = None
                            hh = None
                            hr = None
                        dt_pss, _, _, _, _ = pss_minus_psp_from_pss_slope_with_profile(
                            xm_pss,
                            off_pss,
                            t_pss,
                            vp=float(self.vp_cr.get()),
                            h_profile_x=hx,
                            h_profile=hh,
                            vpratio_profile=hr,
                            h_default=float(self.h_cr.get()),
                            vpratio_default=float(self.vp_cr.get()) / max(float(self.vs_cr.get()), 1e-6),
                            window_points=self._window_points_value(),
                            split_by_sign=True,
                            n_iter=2,
                            smooth_dense_half_win=self._smooth_dense_half_win_value(),
                        )
                    t_psp_new = t_pss - dt_pss
                    t_psp_new_red = t_psp_new - np.abs(off_pss) / vred
                    vv_new = np.isfinite(xm_pss) & np.isfinite(t_psp_new_red)
                    if np.any(vv_new):
                        ax6.scatter(
                            xm_pss[vv_new],
                            t_psp_new_red[vv_new],
                            c=c_psp_new,
                            s=22,
                            alpha=0.85,
                            marker="^",
                        )
                        legend_ax6["psp_theory_pss"] = True
                    if xs_psp_corr.size > 0 and np.any(vv_new):
                        xr = np.round(xs_psp_corr, 3)
                        xt = np.round(xm_pss[vv_new], 3)
                        common = np.intersect1d(xr, xt)
                        if common.size > 0:
                            e = []
                            for xx in common:
                                yc = float(np.mean(ts_psp_corr_real[xr == xx]))
                                yt = float(np.mean(t_psp_new[vv_new][xt == xx]))
                                e.append(yt - yc)
                            ee = np.asarray(e, dtype=float)
                            if ee.size > 0:
                                psp_compare_stats.append((obs_tag, c_psp_new, float(np.sqrt(np.mean(ee**2)))))
                except Exception:
                    pass

        ax2.set_title(
            "Input PPP/PPS/PSS (2Dequi, tx_2Dequiv.in)"
            if self.theory_mode.get() == "2Dequi"
            else "Input PPP/PPS/PSS"
        )
        ax2.set_xlabel("Model distance (km)")
        ax2.set_ylabel("Reduced time (s), Vred=7 km/s")
        ax2.grid(True, alpha=0.5)
        if self.theory_mode.get() in ("2D", "2Dequi"):
            ax3.set_title(f"PPP-PPS diff + fit{title_suffix}")
        else:
            ax3.set_title(f"PPP-PPS & PSS-PSP diff + fit{title_suffix}")
        mode_title = f"Theory vs Obs ({self.theory_mode.get()}){title_suffix}"
        if self.theory2d_notice:
            mode_title += " [fallback]"
        ax5.set_title(mode_title)
        ax6.set_title(f"corrected PSP (phase {int(self.psp_phase_id.get())}) vs raw PSS (reduced)")
        ax6.set_xlabel("Model distance (km)")
        ax6.set_ylabel("Reduced time (s), Vred=4 km/s")
        ax6.grid(True, alpha=0.5)

        ax3.set_xlabel("Model distance (km)")
        ax5.set_xlabel("Model distance (km)")
        ax5.set_ylabel("PPS-PPP (s)")
        ax5.grid(True, alpha=0.5)
        # 图例：根据实际绘制内容动态生成
        ax2_handles = []
        if legend_ax2["ppp"]:
            ax2_handles.append(Line2D([0], [0], marker="o", linestyle="None", color="C0", label=f"Phase {PHASE_PPP}", markersize=6))
        if legend_ax2["pps"]:
            ax2_handles.append(Line2D([0], [0], marker="o", linestyle="None", color="C1", label=f"Phase {PHASE_PPS}", markersize=6))
        if legend_ax2["pss"]:
            ax2_handles.append(Line2D([0], [0], marker="o", linestyle="None", color="C2", label=f"Phase {PHASE_PSS}", markersize=6))
        if legend_ax2["psp_in"]:
            ax2_handles.append(Line2D([0], [0], marker="s", linestyle="None", color="#7B68EE", label=f"Phase {int(self.psp_phase_id.get())} (tx.in)", markersize=6))
        if legend_ax2["equi_ppp"]:
            ax2_handles.append(Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="none", markeredgecolor="C0", label="Equiv PPP (2Dequi)", markersize=6))
        if legend_ax2["equi_pps"]:
            ax2_handles.append(Line2D([0], [0], marker="^", linestyle="None", markerfacecolor="none", markeredgecolor="C1", label="Equiv PPS (2Dequi)", markersize=6))
        if legend_ax2["fit_ppp"]:
            ax2_handles.append(Line2D([0], [0], linestyle="--", color="black", linewidth=2.0, label="PPP fit"))
        if legend_ax2["fit_pps"]:
            ax2_handles.append(Line2D([0], [0], linestyle="-.", color="black", linewidth=2.0, label="PPS fit"))
        if legend_ax2["fit_pss"]:
            ax2_handles.append(Line2D([0], [0], linestyle=":", color="black", linewidth=2.0, label="PSS fit"))
        if legend_ax2["theo_ppp"]:
            ax2_handles.append(Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="none", markeredgecolor="red", label="Theo PPP points", markersize=6))
        if legend_ax2["theo_pps"]:
            ax2_handles.append(Line2D([0], [0], marker="^", linestyle="None", markerfacecolor="none", markeredgecolor="purple", label="Theo PPS points", markersize=6))
        if legend_ax2["theo_psp"]:
            ax2_handles.append(Line2D([0], [0], marker="P", linestyle="None", markerfacecolor="none", markeredgecolor=c_theo_psp, label="Theo PSP (tx.out)", markersize=7))
        if legend_ax2["equi_psp"]:
            ax2_handles.append(Line2D([0], [0], marker="D", linestyle="None", color=c_equi_psp_mean, label="Equiv PSP (PSS−mean Δ)", markersize=6))
        if ax2_handles:
            ax2.legend(handles=ax2_handles, fontsize=8, loc="lower left")

        ax5_handles = []
        if legend_ax5["obs_ppsppp"]:
            ax5_handles.append(Line2D([0], [0], marker="o", linestyle="None", color=c_obs, label="Observed PPS-PPP", markersize=5))
        if legend_ax5["picked_psspsp"]:
            ax5_handles.append(Line2D([0], [0], marker="D", linestyle="None", color="#FF1493", label="Picked PSS-PSP", markersize=5))
        if legend_ax5["theo_ppsppp"]:
            ax5_handles.append(Line2D([0], [0], linestyle="-", color=c_theory, label="Theoretical PPS-PPP"))
        if legend_ax5["theo_psspsp_txout"]:
            # 与右下图 PSP_new（由 Δt_theory(PSS-PSP) 校正）同色，便于视觉对应
            ax5_handles.append(Line2D([0], [0], linestyle="-", color=c_psp_new, linewidth=1.5, label="Theo Δt(PSS−PSP) from tx.out"))
        if legend_ax5["theo_psspsp_1d"]:
            ax5_handles.append(Line2D([0], [0], linestyle="--", color=c_psspsp_dt, label="Theoretical PSS-PSP"))
        if legend_ax5["theo_ppsppp_diag"]:
            ax5_handles.append(Line2D([0], [0], linestyle="--", color=c_theory_diag, label="Theoretical PPS-PPP (PPS slope)"))
        if ax5_handles:
            ax5.legend(handles=ax5_handles, fontsize=8, loc="lower left", framealpha=0.9)

        ax3_handles = []
        if legend_ax3["picked_ppsppp"]:
            ax3_handles.append(Line2D([0], [0], marker="o", linestyle="None", color="gray", label="Picked PPS-PPP", markersize=5))
        if legend_ax3["fit_ppsppp"]:
            ax3_handles.append(Line2D([0], [0], linestyle="-", color=c_theory, label="PPS-PPP fit"))
        if legend_ax3["picked_psspsp"]:
            ax3_handles.append(Line2D([0], [0], marker="D", linestyle="None", color="#FF1493", label="Picked PSS-PSP", markersize=5))
        if legend_ax3["fit_psspsp"]:
            ax3_handles.append(Line2D([0], [0], linestyle="-", color="#FF1493", label="PSS-PSP fit"))
        if ax3_handles:
            ax3.legend(handles=ax3_handles, fontsize=8, loc="lower left")

        ax6_handles_leg = []
        if legend_ax6["psp_txin"]:
            ax6_handles_leg.append(Line2D([0], [0], marker="s", linestyle="None", color=c_psp_input, label=f"PSP from tx.in (phase {int(self.psp_phase_id.get())})", markersize=6))
        if legend_ax6["psp_obs_corr"]:
            ax6_handles_leg.append(Line2D([0], [0], marker="o", linestyle="None", color=c_psp_corr, label=f"PSP from observed PPS-PPP (phase {int(self.psp_phase_id.get())})", markersize=6))
        if legend_ax6["psp_theory_pss"]:
            ax6_handles_leg.append(
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    linestyle="None",
                    color=c_psp_new,
                    label="PSP_new = PSS_pick − Δt_theory(PSS−PSP)",
                    markersize=6,
                )
            )
        if legend_ax6["pss_txin"]:
            ax6_handles_leg.append(Line2D([0], [0], marker="x", linestyle="None", color=c_pss_raw, label=f"PSS from tx.in (phase {PHASE_PSS})", markersize=6))
        if legend_ax6["psp_txout_ppsppp"]:
            ax6_handles_leg.append(Line2D([0], [0], marker="v", linestyle="None", color=c_psp_tx_pp, label="PSP from tx.out PPS-PPP", markersize=6))
        if legend_ax6["psp_txout_theo"]:
            ax6_handles_leg.append(Line2D([0], [0], marker="P", linestyle="None", markerfacecolor="none", markeredgecolor="#228B22", label="Theo PSP (tx.out)", markersize=7))
        if ax6_handles_leg:
            ax6.legend(handles=ax6_handles_leg, fontsize=8, loc="lower left")
        if psp_compare_stats:
            for i, (obs_tag, color, rmsv) in enumerate(psp_compare_stats[:5]):
                ax6.text(
                    0.02,
                    0.98 - i * 0.07,
                    f"{obs_tag}: dPSP_RMS(real)={rmsv:.4f}s",
                    transform=ax6.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=color, alpha=0.85),
                )

        # 多文件模式下可选：图3与图5共享同一 y 轴范围，便于跨图比较
        if multi_mode and self.share_y_var.get():
            pool = [v for v in (y3_values + y5_values) if np.isfinite(v)]
            if pool:
                ymin = min(pool)
                ymax = max(pool)
                if ymin == ymax:
                    ymin -= 0.1
                    ymax += 0.1
                pad = 0.05 * (ymax - ymin)
                ax3.set_ylim(ymin - pad, ymax + pad)
                ax5.set_ylim(ymin - pad, ymax + pad)

        # 右上图标注每个 OBS 的拟合误差统计（max/min/median）
        self._annotate_fit_error_stats(ax3, fit_error_stats, max_items=6)
        # 左下图标注理论-观测误差统计
        self._annotate_theory_obs_error_stats(ax5, theory_obs_error_stats, max_items=6)

        # OBS 号标注：红色三角 + 红色加号（按 obs 对应 model distance）
        # 台站很多时只画标记、不写字，避免 tight_layout 挤爆
        n_obs = len({m[1] for m in obs_marks}) if obs_marks else 0
        label_obs = n_obs <= 24
        self._annotate_obs_marks(
            ax2, obs_marks, y_fixed=float(self.obs_mark_y.get()), with_text=label_obs
        )
        self._annotate_obs_marks(
            ax6, obs_marks, y_fixed=float(self.obs_mark_y.get()), with_text=label_obs
        )
        if multi_mode:
            self._annotate_obs_marks(
                ax3, obs_marks, y_fixed=float(self.obs_mark_y.get()), with_text=label_obs
            )
            self._annotate_obs_marks(
                ax5, obs_marks, y_fixed=float(self.obs_mark_y.get()), with_text=label_obs
            )

        n_files = len(self.files)
        n_units = len(self.results)
        if n_files <= 3:
            names = ", ".join(p.name for p in self.files)
        else:
            names = f"{self.files[0].name} …(+{n_files - 1})"
        title = f"iphase viewer - {names}"
        if n_units > n_files:
            title += f"  [{n_units} OBS]"
        self.fig.suptitle(title, fontsize=11)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Tight layout not applied.*",
                    category=UserWarning,
                )
                self.fig.tight_layout(rect=(0, 0, 1, 0.96))
        except Exception:
            try:
                self.fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.08, hspace=0.28, wspace=0.22)
            except Exception:
                pass
        plot = getattr(self, "plot", None)
        restored = False
        if plot is not None and hasattr(plot, "restore_saved_views"):
            try:
                restored = bool(plot.restore_saved_views())
            except Exception:
                restored = False
        if plot is not None and not restored and hasattr(plot, "schedule_home_refresh"):
            try:
                plot.schedule_home_refresh()
            except Exception:
                pass
        self.canvas.draw_idle()
        unit_note = (
            f"{n_files} 文件/{n_units} OBS"
            if n_units != n_files
            else f"{n_units} 个单元"
        )
        self.status_var.set(
            f"已加载 {unit_note}；理论模式={self.theory_mode.get()}；"
            f"x轴={'model distance' if multi_mode else 'mixed'}；"
            f"共享y={'on' if (multi_mode and self.share_y_var.get()) else 'off'}；"
            f"强制重算={'on' if self.force_recompute.get() else 'off'}"
        )
        if self.theory_mode.get() == "2Dequi" and equi_missing_obs:
            # 去重保序，避免同一 OBS 重复提示
            miss = list(dict.fromkeys(equi_missing_obs))
            self.status_var.set(
                self.status_var.get() + f"；无等效PPP/PPS: {','.join(miss)}"
            )
        if self.results:
            try:
                spec0 = parse_rin_input_files(self.results[0].path.parent)
                tflag = "r.in" if spec0.tfile_from_rin else "fallback->tx.in"
                vflag = "r.in" if spec0.vfile_from_rin else "fallback->v.in"
                self.status_var.set(
                    self.status_var.get()
                    + f"；2D输入(tfile={spec0.t_file.name}[{tflag}],vfile={spec0.v_file.name}[{vflag}])"
                )
            except Exception:
                pass
        if self.theory_mode.get() in ("2D", "2Dequi"):
            if qc_2d_vs_1d:
                n_valid = int(sum(x[1] for x in qc_2d_vs_1d))
                n_total = int(sum(x[2] for x in qc_2d_vs_1d))
                rms_pool = float(np.mean([x[3] for x in qc_2d_vs_1d]))
                self.status_var.set(
                    self.status_var.get()
                    + f"；2D覆盖={n_valid}/{n_total}；2D-1D RMS={rms_pool:.4f}s"
                )
            if self.theory2d_notice:
                self.status_var.set(self.status_var.get() + f"；{self.theory2d_notice}")

    @staticmethod
    def _annotate_obs_marks(
        ax,
        marks: list[tuple[float, str, str]],
        *,
        y_fixed: float = 1.0,
        with_text: bool = True,
    ) -> None:
        """在指定轴上按 x 位置标注 OBS 号（红色三角+号），y 轴位置固定。"""
        if not marks:
            return
        xlim = ax.get_xlim()
        shown = {}
        for x, tag, color in marks:
            if not (xlim[0] <= x <= xlim[1]):
                continue
            # 相同 tag 只标一次；多个文件相同 OBS 号共用一个标注
            if tag in shown:
                continue
            # 再次兜底：文本只留数字
            txt = re.sub(r"\D+", "", str(tag)) or str(tag)
            y = float(y_fixed)
            ax.plot([x], [y], marker="^", color=color, markersize=8, linestyle="None")
            ax.plot(
                [x],
                [y],
                marker="+",
                color=color,
                markersize=9,
                markeredgewidth=1.6,
                linestyle="None",
            )
            if with_text:
                ax.text(x, y + 0.03, txt, color=color, fontsize=9, va="bottom", ha="center")
            shown[tag] = True

    @staticmethod
    def _plot_diff_with_locallinear_fit(
        ax,
        diff_pairs: list[tuple[float, float, float]],
        label: str,
        color: str,
        *,
        x_axis: str,
        window_points: int,
        fit_color: str = "#FF7F00",
        fit_linestyle: str = "--",
    ) -> dict[str, float] | None:
        """在指定 x 轴上用 LocalLinear 绘制 data 与正/负偏移距分组拟合曲线。"""
        if not diff_pairs:
            return None
        arr = np.asarray(diff_pairs, dtype=float)  # (model_dist, true_offset, t_diff)
        x = arr[:, 0] if x_axis == "model_distance" else arr[:, 1]
        to = arr[:, 1]
        td = arr[:, 2]
        v = np.isfinite(x) & np.isfinite(td) & np.isfinite(to)
        if not np.any(v):
            return None
        x = x[v]
        to = to[v]
        td = td[v]

        ax.scatter(x, td, s=20, alpha=0.45, c=color, label=f"{label} data")
        res_all: list[float] = []

        for _sign_name, mask in (
            ("positive", to > 0),
            ("negative", to < 0),
        ):
            if np.sum(mask) < 4:
                continue
            xx = np.asarray(x[mask], dtype=float)
            yy = np.asarray(td[mask], dtype=float)
            so = np.argsort(xx)
            xx = xx[so]
            yy = yy[so]
            y_fit = fit_ppp_time_curve_local_linear(
                xx,
                yy,
                xx,
                window_points=window_points,
                split_by_sign=False,
            )
            vv = np.isfinite(y_fit)
            if not np.any(vv):
                continue
            xs_seg, ys_seg = MainFigurePlotMixin._break_line_on_large_gap(xx[vv], y_fit[vv])
            ax.plot(xs_seg, ys_seg, fit_linestyle, color=fit_color, linewidth=1.8, alpha=0.95)
            res_all.extend((yy[vv] - y_fit[vv]).tolist())

        if not res_all:
            return None
        r = np.asarray(res_all, dtype=float)
        ar = np.abs(r)
        return {
            "abs_max": float(np.nanmax(ar)),
            "abs_min": float(np.nanmin(ar)),
            "abs_median": float(np.nanmedian(ar)),
        }

    @staticmethod
    def _annotate_fit_error_stats(
        ax,
        items: list[tuple[str, str, dict[str, float]]],
        *,
        max_items: int = 6,
    ) -> None:
        """在图3右上角标注每个 OBS 的拟合误差统计。"""
        if not items:
            return
        x0, y0 = 0.985, 0.97
        dy = 0.075
        show = items[: max(1, int(max_items))]
        for i, (obs_tag, color, st) in enumerate(show):
            txt = (
                f"{obs_tag}: "
                f"|res|max={st['abs_max']:.4f}, "
                f"|res|min={st['abs_min']:.4f}, "
                f"|res|med={st['abs_median']:.4f}"
            )
            ax.text(
                x0,
                y0 - i * dy,
                txt,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color=color,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=color, alpha=0.88),
            )
        if len(items) > len(show):
            ax.text(
                x0,
                y0 - len(show) * dy,
                f"… +{len(items) - len(show)} OBS",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#666666",
            )

    @staticmethod
    def _annotate_theory_obs_error_stats(
        ax,
        items: list[tuple[str, str, dict[str, float]]],
        *,
        max_items: int = 6,
    ) -> None:
        """在左下图右上角标注 theory-vs-obs 误差统计。"""
        if not items:
            return
        x0, y0 = 0.985, 0.97
        dy = 0.078
        show = items[: max(1, int(max_items))]
        for i, (obs_tag, color, st) in enumerate(show):
            txt = (
                f"{obs_tag}: "
                f"RMS={st['rms']:.4f}, "
                f"|e|med={st['abs_median']:.4f}, "
                f"|e|max={st['abs_max']:.4f}"
            )
            ax.text(
                x0,
                y0 - i * dy,
                txt,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color=color,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=color, alpha=0.88),
            )
        if len(items) > len(show):
            ax.text(
                x0,
                y0 - len(show) * dy,
                f"… +{len(items) - len(show)} OBS",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#666666",
            )

    @staticmethod
    def _break_line_on_large_gap(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        在 x 的大间隔位置插入 NaN，使折线分段显示（不跨无数据约束区直连）。
        """
        if x.size <= 2:
            return x, y
        dx = np.diff(x)
        dxf = dx[np.isfinite(dx)]
        if dxf.size == 0:
            return x, y
        # 自适应阈值：超过 10 倍中位间隔认为是不连续约束区
        gap_th = 10.0 * float(np.median(dxf))
        if gap_th <= 0:
            return x, y

        xs = [x[0]]
        ys = [y[0]]
        for i in range(1, x.size):
            if (x[i] - x[i - 1]) > gap_th:
                xs.append(np.nan)
                ys.append(np.nan)
            xs.append(x[i])
            ys.append(y[i])
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

