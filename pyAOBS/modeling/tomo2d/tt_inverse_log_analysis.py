# -*- coding: utf-8 -*-
"""
tt_inverse ``-L`` 日志解析与反演结果可视化（与 ``inverse.cc`` 数据行 26 列 + 可选重力列一致）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def format_decimal_log_tick(x, _pos=None) -> str:
    """对数轴刻度标签用十进制（600 而不是 6×10²）。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v) or v <= 0:
        return ""
    av = abs(v)
    nint = round(v)
    if abs(v - nint) <= max(1e-12, 1e-9 * av):
        return str(int(nint))
    if av >= 1:
        return f"{v:.8f}".rstrip("0").rstrip(".")
    return f"{v:.12f}".rstrip("0").rstrip(".")


def _positive_unique(xs) -> np.ndarray:
    arr = np.asarray(list(xs), dtype=float).ravel()
    return np.unique(arr[np.isfinite(arr) & (arr > 0)])


def _nonsingular_log_range(v0, v1) -> tuple[float, float]:
    if v0 > v1:
        v0, v1 = v1, v0
    if not np.isfinite(v0) or not np.isfinite(v1) or v1 <= 0:
        return 0.1, 10.0
    if v0 <= 0:
        v0 = v1 / 100.0 if v1 > 0 else 0.1
    if v1 <= v0 * 1.01:
        mid = float(np.sqrt(max(v0, 1e-12) * max(v1, 1e-12)))
        return mid / 5.0, mid * 5.0
    return float(v0), float(v1)


def decimal_log_tick_values(vmin, vmax, data_xs=()) -> np.ndarray:
    """对数轴线性主刻度：数据点（如 150）以及 1 / 1.5 / 2 / 3 / 5×10ⁿ。"""
    vmin, vmax = _nonsingular_log_range(vmin, vmax)
    data = _positive_unique(data_xs)
    ticks = [float(v) for v in data if vmin <= float(v) <= vmax]
    e0 = int(np.floor(np.log10(max(vmin, 1e-300))))
    e1 = int(np.ceil(np.log10(max(vmax, 1e-300))))
    for e in range(e0, e1 + 1):
        for m in (1.0, 1.5, 2.0, 3.0, 5.0):
            t = m * (10.0 ** e)
            if vmin <= t <= vmax:
                ticks.append(float(t))
    ticks = sorted({round(t, 12) for t in ticks if t > 0})
    if len(ticks) > 8:
        known = {round(float(v), 12) for v in data}
        keep = []
        for t in ticks:
            if t in known:
                keep.append(t)
                continue
            mag = 10.0 ** np.floor(np.log10(t))
            mant = t / mag
            if min(abs(mant - 1.0), abs(mant - 2.0), abs(mant - 5.0)) < 1e-6:
                keep.append(t)
        ticks = keep or ticks[:8]
    return np.asarray(ticks if ticks else [vmin, vmax], dtype=float)


def _decimal_log_locator(data_xs):
    from matplotlib.ticker import Locator

    data = _positive_unique(data_xs)

    class DecimalLogLocator(Locator):
        def nonsingular(self, v0, v1):
            return _nonsingular_log_range(v0, v1)

        def view_limits(self, vmin, vmax):
            vmin, vmax = self.nonsingular(vmin, vmax)
            if data.size:
                lo, hi = float(data.min()), float(data.max())
                g = float(np.exp(np.mean(np.log(data))))
                if hi / lo < 1.05:
                    return g / 5.0, g * 5.0
                if np.log10(hi) - np.log10(lo) < 0.8:
                    return min(vmin, lo / 2.5), max(vmax, hi * 2.5)
            return vmin, vmax

        def tick_values(self, vmin, vmax):
            return decimal_log_tick_values(vmin, vmax, data)

        def __call__(self):
            return self.tick_values(*self.axis.get_view_interval())

    return DecimalLogLocator()


def _use_decimal_log_xaxis(ax, xs=None) -> None:
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.set_xscale("log")
    loc = _decimal_log_locator([] if xs is None else xs)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(FuncFormatter(format_decimal_log_tick))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    vmin, vmax = ax.get_xlim()
    ax.set_xlim(*loc.view_limits(vmin, vmax))

# 列索引（0 起，与 help_docs 中 1–26 对应）
COL_ITER = 0
COL_ISET = 1
COL_REJECTED = 2
COL_RMS_TOT = 3
COL_CHI_TOT = 4
COL_N_PG = 5
COL_RMS_PG = 6
COL_CHI_PG = 7
COL_N_PMP = 8
COL_RMS_PMP = 9
COL_CHI_PMP = 10
COL_W_SV = 13
COL_W_SD = 14
COL_W_DV = 15
COL_W_DD = 16
COL_PRED_CHI = 20
COL_DV_NORM = 21
COL_DD_NORM = 22
COL_LMVH = 23
COL_LMVV = 24
COL_LMD = 25

_MPL_CJK_CONFIGURED = False


def ensure_matplotlib_cjk_font() -> None:
    """
    为 Matplotlib 选择含中日韩字形的无衬线字体，避免 suptitle/坐标轴中文缺字警告。

    仅从 **fontManager 已索引的字体文件** 中识别 CJK（按路径启发式），**不把**一长串可能未安装的
    字体族名写入 ``font.sans-serif``，以免触发大量 ``findfont: ... not found`` 日志。
    若系统未装任何 CJK 字体，则只保留 DejaVu 等回退字体（中文仍可能显示为方块）。
    """
    global _MPL_CJK_CONFIGURED
    if _MPL_CJK_CONFIGURED:
        return
    _MPL_CJK_CONFIGURED = True
    try:
        import matplotlib
        from matplotlib import font_manager as fm

        def _fontfile_has_cjk(path: str) -> bool:
            if not path:
                return False
            pl = path.lower().replace("\\", "/")
            return any(
                k in pl
                for k in (
                    "notosanscjk",
                    "notoserifcjk",
                    "sourcehansans",
                    "source han",
                    "wqy",
                    "wenquanyi",
                    "msyh",
                    "simhei",
                    "simsun",
                    "droidsansfallback",
                    "arphic",
                    "uming",
                    "ukai",
                    "ipaex",
                    "ipagothic",
                    "ipamincho",
                )
            )

        def _cjk_preference_score(fname: str, family: str = "") -> int:
            """越小越优先作为 sans-serif 首选。细体（Light/Thin）会把图上的字画成灰。"""
            pl = fname.lower().replace("\\", "/")
            fam = str(family or "").lower()
            blob = f"{pl} {fam}"
            order = (
                ("notosanscjk", 0),
                ("notoserifcjk", 1),
                ("sourcehansans", 2),
                ("wqy-microhei", 3),
                ("wqy-zenhei", 3),
                ("wqy", 4),
                ("wenquanyi", 4),
                ("droidsansfallback", 5),
                ("msyh", 6),
                ("simhei", 7),
                ("simsun", 8),
                ("arphic", 9),
                ("uming", 10),
                ("ukai", 11),
                ("ipaex", 12),
                ("ipag", 13),
            )
            best = 99
            for key, rank in order:
                if key in pl:
                    best = min(best, rank)
            if any(
                k in blob
                for k in (
                    "msyhl",
                    "msyhs",
                    "ultralight",
                    "extralight",
                    "semilight",
                    "demilight",
                    "light",
                    "thin",
                )
            ):
                best += 80
            return best

        # 只使用磁盘上真实存在的字体；按路径判断 CJK，避免对虚构族名调用 findfont。
        families_ordered: List[str] = []
        seen: set[str] = set()
        scored: List[Tuple[int, int, str]] = []
        for idx, font in enumerate(fm.fontManager.ttflist):
            if not _fontfile_has_cjk(font.fname):
                continue
            name = font.name
            if name in seen:
                continue
            seen.add(name)
            scored.append((_cjk_preference_score(font.fname, name), idx, name))

        scored.sort(key=lambda t: (t[0], t[1]))
        families_ordered = [t[2] for t in scored]

        fallback = ["DejaVu Sans"]
        if families_ordered:
            matplotlib.rcParams["font.sans-serif"] = families_ordered + [
                x for x in fallback if x not in families_ordered
            ]
        # 未检测到任何 CJK 文件：不写入虚构族名列表，避免 findfont 对每个名字告警
        matplotlib.rcParams["axes.unicode_minus"] = False
        matplotlib.rcParams["text.color"] = "black"
        matplotlib.rcParams["axes.labelcolor"] = "black"
        matplotlib.rcParams["xtick.color"] = "black"
        matplotlib.rcParams["ytick.color"] = "black"
        try:
            matplotlib.rcParams["axes.titlecolor"] = "black"
        except KeyError:
            pass
    except Exception:
        pass


def parse_tt_inverse_log(path: Path) -> List[List[float]]:
    """
    读取日志中非 ``#`` 开头的数值行；每行至少 26 列（联合重力时可有第 27 列）。
    无效行跳过。
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    rows: List[List[float]] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue
        if len(vals) < 26:
            continue
        rows.append(vals)
    return rows


def _header_token_map(s: str) -> Dict[str, str]:
    """``# key=val`` 或 ``-TV_percent=20`` 一类空白分隔标记。"""
    out: Dict[str, str] = {}
    for tok in s.split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        k = k.strip().lstrip("-")
        if k:
            out[k] = v.strip()
    return out


def _parse_on_flag(raw: str | None) -> bool | None:
    if raw is None:
        return None
    t = str(raw).strip().lower()
    if t in ("1", "on", "true"):
        return True
    if t in ("0", "off", "false"):
        return False
    return None


def _parse_float_tok(raw: str | None) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def parse_tt_inverse_log_header(path: Path) -> Dict[str, Any]:
    """
    解析 -L 文件开头的 ``#`` 行，识别阻尼模式（-T / -D）与是否开了 -s。
    同时兼容旧头（``# damp_vel`` / ``# fixed_damping`` / smooth_vel 末位为滤波）。
    """
    info: Dict[str, Any] = {
        "damping_mode": None,
        "filter_2d": None,
        "lsqr_precond": None,
        "lsqr_precond_maxd": None,
        "reuse_forward": None,
        "reuse_thresh": None,
        "coarse2fine": None,
        "legacy_baseline": None,
        "jumping": None,
        "robust": None,
        "crit_chi": None,
        "sv_on": None,
        "sv_wmin": None,
        "sv_wmax": None,
        "sd_on": None,
        "sd_wmin": None,
        "sd_wmax": None,
        "tv_percent": None,
        "td_percent": None,
        "dv_weight": None,
        "dd_weight": None,
        "lsqr_atol": None,
        "lines": [],
    }
    raw = path.read_text(encoding="utf-8", errors="replace")
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if not s.startswith("#"):
            break
        info["lines"].append(s)
        sl = s.lower()
        compact = sl.replace(" ", "")
        kv = _header_token_map(s)
        if sl.startswith("# strategy"):
            j = _parse_on_flag(kv.get("jumping"))
            if j is not None:
                info["jumping"] = j
            r = _parse_on_flag(kv.get("robust"))
            if r is not None:
                info["robust"] = r
            chi = _parse_float_tok(kv.get("crit_chi"))
            if chi is not None:
                info["crit_chi"] = chi
        elif sl.startswith("# smooth_vel"):
            on = _parse_on_flag(kv.get("on"))
            if on is not None:
                info["sv_on"] = on
            wmin = _parse_float_tok(kv.get("wmin"))
            wmax = _parse_float_tok(kv.get("wmax"))
            if wmin is not None:
                info["sv_wmin"] = wmin
            if wmax is not None:
                info["sv_wmax"] = wmax
        elif sl.startswith("# smooth_dep"):
            on = _parse_on_flag(kv.get("on"))
            if on is not None:
                info["sd_on"] = on
            wmin = _parse_float_tok(kv.get("wmin"))
            wmax = _parse_float_tok(kv.get("wmax"))
            if wmin is not None:
                info["sd_wmin"] = wmin
            if wmax is not None:
                info["sd_wmax"] = wmax
        if sl.startswith("# lsqr") and "precond" not in sl:
            atol = _parse_float_tok(kv.get("atol"))
            if atol is not None:
                info["lsqr_atol"] = atol
        tv = _parse_float_tok(kv.get("TV_percent") or kv.get("tv_percent"))
        if tv is not None:
            info["tv_percent"] = tv
        td = _parse_float_tok(kv.get("TD_percent") or kv.get("td_percent"))
        if td is not None:
            info["td_percent"] = td
        dv = _parse_float_tok(kv.get("DV") or kv.get("dv"))
        if dv is not None:
            info["dv_weight"] = dv
        dd = _parse_float_tok(kv.get("DD") or kv.get("dd"))
        if dd is not None:
            info["dd_weight"] = dd
        if "mode=fixed" in compact or "damping_modefixed" in compact:
            info["damping_mode"] = "fixed"
        elif s.startswith("# fixed_damping") and info["damping_mode"] is None:
            info["damping_mode"] = "fixed"
        elif "mode=auto" in compact or "damping_modeauto" in compact:
            info["damping_mode"] = "auto"
        elif (
            s.startswith("# damp_vel") or s.startswith("# auto_damp")
        ) and info["damping_mode"] is None:
            info["damping_mode"] = "auto"
        elif "mode=none" in compact or "damping_modenone" in compact:
            info["damping_mode"] = "none"
        if sl.startswith("# filter_-s"):
            # "# filter_-s: ON  (ON=2D ...)" — 只看冒号后第一个词
            after = sl.split(":", 1)[-1].strip() if ":" in sl else sl
            tok = after.split()[0] if after.split() else ""
            if tok == "on":
                info["filter_2d"] = True
            elif tok == "off":
                info["filter_2d"] = False
        elif sl.startswith("# lsqr_precond"):
            after = sl.split(":", 1)[-1].strip() if ":" in sl else sl
            parts = after.split()
            tok = parts[0] if parts else ""
            if tok == "on":
                info["lsqr_precond"] = True
            elif tok == "off":
                info["lsqr_precond"] = False
            for p in parts:
                if p.startswith("maxd="):
                    try:
                        info["lsqr_precond_maxd"] = float(p.split("=", 1)[1])
                    except ValueError:
                        pass
        elif sl.startswith("# accel"):
            after = sl.split(":", 1)[-1].strip() if ":" in sl else sl
            for p in after.split():
                if p.startswith("reuse="):
                    info["reuse_forward"] = p.split("=", 1)[1] == "on"
                elif p.startswith("thresh="):
                    try:
                        info["reuse_thresh"] = float(p.split("=", 1)[1])
                    except ValueError:
                        pass
                elif p.startswith("c2f="):
                    info["coarse2fine"] = p.split("=", 1)[1] == "on"
                elif p.startswith("legacy="):
                    info["legacy_baseline"] = p.split("=", 1)[1] == "on"
        elif sl.startswith("# filter_2d"):
            if "on=1" in compact:
                info["filter_2d"] = True
            elif "on=0" in compact:
                info["filter_2d"] = False
        elif s.startswith("# smooth_vel") and "-SV" not in s and "log10" not in sl:
            parts = s.split()
            if len(parts) >= 3:
                try:
                    info["filter_2d"] = int(float(parts[-1])) != 0
                except ValueError:
                    pass
    return info


def format_tt_inverse_log_header_summary(info: Dict[str, Any]) -> str:
    """已开功能清单（不写「=开」；关的项省略）。"""
    bits: List[str] = []
    mode = info.get("damping_mode")
    if mode == "fixed":
        bits.append("固定阻尼 -D")
    elif mode == "auto":
        bits.append("自动阻尼 -T")
    if info.get("filter_2d") is True:
        bits.append("滤波 -s")
    if info.get("legacy_baseline") is True:
        bits.append("Legacy")
    else:
        if info.get("reuse_forward") is True:
            th = info.get("reuse_thresh")
            bits.append(
                "前向复用" + (f" 阈={th:g}" if th not in (None, 0, 0.0) else "")
            )
        if info.get("coarse2fine") is True:
            bits.append("C2F")
    if info.get("lsqr_precond") is True:
        mx = info.get("lsqr_precond_maxd")
        bits.append("列预条件" + (f" maxD={mx:g}" if mx is not None else ""))
    return " · ".join(bits)


def _fmt_sv_sd(flag: str, on: Any, wmin: Any, wmax: Any) -> str | None:
    if on is False:
        return None
    if wmin is None and wmax is None:
        return flag if on else None
    lo = wmin if wmin is not None else wmax
    hi = wmax if wmax is not None else wmin
    try:
        a, b = float(lo), float(hi)
    except (TypeError, ValueError):
        return f"{flag}{lo}"
    if abs(a - b) < 1e-12:
        return f"{flag}{a:g}"
    return f"{flag}{a:g}…{b:g}"


def format_tt_inverse_run_params(info: Dict[str, Any]) -> str:
    """挑选/对照用：平滑权、阻尼数值、跳跃与已开策略（含 -SV/-TV 等）。"""
    bits: List[str] = []
    if info.get("jumping") is True:
        bits.append("跳跃")
    if info.get("robust") is True:
        chi = info.get("crit_chi")
        bits.append("稳健 -R" + (f" χ={chi:g}" if chi is not None else ""))
    sv = _fmt_sv_sd("-SV", info.get("sv_on"), info.get("sv_wmin"), info.get("sv_wmax"))
    if sv:
        bits.append(sv)
    sd = _fmt_sv_sd("-SD", info.get("sd_on"), info.get("sd_wmin"), info.get("sd_wmax"))
    if sd:
        bits.append(sd)
    mode = info.get("damping_mode")
    if mode == "auto":
        extra = []
        if info.get("tv_percent") is not None:
            extra.append(f"-TV{float(info['tv_percent']):g}%")
        if info.get("td_percent") is not None:
            extra.append(f"-TD{float(info['td_percent']):g}%")
        bits.append("自动阻尼 " + (" ".join(extra) if extra else "-T"))
    elif mode == "fixed":
        extra = []
        if info.get("dv_weight") is not None:
            extra.append(f"-DV{float(info['dv_weight']):g}")
        if info.get("dd_weight") is not None:
            extra.append(f"-DD{float(info['dd_weight']):g}")
        bits.append("固定阻尼 " + (" ".join(extra) if extra else "-D"))
    strat = format_tt_inverse_log_header_summary(info)
    for bit in strat.split(" · "):
        b = bit.strip()
        if not b:
            continue
        if b.startswith("自动阻尼") or b.startswith("固定阻尼"):
            continue
        if b not in bits:
            bits.append(b)
    return " · ".join(bits)


def sort_log_rows(rows: Sequence[Sequence[float]]) -> List[List[float]]:
    return sorted(rows, key=lambda r: (r[COL_ITER], r[COL_ISET]))


def last_row_metrics(rows: Sequence[Sequence[float]]) -> Dict[str, float] | None:
    if not rows:
        return None
    r = sort_log_rows(list(rows))[-1]
    rough = abs(r[COL_LMVH]) + abs(r[COL_LMVV]) + abs(r[COL_LMD])
    return {
        "rms_total": r[COL_RMS_TOT],
        "chi_total": r[COL_CHI_TOT],
        "pred_chi": r[COL_PRED_CHI],
        "w_sv": r[COL_W_SV],
        "w_sd": r[COL_W_SD],
        "w_dv": r[COL_W_DV],
        "w_dd": r[COL_W_DD],
        "roughness_sum": rough,
    }


def composite_score(pred_chi: float, roughness: float, weight: float) -> float:
    """
    启发式综合得分：越小越倾向「拟合好且模型不过度振荡」。
    score = pred_χ² × (1 + weight × R)，R 为速度水平/垂向与深度粗糙度之和。
    """
    return float(pred_chi * (1.0 + weight * max(roughness, 0.0)))


def format_params_compact(m: Dict[str, float]) -> str:
    """末步日志列 14–17：平滑权重 s_v/s_d，阻尼 dv/dd（与 help_docs 一致）。"""
    return (
        f"sv={m['w_sv']:.4g} sd={m['w_sd']:.4g} "
        f"dv={m['w_dv']:.4g} dd={m['w_dd']:.4g}"
    )


def _collect_last_metrics_series(
    series: Dict[str, Sequence[Sequence[float]]],
) -> Tuple[List[str], List[Dict[str, float]]]:
    names: List[str] = []
    metrics: List[Dict[str, float]] = []
    for name, rows in series.items():
        m = last_row_metrics(rows)
        if m is None:
            continue
        names.append(name)
        metrics.append(m)
    return names, metrics


def single_log_curve_data(rows: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """单日志折线数组（iteration / RMS / χ²）。"""
    if not rows:
        raise ValueError("日志无有效数据行（需至少一行 ≥26 列数值）")
    s = sort_log_rows(rows)
    return {
        "iter": np.array([r[COL_ITER] for r in s], dtype=float),
        "rms_pg": np.array([r[COL_RMS_PG] for r in s], dtype=float),
        "rms_pmp": np.array([r[COL_RMS_PMP] for r in s], dtype=float),
        "chi_tot": np.array([r[COL_CHI_TOT] for r in s], dtype=float),
        "pred_chi": np.array([r[COL_PRED_CHI] for r in s], dtype=float),
    }


def overlay_curve_series(
    series: Dict[str, Sequence[Sequence[float]]],
) -> List[Dict[str, Any]]:
    """多日志叠画：每项含 name 与折线数组。"""
    out: List[Dict[str, Any]] = []
    for name, rows in series.items():
        if not rows:
            continue
        d = single_log_curve_data(rows)
        d["name"] = name
        out.append(d)
    if not out:
        raise ValueError("没有可用的多日志数据")
    return out


def pareto_score_data(
    series: Dict[str, Sequence[Sequence[float]]],
    rough_weight: float = 0.001,
) -> Dict[str, Any]:
    names, metrics = _collect_last_metrics_series(series)
    if not names:
        raise ValueError("没有可用的多日志数据")
    pred = np.array([m["pred_chi"] for m in metrics], dtype=float)
    rough = np.array([m["roughness_sum"] for m in metrics], dtype=float)
    scores = np.array(
        [
            composite_score(float(m["pred_chi"]), float(m["roughness_sum"]), float(rough_weight))
            for m in metrics
        ],
        dtype=float,
    )
    params = [format_params_compact(m) for m in metrics]
    order = np.argsort(scores, kind="mergesort")
    return {
        "names": list(names),
        "metrics": metrics,
        "pred_chi": pred,
        "rough": rough,
        "scores": scores,
        "params": params,
        "bar_order": [int(i) for i in order],
        "rough_weight": float(rough_weight),
    }


PARAM_INFLUENCE_AXES: Tuple[Tuple[str, str], ...] = (
    ("w_sv", "weight_s_v（平滑·速度）"),
    ("w_sd", "weight_s_d（平滑·深度）"),
    ("w_dv", "w_dv（阻尼·速度）"),
    ("w_dd", "w_dd（阻尼·深度）"),
)


def param_influence_data(
    series: Dict[str, Sequence[Sequence[float]]],
) -> Dict[str, Any]:
    names, metrics = _collect_last_metrics_series(series)
    if not names:
        raise ValueError("没有可用的多日志数据")
    return {"names": list(names), "metrics": metrics}


def summary_table_ranked(
    series: Dict[str, Sequence[Sequence[float]]],
    rough_weight: float = 0.001,
) -> List[Tuple[float, str, Dict[str, float]]]:
    """末步按 score 升序：``(score, name, metrics)``。"""
    names, metrics = _collect_last_metrics_series(series)
    if not names:
        raise ValueError("没有可用的多日志数据")
    ranked: List[Tuple[float, str, Dict[str, float]]] = []
    for nm, m in zip(names, metrics):
        sc = composite_score(
            float(m["pred_chi"]),
            float(m["roughness_sum"]),
            float(rough_weight),
        )
        ranked.append((sc, nm, m))
    ranked.sort(key=lambda t: (t[0], t[1]))
    return ranked


def build_figure_multi_param_influence(
    series: Dict[str, Sequence[Sequence[float]]],
    title: str = "",
):
    """
    末步：各反演在日志中记录的平滑/阻尼权重与 pred χ²、粗糙度 R 的关系（每组反演一个点）。
    横轴为对数刻度（十进制标签，含图上实际参数值如 150）。不附图例与点旁文件名；右击点识别该次日志。
    """
    import matplotlib.pyplot as plt

    ensure_matplotlib_cjk_font()
    names, metrics = _collect_last_metrics_series(series)
    if not names:
        raise ValueError("没有可用的多日志数据")

    param_axes: List[Tuple[str, str]] = [
        ("w_sv", "weight_s_v（平滑·速度）"),
        ("w_sd", "weight_s_d（平滑·深度）"),
        ("w_dv", "w_dv（阻尼·速度）"),
        ("w_dd", "w_dd（阻尼·深度）"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(10.5, 11.0), constrained_layout=True)
    fig.suptitle(
        title or "反演参数影响（末步）：日志列 14–17 与 pred χ² / 粗糙度 R",
        fontsize=11,
    )
    scatters: List[Any] = []

    for i, (key, xlab) in enumerate(param_axes):
        w_raw = np.array([float(m[key]) for m in metrics], dtype=float)
        w_plot = np.where(np.isfinite(w_raw) & (w_raw > 0.0), w_raw, np.nan)
        pred = np.array([m["pred_chi"] for m in metrics], dtype=float)
        rough = np.array([m["roughness_sum"] for m in metrics], dtype=float)

        ax_l, ax_r = axes[i, 0], axes[i, 1]
        sl = ax_l.scatter(
            w_plot, pred, s=72, c=range(len(names)), cmap="tab10", zorder=3, picker=8
        )
        sr = ax_r.scatter(
            w_plot, rough, s=72, c=range(len(names)), cmap="tab10", zorder=3, picker=8
        )
        scatters.extend([sl, sr])
        _use_decimal_log_xaxis(ax_l, w_plot)
        ax_l.set_ylabel("pred χ² (LSQR)")
        ax_l.set_xlabel(xlab)
        ax_l.grid(True, alpha=0.3)
        ax_l.set_title("pred χ²")

        _use_decimal_log_xaxis(ax_r, w_plot)
        ax_r.set_ylabel("R = |Lmvh|+|Lmvv|+|Lmd|")
        ax_r.set_xlabel(xlab)
        ax_r.grid(True, alpha=0.3)
        ax_r.set_title("粗糙度 R")

    fig._pyaobs_pareto = {  # type: ignore[attr-defined]
        "scatters": scatters,
        "names": list(names),
    }
    return fig


def build_figure_multi_summary_table(
    series: Dict[str, Sequence[Sequence[float]]],
    rough_weight: float = 0.001,
    title: str = "",
):
    """末步数值表：按综合得分升序（越小越好），对照文件名与平滑/阻尼及指标。"""
    import matplotlib.pyplot as plt

    ensure_matplotlib_cjk_font()
    names, metrics = _collect_last_metrics_series(series)
    if not names:
        raise ValueError("没有可用的多日志数据")

    nrows = len(names)
    # 字号较大时行高略增，避免裁切
    fig_h = min(2.6 + nrows * 0.48, 28.0)
    fig, ax = plt.subplots(figsize=(14.5, fig_h))
    ax.axis("off")
    col_labels = [
        "Run",
        "s_v",
        "s_d",
        "dv",
        "dd",
        "pred chi2",
        "R",
        f"score↑ (w={rough_weight:g})",
        "RMS",
    ]
    ranked: List[Tuple[float, str, Dict[str, float]]] = []
    for nm, m in zip(names, metrics):
        sc = composite_score(
            float(m["pred_chi"]),
            float(m["roughness_sum"]),
            float(rough_weight),
        )
        ranked.append((sc, nm, m))
    ranked.sort(key=lambda t: (t[0], t[1]))
    cell_text: List[List[str]] = []
    for sc, nm, m in ranked:
        cell_text.append(
            [
                nm[:36] + ("…" if len(nm) > 36 else ""),
                f"{m['w_sv']:.5g}",
                f"{m['w_sd']:.5g}",
                f"{m['w_dv']:.5g}",
                f"{m['w_dd']:.5g}",
                f"{m['pred_chi']:.5g}",
                f"{m['roughness_sum']:.5g}",
                f"{sc:.5g}",
                f"{m['rms_total']:.5g}",
            ]
        )
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.08, 2.05)
    fig.suptitle(
        title
        or f"末步汇总（按 score 升序，越小越好）：平滑/阻尼与 pred chi2、R、score（w={rough_weight:g}）、RMS",
        fontsize=13,
        y=0.99,
    )
    plt.subplots_adjust(top=0.88, left=0.04, right=0.98, bottom=0.02)
    order = [nm for _sc, nm, _m in ranked]
    fig._pyaobs_summary_order = order  # type: ignore[attr-defined]
    fig._pyaobs_pareto = {  # type: ignore[attr-defined]
        "names": list(order),
        "table": tbl,
    }
    return fig


def build_figure_single_log(rows: Sequence[Sequence[float]], title: str = ""):
    """单日志：折射/反射 RMS 与卡方随迭代（横轴为 iteration）。"""
    import matplotlib.pyplot as plt

    ensure_matplotlib_cjk_font()
    if not rows:
        raise ValueError("日志无有效数据行（需至少一行 ≥26 列数值）")
    s = sort_log_rows(rows)
    it = np.array([r[COL_ITER] for r in s], dtype=float)
    rms_pg = np.array([r[COL_RMS_PG] for r in s])
    rms_pmp = np.array([r[COL_RMS_PMP] for r in s])
    chi0 = np.array([r[COL_CHI_TOT] for r in s])
    pred = np.array([r[COL_PRED_CHI] for r in s])

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True, constrained_layout=True)
    fig.suptitle(title or "tt_inverse 日志：折射/反射 RMS 与卡方随迭代", fontsize=12)

    axes[0].plot(it, rms_pg, "b-o", markersize=4, lw=1.2)
    axes[0].set_ylabel("RMS 折射 (Pg)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(it, rms_pmp, "r-s", markersize=4, lw=1.2)
    axes[1].set_ylabel("RMS 反射 (PmP)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(it, chi0, "g-s", markersize=4, lw=1.2, label="initial χ² (合并)")
    axes[2].plot(it, pred, "m-^", markersize=4, lw=1.2, label="pred χ² (LSQR)")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("χ²")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    return fig


def build_figure_multi_overlay(
    series: Dict[str, Sequence[Sequence[float]]],
    title: str = "",
):
    """多日志：折射/反射 RMS、pred χ² 随 iteration 叠画（无图例；右击曲线识别该次日志）。"""
    import matplotlib.pyplot as plt

    ensure_matplotlib_cjk_font()
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True, constrained_layout=True)
    fig.suptitle(
        title or "多反演：折射/反射 RMS 与 pred χ² 随 iteration（叠画）",
        fontsize=12,
    )
    cmap = plt.get_cmap("tab10")
    names: List[str] = []
    lines: List[Any] = []
    line_series_index: List[int] = []

    for name, rows in series.items():
        if not rows:
            continue
        s = sort_log_rows(rows)
        it = np.array([r[COL_ITER] for r in s], dtype=float)
        rms_pg = np.array([r[COL_RMS_PG] for r in s])
        rms_pmp = np.array([r[COL_RMS_PMP] for r in s])
        pred = np.array([r[COL_PRED_CHI] for r in s])
        idx = len(names)
        color = cmap(idx % 10)
        (ln0,) = axes[0].plot(
            it, rms_pg, "-o", markersize=3, lw=1.0, color=color, picker=8
        )
        (ln1,) = axes[1].plot(
            it, rms_pmp, "-s", markersize=3, lw=1.0, color=color, picker=8
        )
        (ln2,) = axes[2].plot(
            it, pred, "-^", markersize=3, lw=1.0, color=color, picker=8
        )
        names.append(name)
        lines.extend([ln0, ln1, ln2])
        line_series_index.extend([idx, idx, idx])

    if not names:
        raise ValueError("没有可用的多日志数据")

    axes[0].set_ylabel("RMS 折射 (Pg)")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("RMS 反射 (PmP)")
    axes[1].grid(True, alpha=0.3)
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("pred χ² (LSQR)")
    axes[2].grid(True, alpha=0.3)
    fig._pyaobs_pareto = {  # type: ignore[attr-defined]
        "names": names,
        "lines": lines,
        "line_series_index": line_series_index,
    }
    return fig


def build_figure_multi_pareto_and_score(
    series: Dict[str, Sequence[Sequence[float]]],
    rough_weight: float = 0.001,
    title: str = "",
):
    """
    策略：pred χ² vs 综合粗糙度 (|Lmvh|+|Lmvv|+|Lmd|) 散点；并给出加权得分
    score = pred_χ² × (1 + w×R)，越小越优（启发式，非唯一准则）。
    """
    import matplotlib.pyplot as plt

    ensure_matplotlib_cjk_font()
    names: List[str] = []
    pred_chi: List[float] = []
    rough: List[float] = []
    scores: List[float] = []

    param_lines: List[str] = []
    for name, rows in series.items():
        m = last_row_metrics(rows)
        if m is None:
            continue
        names.append(name)
        param_lines.append(format_params_compact(m))
        pc = m["pred_chi"]
        rg = m["roughness_sum"]
        pred_chi.append(pc)
        rough.append(rg)
        scores.append(composite_score(pc, rg, rough_weight))

    if not names:
        raise ValueError("没有可用的多日志数据")

    n = len(names)
    norm = plt.Normalize(vmin=0, vmax=max(n - 1, 1))
    cmap = plt.get_cmap("tab10")
    point_colors = [cmap(norm(i)) for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    fig.suptitle(
        title or f"反演质量：pred χ²–粗糙度 与 综合得分（w={rough_weight:g}）",
        fontsize=11,
    )

    ax0 = axes[0]
    scatter = ax0.scatter(
        rough,
        pred_chi,
        c=range(n),
        cmap=cmap,
        norm=norm,
        s=80,
        zorder=3,
        picker=8,
    )
    ax0.set_xlabel("R = |Lmvh|+|Lmvv|+|Lmd|（末步粗糙度）")
    ax0.set_ylabel("pred χ² (LSQR, 末步)")
    ax0.grid(True, alpha=0.3)
    ax0.set_title("Pareto 式：左下区域通常更优（拟合好且更平滑）")

    ax1 = axes[1]
    order = np.argsort(scores)
    xo = np.arange(len(names))
    bar_colors = [point_colors[i] for i in order]
    bars = ax1.barh(xo, [scores[i] for i in order], color=bar_colors, alpha=0.85)
    ax1.set_yticks(xo)
    ax1.set_yticklabels(
        [f"{names[i]}\n{param_lines[i]}" for i in order],
        fontsize=6,
    )
    ax1.set_xlabel(f"score = pred_χ² × (1 + {rough_weight:g} × R)（越小越好）")
    ax1.set_title("综合得分排序（启发式）")
    ax1.grid(True, axis="x", alpha=0.3)
    fig._pyaobs_pareto = {  # type: ignore[attr-defined]
        "scatter": scatter,
        "bars": bars,
        "bar_series_index": [int(i) for i in order],
        "names": list(names),
    }

    return fig
