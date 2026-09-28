"""棋盘格分辨率测试 / 蒙特卡洛不确定性 — 无 UI 工作流。"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from ...tomand import validate_tomo2d_geom_data_format
from ...tt_inverse_bundle import classify_tt_inverse_outputs
from ..state.form_state import FormState
from .collectors import collect_tt_inverse_args
from .paths import resolve_existing_file, to_workdir_relative
from .smesh_ops import (
    apply_checkerboard_to_file,
    checkerboard_velocity_fields,
    find_latest_inverse_smesh,
    load_interface_mean_std,
    realization_refl_path,
    stack_mean_std,
    stack_reflector_mean_std,
    write_interface_mean_std,
    write_interface_xz,
    write_velocity_grid_as_smesh,
)
from .ttimes_noise import add_traveltime_noise


def _abs(work: Path, p: str) -> Path:
    path = Path(p).expanduser()
    if path.is_absolute():
        return path.resolve()
    rel = to_workdir_relative(p, work, warn_outside=False).value
    return (work / rel).resolve()


@dataclass
class QcRunResult:
    title: str
    run_dir: Path
    messages: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)


def _copy_inv_kwargs(state: FormState) -> dict:
    _, _, kwargs = collect_tt_inverse_args(state)
    # QC 自管输出路径，去掉用户表单里的落盘项避免写到别处
    for k in ("log_file", "out_root", "dws_file"):
        kwargs.pop(k, None)
    grav = kwargs.get("gravity_opts") or kwargs.get("grav_opts")
    if isinstance(grav, dict):
        grav = dict(grav)
        grav.pop("grav_dws", None)
        kwargs["gravity_opts"] = grav
        kwargs.pop("grav_opts", None)
    return kwargs


def _checkerboard_bg_path(state: FormState) -> str:
    return (
        state.get_str("cb.bg_smesh")
        or state.get_str("inv.mesh")
        or state.get_str("fwd.smesh")
    )


def _companion_refl_rel(work: Path, bg: str | None) -> str | None:
    """背景 smesh 旁边的同轮 ``*.refl.<iter>.<iset>``，相对工区。"""
    from .smesh_ops import companion_inverse_refl

    if not bg:
        return None
    try:
        bg_abs = resolve_existing_file(bg, work)
    except FileNotFoundError:
        bg_abs = _abs(work, bg)
    if not bg_abs.is_file():
        return None
    hit = companion_inverse_refl(bg_abs)
    if hit is None:
        return None
    rel = to_workdir_relative(str(hit), work, warn_outside=False).value
    return rel or str(hit)


def fill_cb_refl_from_companion(state: FormState, work: Path) -> str | None:
    """有同轮界面则写入 ``cb.refl_file``，返回写入的路径。"""
    rel = _companion_refl_rel(work, _checkerboard_bg_path(state))
    if not rel:
        return None
    cur = (state.get_str("cb.refl_file") or "").strip().replace("\\", "/")
    if cur != rel.replace("\\", "/"):
        state.set("cb.refl_file", rel)
    return rel


@dataclass(frozen=True)
class CheckerboardInputs:
    """预览文案 / 预览图 / 运行共用的一份输入（预览即所用）。"""

    bg: str
    bg_key: str
    bg_abs: Path | None
    geom: str
    geom_key: str
    refl: str | None
    refl_src: str
    from_companion: bool
    amp: float | None
    h_len: float | None
    v_len: float | None

    @property
    def staged_refl(self) -> str | None:
        if not self.refl:
            return None
        return f"inputs/{Path(self.refl).name}"


def resolve_checkerboard_inputs(state: FormState, work: Path) -> CheckerboardInputs:
    """先同步同轮面到 ``cb.refl_file``，再读本页字段。预览与运行都走这里。"""
    from .collectors import to_number

    fill_cb_refl_from_companion(state, work)
    bg_key, bg = _filled_field(state, ("cb.bg_smesh", "inv.mesh", "fwd.smesh"))
    geom_key, geom = _filled_field(state, ("cb.geom", "fwd.geom"))
    companion = _companion_refl_rel(work, bg) if bg else None
    refl = (state.get_str("cb.refl_file") or "").strip() or None
    if companion:
        refl = companion
        state.set("cb.refl_file", companion)
        refl_src = "背景同轮界面"
        from_companion = True
    else:
        refl_src = "cb.refl_file"
        from_companion = False
    bg_abs = None
    if bg:
        try:
            bg_abs = resolve_existing_file(bg, work)
        except FileNotFoundError:
            cand = _abs(work, bg)
            bg_abs = cand if cand.is_file() else cand
    return CheckerboardInputs(
        bg=bg,
        bg_key=bg_key,
        bg_abs=bg_abs if bg_abs is not None and bg_abs.is_file() else None,
        geom=geom,
        geom_key=geom_key,
        refl=refl,
        refl_src=refl_src,
        from_companion=from_companion,
        amp=to_number(state.get_str("cb.amp") or "3"),
        h_len=to_number(state.get_str("cb.h_len") or "10"),
        v_len=to_number(state.get_str("cb.v_len") or "5"),
    )


def _resolve_checkerboard_refl(
    state: FormState, work: Path, bg: str | None = None
) -> tuple[str | None, str, bool]:
    """兼容旧调用：与 ``resolve_checkerboard_inputs`` 同一套决议。"""
    _ = bg
    inp = resolve_checkerboard_inputs(state, work)
    return inp.refl, inp.refl_src, inp.from_companion


def _fwd_numeric_from_inv_or_fwd(state: FormState) -> dict:
    """正演 -N：优先用 tt_forward 表单，否则用 tt_inverse 同名字段。"""
    from .collectors import to_number

    kwargs: dict = {}
    has_fwd_n = any(
        state.get_str(f"fwd.{k}") for k in ("xorder", "zorder", "clen", "nintp")
    )
    src = "fwd" if has_fwd_n else "inv"
    numeric = {}
    for key in ["xorder", "zorder", "clen", "nintp"]:
        raw = state.get_str(f"{src}.{key}")
        if raw:
            numeric[key] = to_number(raw)
    cg = state.get_str(f"{src}.bend_cg_tol")
    br = state.get_str(f"{src}.bend_br_tol")
    if cg:
        numeric["tol1"] = to_number(cg)
    if br:
        numeric["tol2"] = to_number(br)
    if len(numeric) == 6:
        kwargs.update(numeric)
    if state.get_bool("fwd.do_full_refl") or state.get_bool("inv.do_full_refl"):
        kwargs["do_full_refl"] = True
    return kwargs


# 棋盘格：要折射+反射走时，但不让界面动。tt_inverse -u 可硬冻 -F，
# 本流程仍用极小 -TD 或很大 -DD 软锁（与现网棋盘格习惯一致；-W=0 会被源码忽略）。
_CB_FREEZE_TD = 1e-8
_CB_FREEZE_DD = 1.0e6


def _stage_run_input(
    work: Path, run_dir: Path, src: str | Path, *, stride: int = 1
) -> str:
    """把工区里的输入拷进本次包 ``inputs/``，返回相对 ``run_dir`` 的路径。

    棋盘格/蒙特卡洛把 ``proc_cwd`` 切到 ``runs/<包>/``，表单里的
    ``outputs/vpfd42.refl`` 等工区相对路径在包内不存在。正规反演走
    ``TtInverseBundleBuilder`` 已经会拷贝；QC 必须自己做同样的事。
    ``stride>1`` 时写入抽稀结果（不在源文件旁另存 ``*_sN``）。
    """
    raw = str(src).strip().strip('"')
    if not raw:
        raise FileNotFoundError("输入路径为空")
    try:
        abs_p = resolve_existing_file(raw, work)
    except FileNotFoundError:
        cand = Path(raw)
        abs_p = cand if cand.is_absolute() else _abs(work, raw)
        if not abs_p.is_file():
            raise FileNotFoundError(
                f"找不到输入文件: {raw}\n"
                f"  工区: {work}\n"
                f"  解析为: {abs_p}"
            ) from None
    dest = run_dir / "inputs" / abs_p.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    from .refl_stride import copy_or_stride_refl

    if dest.resolve() != abs_p.resolve() or stride > 1:
        copy_or_stride_refl(abs_p, dest, stride)
    # 表单常写 outputs/*.refl；若命令行仍带这个相对路径，包内同位置也要有一份
    posix = raw.replace("\\", "/")
    if (
        not Path(raw).is_absolute()
        and posix not in (f"inputs/{abs_p.name}", dest.as_posix())
        and not posix.startswith("inputs/")
    ):
        mirror = run_dir / Path(posix)
        try:
            if mirror.resolve() != dest.resolve() and mirror.resolve() != abs_p.resolve():
                mirror.parent.mkdir(parents=True, exist_ok=True)
                copy_or_stride_refl(abs_p, mirror, stride)
        except OSError:
            pass
    return f"inputs/{abs_p.name}"


def _stage_inv_kwargs_files(work: Path, run_dir: Path, inv_kw: dict) -> list[str]:
    """把反演 kwargs 里所有输入文件拷进包，路径改成 ``inputs/<名>``。"""
    notes: list[str] = []
    from .refl_stride import peek_refl_stride, pop_refl_stride

    refl_stride = peek_refl_stride(inv_kw)

    def stage_one(container: dict, key: str, label: str, *, file_stride: int = 1) -> None:
        val = container.get(key)
        if not val:
            return
        old = str(val)
        container[key] = _stage_run_input(work, run_dir, old, stride=file_stride)
        if old.replace("\\", "/") != container[key]:
            notes.append(f"{label}: {old} → {container[key]}")

    stage_one(inv_kw, "refl_file", "-F", file_stride=refl_stride)
    if refl_stride > 1 and inv_kw.get("refl_file"):
        notes.append(f"-F 按步长 {refl_stride} 抽稀后写入 inputs/（不另存 *_sN）")
    pop_refl_stride(inv_kw)
    stage_one(inv_kw, "filter_bound_file", "-s")

    smooth = inv_kw.get("smooth_opts")
    if isinstance(smooth, dict):
        smooth = dict(smooth)
        stage_one(smooth, "corr_v_fn", "-CV")
        stage_one(smooth, "corr_d_fn", "-CD")
        inv_kw["smooth_opts"] = smooth

    damp = inv_kw.get("damp_opts")
    if isinstance(damp, dict):
        damp = dict(damp)
        stage_one(damp, "damp_v_fn", "-DQ")
        inv_kw["damp_opts"] = damp

    g = inv_kw.get("gravity_opts")
    if isinstance(g, dict):
        g = dict(g)
        stage_one(g, "grav_file", "-ZG")
        if g.get("continent"):
            p, iconv = g["continent"]
            g["continent"] = (_stage_run_input(work, run_dir, str(p)), iconv)
        ou = g.get("ocean_upper")
        if ou:
            up, lo, iconv = ou
            g["ocean_upper"] = (
                _stage_run_input(work, run_dir, str(up)),
                _stage_run_input(work, run_dir, str(lo)),
                iconv,
            )
        ol = g.get("ocean_lower")
        if ol:
            up, iconv = ol
            g["ocean_lower"] = (_stage_run_input(work, run_dir, str(up)), iconv)
        sed = g.get("sediment")
        if sed:
            up, lo, iconv = sed
            g["sediment"] = (
                _stage_run_input(work, run_dir, str(up)),
                _stage_run_input(work, run_dir, str(lo)),
                iconv,
            )
        inv_kw["gravity_opts"] = g
    return notes


def _freeze_checkerboard_interface(inv_kw: dict) -> str:
    """锁住界面深度节点；反射走时仍进核。"""
    if not inv_kw.get("refl_file"):
        return "未提供 -F：只有折射走时。"
    smooth = dict(inv_kw.get("smooth_opts") or {})
    if any(k in smooth for k in ("dep", "dep_log10", "corr_d_fn")):
        smooth.pop("dep", None)
        smooth.pop("dep_log10", None)
        smooth.pop("corr_d_fn", None)
        if smooth:
            inv_kw["smooth_opts"] = smooth
        else:
            inv_kw.pop("smooth_opts", None)
    damp = dict(inv_kw.get("damp_opts") or {})
    has_fixed = any(damp.get(k) is not None for k in ("vel", "dep", "damp_v_fn"))
    if has_fixed:
        damp["dep"] = _CB_FREEZE_DD
        inv_kw["damp_opts"] = damp
        inv_kw.pop("auto_damp_max_dd", None)
        return (
            "有 -F：折射+反射走时都用；界面用很大 -DD 锁住（棋盘格不改反射面）。"
        )
    inv_kw["auto_damp_max_dd"] = _CB_FREEZE_TD
    return (
        "有 -F：折射+反射走时都用；界面用极小 -TD 锁住（棋盘格不改反射面）。"
    )


def _checkerboard_fwd_inv_kwargs(
    state: FormState,
    work: Path,
    *,
    run_dir: Path | None = None,
) -> tuple[dict, dict, list[str]]:
    """正演/反演共用同一份 -F；反演锁界面。"""
    inp = resolve_checkerboard_inputs(state, work)
    fwd_kw = dict(_fwd_numeric_from_inv_or_fwd(state))
    inv_kw = _copy_inv_kwargs(state)
    notes: list[str] = []
    # 丢掉 tt_inverse 起始面；只用 CheckerboardInputs（与预览同一份）
    inv_kw.pop("refl_file", None)
    inv_kw.pop("_refl_stride", None)
    fwd_kw.pop("refl_file", None)
    refl, refl_src, from_companion = inp.refl, inp.refl_src, inp.from_companion
    if from_companion:
        notes.append(
            f"-F 用{refl_src} {refl}（与棋盘预览图一致；不是 inv.refl_file 初始面）"
        )
    elif refl:
        notes.append(
            f"-F 用本页自选 {refl}  ← {refl_src}（背景无同轮界面）"
        )
    if refl:
        inv_kw["refl_file"] = str(refl)
        if run_dir is not None:
            notes.extend(_stage_inv_kwargs_files(work, run_dir, inv_kw))
            fwd_kw["refl_file"] = inv_kw["refl_file"]
        else:
            staged = inp.staged_refl or f"inputs/{Path(str(refl)).name}"
            inv_kw["refl_file"] = staged
            fwd_kw["refl_file"] = staged
        notes.append(_freeze_checkerboard_interface(inv_kw))
    else:
        if run_dir is not None:
            notes.extend(_stage_inv_kwargs_files(work, run_dir, inv_kw))
        notes.append(
            "背景 smesh 无同轮 *.refl.<iter>.<iset>，本页也未自选 cb.refl_file："
            "不传 -F，合成走时只有折射。"
            "要用反射走时请选带同轮界面的反演模型，或在本页自选该模型的界面"
            "（不要填 inv.refl_file 初始面），且 geom 含反射震相。"
        )
    inv_kw["log_file"] = "outputs/tt_inverse.log"
    inv_kw["out_root"] = "outputs/out"
    inv_kw["dws_file"] = "outputs/dws.dat"
    return fwd_kw, inv_kw, notes


def run_checkerboard_test(state: FormState, work: Path, tomo) -> QcRunResult:
    """
    棋盘格：背景 → 真模型(只扰速度) → 正演(折射+反射) → 从背景反演(锁界面) → 百分异常。
    """
    inp = resolve_checkerboard_inputs(state, work)
    bg, geom = inp.bg, inp.geom
    if not bg:
        raise ValueError("棋盘格测试需要背景 smesh（cb.bg_smesh 或 inv.mesh / fwd.smesh）")
    if not geom:
        raise ValueError("棋盘格测试需要几何 geom（cb.geom 或 fwd.geom）")

    amp, h_len, v_len = inp.amp, inp.h_len, inp.v_len
    if amp is None or h_len is None or v_len is None:
        raise ValueError("棋盘格 amp / h_len / v_len 须为数值")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = work / "runs" / f"checkerboard_{stamp}"
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)

    bg_abs = inp.bg_abs or _abs(work, bg)
    geom_abs = _abs(work, geom)
    if not bg_abs.is_file():
        raise FileNotFoundError(f"背景 smesh 不存在: {bg_abs}")
    if not geom_abs.is_file():
        raise FileNotFoundError(f"geom 不存在: {geom_abs}")

    bg_copy = run_dir / "inputs" / "background.smesh"
    geom_copy = run_dir / "inputs" / "geom.dat"
    shutil.copy2(bg_abs, bg_copy)
    shutil.copy2(geom_abs, geom_copy)

    true_smesh = run_dir / "inputs" / "true_checkerboard.smesh"
    apply_checkerboard_to_file(
        bg_copy, true_smesh, amp_percent=float(amp), h_len=float(h_len), v_len=float(v_len)
    )

    syn_tt = run_dir / "inputs" / "syn_ttimes.dat"
    fwd_kw, inv_kw, refl_notes = _checkerboard_fwd_inv_kwargs(
        state, work, run_dir=run_dir
    )
    out_opts = dict(fwd_kw.pop("out_opts", {}) or {})
    out_opts["ttime"] = "inputs/syn_ttimes.dat"
    # 相对 run_dir 跑子进程
    old_cwd = tomo.proc_cwd
    tomo.proc_cwd = str(run_dir)
    logs: list[str] = []
    logs.extend(refl_notes)
    staged_f = fwd_kw.get("refl_file")
    if staged_f:
        staged_abs = run_dir / Path(str(staged_f))
        if not staged_abs.is_file():
            raise FileNotFoundError(
                f"棋盘格正演 -F 未拷进运行包: {staged_f}\n"
                f"  期望: {staged_abs}\n"
                f"  表单路径相对工区，子进程 cwd 已切到 {run_dir}"
            )
        logs.append(f"正演/反演 -F → {staged_f}  （已从工区复制）")
    try:
        logs.append("== tt_forward (true checkerboard) ==")
        tomo.tt_forward(
            smesh="inputs/true_checkerboard.smesh",
            geom="inputs/geom.dat",
            out_opts=out_opts,
            **{k: v for k, v in fwd_kw.items() if k != "out_opts"},
        )
        nsrc, nrcv = validate_tomo2d_geom_data_format(run_dir / "inputs" / "syn_ttimes.dat")
        logs.append(
            f"合成走时 → inputs/syn_ttimes.dat  （{nsrc} 炮 / {nrcv} 道；"
            "来自 tt_forward stdout，不是原生 -T 折合图）"
        )

        logs.append("== tt_inverse (start from background, interface frozen) ==")
        r2 = tomo.tt_inverse(
            mesh="inputs/background.smesh",
            data="inputs/syn_ttimes.dat",
            **inv_kw,
        )
        if r2 is not None and getattr(r2, "stdout", None):
            logs.append(str(r2.stdout)[:2000])
    finally:
        tomo.proc_cwd = old_cwd

    classify_tt_inverse_outputs(run_dir / "outputs")
    dws_path = _find_classified_dws(run_dir / "outputs")
    recovered = find_latest_inverse_smesh(run_dir / "outputs" / "out")
    # QC 对照用别名（非「最优」评定）；放在 outputs/ 根便于辨认
    recovered_copy = run_dir / "outputs" / "recovered.smesh"
    shutil.copy2(recovered, recovered_copy)

    # 百分异常：真模型相对背景、反演相对背景
    from .smesh_ops import _load_mesh

    bg_m = _load_mesh(bg_copy)
    true_m = _load_mesh(true_smesh)
    rec_m = _load_mesh(recovered_copy)
    true_anom = (true_m.vgrid - bg_m.vgrid) / np_max(bg_m.vgrid) * 100.0
    rec_anom = (rec_m.vgrid - bg_m.vgrid) / np_max(bg_m.vgrid) * 100.0
    # 写成「伪速度」网格便于用同一 smesh 查看器：100+anom 或直接写 anom 到 vgrid
    # 更清晰：写出 anomaly 场为独立 smesh（vgrid=百分异常）
    true_anom_path = run_dir / "outputs" / "true_anomaly_pct.smesh"
    rec_anom_path = run_dir / "outputs" / "recovered_anomaly_pct.smesh"
    write_velocity_grid_as_smesh(bg_copy, true_anom, true_anom_path, allow_nonpositive=True)
    write_velocity_grid_as_smesh(bg_copy, rec_anom, rec_anom_path, allow_nonpositive=True)
    recovery = checkerboard_recovery_stats(true_anom, rec_anom)
    rec_stat_path = run_dir / "outputs" / "recovery_stats.json"
    rec_stat_path.write_text(
        json.dumps(recovery, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "type": "checkerboard",
        "amp_percent": float(amp),
        "h_len": float(h_len),
        "v_len": float(v_len),
        "background": str(bg_copy),
        "true_smesh": str(true_smesh),
        "recovered_smesh": str(recovered_copy),
        "true_anomaly_pct": str(true_anom_path),
        "recovered_anomaly_pct": str(rec_anom_path),
        "syn_ttimes": str(syn_tt),
        "dws_file": str(dws_path) if dws_path else "outputs/dws.dat",
        "recovery": recovery,
        "refl_file": inv_kw.get("refl_file"),
        "interface_frozen": bool(inv_kw.get("refl_file")),
    }
    man_path = run_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return QcRunResult(
        title="checkerboard",
        run_dir=run_dir,
        messages=logs
        + [
            f"真模型异常 → {true_anom_path.name}",
            f"反演恢复异常 → {rec_anom_path.name}",
            (
                f"格统计：恢复振幅 {recovery['amp_recovery_pct']:.0f}%  "
                f"格中位 {recovery['cell_median_pct']:.0f}%  "
                f"r={recovery['corr']:.2f}"
                if np_isfinite(recovery.get("corr"))
                else f"格统计：恢复振幅 {recovery['amp_recovery_pct']:.0f}%  "
                f"格中位 {recovery['cell_median_pct']:.0f}%"
            ),
            *(
                [f"DWS → {dws_path}"]
                if dws_path
                else ["DWS：未写出（检查反演是否带了 -Koutputs/dws.dat）"]
            ),
            f"清单 → {man_path}",
        ],
        artifacts={
            "run_dir": str(run_dir),
            "true_anomaly_pct": str(true_anom_path),
            "recovered_anomaly_pct": str(rec_anom_path),
            "recovered_smesh": str(recovered_copy),
            "dws_file": str(dws_path) if dws_path else "",
            "manifest": str(man_path),
        },
    )


def _find_classified_dws(outputs_dir: Path) -> Path | None:
    """classify 后 DWS 通常在 outputs/dws/；旧式也可能还在 outputs/dws.dat。"""
    root = Path(outputs_dir)
    for p in (root / "dws" / "dws.dat", root / "dws.dat"):
        if p.is_file() and p.stat().st_size > 0:
            return p
    ddir = root / "dws"
    if ddir.is_dir():
        hits = [
            p
            for p in ddir.iterdir()
            if p.is_file() and p.stat().st_size > 0 and "dws" in p.name.lower()
        ]
        if hits:
            return max(hits, key=lambda p: p.stat().st_mtime)
    return None


def np_max(a):
    import numpy as np

    return np.maximum(a, 1e-9)


MC_CHI_MAX_DEFAULT = 1.8


def mc_chi_max_from_state(state: FormState | None) -> float:
    """叠均值前的 pred χ² 上限（严格小于）。空/非法回退 1.8。"""
    from .collectors import to_number

    raw = ""
    if state is not None:
        raw = str(state.get_str("mc.chi_max") or "").strip()
    if not raw:
        return MC_CHI_MAX_DEFAULT
    parsed = to_number(raw)
    try:
        chi = float(parsed)
    except (TypeError, ValueError):
        return MC_CHI_MAX_DEFAULT
    if not math.isfinite(chi) or chi <= 0.0:
        return MC_CHI_MAX_DEFAULT
    return chi


def find_realization_tt_inverse_log(real_dir: Path | str) -> Path | None:
    """单次实现目录里的 ``-L`` 日志（归类前后均可）。"""
    real = Path(real_dir)
    for cand in (real / "tt_inverse.log", real / "logs" / "tt_inverse.log"):
        try:
            if cand.is_file() and cand.stat().st_size > 0:
                return cand
        except OSError:
            continue
    logs = real / "logs"
    if not logs.is_dir():
        return None
    try:
        kids = sorted(logs.glob("*.log"))
    except OSError:
        return None
    for p in kids:
        try:
            if p.is_file() and p.stat().st_size > 0:
                return p
        except OSError:
            continue
    return None


def realization_pred_chi(real_dir: Path | str) -> float | None:
    """日志末行 pred χ²；读不到则 None。"""
    from ...tt_inverse_log_analysis import last_row_metrics, parse_tt_inverse_log

    log = find_realization_tt_inverse_log(real_dir)
    if log is None:
        return None
    try:
        rows = parse_tt_inverse_log(log)
    except OSError:
        return None
    metrics = last_row_metrics(rows)
    if not metrics:
        return None
    raw = metrics.get("pred_chi")
    if raw is None or not math.isfinite(float(raw)):
        raw = metrics.get("chi_total")
    if raw is None:
        return None
    try:
        chi = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(chi):
        return None
    return chi


def _finite_floats(vals) -> list[float]:
    out: list[float] = []
    for raw in vals:
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def summarize_chi_values(values) -> dict:
    """保留实现的 pred χ²：n / 均值 / 最小 / 最大 / 中位。"""
    xs = sorted(_finite_floats(values))
    if not xs:
        return {}
    n = len(xs)
    mid = n // 2
    median = float(xs[mid]) if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])
    return {
        "n": n,
        "mean": sum(xs) / n,
        "min": xs[0],
        "max": xs[-1],
        "median": median,
    }


def resolve_mc_chi_max(state: FormState | None = None, man_info: dict | None = None) -> float:
    """结果图 / 叠均值用的 pred χ² 阈值：表单 > 清单 > 默认 1.8。"""
    if state is not None:
        raw = str(state.get_str("mc.chi_max") or "").strip()
        if raw:
            return mc_chi_max_from_state(state)
    if isinstance(man_info, dict) and man_info.get("chi_max") is not None:
        try:
            v = float(man_info["chi_max"])
        except (TypeError, ValueError):
            v = None
        if v is not None and math.isfinite(v) and v > 0.0:
            return v
    return MC_CHI_MAX_DEFAULT


def _mc_real_dir_of(path: Path) -> Path | None:
    cur = Path(path)
    for anc in (cur if cur.is_dir() else cur.parent, *cur.parents):
        try:
            if anc.parent.name == "reals":
                return anc
        except Exception:
            break
    return None


def collect_mc_realization_records(
    run: Path | str,
    man_info: dict | None = None,
) -> list[dict]:
    """各次实现：smesh、pred χ²、界面。不按卡方筛选。"""
    recs: list[dict] = []
    root = Path(run)
    reals = root / "reals"
    if reals.is_dir():
        try:
            kids = sorted(reals.iterdir())
        except OSError:
            kids = []
        for d in kids:
            if not d.is_dir():
                continue
            try:
                smesh = find_latest_inverse_smesh(d / "out")
            except FileNotFoundError:
                continue
            recs.append(
                {
                    "smesh": smesh,
                    "real_dir": d,
                    "pred_chi": realization_pred_chi(d),
                    "refl": realization_refl_path(smesh, d),
                }
            )
        if recs:
            return recs
    info = man_info if isinstance(man_info, dict) else {}
    rows = info.get("realization_chi")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw = row.get("smesh")
            if not raw:
                continue
            smesh = Path(str(raw))
            if not smesh.is_file():
                continue
            real_dir = _mc_real_dir_of(smesh)
            chi = row.get("pred_chi")
            if chi is None and real_dir is not None:
                chi = realization_pred_chi(real_dir)
            recs.append(
                {
                    "smesh": smesh,
                    "real_dir": real_dir or smesh.parent,
                    "pred_chi": chi,
                    "refl": realization_refl_path(smesh, real_dir),
                }
            )
        if recs:
            return recs
    for raw in info.get("realizations") or []:
        smesh = Path(str(raw))
        if not smesh.is_file():
            continue
        real_dir = _mc_real_dir_of(smesh)
        recs.append(
            {
                "smesh": smesh,
                "real_dir": real_dir or smesh.parent,
                "pred_chi": realization_pred_chi(real_dir) if real_dir else None,
                "refl": realization_refl_path(smesh, real_dir),
            }
        )
    return recs


def filter_mc_records_by_chi(records: list[dict], chi_max: float) -> list[dict]:
    """只保留 pred χ² < 阈值；读不到 χ² 的不进平均。"""
    thr = float(chi_max)
    kept: list[dict] = []
    for rec in records:
        raw = rec.get("pred_chi")
        try:
            chi = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(chi) and chi < thr:
            kept.append(rec)
    return kept


def kept_pred_chi_values(
    man_info: dict,
    run: Path | None = None,
    *,
    chi_max: float | None = None,
) -> list[float]:
    """通过卡方阈值的 pred χ²。不信任清单里的 kept（旧包可能全标保留）。"""
    thr = float(chi_max) if chi_max is not None else resolve_mc_chi_max(None, man_info)
    recs = collect_mc_realization_records(run, man_info) if run is not None else []
    if recs:
        return [float(r["pred_chi"]) for r in filter_mc_records_by_chi(recs, thr)]
    vals: list[float] = []
    rows = man_info.get("realization_chi") if isinstance(man_info, dict) else None
    if isinstance(rows, list) and rows:
        for row in rows:
            if isinstance(row, dict):
                vals.extend(_finite_floats([row.get("pred_chi")]))
    else:
        cached = (man_info or {}).get("chi_stats") if isinstance(man_info, dict) else None
        if isinstance(cached, dict) and isinstance(cached.get("values"), list):
            vals = _finite_floats(cached.get("values"))
    return [v for v in vals if v < thr]


def format_mc_result_notes(
    man_info: dict,
    *,
    run: Path | None = None,
    state: FormState | None = None,
    chi_max: float | None = None,
    kept_chi: list[float] | None = None,
    n_kept: int | None = None,
    n_all: int | None = None,
) -> tuple[str, str, str]:
    """结果图：均值标题、误差标题、底栏说明。"""
    info = man_info if isinstance(man_info, dict) else {}
    thr = (
        float(chi_max)
        if chi_max is not None
        else resolve_mc_chi_max(state, info)
    )
    if n_all is None:
        try:
            n_all = int(info["n_runs"]) if info.get("n_runs") is not None else None
        except (TypeError, ValueError):
            n_all = None
    chi_vals = (
        list(kept_chi)
        if kept_chi is not None
        else kept_pred_chi_values(info, run, chi_max=thr)
    )
    chi_sum = summarize_chi_values(chi_vals)
    if n_kept is None:
        n_kept = int(chi_sum["n"]) if chi_sum else None
    try:
        n_all_i = int(n_all) if n_all is not None else None
    except (TypeError, ValueError):
        n_all_i = None
    try:
        n_kept_i = int(n_kept) if n_kept is not None else None
    except (TypeError, ValueError):
        n_kept_i = None

    if n_kept_i is not None and n_all_i is not None:
        n_txt = f"用 {n_kept_i}/{n_all_i} 个模型平均"
    elif n_kept_i is not None:
        n_txt = f"用 {n_kept_i} 个模型平均"
    else:
        n_txt = "用保留模型平均"
    n_txt += f"（pred χ² < {thr:g}）"

    chi_txt = ""
    if chi_sum:
        chi_txt = (
            f"保留 χ²：均值 {chi_sum['mean']:.2f}  "
            f"最小 {chi_sum['min']:.2f}  "
            f"最大 {chi_sum['max']:.2f}  "
            f"中位 {chi_sum['median']:.2f}"
        )
    mean_title = f"均值 Vp  ·  {n_txt}"
    if chi_txt:
        mean_title += f"\n{chi_txt}"
    std_n = f"{n_kept_i} 个模型" if n_kept_i is not None else "同上"
    std_title = f"误差 σ (km/s)  ·  {std_n}  ·  红线=界面均值，色带=±σ"
    hint = n_txt
    if chi_txt:
        hint += f"  ·  {chi_txt}"
    hint += "  ·  红线/色带：界面均值 ±σ  ·  右键保存 / 写入表单"
    return mean_title, std_title, hint


def run_monte_carlo(state: FormState, work: Path, tomo) -> QcRunResult:
    """
    蒙特卡洛：可选随机初始模型 + 可选走时噪声，多次反演后输出均值与标准差 smesh。
    """
    from .collectors import to_number

    mesh = state.get_str("mc.base_mesh") or state.get_str("inv.mesh")
    data = state.get_str("mc.data") or state.get_str("inv.data")
    if not mesh or not data:
        raise ValueError("蒙特卡洛需要 base_mesh 与 data（或填写 tt_inverse 页 mesh/data）")

    n = int(to_number(state.get_str("mc.n_runs") or "10") or 10)
    if n < 2:
        raise ValueError("蒙特卡洛至少 2 次实现")
    seed0 = int(to_number(state.get_str("mc.seed") or "1") or 1)
    from .mc_init_models import (
        apply_mc_init_to_file,
        layer1d_bounds_from_state,
        resolve_mc_init_mode,
    )
    from .mc_vin_layers import resolve_mc_vin_path, vin_spec_from_state

    init_mode = resolve_mc_init_mode(state)
    init_amp = float(to_number(state.get_str("mc.init_amp") or "2") or 2)
    layer_bounds = layer1d_bounds_from_state(state)
    vin_path = resolve_mc_vin_path(state, work)
    vin_spec = None
    if init_mode == "vinlayers":
        if vin_path is None:
            raise FileNotFoundError("扰动 v.in 分层须指定存在的 v.in（本页或 gen_smesh）")
        vin_spec = vin_spec_from_state(state, vin_path)
    do_noise = state.get_bool("mc.tt_noise")
    sigma = float(to_number(state.get_str("mc.noise_sigma") or "0.01") or 0.01)
    rel_u = state.get_bool("mc.noise_relative_u")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = work / "runs" / f"montecarlo_{stamp}"
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "reals").mkdir(parents=True, exist_ok=True)

    mesh_abs = _abs(work, mesh)
    data_abs = _abs(work, data)
    if not mesh_abs.is_file():
        raise FileNotFoundError(f"base mesh 不存在: {mesh_abs}")
    if not data_abs.is_file():
        raise FileNotFoundError(f"data 不存在: {data_abs}")

    base_mesh = run_dir / "inputs" / "base.smesh"
    base_data = run_dir / "inputs" / "base_ttimes.dat"
    shutil.copy2(mesh_abs, base_mesh)
    shutil.copy2(data_abs, base_data)
    if init_mode == "vinlayers" and vin_path is not None:
        shutil.copy2(vin_path, run_dir / "inputs" / Path(vin_path).name)

    inv_kw_base = _copy_inv_kwargs(state)
    stage_notes = _stage_inv_kwargs_files(work, run_dir, inv_kw_base)
    old_cwd = tomo.proc_cwd
    tomo.proc_cwd = str(run_dir)
    records: list[dict] = []
    logs: list[str] = []
    logs.extend(stage_notes)
    chi_max = mc_chi_max_from_state(state)

    try:
        for i in range(n):
            seed = seed0 + i
            real_dir = run_dir / "reals" / f"{i:03d}"
            real_dir.mkdir(parents=True, exist_ok=True)
            mesh_i = real_dir / "init.smesh"
            data_i = real_dir / "data.dat"
            refl_i = (
                real_dir / "moho.refl"
                if init_mode in {"layers1d", "vinlayers"}
                else None
            )
            _mesh_p, refl_p = apply_mc_init_to_file(
                base_mesh,
                mesh_i,
                mode=init_mode,
                seed=seed,
                amp_percent=init_amp,
                bounds=layer_bounds,
                refl_dst=refl_i,
                vin_path=vin_path,
                vin_spec=vin_spec,
            )
            if do_noise:
                add_traveltime_noise(
                    base_data,
                    data_i,
                    sigma=sigma,
                    seed=seed + 10007,
                    relative_to_u=rel_u,
                )
            else:
                shutil.copy2(base_data, data_i)

            inv_kw = dict(inv_kw_base)
            inv_kw["log_file"] = f"reals/{i:03d}/tt_inverse.log"
            inv_kw["out_root"] = f"reals/{i:03d}/out"
            inv_kw["dws_file"] = f"reals/{i:03d}/dws.dat"
            if init_mode in {"layers1d", "vinlayers"}:
                inv_kw["refl_file"] = f"reals/{i:03d}/moho.refl"
            logs.append(
                f"== MC realization {i + 1}/{n} (seed={seed} mode={init_mode}) =="
            )
            if refl_p is not None:
                logs.append(
                    f"  -F {inv_kw.get('refl_file')}  (Moho=h_sed+h_uc+h_lc，随海底)"
                )
            tomo.tt_inverse(
                mesh=f"reals/{i:03d}/init.smesh",
                data=f"reals/{i:03d}/data.dat",
                **inv_kw,
            )
            classify_tt_inverse_outputs(real_dir)
            # 取最后写出的网格（非最优评判）；不写 final.smesh
            latest = find_latest_inverse_smesh(real_dir / "out")
            chi = realization_pred_chi(real_dir)
            kept = chi is not None and chi < chi_max
            chi_txt = f"{chi:.4g}" if chi is not None else "（无日志）"
            logs.append(
                f"  pred χ²={chi_txt}  阈值<{chi_max:g}  → "
                + ("保留" if kept else "剔除")
            )
            records.append(
                {
                    "index": i,
                    "seed": seed,
                    "smesh": str(latest),
                    "real_dir": str(real_dir),
                    "refl": (
                        str(realization_refl_path(latest, real_dir) or "")
                    ),
                    "pred_chi": chi,
                    "kept": kept,
                }
            )
    finally:
        tomo.proc_cwd = old_cwd

    kept_recs = [r for r in records if r.get("kept")]
    if len(kept_recs) < 2:
        raise ValueError(
            f"卡方筛选后不足 2 个实现（pred χ² < {chi_max:g}，"
            f"保留 {len(kept_recs)}/{len(records)}）"
        )
    recovered_list = [Path(r["smesh"]) for r in kept_recs]
    refl_list = [Path(r["refl"]) for r in kept_recs if r.get("refl")]
    from .dws_plot import (
        dws_xyz_for_plot,
        mean_dws_xyz_from_arrays,
        write_dws_xyz,
    )

    # 与统计均值相同：各点只平均该处 DWS>0 的成员
    dws_each = [
        dws_xyz_for_plot(
            state,
            work,
            Path(r["smesh"]),
            run_dir=Path(r["real_dir"]),
            enabled=True,
        )
        for r in kept_recs
    ]
    n_dws_hit = sum(1 for xyz in dws_each if xyz is not None)
    logs.append(
        f"卡方筛选 pred χ² < {chi_max:g}：保留 {len(kept_recs)}/{len(records)}；"
        f"DWS {n_dws_hit}/{len(kept_recs)}"
    )
    _, mean_v, std_v = stack_mean_std(recovered_list, dws_xyz_list=dws_each)
    mean_dws, _n_dws = mean_dws_xyz_from_arrays(dws_each)
    mean_dws_path = None
    if mean_dws is not None:
        mean_dws_path = write_dws_xyz(
            run_dir / "outputs" / "dws.dat", mean_dws
        )
        write_dws_xyz(run_dir / "outputs" / "dws" / "dws.dat", mean_dws)
    mean_path = run_dir / "outputs" / "mean_velocity.smesh"
    std_path = run_dir / "outputs" / "std_velocity.smesh"
    # 相对不确定性 %
    rel_std = std_v / np_max(mean_v) * 100.0
    rel_path = run_dir / "outputs" / "uncertainty_pct.smesh"
    write_velocity_grid_as_smesh(base_mesh, mean_v, mean_path)
    write_velocity_grid_as_smesh(base_mesh, std_v, std_path, allow_nonpositive=True)
    write_velocity_grid_as_smesh(base_mesh, rel_std, rel_path, allow_nonpositive=True)

    # 简单 1D 平均剖面（对 x 平均）
    profile_path = run_dir / "outputs" / "mean_profile.txt"
    from .smesh_ops import _load_mesh
    import numpy as np

    m = _load_mesh(mean_path)
    z = m.zpos
    v_mean_1d = np.mean(mean_v, axis=0)
    v_std_1d = np.mean(std_v, axis=0)
    with profile_path.open("w", encoding="utf-8") as f:
        f.write("# z_rel_km  v_mean  v_std\n")
        for zi, vm, vs in zip(z, v_mean_1d, v_std_1d):
            f.write(f"{zi:.6f}  {vm:.6f}  {vs:.6f}\n")

    mean_moho_path = None
    moho_std_path = None
    if len(refl_list) >= 2:
        rx, rz, rs = stack_reflector_mean_std(refl_list)
        mean_moho_path = write_interface_xz(
            rx,
            rz,
            run_dir / "outputs" / "mean_moho.refl",
            header=f"MC mean Moho  n={len(refl_list)}",
        )
        moho_std_path = write_interface_mean_std(
            rx,
            rz,
            rs,
            run_dir / "outputs" / "moho_mean_std.txt",
            header=f"MC Moho mean±σ  n={len(refl_list)}",
        )

    manifest = {
        "type": "monte_carlo",
        "n_runs": n,
        "n_kept": len(kept_recs),
        "n_rejected": len(records) - len(kept_recs),
        "chi_max": chi_max,
        "chi_metric": "pred_chi",
        "seed0": seed0,
        "init_mode": init_mode,
        "init_amp_percent": init_amp if init_mode == "perturb" else None,
        "moho_from_layers": init_mode in {"layers1d", "vinlayers"},
        "vin_path": str(vin_path) if init_mode == "vinlayers" and vin_path else None,
        "vin_units": list(vin_spec.units) if vin_spec is not None else None,
        "tt_noise": do_noise,
        "noise_sigma": sigma if do_noise else None,
        "noise_relative_to_u": rel_u if do_noise else None,
        "mean_velocity": str(mean_path),
        "std_velocity": str(std_path),
        "uncertainty_pct": str(rel_path),
        "mean_dws": str(mean_dws_path) if mean_dws_path else None,
        "mean_profile": str(profile_path),
        "mean_moho": str(mean_moho_path) if mean_moho_path else None,
        "moho_mean_std": str(moho_std_path) if moho_std_path else None,
        "n_refl": len(refl_list),
        "n_dws": n_dws_hit,
        "realizations": [str(p) for p in recovered_list],
        "realization_chi": [
            {
                "index": r["index"],
                "seed": r["seed"],
                "pred_chi": r["pred_chi"],
                "kept": r["kept"],
                "smesh": r["smesh"],
            }
            for r in records
        ],
        "chi_stats": {
            **summarize_chi_values(
                [r["pred_chi"] for r in kept_recs if r.get("pred_chi") is not None]
            ),
            "values": [
                r["pred_chi"] for r in kept_recs if r.get("pred_chi") is not None
            ],
            "threshold": chi_max,
            "metric": "pred_chi",
        },
    }
    man_path = run_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return QcRunResult(
        title="monte_carlo",
        run_dir=run_dir,
        messages=logs[-8:]
        + [
            f"卡方筛选 pred χ² < {chi_max:g}：保留 {len(kept_recs)}/{n}",
            f"均值速度 → {mean_path.name}",
            f"速度标准差 → {std_path.name}",
            f"相对不确定度% → {rel_path.name}",
            f"平均剖面 → {profile_path.name}",
            *(
                [f"平均 DWS → {mean_dws_path.name}"]
                if mean_dws_path is not None
                else []
            ),
            *(
                [f"界面均值 → {mean_moho_path.name}", f"界面误差 → {moho_std_path.name}"]
                if mean_moho_path is not None and moho_std_path is not None
                else []
            ),
            f"清单 → {man_path}",
        ],
        artifacts={
            "run_dir": str(run_dir),
            "mean_velocity": str(mean_path),
            "std_velocity": str(std_path),
            "uncertainty_pct": str(rel_path),
            "mean_dws": str(mean_dws_path) if mean_dws_path else "",
            "mean_profile": str(profile_path),
            "mean_moho": str(mean_moho_path) if mean_moho_path else "",
            "moho_mean_std": str(moho_std_path) if moho_std_path else "",
            "manifest": str(man_path),
        },
    )


def _mc_run_complete(p: Path) -> bool:
    return (p / "outputs" / "mean_velocity.smesh").is_file() and (
        p / "outputs" / "std_velocity.smesh"
    ).is_file()


def list_monte_carlo_runs(work: Path, *, limit: int = 80) -> list[Path]:
    """工区 ``runs/montecarlo_*`` 中完整结果包，新→旧。"""
    root = Path(work) / "runs"
    if not root.is_dir():
        return []
    cands: list[tuple[float, Path]] = []
    try:
        kids = list(root.iterdir())
    except OSError:
        return []
    for p in kids:
        if not p.is_dir() or not p.name.startswith("montecarlo_"):
            continue
        if not _mc_run_complete(p):
            continue
        try:
            mt = (p / "outputs" / "mean_velocity.smesh").stat().st_mtime
        except OSError:
            continue
        cands.append((mt, p))
    cands.sort(key=lambda t: t[0], reverse=True)
    return [p for _mt, p in cands[: max(1, int(limit))]]


def monte_carlo_run_label(run: Path) -> str:
    bits = [Path(run).name]
    man = Path(run) / "manifest.json"
    if man.is_file():
        try:
            info = json.loads(man.read_text(encoding="utf-8"))
            n = info.get("n_runs")
            n_kept = info.get("n_kept")
            mode = info.get("init_mode")
            extra = []
            if n_kept is not None and n is not None and int(n_kept) != int(n):
                extra.append(f"N={int(n_kept)}/{int(n)}")
            elif n is not None:
                extra.append(f"N={int(n)}")
            if mode:
                extra.append(str(mode))
            if extra:
                bits.append("  ".join(extra))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return "  ·  ".join(bits)


def find_latest_monte_carlo_run(work: Path) -> Path | None:
    hits = list_monte_carlo_runs(work, limit=1)
    return hits[0] if hits else None


def resolve_monte_carlo_run_dir(
    work: Path, run_dir: Path | str | None = None
) -> Path:
    if run_dir is not None and str(run_dir).strip():
        p = Path(str(run_dir))
        if not p.is_absolute():
            p = Path(work) / p
        if _mc_run_complete(p):
            return p
        raise FileNotFoundError(
            f"蒙特卡洛结果不完整（缺 mean/std_velocity.smesh）:\n{p}"
        )
    hit = find_latest_monte_carlo_run(work)
    if hit is None:
        raise FileNotFoundError(
            "没有完整的蒙特卡洛结果。请先「运行蒙特卡洛分析」。"
        )
    return hit


def mc_interface_mean_std_overlays(
    x,
    z_mean,
    z_std=None,
    *,
    name: str = "界面",
) -> list[dict]:
    """均值实线 + ±σ 色带/虚线。"""
    import numpy as np

    xx = np.asarray(x, dtype=float)
    zm = np.asarray(z_mean, dtype=float)
    extra = [
        {
            "x": xx,
            "z": zm,
            "label": f"{name} 均值",
            "color": "#dc143c",
            "linewidth": 1.8,
            "linestyle": "-",
        }
    ]
    if z_std is None:
        return extra
    zs = np.asarray(z_std, dtype=float)
    extra[0]["z_lo"] = zm - zs
    extra[0]["z_hi"] = zm + zs
    extra[0]["fill_alpha"] = 0.22
    extra.append(
        {
            "x": xx,
            "z": zm - zs,
            "label": f"{name} −σ",
            "color": "#dc143c",
            "linewidth": 1.0,
            "linestyle": ":",
        }
    )
    extra.append(
        {
            "x": xx,
            "z": zm + zs,
            "label": f"{name} +σ",
            "color": "#dc143c",
            "linewidth": 1.0,
            "linestyle": ":",
        }
    )
    return extra


def load_mc_interface_stats(run: Path):
    """从结果包读界面均值/σ；旧包则从各次 refl 重算。"""
    import numpy as np

    packed = load_interface_mean_std(run / "outputs" / "moho_mean_std.txt")
    if packed is not None:
        return packed
    mean_p = run / "outputs" / "mean_moho.refl"
    if mean_p.is_file():
        from pyAOBS.model_building.tomoform import load_tomo2d_interface_file

        x, z = load_tomo2d_interface_file(str(mean_p))
        return np.asarray(x, dtype=float), np.asarray(z, dtype=float), None
    refls: list[Path] = []
    reals = run / "reals"
    if reals.is_dir():
        for d in sorted(reals.iterdir()):
            if not d.is_dir():
                continue
            hit = d / "moho.refl"
            if hit.is_file():
                refls.append(hit)
                continue
            out = d / "out"
            if out.is_dir():
                try:
                    latest = find_latest_inverse_smesh(out)
                except Exception:
                    latest = None
                if latest is not None:
                    rp = realization_refl_path(latest, d)
                    if rp is not None:
                        refls.append(rp)
    if len(refls) >= 2:
        return stack_reflector_mean_std(refls)
    return None


def stack_filtered_mc_ensemble(
    state: FormState,
    work: Path,
    run: Path,
    man_info: dict,
    *,
    chi_max: float | None = None,
):
    """按 pred χ² 重叠均值/σ（打开旧结果图也会筛）。筛不够则 None。"""
    from .dws_plot import dws_xyz_for_plot, mean_dws_xyz_from_arrays

    thr = float(chi_max) if chi_max is not None else resolve_mc_chi_max(state, man_info)
    recs = collect_mc_realization_records(run, man_info)
    if len(recs) < 2:
        return None
    n_chi = 0
    for rec in recs:
        try:
            if rec.get("pred_chi") is not None and math.isfinite(float(rec["pred_chi"])):
                n_chi += 1
        except (TypeError, ValueError):
            continue
    if n_chi < 2:
        return None
    kept = filter_mc_records_by_chi(recs, thr)
    if len(kept) < 2:
        raise ValueError(
            f"卡方筛选后不足 2 个实现（pred χ² < {thr:g}，"
            f"保留 {len(kept)}/{len(recs)}）"
        )
    smeshes = [Path(r["smesh"]) for r in kept]
    dws_each = [
        dws_xyz_for_plot(
            state,
            work,
            Path(r["smesh"]),
            run_dir=r.get("real_dir"),
            enabled=True,
        )
        for r in kept
    ]
    template, mean_v, std_v = stack_mean_std(smeshes, dws_xyz_list=dws_each)
    dws_xyz, n_dws = mean_dws_xyz_from_arrays(dws_each)
    refls = [Path(r["refl"]) for r in kept if r.get("refl")]
    rx = rz = rs = None
    if len(refls) >= 2:
        rx, rz, rs = stack_reflector_mean_std(refls)
    return {
        "chi_max": thr,
        "records": recs,
        "kept": kept,
        "template": template,
        "mean_v": mean_v,
        "std_v": std_v,
        "dws_xyz": dws_xyz,
        "n_dws": n_dws,
        "refl_x": rx,
        "refl_mean": rz,
        "refl_std": rs,
        "n_refl": len(refls),
    }


def _mc_result_dws_xyz(
    state: FormState,
    work: Path,
    run: Path,
    mean_p: Path,
    man_info: dict,
):
    """结果图 DWS：优先 outputs 平均覆盖，否则由保留实现再平均。"""
    from .dws_plot import (
        dws_mask_enabled,
        dws_xyz_for_plot,
        mean_dws_xyz_from_arrays,
    )

    if not dws_mask_enabled(state):
        return None
    xyz = dws_xyz_for_plot(state, work, mean_p, run_dir=run, enabled=True)
    if xyz is not None:
        return xyz
    paths = list(man_info.get("realizations") or [])
    if len(paths) < 1:
        return None
    arrays = [
        dws_xyz_for_plot(state, work, p, enabled=True) for p in paths
    ]
    mean_xyz, _n = mean_dws_xyz_from_arrays(arrays)
    return mean_xyz


def paint_monte_carlo_result(
    widget,
    state: FormState,
    work: Path,
    *,
    run_dir: Path | str | None = None,
    reset_home: bool = True,
) -> Path:
    """结果窗：上均值 Vp、下误差 σ；叠界面均值 ±σ。"""
    import numpy as np

    from ..plots.velocity_contours import (
        auto_contour_specs,
        contour_specs_for_state,
        contours_enabled,
    )
    from .result_nav import (
        diff_vlim_half_range,
        resample_vgrid_fields_to_xarray,
        resolve_sigma_cmap_and_limits,
    )
    from .smesh_ops import _load_mesh
    from .smesh_plot_core import resolve_plot_smesh_cmap

    run = resolve_monte_carlo_run_dir(work, run_dir)
    mean_p = run / "outputs" / "mean_velocity.smesh"
    std_p = run / "outputs" / "std_velocity.smesh"
    man = run / "manifest.json"
    man_info: dict = {}
    if man.is_file():
        try:
            loaded = json.loads(man.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                man_info = loaded
        except (OSError, json.JSONDecodeError, TypeError):
            man_info = {}
    chi_max = resolve_mc_chi_max(state, man_info)
    stacked = stack_filtered_mc_ensemble(
        state, work, run, man_info, chi_max=chi_max
    )
    from .model_compare import EnsembleStatResult

    if stacked is not None:
        mean_v = np.asarray(stacked["mean_v"], dtype=float)
        std_v = np.asarray(stacked["std_v"], dtype=float)
        mesh = stacked["template"]
        write_velocity_grid_as_smesh(mean_p, mean_v, mean_p)
        write_velocity_grid_as_smesh(std_p, std_v, std_p, allow_nonpositive=True)
        dws_xyz = stacked["dws_xyz"]
        if dws_xyz is None:
            dws_xyz = _mc_result_dws_xyz(state, work, run, mean_p, man_info)
        extra = None
        if stacked["refl_x"] is not None:
            extra = mc_interface_mean_std_overlays(
                stacked["refl_x"],
                stacked["refl_mean"],
                stacked["refl_std"],
                name="界面",
            )
            write_interface_xz(
                stacked["refl_x"],
                stacked["refl_mean"],
                run / "outputs" / "mean_moho.refl",
                header=f"MC mean Moho  n={stacked['n_refl']}",
            )
            if stacked["refl_std"] is not None:
                write_interface_mean_std(
                    stacked["refl_x"],
                    stacked["refl_mean"],
                    stacked["refl_std"],
                    run / "outputs" / "moho_mean_std.txt",
                    header=f"MC Moho mean±σ  n={stacked['n_refl']}",
                )
        kept_chi = [float(r["pred_chi"]) for r in stacked["kept"]]
        mean_title, std_title, hint = format_mc_result_notes(
            man_info,
            run=run,
            state=state,
            chi_max=chi_max,
            kept_chi=kept_chi,
            n_kept=len(stacked["kept"]),
            n_all=len(stacked["records"]),
        )
        stat = EnsembleStatResult(
            n=len(stacked["kept"]),
            paths=[Path(r["smesh"]) for r in stacked["kept"]],
            template_path=mean_p,
            mean_v=mean_v,
            std_v=std_v,
            refl_x=stacked["refl_x"],
            refl_mean=stacked["refl_mean"],
            refl_std=stacked["refl_std"],
            n_refl=int(stacked["n_refl"]),
            n_dws=int(stacked["n_dws"] or 0),
            mean_smesh=mean_p,
            std_smesh=std_p,
            mean_refl=(
                run / "outputs" / "mean_moho.refl"
                if (run / "outputs" / "mean_moho.refl").is_file()
                else None
            ),
            mesh=mesh,
            dws_xyz=dws_xyz,
        )
    else:
        mesh = _load_mesh(mean_p)
        mean_v = np.asarray(mesh.vgrid, dtype=float)
        std_v = np.asarray(_load_mesh(std_p).vgrid, dtype=float)
        if std_v.shape != mean_v.shape:
            raise ValueError(f"均值与误差网格不一致: {mean_v.shape} vs {std_v.shape}")
        mean_title, std_title, hint = format_mc_result_notes(
            man_info, run=run, state=state, chi_max=chi_max
        )
        iface = load_mc_interface_stats(run)
        extra = None
        if iface is not None:
            extra = mc_interface_mean_std_overlays(*iface, name="界面")
        dws_xyz = _mc_result_dws_xyz(state, work, run, mean_p, man_info)
        rx = rz = rs = None
        n_refl = 0
        if iface is not None:
            rx, rz, rs = iface[0], iface[1], iface[2] if len(iface) > 2 else None
            n_refl = 2
        stat = EnsembleStatResult(
            n=int(man_info.get("n_kept") or man_info.get("n_runs") or 0),
            paths=[],
            template_path=mean_p,
            mean_v=mean_v,
            std_v=std_v,
            refl_x=rx,
            refl_mean=rz,
            refl_std=rs,
            n_refl=n_refl,
            mean_smesh=mean_p,
            std_smesh=std_p,
            mean_refl=(
                run / "outputs" / "mean_moho.refl"
                if (run / "outputs" / "mean_moho.refl").is_file()
                else None
            ),
            mesh=mesh,
            dws_xyz=dws_xyz,
        )
    from .dws_plot import dws_mask_enabled

    cmap = resolve_plot_smesh_cmap(state, work)
    draw_c = contours_enabled(state)
    contours = contour_specs_for_state(state)
    half_range = diff_vlim_half_range(state, mode="abs")
    sigma_cmap, s_lo, s_hi, sigma_gamma = resolve_sigma_cmap_and_limits(
        std_v, auto=True, half_range=float(half_range)
    )
    std_contours = auto_contour_specs(0.0, max(s_hi, 1e-6)) if draw_c else []
    ds_mean, ds_std = resample_vgrid_fields_to_xarray(mesh, [mean_v, std_v])
    plot_dws = dws_xyz if dws_mask_enabled(state) else None
    widget.set_save_dir(work)
    widget.set_model_source(mean_p, state)
    setattr(widget, "_mc_stat", stat)
    widget.set_velocity_stack(
        [
            {
                "ds": ds_mean,
                "mesh": mesh,
                "extra": extra,
                "cmap_spec": cmap,
                "title": mean_title,
                "cb_label": "km/s",
                "draw_contours": draw_c,
                "contour_specs": contours,
            },
            {
                "ds": ds_std,
                "mesh": mesh,
                "extra": extra,
                "cmap_spec": sigma_cmap,
                "title": std_title,
                "cb_label": "km/s",
                "draw_contours": draw_c,
                "contour_specs": std_contours,
                "vlim": (s_lo, s_hi),
                "norm_gamma": sigma_gamma,
            },
        ],
        dws_xyz=plot_dws,
        reset_home=reset_home,
    )
    widget.set_interaction_hint(hint)
    return run


def checkerboard_preview_params(state: FormState, work: Path):
    inp = resolve_checkerboard_inputs(state, work)
    return inp.bg, inp.bg_abs, inp.amp, inp.h_len, inp.v_len


def paint_checkerboard_preview(
    widget, state: FormState, work: Path, *, reset_home: bool = True
) -> None:
    """预览窗：上 ΔV、中棋盘后 Vp、下棋盘前 Vp（不写盘）。"""
    import numpy as np

    from .dws_plot import dws_xyz_for_plot
    from .result_nav import resample_vgrid_field_to_xarray
    from .smesh_plot_core import load_smesh_plot_data, resolve_plot_smesh_cmap
    from ..plots.velocity_contours import contour_specs_for_state, contours_enabled

    inp = resolve_checkerboard_inputs(state, work)
    bg, bg_abs, amp, h_len, v_len = inp.bg, inp.bg_abs, inp.amp, inp.h_len, inp.v_len
    if not bg or bg_abs is None or not bg_abs.is_file():
        widget.show_empty_stack("选择背景 smesh 后点「棋盘预览图…」")
        return
    if amp is None or h_len is None or v_len is None:
        widget.show_empty_stack("振幅 A、水平波长 h、垂向波长 v 须为数值")
        return
    mesh, v_bg, v_cb, dv = checkerboard_velocity_fields(
        bg_abs, amp_percent=float(amp), h_len=float(h_len), v_len=float(v_len)
    )
    refl_abs = None
    if inp.refl:
        try:
            refl_abs = resolve_existing_file(inp.refl, work)
        except FileNotFoundError:
            cand = _abs(work, inp.refl)
            refl_abs = cand if cand.is_file() else None
    _m, _ds, extra = load_smesh_plot_data(
        bg_abs, str(refl_abs) if refl_abs else None, with_xarray=False
    )
    ds_bg = mesh.to_xarray()
    mesh.vgrid = v_cb
    mesh.pgrid = 1.0 / np.maximum(v_cb, 1e-9)
    ds_cb = mesh.to_xarray()
    mesh.vgrid = v_bg
    mesh.pgrid = 1.0 / np.maximum(v_bg, 1e-9)
    ds_dv = resample_vgrid_field_to_xarray(mesh, dv)
    cmap = resolve_plot_smesh_cmap(state, work)
    peak = float(np.nanmax(np.abs(dv))) if np.isfinite(dv).any() else 0.05
    if not np.isfinite(peak) or peak <= 0:
        peak = 0.05
    draw_c = contours_enabled(state)
    contours = contour_specs_for_state(state)
    dws_xyz = dws_xyz_for_plot(state, work, bg_abs)
    widget.set_save_dir(work)
    widget.set_model_source(bg_abs, state)
    widget.set_velocity_stack(
        [
            {
                "ds": ds_dv,
                "mesh": mesh,
                "extra": extra,
                "cmap_spec": "seismic_r",
                "title": f"扰动 ΔV = 棋盘 − 背景  （峰值 ±{peak:.3g} km/s）",
                "vlim": (-peak, peak),
                "cb_label": "ΔV (km/s)",
                "draw_contours": False,
            },
            {
                "ds": ds_cb,
                "mesh": mesh,
                "extra": extra,
                "cmap_spec": cmap,
                "title": (
                    f"棋盘后 Vp  A={float(amp):g}%  "
                    f"h={float(h_len):g}  v={float(v_len):g}"
                ),
                "cb_label": "km/s",
                "draw_contours": draw_c,
                "contour_specs": contours,
            },
            {
                "ds": ds_bg,
                "mesh": mesh,
                "extra": extra,
                "cmap_spec": cmap,
                "title": f"棋盘前 Vp（背景）  {bg_abs.name}",
                "cb_label": "km/s",
                "draw_contours": draw_c,
                "contour_specs": contours,
            },
        ],
        dws_xyz=dws_xyz,
        reset_home=reset_home,
    )
    widget.set_interaction_hint(
        "上：扰动 ΔV · 中：棋盘后 Vp · 下：棋盘前 Vp · 右键写入表单"
    )


def monte_carlo_preview_params(state: FormState, work: Path):
    from .collectors import to_number
    from .mc_init_models import resolve_mc_init_mode

    mesh = state.get_str("mc.base_mesh") or state.get_str("inv.mesh")
    amp = to_number(state.get_str("mc.init_amp") or "2")
    seed = to_number(state.get_str("mc.seed") or "1")
    mode = resolve_mc_init_mode(state)
    mesh_abs = None
    if mesh:
        try:
            mesh_abs = resolve_existing_file(mesh, work)
        except FileNotFoundError:
            cand = _abs(work, mesh)
            mesh_abs = cand if cand.is_file() else None
    seed_i = int(seed) if seed is not None else None
    return mesh, mesh_abs, mode, amp, seed_i


def paint_monte_carlo_preview(
    widget,
    state: FormState,
    work: Path,
    *,
    reset_home: bool = True,
    profile_plot=None,
) -> None:
    """预览窗：上 ΔV、中第 1 次实现、下基础模型；右侧叠绘全部 1D（不写盘）。"""
    import numpy as np

    from .collectors import to_number
    from .dws_plot import dws_xyz_for_plot
    from .result_nav import resample_vgrid_field_to_xarray
    from .mc_init_models import (
        collect_layered_1d_profiles,
        layer1d_bounds_from_state,
        layered_1d_preview_overlays,
        mc_init_velocity_fields,
    )
    from .smesh_plot_core import (
        load_smesh_plot_data,
        resolve_plot_refl_for_smesh,
        resolve_plot_smesh_cmap,
    )
    from ..plots.velocity_contours import contour_specs_for_state, contours_enabled

    mesh_s, mesh_abs, init_mode, amp, seed = monte_carlo_preview_params(state, work)
    if not mesh_s or mesh_abs is None or not mesh_abs.is_file():
        widget.show_empty_stack("选择初始/背景 mesh 后点「蒙特卡洛预览图…」")
        if profile_plot is not None:
            profile_plot.show_empty("先指定基础 mesh")
        return
    if init_mode == "perturb" and amp is None:
        widget.show_empty_stack("初始扰动幅度须为数值")
        if profile_plot is not None:
            profile_plot.show_empty("扰动幅度无效")
        return
    if seed is None:
        seed = 1
    from .mc_vin_layers import (
        collect_vin_1d_profiles,
        load_zelt,
        resolve_mc_vin_path,
        vin_named_iface_overlays,
        vin_spec_from_state,
    )

    vin_path = resolve_mc_vin_path(state, work)
    vin_spec = None
    if init_mode == "vinlayers":
        if vin_path is None or not Path(vin_path).is_file():
            widget.show_empty_stack("扰动 v.in 分层须指定存在的 v.in")
            if profile_plot is not None:
                profile_plot.show_empty("先指定 v.in")
            return
        vin_spec = vin_spec_from_state(state, vin_path)
    mesh, v_bg, v_pert, dv = mc_init_velocity_fields(
        mesh_abs,
        mode=init_mode,
        seed=int(seed),
        amp_percent=float(amp or 0),
        bounds=layer1d_bounds_from_state(state),
        vin_path=vin_path,
        vin_spec=vin_spec,
    )
    refl = resolve_plot_refl_for_smesh(mesh_abs, state, work)
    _m, _ds, extra = load_smesh_plot_data(
        mesh_abs, str(refl) if refl else None, with_xarray=False
    )
    extra_pert = extra
    profiles = None
    z_max_1d = 0.0
    note_1d = ""
    datum_role = "seafloor"
    n_1d = max(int(to_number(state.get_str("mc.n_runs") or "10") or 10), 1)
    if n_1d > 400:
        note_1d = f"（预览只画 400 / {n_1d}）"
        n_1d = 400
    if init_mode == "layers1d":
        profiles, z_max_1d, _vw = collect_layered_1d_profiles(
            mesh_abs,
            n=n_1d,
            seed0=int(seed),
            bounds=layer1d_bounds_from_state(state),
        )
        extra_pert = layered_1d_preview_overlays(mesh, profiles)
    elif init_mode == "vinlayers" and vin_path is not None and vin_spec is not None:
        zelt = load_zelt(vin_path, clone=False)
        extra_pert = vin_named_iface_overlays(zelt, vin_spec.marks)
        extra = list(extra or []) + extra_pert
        vw = float(getattr(mesh, "v_water", 1.5) or 1.5)
        profiles, z_max_1d, datum_role = collect_vin_1d_profiles(
            vin_path,
            spec=vin_spec,
            n=n_1d,
            seed0=int(seed),
            bounds=layer1d_bounds_from_state(state),
            v_water=vw,
        )
    ds_bg = mesh.to_xarray()
    mesh.vgrid = v_pert
    mesh.pgrid = 1.0 / np.maximum(v_pert, 1e-9)
    ds_pert = mesh.to_xarray()
    mesh.vgrid = v_bg
    mesh.pgrid = 1.0 / np.maximum(v_bg, 1e-9)
    ds_dv = resample_vgrid_field_to_xarray(mesh, dv)
    cmap = resolve_plot_smesh_cmap(state, work)
    peak = float(np.nanmax(np.abs(dv))) if np.isfinite(dv).any() else 0.05
    if not np.isfinite(peak) or peak <= 0:
        peak = 0.05
    draw_c = contours_enabled(state)
    contours = contour_specs_for_state(state)
    dws_xyz = dws_xyz_for_plot(state, work, mesh_abs)
    if init_mode == "layers1d":
        mid_title = (
            f"第 1 次实现初始模型  分段随机 1D  seed={int(seed)}  "
            "虚线=各次 Moho（随海底）"
        )
    elif init_mode == "vinlayers":
        units = " ".join(vin_spec.units) if vin_spec is not None else ""
        mid_title = (
            f"第 1 次实现  扰动 v.in 分层  seed={int(seed)}  "
            f"层 {units or '—'}  -F=莫霍"
        )
    elif init_mode == "perturb":
        mid_title = (
            f"第 1 次实现初始模型  扰动 smesh  seed={int(seed)}  "
            f"amp={float(amp or 0):g}%"
        )
    else:
        mid_title = "第 1 次实现初始模型  （不随机，与基础相同）"
    widget.set_save_dir(work)
    widget.set_model_source(mesh_abs, state)
    widget.set_velocity_stack(
        [
            {
                "ds": ds_dv,
                "mesh": mesh,
                "extra": extra_pert,
                "cmap_spec": "seismic_r",
                "title": (
                    f"ΔV = 第1次实现 − 基础  （峰值 ±{peak:.3g} km/s）"
                    if init_mode != "off"
                    else "ΔV  （不随机）"
                ),
                "vlim": (-peak, peak),
                "cb_label": "ΔV (km/s)",
                "draw_contours": False,
            },
            {
                "ds": ds_pert,
                "mesh": mesh,
                "extra": extra_pert,
                "cmap_spec": cmap,
                "title": mid_title,
                "cb_label": "km/s",
                "draw_contours": draw_c,
                "contour_specs": contours,
            },
            {
                "ds": ds_bg,
                "mesh": mesh,
                "extra": extra,
                "cmap_spec": cmap,
                "title": f"基础模型 Vp  {mesh_abs.name}",
                "cb_label": "km/s",
                "draw_contours": draw_c,
                "contour_specs": contours,
            },
        ],
        dws_xyz=dws_xyz,
        reset_home=reset_home,
    )
    if profile_plot is not None:
        if init_mode == "layers1d" and profiles:
            profile_plot.set_profiles(
                profiles,
                z_max=z_max_1d,
                title=(
                    f"分段随机 1D  全部 {len(profiles)} 次  "
                    f"seed={int(seed)}…{int(seed) + len(profiles) - 1}{note_1d}"
                    "  红粗=第1次"
                ),
                highlight=0,
            )
        elif init_mode == "vinlayers" and profiles:
            from .mc_vin_layers import DATUM_ROLE_LABEL, DATUM_YLABEL

            lab = DATUM_ROLE_LABEL.get(datum_role, datum_role)
            profile_plot.set_profiles(
                profiles,
                z_max=z_max_1d,
                title=(
                    f"v.in 1D  从{lab}起  全部 {len(profiles)} 次  "
                    f"seed={int(seed)}…{int(seed) + len(profiles) - 1}{note_1d}"
                    "  红粗=第1次  中点柱"
                ),
                highlight=0,
                ylabel=DATUM_YLABEL.get(datum_role),
            )
        else:
            msg = (
                "选层后这里叠绘从最浅勾选层顶起的 1D"
                if init_mode == "vinlayers"
                else "模型方式选「smesh」后，这里叠绘全部实现与 Moho"
            )
            profile_plot.show_empty(msg)
    widget.set_interaction_hint(
        "左上：ΔV · 左中：第1次实现 · 左下：基础 · 右：红粗线=第1次 1D · 右键写入表单"
        if init_mode in {"layers1d", "vinlayers"}
        else "左上：ΔV · 左中：第1次实现 · 左下：基础模型 · 右：全部 1D · 右键写入表单"
    )


def _checkerboard_run_complete(p: Path) -> bool:
    return (p / "outputs" / "true_anomaly_pct.smesh").is_file() and (
        p / "outputs" / "recovered_anomaly_pct.smesh"
    ).is_file()


def list_checkerboard_runs(work: Path, *, limit: int = 80) -> list[Path]:
    """工区 ``runs/checkerboard_*`` 中完整结果包，新→旧。"""
    root = Path(work) / "runs"
    if not root.is_dir():
        return []
    cands: list[tuple[float, Path]] = []
    try:
        kids = list(root.iterdir())
    except OSError:
        return []
    for p in kids:
        if not p.is_dir() or not p.name.startswith("checkerboard_"):
            continue
        if not _checkerboard_run_complete(p):
            continue
        try:
            mt = max(
                (p / "outputs" / "true_anomaly_pct.smesh").stat().st_mtime,
                (p / "outputs" / "recovered_anomaly_pct.smesh").stat().st_mtime,
            )
        except OSError:
            continue
        cands.append((mt, p))
    cands.sort(key=lambda t: t[0], reverse=True)
    return [p for _mt, p in cands[: max(1, int(limit))]]


def checkerboard_run_label(run: Path) -> str:
    """下拉显示：目录名 + 清单里的 A/h/v。"""
    bits = [Path(run).name]
    man = Path(run) / "manifest.json"
    if man.is_file():
        try:
            info = json.loads(man.read_text(encoding="utf-8"))
            amp = info.get("amp_percent")
            h_len = info.get("h_len")
            v_len = info.get("v_len")
            if amp is not None and h_len is not None and v_len is not None:
                bits.append(
                    f"A={float(amp):g}%  h={float(h_len):g}  v={float(v_len):g}"
                )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return "  ·  ".join(bits)


def find_latest_checkerboard_run(work: Path) -> Path | None:
    """``work/runs/checkerboard_*`` 中最新一份含真/恢复异常场的包。"""
    hits = list_checkerboard_runs(work, limit=1)
    return hits[0] if hits else None


def resolve_checkerboard_run_dir(
    work: Path, run_dir: Path | str | None = None
) -> Path:
    if run_dir is not None and str(run_dir).strip():
        p = Path(str(run_dir))
        if not p.is_absolute():
            p = Path(work) / p
        true_p = p / "outputs" / "true_anomaly_pct.smesh"
        rec_p = p / "outputs" / "recovered_anomaly_pct.smesh"
        if true_p.is_file() and rec_p.is_file():
            return p
        raise FileNotFoundError(
            f"棋盘格结果不完整（缺 true/recovered_anomaly_pct.smesh）:\n{p}"
        )
    hit = find_latest_checkerboard_run(work)
    if hit is None:
        raise FileNotFoundError(
            "没有完整的棋盘格结果。请先「运行棋盘格测试」。"
        )
    return hit


def checkerboard_recovery_stats(
    true_a,
    rec_a,
    *,
    cover=None,
    frac_cut: float = 0.25,
) -> dict:
    """按网格统计棋盘恢复程度。

    排除 |真异常| 过小的零线附近格（比值会爆）。``cover`` 为与网格同形的
    权重（如 DWS），>0 才计入。
    """
    import numpy as np

    t = np.asarray(true_a, dtype=float)
    r = np.asarray(rec_a, dtype=float)
    if t.shape != r.shape:
        raise ValueError(f"真/恢复异常网格不一致: {t.shape} vs {r.shape}")
    ok = np.isfinite(t) & np.isfinite(r)
    used_dws = False
    if cover is not None:
        c = np.asarray(cover, dtype=float)
        if c.shape == t.shape:
            ok = ok & np.isfinite(c) & (c > 0)
            used_dws = True
    n_ok = int(np.count_nonzero(ok))
    peak = float(np.nanmax(np.abs(t[ok]))) if n_ok else 0.0
    if not np.isfinite(peak):
        peak = 0.0
    cut = max(peak * float(frac_cut), 1e-9)
    use = ok & (np.abs(t) >= cut)
    n_use = int(np.count_nonzero(use))
    if n_use < 8:
        use = ok
        n_use = n_ok
        cut = 0.0
    tv = t[use]
    rv = r[use]
    resid = rv - tv
    true_rms = float(np.sqrt(np.mean(tv * tv))) if n_use else 0.0
    rec_rms = float(np.sqrt(np.mean(rv * rv))) if n_use else 0.0
    resid_rms = float(np.sqrt(np.mean(resid * resid))) if n_use else 0.0
    amp_pct = (rec_rms / true_rms * 100.0) if true_rms > 1e-12 else 0.0
    ratio = rv / tv if n_use else np.asarray([], dtype=float)
    if n_use:
        ratio = np.where(np.isfinite(ratio), ratio, np.nan)
    mean_pct = float(np.nanmean(ratio) * 100.0) if n_use else 0.0
    med_pct = float(np.nanmedian(ratio) * 100.0) if n_use else 0.0
    if n_use >= 2 and float(np.std(tv)) > 1e-12 and float(np.std(rv)) > 1e-12:
        corr = float(np.corrcoef(tv, rv)[0, 1])
    else:
        corr = float("nan")
    return {
        "n_ok": n_ok,
        "n_use": n_use,
        "cut_abs_pct": cut,
        "used_dws": used_dws,
        "true_rms": true_rms,
        "rec_rms": rec_rms,
        "resid_rms": resid_rms,
        "amp_recovery_pct": amp_pct,
        "cell_mean_pct": mean_pct,
        "cell_median_pct": med_pct,
        "corr": corr,
        "resid_over_true_pct": (
            resid_rms / true_rms * 100.0 if true_rms > 1e-12 else 0.0
        ),
    }


def format_checkerboard_recovery_notes(stat: dict) -> tuple[str, str, str]:
    """返回 (真异常注记, 恢复注记, 残差注记)。"""
    n = int(stat.get("n_use") or 0)
    cut = float(stat.get("cut_abs_pct") or 0)
    tag = "DWS>0 且 " if stat.get("used_dws") else ""
    true_note = f"{tag}|真|≥{cut:.2g}% 的格  n={n}"
    corr = stat.get("corr")
    corr_s = f"{float(corr):.2f}" if corr is not None and np_isfinite(corr) else "—"
    rec_note = (
        f"恢复振幅 {float(stat.get('amp_recovery_pct') or 0):.0f}%\n"
        f"格中位 {float(stat.get('cell_median_pct') or 0):.0f}%  "
        f"均值 {float(stat.get('cell_mean_pct') or 0):.0f}%\n"
        f"相关 r={corr_s}  n={n}"
    )
    res_note = (
        f"残差 RMS {float(stat.get('resid_rms') or 0):.2g}%\n"
        f"相对真场 {float(stat.get('resid_over_true_pct') or 0):.0f}%"
    )
    return true_note, rec_note, res_note


def np_isfinite(x) -> bool:
    import math

    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def paint_checkerboard_result(
    widget,
    state: FormState,
    work: Path,
    *,
    run_dir: Path | str | None = None,
    reset_home: bool = True,
) -> Path:
    """结果窗：上真异常%、中恢复异常%、下残差（恢复−真）。"""
    import json

    import numpy as np

    from .dws_plot import dws_xyz_for_plot
    from .result_nav import resample_vgrid_field_to_xarray
    from .smesh_ops import _dws_xyz_to_vgrid, _load_mesh
    from .smesh_plot_core import load_smesh_plot_data

    run = resolve_checkerboard_run_dir(work, run_dir)
    true_p = run / "outputs" / "true_anomaly_pct.smesh"
    rec_p = run / "outputs" / "recovered_anomaly_pct.smesh"
    rec_smesh = run / "outputs" / "recovered.smesh"
    bg_p = run / "inputs" / "background.smesh"
    mesh = _load_mesh(true_p)
    true_a = np.asarray(mesh.vgrid, dtype=float)
    rec_a = np.asarray(_load_mesh(rec_p).vgrid, dtype=float)
    if rec_a.shape != true_a.shape:
        raise ValueError(
            f"真异常与恢复异常网格不一致: {true_a.shape} vs {rec_a.shape}"
        )
    resid = rec_a - true_a
    amp = h_len = v_len = None
    man = run / "manifest.json"
    if man.is_file():
        try:
            info = json.loads(man.read_text(encoding="utf-8"))
            amp = info.get("amp_percent")
            h_len = info.get("h_len")
            v_len = info.get("v_len")
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    scale = ""
    if amp is not None and h_len is not None and v_len is not None:
        scale = f"  A={float(amp):g}%  h={float(h_len):g}  v={float(v_len):g}"
    overlay = rec_smesh if rec_smesh.is_file() else (
        bg_p if bg_p.is_file() else true_p
    )
    refl_abs = None
    inp = run / "inputs"
    if inp.is_dir():
        staged = sorted(
            p
            for p in inp.iterdir()
            if p.is_file() and (".refl" in p.name.lower() or p.suffix.lower() == ".refl")
        )
        if staged:
            refl_abs = staged[0]
    if refl_abs is None:
        bg_form = _checkerboard_bg_path(state)
        refl_rel, _src, _comp = _resolve_checkerboard_refl(state, work, bg_form)
        if refl_rel:
            try:
                refl_abs = resolve_existing_file(refl_rel, work)
            except FileNotFoundError:
                cand = _abs(work, refl_rel)
                refl_abs = cand if cand.is_file() else None
    _m, _ds, extra = load_smesh_plot_data(
        overlay, str(refl_abs) if refl_abs else None, with_xarray=False
    )
    ds_true = resample_vgrid_field_to_xarray(mesh, true_a)
    ds_rec = resample_vgrid_field_to_xarray(mesh, rec_a)
    ds_res = resample_vgrid_field_to_xarray(mesh, resid)
    peak = float(np.nanmax(np.abs(np.concatenate([true_a.ravel(), rec_a.ravel()]))))
    if not np.isfinite(peak) or peak <= 0:
        peak = 3.0
    rpeak = float(np.nanmax(np.abs(resid))) if np.isfinite(resid).any() else peak
    if not np.isfinite(rpeak) or rpeak <= 0:
        rpeak = peak
    dws_xyz = dws_xyz_for_plot(
        state,
        work,
        overlay,
        run_dir=run,
        out_root=run / "outputs" / "out",
    )
    cover = _dws_xyz_to_vgrid(mesh, dws_xyz) if dws_xyz is not None else None
    stat = checkerboard_recovery_stats(true_a, rec_a, cover=cover)
    note_t, note_r, note_e = format_checkerboard_recovery_notes(stat)
    try:
        (run / "outputs" / "recovery_stats.json").write_text(
            json.dumps(stat, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass
    corr = stat.get("corr")
    corr_s = f"{float(corr):.2f}" if np_isfinite(corr) else "—"
    widget.set_save_dir(run)
    widget.set_model_source(overlay, state)
    widget.set_velocity_stack(
        [
            {
                "ds": ds_true,
                "mesh": mesh,
                "extra": extra,
                "cmap_spec": "seismic_r",
                "title": f"真异常 %（棋盘相对背景）{scale}",
                "vlim": (-peak, peak),
                "cb_label": "%",
                "draw_contours": False,
                "note": note_t,
            },
            {
                "ds": ds_rec,
                "mesh": mesh,
                "extra": extra,
                "cmap_spec": "seismic_r",
                "title": f"恢复异常 %（反演相对背景）  {run.name}",
                "vlim": (-peak, peak),
                "cb_label": "%",
                "draw_contours": False,
                "note": note_r,
            },
            {
                "ds": ds_res,
                "mesh": mesh,
                "extra": extra,
                "cmap_spec": "seismic_r",
                "title": f"残差 %（恢复 − 真）  峰值 ±{rpeak:.3g}",
                "vlim": (-rpeak, rpeak),
                "cb_label": "%",
                "draw_contours": False,
                "note": note_e,
            },
        ],
        dws_xyz=dws_xyz,
        reset_home=reset_home,
    )
    widget.set_interaction_hint(
        f"恢复振幅 {float(stat['amp_recovery_pct']):.0f}% · "
        f"格中位 {float(stat['cell_median_pct']):.0f}% · "
        f"r={corr_s} · 上真 / 中恢复 / 下残差 · 右键写入表单"
    )
    return run


def _filled_field(state: FormState, keys: tuple[str, ...]) -> tuple[str, str]:
    for key in keys:
        raw = state.get_str(key)
        if raw:
            return key, raw
    return keys[0], ""


def _call_preview(name: str, first: dict, rest: dict) -> str:
    pieces = [f"  {k}={v!r}" for k, v in first.items()]
    pieces.extend(f"  {k}={rest[k]!r}" for k in sorted(rest.keys()))
    return f"{name}(\n" + ",\n".join(pieces) + "\n)"


def _resolved_cmd_block(label: str, resolve) -> str:
    from .preview import format_resolved_cmdline

    try:
        cmd = resolve()
        if cmd:
            return f"\n\n# 解析后命令行（{label}）:\n{format_resolved_cmdline(cmd)}"
        return f"\n\n# 解析后命令行（{label}）: （参数不齐，未拼 argv）"
    except Exception as e:
        return f"\n\n# 解析后命令行（{label}）: （无法解析: {e}）"


def preview_checkerboard(state: FormState, work: Path, tomo=None) -> str:
    """列出棋盘格本页参数，以及实际会传给 tt_forward / tt_inverse 的项。"""
    inp = resolve_checkerboard_inputs(state, work)
    bg, bg_key = inp.bg, inp.bg_key
    geom, geom_key = inp.geom, inp.geom_key
    refl, refl_src = inp.refl, inp.refl_src
    from .refl_stride import estimate_refl_dx, format_dx

    # 所用界面按文件本身点距，不套 inv.refl_stride
    stride_disp = "1"
    dx_txt = ""
    if refl:
        try:
            dx0 = estimate_refl_dx(resolve_existing_file(refl, work))
        except FileNotFoundError:
            dx0 = None
        if dx0 is not None:
            dx_txt = f" dx={format_dx(dx0)}"
    amp = state.get_str("cb.amp") or "3"
    h_len = state.get_str("cb.h_len") or "10"
    v_len = state.get_str("cb.v_len") or "5"

    has_fwd_n = any(
        state.get_str(f"fwd.{k}") for k in ("xorder", "zorder", "clen", "nintp")
    )
    n_src = "tt_forward 页" if has_fwd_n else "tt_inverse 页"
    fwd_kw, inv_kw, refl_notes = _checkerboard_fwd_inv_kwargs(state, work)
    out_opts = dict(fwd_kw.pop("out_opts", {}) or {})
    out_opts["ttime"] = "inputs/syn_ttimes.dat"
    fwd_rest = dict(fwd_kw)
    if out_opts:
        fwd_rest["out_opts"] = out_opts
    n_note = (
        f"正演 -N / 弯射线容差取自 {n_src}"
        if all(k in fwd_kw for k in ("xorder", "zorder", "clen", "nintp", "tol1", "tol2"))
        else (
            f"正演 -N 未凑齐六项（xorder/zorder/clen/nintp + 两个 bend 容差），"
            f"运行时不传 -N（当前按 {n_src} 查找）"
        )
    )

    lines = [
        "棋盘格分辨率测试",
        f"  背景 smesh: {bg or '(未填)'}  ← {bg_key}",
        f"  geom: {geom or '(未填)'}  ← {geom_key}",
        f"  refl: {refl or '(未填)'}  ← {refl_src}  抽稀步长: {stride_disp}{dx_txt}",
        f"  运行 -F: {inp.staged_refl or '(不传)'}",
        f"  amp%={amp}  h={h_len} km  v={v_len} km",
        "  扰动: V ← V×(1 + 0.01×A×sin(2πx/h)×sin(2πz_abs/v))  （同 edit_smesh -Cc）",
        "",
        "步骤: 造真模型 → tt_forward → tt_inverse(自背景) → 输出百分异常场",
        f"work_dir: {work}",
        "输出目录: work_dir/runs/checkerboard_<时间戳>/",
        "  inputs/background.smesh",
        "  inputs/true_checkerboard.smesh",
        "  inputs/geom.dat",
        "  inputs/<反射面>  （有 -F 时从工区复制进来；命令行用 inputs/ 而非 outputs/）",
        "  inputs/syn_ttimes.dat  （tt_forward stdout，与 tt_inverse -G 同构；不是原生 -T）",
        "  outputs/out  （-O，覆盖表单 inv.out_root）",
        "  outputs/tt_inverse.log  （-L，覆盖表单 inv.log_file）",
        "  outputs/dws.dat  （-K；结束后归入 outputs/dws/）",
        "页签「棋盘预览图…」只画扰动；「棋盘结果图…」画最近一次真/恢复异常与残差。",
        "",
        f"# {n_note}",
        *[f"# {n}" for n in refl_notes],
        "# 棋盘只扰动速度；界面不扰动、反演时锁定。",
        "# 不用 tt_forward 页的 smesh / 输出路径 / vred / 钟差等；geom 仅作回退。",
        "# 不用 tt_inverse 页的 inv.mesh / inv.data；速度阻尼/光滑/迭代仍照用。",
        "",
        _call_preview(
            "tomo.tt_forward",
            {
                "smesh": "inputs/true_checkerboard.smesh",
                "geom": "inputs/geom.dat",
            },
            fwd_rest,
        ),
        "",
        _call_preview(
            "tomo.tt_inverse",
            {
                "mesh": "inputs/background.smesh",
                "data": "inputs/syn_ttimes.dat",
            },
            inv_kw,
        ),
    ]
    text = "\n".join(lines)
    if tomo is None:
        return text
    text += _resolved_cmd_block(
        "tt_forward",
        lambda: tomo.resolve_cmdline_tt_forward(
            smesh="inputs/true_checkerboard.smesh",
            geom="inputs/geom.dat",
            out_opts=out_opts,
            **fwd_kw,
        ),
    )
    text += _resolved_cmd_block(
        "tt_inverse",
        lambda: tomo.resolve_cmdline_tt_inverse(
            mesh="inputs/background.smesh",
            data="inputs/syn_ttimes.dat",
            **inv_kw,
        ),
    )
    return text


def _relabel_inv_files_to_inputs(inv_kw: dict) -> list[str]:
    """预览用：把文件路径改成运行包内 ``inputs/<文件名>``（与拷包后一致，不落地）。"""
    notes: list[str] = []
    from .refl_stride import peek_refl_stride, pop_refl_stride

    stride = peek_refl_stride(inv_kw)

    def one(container: dict, key: str, label: str) -> None:
        val = container.get(key)
        if not val:
            return
        old = str(val)
        new = f"inputs/{Path(old).name}"
        container[key] = new
        if old.replace("\\", "/") != new:
            notes.append(f"{label}: {old} → {new}")

    one(inv_kw, "refl_file", "-F")
    if stride > 1 and inv_kw.get("refl_file"):
        notes.append(f"-F 运行时按步长 {stride} 抽稀后写入 inputs/")
    pop_refl_stride(inv_kw)
    one(inv_kw, "seafloor_file", "-Y")
    one(inv_kw, "filter_bound_file", "-s")

    smooth = inv_kw.get("smooth_opts")
    if isinstance(smooth, dict):
        smooth = dict(smooth)
        one(smooth, "corr_v_fn", "-CV")
        one(smooth, "corr_d_fn", "-CD")
        inv_kw["smooth_opts"] = smooth

    damp = inv_kw.get("damp_opts")
    if isinstance(damp, dict):
        damp = dict(damp)
        one(damp, "damp_v_fn", "-DQ")
        inv_kw["damp_opts"] = damp

    g = inv_kw.get("gravity_opts")
    if isinstance(g, dict):
        g = dict(g)
        one(g, "grav_file", "-ZG")

        def _in(p) -> str:
            return f"inputs/{Path(str(p)).name}"

        if g.get("continent"):
            p, iconv = g["continent"]
            g["continent"] = (_in(p), iconv)
        ou = g.get("ocean_upper")
        if ou:
            up, lo, iconv = ou
            g["ocean_upper"] = (_in(up), _in(lo), iconv)
        ol = g.get("ocean_lower")
        if ol:
            up, iconv = ol
            g["ocean_lower"] = (_in(up), iconv)
        sed = g.get("sediment")
        if sed:
            up, lo, iconv = sed
            g["sediment"] = (_in(up), _in(lo), iconv)
        inv_kw["gravity_opts"] = g
    return notes


def _monte_carlo_preview_inv_kw(state: FormState) -> tuple[dict, list[str], str]:
    """第 1 次实现（reals/000）的 tt_inverse kwargs，与运行时一致。"""
    from .mc_init_models import resolve_mc_init_mode

    inv_kw = _copy_inv_kwargs(state)
    init_mode = resolve_mc_init_mode(state)
    if init_mode in {"layers1d", "vinlayers"}:
        # 运行时也会覆盖 tt_inverse 页的 -F，预览不要把它标成 inputs/ 再用
        inv_kw.pop("refl_file", None)
    notes = _relabel_inv_files_to_inputs(inv_kw)
    if init_mode == "layers1d":
        notes.append(
            "分段 1D：每次 -F 用该次 moho.refl（覆盖 tt_inverse 页原来的反射面）"
        )
        inv_kw["refl_file"] = "reals/000/moho.refl"
    elif init_mode == "vinlayers":
        notes.append(
            "扰动 v.in：每次 -F 用选定的莫霍界面（moho.refl），覆盖 tt_inverse 页"
        )
        inv_kw["refl_file"] = "reals/000/moho.refl"
    elif inv_kw.get("refl_file"):
        notes.append(f"-F 用 tt_inverse 页界面（已改成 {inv_kw['refl_file']}）")
    else:
        notes.append("tt_inverse 页未填 -F：各次反演不传反射面。")
    inv_kw["log_file"] = "reals/000/tt_inverse.log"
    inv_kw["out_root"] = "reals/000/out"
    inv_kw["dws_file"] = "reals/000/dws.dat"
    return inv_kw, notes, init_mode


def preview_monte_carlo(state: FormState, work: Path, tomo=None) -> str:
    """列出蒙特卡洛本页参数，以及第 1 次实现实际会传给 tt_inverse 的项。"""
    from .collectors import to_number
    from .mc_init_models import resolve_mc_init_mode

    mesh_key, mesh = _filled_field(state, ("mc.base_mesh", "inv.mesh"))
    data_key, data = _filled_field(state, ("mc.data", "inv.data"))
    n = int(to_number(state.get_str("mc.n_runs") or "10") or 10)
    seed0 = int(to_number(state.get_str("mc.seed") or "1") or 1)
    chi_max = mc_chi_max_from_state(state)
    init_mode = resolve_mc_init_mode(state)
    mode_label = (state.get_str("mc.init_mode") or "").strip() or init_mode
    inv_kw, notes, _mode = _monte_carlo_preview_inv_kw(state)

    if init_mode == "vinlayers":
        from .mc_vin_layers import resolve_mc_vin_path

        vin = resolve_mc_vin_path(state, work)
        iface = (
            f"  界面: 海底={state.get_str('mc.vin_seafloor') or '默认界面2'}"
            f"  基底={state.get_str('mc.vin_basement') or '—'}"
        )
        conrad = (state.get_str("mc.vin_conrad") or "").strip()
        if conrad:
            iface += f"  Conrad={conrad}"
        iface += f"  莫霍={state.get_str('mc.vin_moho') or '默认'}"
        init_lines = [
            f"  起始方式: {mode_label or 'v.in'}",
            f"  v.in: {vin or '(未找到，请填本页或 gen_smesh)'}",
            iface,
            f"  扰动层: {state.get_str('mc.vin_units') or '可用层全部'}"
            "  （速度区间同分段 1D；未选层保持 v.in）",
            "  每次 -F 用选定的莫霍界面。",
        ]
    else:
        init_lines = [
            f"  起始方式: {mode_label or 'smesh'}  （分段随机 1D）",
            f"  沉积厚度: {state.get_str('mc.sed_h') or '0.2 2.5'} km"
            f"  顶底速度: {state.get_str('mc.sed_v') or '1.7 3.6'} km/s",
            f"  上地壳厚度: {state.get_str('mc.uc_h') or '6 11'} km"
            f"  顶底速度: {state.get_str('mc.uc_v') or '4.0 6.5'} km/s",
            f"  下地壳厚度: {state.get_str('mc.lc_h') or '10 25'} km"
            f"  顶底速度: {state.get_str('mc.lc_v') or '6.6 7.5'} km/s",
            f"  地幔顶底速度: {state.get_str('mc.mantle_v') or '7.6 8.2'} km/s"
            "  （厚度接到网格底）",
            "  上/下地壳交界面速度连续；水层/空气层不改。",
            "  每次 Moho = 沉积+上地壳+下地壳，沿 topo 写成 reals/iii/moho.refl。",
        ]

    noise_on = state.get_bool("mc.tt_noise")
    if noise_on:
        noise_line = (
            f"  走时噪声: 开  σ={state.get_str('mc.noise_sigma') or '0.01'}"
            + (
                "  （相对该行 u）"
                if state.get_bool("mc.noise_relative_u")
                else "  秒"
            )
        )
    else:
        noise_line = "  走时噪声: 关"

    lines = [
        "蒙特卡洛不确定性",
        f"  基础 mesh: {mesh or '(未填)'}  ← {mesh_key}",
        f"  走时 data: {data or '(未填)'}  ← {data_key}",
        f"  实现次数 N={n}  种子基数 seed={seed0}  （第 i 次 seed+i）",
        f"  卡方筛选: 只保留 pred χ² < {chi_max:g} 的实现后再算均值/σ（默认 1.8）",
        *init_lines,
        noise_line,
        "",
        "步骤: 拷贝 base → 第 i 次造 init.smesh（+可选 moho.refl / 走时噪声）"
        " → tt_inverse → 按 pred χ² 筛选 → 按各次 DWS 叠均值/σ/界面",
        f"work_dir: {work}",
        "输出目录: work_dir/runs/montecarlo_<时间戳>/",
        "  inputs/base.smesh",
        "  inputs/base_ttimes.dat",
        "  inputs/<阻尼、光滑、原 -F 等>  （从 tt_inverse 页拷入）",
        "  reals/000/init.smesh",
        "  reals/000/data.dat",
        "  reals/000/moho.refl  （分段 1D 时；-F）",
        "  reals/000/tt_inverse.log  （-L，覆盖表单 inv.log_file）",
        "  reals/000/out  （-O，覆盖表单 inv.out_root）",
        "  reals/000/dws.dat  （-K，覆盖表单 inv.dws_file）",
        "  outputs/mean_velocity.smesh",
        "  outputs/std_velocity.smesh",
        "  outputs/uncertainty_pct.smesh",
        "  outputs/dws.dat  （保留实现的平均覆盖，供结果图遮罩）",
        "  outputs/mean_profile.txt",
        "  outputs/mean_moho.refl / moho_mean_std.txt  （有界面时）",
        "页签「蒙特卡洛预览图…」只画第 1 次初始模型；「蒙特卡洛结果图…」画均值 Vp、误差 σ 与界面 ±σ。",
        "",
        f"# 下列为第 1 次实现（i=000, seed={seed0}）的 tt_inverse；"
        "其余 N-1 次只把路径里的 000 换成 iii。",
        "# 不用 tt_inverse 页的 inv.mesh / inv.data / 输出路径；速度阻尼/光滑/迭代仍照用。",
        *[f"# {note}" for note in notes],
        "",
        _call_preview(
            "tomo.tt_inverse",
            {
                "mesh": "reals/000/init.smesh",
                "data": "reals/000/data.dat",
            },
            inv_kw,
        ),
    ]
    text = "\n".join(lines)
    if tomo is None:
        return text
    text += _resolved_cmd_block(
        "tt_inverse  第 1 次",
        lambda: tomo.resolve_cmdline_tt_inverse(
            mesh="reals/000/init.smesh",
            data="reals/000/data.dat",
            **inv_kw,
        ),
    )
    return text
