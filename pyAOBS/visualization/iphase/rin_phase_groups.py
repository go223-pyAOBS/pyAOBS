"""
RAYINVR r.in 相位组（ray/nrbnd/rbnd/ncbnd/cbnd/nray/ivray）读写工具。

目标：
1) 将 r.in 中数组参数解析为可编辑的 PhaseGroup 列表；
2) 将 PhaseGroup 编译回数组；
3) 将数组安全写回 r.in 的 TRAPAR/INVPAR 段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Literal


@dataclass
class ReflectionEvent:
    boundary: int
    direction: Literal["down_reflect_up", "up_reflect_down"]


@dataclass
class ConversionEvent:
    encounter_idx: int


@dataclass
class PhaseGroup:
    enabled: bool = True
    name: str = ""
    ray_code: float = 1.1
    nray: int = 10
    ivray: int = 1
    source_wave: Literal["P", "S"] = "P"
    reflections: list[ReflectionEvent] = field(default_factory=list)
    conversions: list[ConversionEvent] = field(default_factory=list)
    comment: str = ""


@dataclass
class PhaseGroupParseResult:
    groups: list[PhaseGroup]
    diagnostics: list[str]


@dataclass
class RinPhaseArrays:
    ray: list[float]
    nrbnd: list[int]
    ncbnd: list[int]
    nray: list[int]
    ivray: list[int]
    rbnd: list[int]
    cbnd: list[int]
    diagnostics: list[str]


@dataclass
class RinMiscParams:
    """r.in 中常用的全局参数子集（用于 GUI 额外面板编辑）。"""

    xshot: list[float] | None = None
    zshot: list[float] | None = None
    imodf: int | None = None
    i2pt: int | None = None
    isrch: int | None = None
    aamin: float | None = None
    aamax: float | None = None
    ishot: list[int] | None = None
    n2pt: int | None = None
    space: list[float] | None = None
    x2pt: float | None = None
    vred: float | None = None
    diagnostics: list[str] = field(default_factory=list)


TRAPAR_KEYS = ("ray", "nrbnd", "rbnd", "ncbnd", "cbnd", "nray")
INVPAR_KEYS = ("ivray",)
TRAPAR_MISC_KEYS = (
    # 相位组数组 + 常用全局字段（用于安全替换 xshot/zshot）
    "ray",
    "nrbnd",
    "rbnd",
    "ncbnd",
    "cbnd",
    "nray",
    "xshot",
    "zshot",
    "imodf",
    "i2pt",
    "isrch",
    "aamin",
    "aamax",
    "ishot",
    "n2pt",
    "space",
    "x2pt",
    "tfile",
    "vfile",
)
PLTPAR_KEYS = ("vred",)
NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")
ASSIGN_RE = re.compile(r"(?is)\b([a-z_]\w*)\s*=")


def _extract_namelist_block(text: str, name: str) -> str:
    m = re.search(rf"(?is)&{name}\b(.*?)&end", text)
    if not m:
        raise ValueError(f"未找到 &{name} ... &end 段")
    return m.group(1)


def _build_assignment_ranges(block: str, known_keys: tuple[str, ...]) -> dict[str, str]:
    """
    线性扫描 namelist 赋值区间，避免复杂正则在长文本上回溯卡顿。
    """
    allowed = {k.lower() for k in known_keys}
    found: list[tuple[str, int, int]] = []  # (key, rhs_start, rhs_end)

    matches = list(ASSIGN_RE.finditer(block))
    for i, m in enumerate(matches):
        k = m.group(1).lower()
        if k not in allowed:
            continue
        rhs_start = m.end()
        # 截止到“下一个任意 key=”，而不是下一个关心的 key，
        # 防止 ray= 吞并 xshot/zshot/isrch 等大量字段。
        rhs_end = matches[i + 1].start() if (i + 1) < len(matches) else len(block)
        found.append((k, rhs_start, rhs_end))

    out: dict[str, str] = {}
    # 如果同键出现多次，保留最后一次（通常不会出现）
    for k, s, e in found:
        out[k] = block[s:e]
    return out


def _extract_assignment_rhs(block: str, key: str, known_keys: tuple[str, ...]) -> str:
    d = _build_assignment_ranges(block, known_keys)
    k = key.lower()
    if k not in d:
        raise ValueError(f"未找到参数 {key}=")
    return d[k]


def _extract_assignment_rhs_optional(block: str, key: str, known_keys: tuple[str, ...]) -> str | None:
    d = _build_assignment_ranges(block, known_keys)
    return d.get(key.lower())


def _get_required_rhs(assignments: dict[str, str], key: str) -> str:
    rhs = assignments.get(key.lower())
    if rhs is None:
        raise ValueError(f"未找到参数 {key}=")
    return rhs


def _parse_float_array(rhs: str) -> list[float]:
    return [float(x.replace("d", "e").replace("D", "E")) for x in NUMBER_RE.findall(rhs)]


def _parse_int_array(rhs: str) -> list[int]:
    return [int(round(v)) for v in _parse_float_array(rhs)]


def _expand_or_validate_main_array(values: list[int], n: int, name: str) -> list[int]:
    """
    对 nrbnd/ncbnd/nray/ivray 等主数组做宽松处理：
    - 仅给 1 个值时，按 RAYINVR 数组默认规则扩展到 n；
    - 给值多于 n 时，仅取前 n 个（附带诊断由上层决定）；
    - 给值少于 n 且不为 1 时，报错。
    """
    if len(values) == n:
        return values
    if len(values) == 1 and n > 1:
        return [values[0]] * n
    if len(values) > n:
        return values[:n]
    raise ValueError(f"{name} 长度不足：期望 {n}，实际 {len(values)}")


def _decode_rbnd_token(v: int) -> ReflectionEvent:
    if v > 0:
        return ReflectionEvent(boundary=v, direction="down_reflect_up")
    if v < 0:
        return ReflectionEvent(boundary=abs(v), direction="up_reflect_down")
    raise ValueError("rbnd 中不允许 0")


def _encode_rbnd_event(evt: ReflectionEvent) -> int:
    return evt.boundary if evt.direction == "down_reflect_up" else -evt.boundary


def parse_rin_phase_arrays_from_text(
    rin_text: str,
    *,
    parse_path_arrays: bool = True,
) -> RinPhaseArrays:
    """
    严格按 r.in 的 namelist 格式读取并提取目标数组参数：
    TRAPAR: ray/nrbnd/ncbnd/nray/rbnd/cbnd
    INVPAR: ivray
    """
    trapar = _extract_namelist_block(rin_text, "trapar")
    invpar = _extract_namelist_block(rin_text, "invpar")
    ta = _build_assignment_ranges(trapar, TRAPAR_KEYS)
    ia = _build_assignment_ranges(invpar, INVPAR_KEYS)

    ray = _parse_float_array(_get_required_rhs(ta, "ray"))
    nrbnd_raw = _parse_int_array(_get_required_rhs(ta, "nrbnd"))
    ncbnd_raw = _parse_int_array(_get_required_rhs(ta, "ncbnd"))
    nray_raw = _parse_int_array(_get_required_rhs(ta, "nray"))
    ivray_raw = _parse_int_array(_get_required_rhs(ia, "ivray"))

    # 始终从文件读取 rbnd/cbnd 池（若存在），否则保存时各组 conversions/reflections 为空会错误重建数组并覆盖其它组
    rbnd = _parse_int_array(ta["rbnd"]) if "rbnd" in ta else []
    cbnd = _parse_int_array(ta["cbnd"]) if "cbnd" in ta else []

    n = len(ray)
    nrbnd = _expand_or_validate_main_array(nrbnd_raw, n, "nrbnd")
    ncbnd = _expand_or_validate_main_array(ncbnd_raw, n, "ncbnd")
    nray = _expand_or_validate_main_array(nray_raw, n, "nray")
    ivray = _expand_or_validate_main_array(ivray_raw, n, "ivray")

    diagnostics: list[str] = []
    need_rbnd = sum(nrbnd)
    need_cbnd = sum(ncbnd)
    if need_rbnd > 0 and len(rbnd) < need_rbnd:
        raise ValueError(f"rbnd 数量不足：需求 {need_rbnd}，实际 {len(rbnd)}")
    if need_cbnd > 0 and len(cbnd) < need_cbnd:
        raise ValueError(f"cbnd 数量不足：需求 {need_cbnd}，实际 {len(cbnd)}")
    if parse_path_arrays:
        if len(rbnd) > need_rbnd:
            diagnostics.append(f"rbnd 提供了 {len(rbnd)} 个值，仅使用前 {need_rbnd} 个（由 nrbnd 决定）")
        if len(cbnd) > need_cbnd:
            diagnostics.append(f"cbnd 提供了 {len(cbnd)} 个值，仅使用前 {need_cbnd} 个（由 ncbnd 决定）")
    else:
        diagnostics.append("安全模式：已加载 rbnd/cbnd 池；尾部长度提示已省略")

    return RinPhaseArrays(
        ray=ray,
        nrbnd=nrbnd,
        ncbnd=ncbnd,
        nray=nray,
        ivray=ivray,
        rbnd=rbnd,
        cbnd=cbnd,
        diagnostics=diagnostics,
    )


def parse_rin_phase_arrays_from_file(
    rin_path: str | Path,
    *,
    parse_path_arrays: bool = True,
) -> RinPhaseArrays:
    text = Path(rin_path).read_text(encoding="utf-8", errors="ignore")
    return parse_rin_phase_arrays_from_text(text, parse_path_arrays=parse_path_arrays)


def parse_rin_misc_params_from_text(rin_text: str) -> RinMiscParams:
    diagnostics: list[str] = []
    # &trapar: 常用全局参数
    xshot: list[float] | None = None
    zshot: list[float] | None = None
    imodf: int | None = None
    i2pt: int | None = None
    isrch: int | None = None
    aamin: float | None = None
    aamax: float | None = None
    ishot: list[int] | None = None
    n2pt: int | None = None
    space: list[float] | None = None
    x2pt: float | None = None
    try:
        trapar = _extract_namelist_block(rin_text, "trapar")
        rhs_x = _extract_assignment_rhs_optional(trapar, "xshot", TRAPAR_MISC_KEYS)
        rhs_z = _extract_assignment_rhs_optional(trapar, "zshot", TRAPAR_MISC_KEYS)
        xshot = _parse_float_array(rhs_x) if rhs_x is not None else None
        zshot = _parse_float_array(rhs_z) if rhs_z is not None else None
        rhs_imodf = _extract_assignment_rhs_optional(trapar, "imodf", TRAPAR_MISC_KEYS)
        rhs_i2pt = _extract_assignment_rhs_optional(trapar, "i2pt", TRAPAR_MISC_KEYS)
        rhs_isrch = _extract_assignment_rhs_optional(trapar, "isrch", TRAPAR_MISC_KEYS)
        rhs_aamin = _extract_assignment_rhs_optional(trapar, "aamin", TRAPAR_MISC_KEYS)
        rhs_aamax = _extract_assignment_rhs_optional(trapar, "aamax", TRAPAR_MISC_KEYS)
        rhs_ishot = _extract_assignment_rhs_optional(trapar, "ishot", TRAPAR_MISC_KEYS)
        rhs_n2pt = _extract_assignment_rhs_optional(trapar, "n2pt", TRAPAR_MISC_KEYS)
        rhs_space = _extract_assignment_rhs_optional(trapar, "space", TRAPAR_MISC_KEYS)
        rhs_x2pt = _extract_assignment_rhs_optional(trapar, "x2pt", TRAPAR_MISC_KEYS)
        imodf = int(_parse_int_array(rhs_imodf)[0]) if rhs_imodf is not None and _parse_int_array(rhs_imodf) else None
        i2pt = int(_parse_int_array(rhs_i2pt)[0]) if rhs_i2pt is not None and _parse_int_array(rhs_i2pt) else None
        isrch = int(_parse_int_array(rhs_isrch)[0]) if rhs_isrch is not None and _parse_int_array(rhs_isrch) else None
        aamin = float(_parse_float_array(rhs_aamin)[0]) if rhs_aamin is not None and _parse_float_array(rhs_aamin) else None
        aamax = float(_parse_float_array(rhs_aamax)[0]) if rhs_aamax is not None and _parse_float_array(rhs_aamax) else None
        ishot = _parse_int_array(rhs_ishot) if rhs_ishot is not None else None
        n2pt = int(_parse_int_array(rhs_n2pt)[0]) if rhs_n2pt is not None and _parse_int_array(rhs_n2pt) else None
        space = _parse_float_array(rhs_space) if rhs_space is not None else None
        x2pt = float(_parse_float_array(rhs_x2pt)[0]) if rhs_x2pt is not None and _parse_float_array(rhs_x2pt) else None
    except Exception as e:
        diagnostics.append(f"读取 &trapar xshot/zshot 失败: {e}")

    # &pltpar: vred（标量；pltpar 段可能不存在）
    vred: float | None = None
    try:
        pltpar = _extract_namelist_block(rin_text, "pltpar")
        rhs_v = _extract_assignment_rhs_optional(pltpar, "vred", PLTPAR_KEYS)
        if rhs_v is not None:
            vals = _parse_float_array(rhs_v)
            if vals:
                vred = float(vals[0])
    except Exception:
        pass

    return RinMiscParams(
        xshot=xshot,
        zshot=zshot,
        imodf=imodf,
        i2pt=i2pt,
        isrch=isrch,
        aamin=aamin,
        aamax=aamax,
        ishot=ishot,
        n2pt=n2pt,
        space=space,
        x2pt=x2pt,
        vred=vred,
        diagnostics=diagnostics,
    )


def parse_rin_misc_params_from_file(rin_path: str | Path) -> RinMiscParams:
    text = Path(rin_path).read_text(encoding="utf-8", errors="ignore")
    return parse_rin_misc_params_from_text(text)


def parse_phase_groups_from_arrays(
    arr: RinPhaseArrays,
    *,
    max_layer: int | None = None,
    parse_path_arrays: bool = True,
) -> PhaseGroupParseResult:
    """基于已解析数组构建 PhaseGroup（避免重复读取/扫描 r.in）。"""
    ray = arr.ray
    nrbnd = arr.nrbnd
    ncbnd = arr.ncbnd
    nray = arr.nray
    ivray = arr.ivray
    rbnd = arr.rbnd
    cbnd = arr.cbnd
    diagnostics: list[str] = list(arr.diagnostics)

    groups: list[PhaseGroup] = []
    ir = 0
    ic = 0
    signatures: dict[float, set[tuple[tuple[int, ...], tuple[int, ...]]]] = {}
    n = len(ray)

    for i in range(n):
        # 始终按 nrbnd/ncbnd 从池中切片，保证与 r.in 一致（保存时不会抹掉未编辑组的 cbnd/rbnd）
        local_rbnd = rbnd[ir : ir + nrbnd[i]]
        local_cbnd = cbnd[ic : ic + ncbnd[i]]
        ir += nrbnd[i]
        ic += ncbnd[i]

        reflections = [_decode_rbnd_token(v) for v in local_rbnd]
        conversions = [ConversionEvent(encounter_idx=v) for v in local_cbnd]

        g = PhaseGroup(
            enabled=True,
            name=f"G{i+1}",
            ray_code=float(ray[i]),
            nray=int(nray[i]),
            ivray=int(ivray[i]),
            source_wave="S" if any(c.encounter_idx == 0 for c in conversions) else "P",
            reflections=reflections,
            conversions=conversions,
        )
        groups.append(g)

        if max_layer is not None:
            lnum = int(g.ray_code)
            if lnum < 1 or lnum > max_layer:
                diagnostics.append(f"组{i+1}: ray={g.ray_code} 层号超界 [1,{max_layer}]")
            for r in reflections:
                if r.boundary < 1 or r.boundary > max_layer:
                    diagnostics.append(f"组{i+1}: rbnd 边界 {r.boundary} 超界 [1,{max_layer}]")

        sig = (
            tuple(_encode_rbnd_event(r) for r in reflections),
            tuple(c.encounter_idx for c in conversions),
        )
        signatures.setdefault(g.ray_code, set()).add(sig)

    for ray_code, sigset in signatures.items():
        if len(sigset) > 1:
            diagnostics.append(f"ray code {ray_code} 存在多路径，建议 isrch=1")

    return PhaseGroupParseResult(groups=groups, diagnostics=diagnostics)


def parse_phase_groups_from_rin_text(
    rin_text: str,
    *,
    max_layer: int | None = None,
    parse_path_arrays: bool = True,
) -> PhaseGroupParseResult:
    """
    从 r.in 文本中解析 PhaseGroup 列表。
    """
    arr = parse_rin_phase_arrays_from_text(rin_text, parse_path_arrays=parse_path_arrays)
    return parse_phase_groups_from_arrays(
        arr,
        max_layer=max_layer,
        parse_path_arrays=parse_path_arrays,
    )


def parse_phase_groups_from_rin_file(
    rin_path: str | Path,
    *,
    max_layer: int | None = None,
    parse_path_arrays: bool = True,
) -> PhaseGroupParseResult:
    arr = parse_rin_phase_arrays_from_file(rin_path, parse_path_arrays=parse_path_arrays)
    return parse_phase_groups_from_arrays(
        arr,
        max_layer=max_layer,
        parse_path_arrays=parse_path_arrays,
    )


def compile_phase_groups_to_arrays(
    groups: list[PhaseGroup],
    *,
    max_layer: int | None = None,
) -> dict[str, list[int | float] | list[str]]:
    """
    将 PhaseGroup 编译为 r.in 数组。

    注意：必须输出与 ivray 等长的全部相位组。禁用组仍写出 ray/rbnd/cbnd 结构，仅将 nray 置 0，
    避免以前按 enabled 过滤导致数组变短、误覆盖其它组参数。
    """
    diagnostics: list[str] = []

    ray: list[float] = []
    nrbnd: list[int] = []
    rbnd: list[int] = []
    ncbnd: list[int] = []
    cbnd: list[int] = []
    nray: list[int] = []
    ivray: list[int] = []
    signatures: dict[float, set[tuple[tuple[int, ...], tuple[int, ...]]]] = {}

    for i, g in enumerate(groups):
        nray_val = int(g.nray) if g.enabled else 0
        if g.nray < 0:
            raise ValueError(f"组{i+1}: nray 不能为负数")
        if g.ivray == 0:
            raise ValueError(f"组{i+1}: ivray 不能为 0")

        lnum = int(g.ray_code)
        frac = round(g.ray_code - lnum, 1)
        if frac not in (0.0, 0.1, 0.2, 0.3):
            raise ValueError(f"组{i+1}: ray_code={g.ray_code} 非法（应为 L.0/L.1/L.2/L.3）")
        if max_layer is not None and (lnum < 1 or lnum > max_layer):
            raise ValueError(f"组{i+1}: ray 层号超界 [1,{max_layer}]")

        local_rbnd: list[int] = []
        for evt in g.reflections:
            if max_layer is not None and (evt.boundary < 1 or evt.boundary > max_layer):
                raise ValueError(f"组{i+1}: rbnd 边界超界 [1,{max_layer}]")
            local_rbnd.append(_encode_rbnd_event(evt))

        local_cbnd: list[int] = []
        for evt in g.conversions:
            if evt.encounter_idx < 0:
                raise ValueError(f"组{i+1}: cbnd encounter_idx 不能为负数")
            local_cbnd.append(int(evt.encounter_idx))
        if g.source_wave == "S" and 0 not in local_cbnd:
            local_cbnd.insert(0, 0)

        ray.append(float(g.ray_code))
        nrbnd.append(len(local_rbnd))
        rbnd.extend(local_rbnd)
        ncbnd.append(len(local_cbnd))
        cbnd.extend(local_cbnd)
        nray.append(nray_val)
        ivray.append(int(g.ivray))

        sig = (tuple(local_rbnd), tuple(local_cbnd))
        signatures.setdefault(g.ray_code, set()).add(sig)

    for ray_code, sigset in signatures.items():
        if len(sigset) > 1:
            diagnostics.append(f"ray code {ray_code} 存在多路径，建议 isrch=1")

    return {
        "ray": ray,
        "nrbnd": nrbnd,
        "rbnd": rbnd,
        "ncbnd": ncbnd,
        "cbnd": cbnd,
        "nray": nray,
        "ivray": ivray,
        "diagnostics": diagnostics,
    }


def describe_ray_wave_phrase(ray_code: float) -> str:
    """
    ray 码对应的短名，用于「震相…，第L层…波，…」模板（RAYINVR：L.0–L.3）。
    """
    lnum = int(ray_code)
    frac = round(ray_code - lnum, 1)
    if frac == 0.1:
        return f"第{lnum}层内折射波"
    if frac == 0.2:
        return f"第{lnum}层底反射波"
    if frac == 0.3:
        return f"第{lnum}层底首波"
    if frac == 0.0:
        return f"第{lnum}层手动角射线"
    return f"未识别ray码{ray_code}"


def describe_ray_code_zh(ray_code: float) -> str:
    """将 ray 码译为中文，附 L.x 标注（略长于模板用词）。"""
    wp = describe_ray_wave_phrase(ray_code)
    if wp.startswith("未识别"):
        return wp
    lnum = int(ray_code)
    frac = round(ray_code - lnum, 1)
    label = {0.1: "L.1", 0.2: "L.2", 0.3: "L.3", 0.0: "L.0"}.get(frac, "")
    if not label:
        return wp
    return f"{wp}（ray={ray_code:.1f}，{label}）"


def describe_rbnd_token_zh(v: int) -> str:
    """单个 rbnd 整数在说明中的中文。"""
    if v > 0:
        return f"下行至第{v}层底界后向上反射"
    if v < 0:
        return f"上行至第{abs(v)}层顶界后向下反射"
    return "rbnd=0（非法）"


def describe_phase_group_short_sentence(g: PhaseGroup) -> str:
    """
    短句模板（逗号分隔）。

    cbnd=1 且震源为 P：仅当震相归类为 PPS/PSS 时，才视为「名义 P、即刻按 S 出射」的出射设置，
    不计入真实转换次数；PSP 等两次真实转换不应写 cbnd=1，也不做此项扣减。

    命名阶段若将「物理炮点(OBS)↔物理接收(气枪)」对调表述，则 PPS/PSS 在仅一次真实转换
    且含 cbnd=1 出射设置时，界面处箭头按显示惯例取与 Rayinvr 逐次翻转相反的方向（例如 S→P 写为 P→S）。
    """
    segs: list[str] = []
    if not g.enabled:
        segs.append("【停用】")
    segs.append(f"震相{g.ivray}")
    segs.append(describe_ray_wave_phrase(g.ray_code))

    nr = len(g.reflections)
    if nr:
        rb_txt = "、".join(str(_encode_rbnd_event(e)) for e in g.reflections)
        segs.append(f"反射{nr}次（rbnd {rb_txt}）")

    cb_vals = [c.encounter_idx for c in g.conversions]
    pos = [x for x in cb_vals if x > 0]
    phase_hint = _classify_conversion_phase_name(g, pos)
    source_setup_1 = bool(pos) and g.source_wave == "P" and pos[0] == 1 and phase_hint in ("PPS", "PSS")
    real_ifc = pos[1:] if source_setup_1 else pos

    if g.source_wave == "S":
        segs.append("震源S")
    else:
        segs.append("震源P")

    if not pos:
        segs.append("无界面转换")
        segs.append(_short_phase_ending(g, source_setup_1=source_setup_1, has_real_conv=False))
        return "，".join(segs)

    if len(pos) == 1 and pos[0] == 1 and g.source_wave == "P" and phase_hint is None:
        segs.append("仅cbnd=1：通常须再给出第二遇界（用于PPS/PSS出射设置）")
        segs.append(_short_phase_ending(g, source_setup_1=False, has_real_conv=False))
        return "，".join(segs)

    if source_setup_1 and not real_ifc:
        segs.append("无真实界面转换")
        segs.append("cbnd=1为炮点名义P即刻出S（出射设置）")
        segs.append(_short_phase_ending(g, source_setup_1=source_setup_1, has_real_conv=False))
        return "，".join(segs)

    if real_ifc:
        n_real = len(real_ifc)
        eff = _effective_wave_after_cbnd1_setup(g, source_setup_1)
        stages: list[str] = [eff]
        cur = eff
        for _ in real_ifc:
            cur = "S" if cur == "P" else "P"
            stages.append(cur)
        chain = "→".join(stages)

        segs.append(f"{n_real}次转换")
        if n_real == 1:
            a0, a1 = stages[0], stages[1]
            pname = phase_hint
            if _use_naming_axis_flip_for_single_step(source_setup_1, pname):
                a0, a1 = a1, a0
            segs.append(f"界面{real_ifc[0]}处{a0}→{a1}")
        else:
            segs.append(f"界面{_join_iface_numbers(real_ifc)}处{chain}")

    segs.append(_short_phase_ending(g, source_setup_1=source_setup_1, has_real_conv=True))
    return "，".join(segs)

def infer_phase_type_label(g: PhaseGroup) -> str:
    """
    推断相位组震相类型（用于列表快速浏览）。

    - PPP：纯震相（无 P/S 转换）
    - PPS / PSS / PSP：按用户口径的转换波命名
    - CONV：存在转换但无法归类
    - UNK：仅 cbnd=1 等不完整/不确定写法
    """
    cb_vals = [c.encounter_idx for c in g.conversions]
    pos = [x for x in cb_vals if x > 0]
    if not pos:
        return "PPP"
    name = _classify_conversion_phase_name(g, pos)
    if name:
        return name
    if len(pos) == 1 and pos[0] == 1 and g.source_wave == "P":
        return "UNK"
    return "CONV"


def _format_encounter_phrase(pos: list[int]) -> str:
    """cbnd 正整数：第几次遇到层界面；两项用「和」，多项用顿号。"""
    if not pos:
        return "无界面序号"
    if len(pos) == 1:
        return f"界面{pos[0]}"
    if len(pos) == 2:
        return f"界面{pos[0]}和{pos[1]}"
    return "界面" + "、".join(str(x) for x in pos)


def _effective_wave_after_cbnd1_setup(g: PhaseGroup, source_setup_1: bool) -> str:
    """经 cbnd=1（炮点即刻 P→S）后的有效出射波型；否则与震源类型一致。"""
    if source_setup_1:
        return "S"
    return "S" if g.source_wave == "S" else "P"


def _use_naming_axis_flip_for_single_step(source_setup_1: bool, phase_name: str | None) -> bool:
    """
    PPS/PSS 且带 cbnd=1 出射设置、仅一步真实转换时：按命名坐标（海面炮点/海底接收）
    在文案中反向书写界面处的 P/S（与 r.in 物理炮点起算的翻转对偶）。
    """
    return bool(source_setup_1 and phase_name in ("PPS", "PSS"))


def _join_iface_numbers(xs: list[int]) -> str:
    if len(xs) == 1:
        return str(xs[0])
    if len(xs) == 2:
        return f"{xs[0]}和{xs[1]}"
    return "、".join(str(x) for x in xs)


def _short_phase_ending(
    g: PhaseGroup,
    *,
    source_setup_1: bool,
    has_real_conv: bool,
) -> str:
    """
    模板末尾震相名（PPS / PSS / PSP），分类仍基于完整 cbnd 正整数序列。
    """
    cb = [c.encounter_idx for c in g.conversions]
    pos = [x for x in cb if x > 0]
    if not pos:
        return "纯震相"
    if not has_real_conv:
        return "纯震相"

    name = _classify_conversion_phase_name(g, pos)
    if name:
        return f"{name}波"
    return "转换波"


def _classify_conversion_phase_name(g: PhaseGroup, pos: list[int]) -> str | None:
    """
    依据用户口径对转换波粗分：PPS / PSS / PSP。

    说明：缺少完整几何（总遇界次数、炮点/接收点侧界面位置）时只能启发式判断。
    """
    if len(pos) == 1:
        if pos[0] == 1 and g.source_wave == "P":
            return None
        return "PPS" if pos[0] <= 3 else "PSS"

    if len(pos) == 2:
        a, b = pos[0], pos[1]
        if a == 1 and g.source_wave == "P":
            return "PPS" if b <= 3 else "PSS"
        return "PSP"

    return None


def describe_phase_group_line(group_index: int, g: PhaseGroup) -> str:
    """
    前半为 r.in 参数摘要，后半为短句模板释义。

    cbnd 正整数：第几次遇到层界面处发生 P/S 转换；0：震源 S 出射（RAYINVR）。
    """
    idx = group_index + 1
    parts: list[str] = [f"组{idx:03d}: ray={g.ray_code:.1f},"]

    nr = len(g.reflections)
    nc = len(g.conversions)
    if nr > 0:
        rb_txt = ",".join(str(_encode_rbnd_event(e)) for e in g.reflections)
        parts.append(f" nrbnd={nr}, rbnd={rb_txt},")
    if nc > 0:
        cb_txt = ",".join(str(c.encounter_idx) for c in g.conversions)
        parts.append(f" ncbnd={nc}, cbnd={cb_txt},")
    else:
        parts.append(" ncbnd=0,")

    parts.append(f" nray={g.nray}, ivray={g.ivray}")
    head = "".join(parts)

    short = describe_phase_group_short_sentence(g)
    return f"{head}  —  {short}"


def describe_all_phase_groups(groups: list[PhaseGroup]) -> list[str]:
    """返回每组一行说明（1-based 组号与列表索引一致）。"""
    return [describe_phase_group_line(i, g) for i, g in enumerate(groups)]


def _format_numeric_array(values: list[int | float], *, is_float: bool) -> str:
    if is_float:
        parts = [f"{float(v):.1f}" for v in values]
    else:
        parts = [str(int(v)) for v in values]
    return ", ".join(parts)


def _format_float_array(values: list[float], *, decimals: int = 3) -> str:
    fmt = f"{{:.{int(decimals)}f}}"
    return ", ".join(fmt.format(float(v)) for v in values)


def _format_int_array(values: list[int]) -> str:
    return ", ".join(str(int(v)) for v in values)


def _replace_or_insert_assignment(block: str, key: str, rhs: str, known_keys: tuple[str, ...]) -> str:
    other = [k for k in known_keys if k.lower() != key.lower()]
    lookahead = rf"(?=\b(?:{'|'.join(other)})\s*=|$)" if other else r"(?=$)"
    pat = re.compile(rf"(?is)(\b{key}\s*=\s*)(.*?){lookahead}")
    # 必须用函数替换：若 rhs 以数字开头，字符串 r"\1" + rhs 会变成 \11… 被误解析为第 11 捕获组
    def _repl(m: re.Match[str]) -> str:
        # 保留原 RHS 后的尾部分隔符/空白，避免反复保存导致空行累积
        old_rhs = m.group(2)
        m_tail = re.search(r"[\s,]*\Z", old_rhs)
        tail = m_tail.group(0) if m_tail else ""
        return m.group(1) + rhs + tail

    if pat.search(block):
        return pat.sub(_repl, block, count=1)
    return block + f"\n           {key}={rhs},\n           "


def _replace_or_insert_assignment_scalar(block: str, key: str, rhs: str, known_keys: tuple[str, ...]) -> str:
    """
    标量/短数组写回用：不强行追加额外缩进换行，避免每次保存产生“额外空行”。
    """
    other = [k for k in known_keys if k.lower() != key.lower()]
    lookahead = rf"(?=\b(?:{'|'.join(other)})\s*=|$)" if other else r"(?=$)"
    pat = re.compile(rf"(?is)(\b{key}\s*=\s*)(.*?){lookahead}")

    def _repl(m: re.Match[str]) -> str:
        # 仅替换 RHS，同时尽量保留原有尾部空白/逗号/换行，避免反复保存产生空行或排版漂移
        old_rhs = m.group(2)
        m_tail = re.search(r"[\s,]*\Z", old_rhs)
        tail = m_tail.group(0) if m_tail else ""

        out = m.group(1) + rhs
        # 若旧尾部不含逗号，则确保该赋值以逗号结束
        if "," not in tail and not out.rstrip().endswith(","):
            out = out.rstrip() + ","
        return out + tail

    if pat.search(block):
        return pat.sub(_repl, block, count=1)
    # 插入新字段：放到 block 末尾并保持一行
    base = block.rstrip()
    # 注意：不要在末尾留下“单独一行缩进”，否则在 &end 前看起来像多了空行
    return base + f"\n           {key}={rhs},"


def _rewrite_namelist_block(text: str, name: str, updater) -> str:
    pat = re.compile(rf"(?is)(&{name}\b)(.*?)(\s*&end)")
    m = pat.search(text)
    if not m:
        raise ValueError(f"未找到 &{name} ... &end 段")
    prefix, body, suffix = m.group(1), m.group(2), m.group(3)
    new_body = updater(body)
    return text[: m.start()] + prefix + new_body + suffix + text[m.end() :]


def apply_phase_groups_to_rin_text(rin_text: str, groups: list[PhaseGroup]) -> tuple[str, list[str]]:
    arr = compile_phase_groups_to_arrays(groups)
    diagnostics = list(arr["diagnostics"])  # type: ignore[arg-type]

    def update_trapar(body: str) -> str:
        b = body
        b = _replace_or_insert_assignment(
            b, "ray", _format_numeric_array(arr["ray"], is_float=True), TRAPAR_KEYS  # type: ignore[arg-type]
        )
        b = _replace_or_insert_assignment(
            b, "nrbnd", _format_numeric_array(arr["nrbnd"], is_float=False), TRAPAR_KEYS  # type: ignore[arg-type]
        )
        b = _replace_or_insert_assignment(
            b, "rbnd", _format_numeric_array(arr["rbnd"], is_float=False), TRAPAR_KEYS  # type: ignore[arg-type]
        )
        b = _replace_or_insert_assignment(
            b, "ncbnd", _format_numeric_array(arr["ncbnd"], is_float=False), TRAPAR_KEYS  # type: ignore[arg-type]
        )
        b = _replace_or_insert_assignment(
            b, "cbnd", _format_numeric_array(arr["cbnd"], is_float=False), TRAPAR_KEYS  # type: ignore[arg-type]
        )
        b = _replace_or_insert_assignment(
            b, "nray", _format_numeric_array(arr["nray"], is_float=False), TRAPAR_KEYS  # type: ignore[arg-type]
        )
        return b

    def update_invpar(body: str) -> str:
        # invpar 通常只含 ivray；用紧凑写回避免多次保存产生“尾部空行/缩进堆积”
        return _replace_or_insert_assignment_scalar(
            body,
            "ivray",
            _format_numeric_array(arr["ivray"], is_float=False),  # type: ignore[arg-type]
            INVPAR_KEYS,
        )

    out = _rewrite_namelist_block(rin_text, "trapar", update_trapar)
    out = _rewrite_namelist_block(out, "invpar", update_invpar)
    return out, diagnostics


def apply_phase_groups_to_rin_file(rin_path: str | Path, groups: list[PhaseGroup]) -> tuple[bool, str, list[str]]:
    p = Path(rin_path)
    if not p.exists():
        return False, f"r.in 不存在: {p}", []
    old = p.read_text(encoding="utf-8", errors="ignore")
    try:
        new, diagnostics = apply_phase_groups_to_rin_text(old, groups)
    except Exception as e:
        return False, f"写回失败: {e}", []
    p.write_text(new, encoding="utf-8")
    return True, "已写回 phase groups 到 r.in", diagnostics


def apply_rin_misc_params_to_rin_text(rin_text: str, misc: RinMiscParams) -> tuple[str, list[str]]:
    diagnostics: list[str] = []
    out = rin_text

    def update_trapar(body: str) -> str:
        b = body
        if misc.imodf is not None:
            b = _replace_or_insert_assignment_scalar(b, "imodf", str(int(misc.imodf)), TRAPAR_MISC_KEYS)
        if misc.i2pt is not None:
            b = _replace_or_insert_assignment_scalar(b, "i2pt", str(int(misc.i2pt)), TRAPAR_MISC_KEYS)
        if misc.isrch is not None:
            b = _replace_or_insert_assignment_scalar(b, "isrch", str(int(misc.isrch)), TRAPAR_MISC_KEYS)
        if misc.xshot is not None:
            b = _replace_or_insert_assignment_scalar(
                b, "xshot", _format_float_array(misc.xshot, decimals=3), TRAPAR_MISC_KEYS
            )
        if misc.zshot is not None:
            b = _replace_or_insert_assignment_scalar(
                b, "zshot", _format_float_array(misc.zshot, decimals=3), TRAPAR_MISC_KEYS
            )
        if misc.aamin is not None:
            b = _replace_or_insert_assignment_scalar(b, "aamin", f"{float(misc.aamin):.3f}", TRAPAR_MISC_KEYS)
        if misc.aamax is not None:
            b = _replace_or_insert_assignment_scalar(b, "aamax", f"{float(misc.aamax):.3f}", TRAPAR_MISC_KEYS)
        if misc.ishot is not None:
            b = _replace_or_insert_assignment_scalar(b, "ishot", _format_int_array(misc.ishot), TRAPAR_MISC_KEYS)
        if misc.n2pt is not None:
            b = _replace_or_insert_assignment_scalar(b, "n2pt", str(int(misc.n2pt)), TRAPAR_MISC_KEYS)
        if misc.space is not None:
            b = _replace_or_insert_assignment_scalar(b, "space", _format_float_array(misc.space, decimals=3), TRAPAR_MISC_KEYS)
        if misc.x2pt is not None:
            b = _replace_or_insert_assignment_scalar(b, "x2pt", f"{float(misc.x2pt):.3f}", TRAPAR_MISC_KEYS)
        return b

    try:
        out = _rewrite_namelist_block(out, "trapar", update_trapar)
    except Exception as e:
        diagnostics.append(f"写回 &trapar xshot/zshot 失败: {e}")

    if misc.vred is not None:

        def update_pltpar(body: str) -> str:
            return _replace_or_insert_assignment_scalar(body, "vred", f"{float(misc.vred):.3f}", PLTPAR_KEYS)

        try:
            out = _rewrite_namelist_block(out, "pltpar", update_pltpar)
        except Exception as e:
            diagnostics.append(f"写回 &pltpar vred 失败: {e}")

    return out, diagnostics


def apply_phase_groups_and_misc_to_rin_text(
    rin_text: str,
    groups: list[PhaseGroup],
    misc: RinMiscParams,
) -> tuple[str, list[str]]:
    t1, d1 = apply_phase_groups_to_rin_text(rin_text, groups)
    t2, d2 = apply_rin_misc_params_to_rin_text(t1, misc)
    return t2, [*d1, *d2, *misc.diagnostics]


def apply_phase_groups_and_misc_to_rin_file(
    rin_path: str | Path,
    groups: list[PhaseGroup],
    misc: RinMiscParams,
) -> tuple[bool, str, list[str]]:
    p = Path(rin_path)
    if not p.exists():
        return False, f"r.in 不存在: {p}", []
    old = p.read_text(encoding="utf-8", errors="ignore")
    try:
        new, diagnostics = apply_phase_groups_and_misc_to_rin_text(old, groups, misc)
    except Exception as e:
        return False, f"写回失败: {e}", []
    p.write_text(new, encoding="utf-8")
    return True, "已写回 phase groups + 全局参数 到 r.in", diagnostics

