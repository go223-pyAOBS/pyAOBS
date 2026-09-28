"""smesh 棋盘格 / 随机扰动 / 多模型均值与标准差。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_mesh(path: str | Path):
    try:
        from pyAOBS.model_building.tomoform import SlownessMesh2D
    except ImportError:  # pragma: no cover
        from pyAOBS.model_building.tomoform import SlownessMesh2D  # type: ignore

    return SlownessMesh2D.from_file(str(path))


def checkerboard_relative_pattern(
    mesh,
    *,
    amp_percent: float,
    h_len: float,
    v_len: float,
) -> np.ndarray:
    """与 edit_smesh -Cc 同形：相对扰动 ``A% × sin(2πx/h) × sin(2πz_abs/v)``。"""
    if h_len <= 0 or v_len <= 0:
        raise ValueError("棋盘格水平/垂向波长须为正")
    x = mesh.xpos[:, np.newaxis]
    z_abs = mesh.topo[:, np.newaxis] + mesh.zpos[np.newaxis, :]
    return (
        float(amp_percent)
        * 0.01
        * np.sin(2.0 * np.pi * x / float(h_len))
        * np.sin(2.0 * np.pi * z_abs / float(v_len))
    )


def checkerboard_velocity_fields(
    src: str | Path,
    *,
    amp_percent: float,
    h_len: float,
    v_len: float,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    """返回 ``(mesh, v_背景, v_棋盘, ΔV)``，不写盘。``mesh.vgrid`` 仍为背景。"""
    mesh = _load_mesh(src)
    v_bg = np.asarray(mesh.vgrid, dtype=float).copy()
    pattern = checkerboard_relative_pattern(
        mesh, amp_percent=amp_percent, h_len=h_len, v_len=v_len
    )
    v_cb = v_bg * (1.0 + pattern)
    return mesh, v_bg, v_cb, v_cb - v_bg


def apply_checkerboard_to_file(
    src: str | Path,
    dst: str | Path,
    *,
    amp_percent: float,
    h_len: float,
    v_len: float,
) -> Path:
    """背景模型叠加棋盘格扰动并写出（与 edit_smesh -Cc 同形：百分数 × sin×sin）。"""
    mesh, _bg, v_cb, _dv = checkerboard_velocity_fields(
        src, amp_percent=amp_percent, h_len=h_len, v_len=v_len
    )
    mesh.vgrid = v_cb
    mesh.pgrid = 1.0 / np.maximum(mesh.vgrid, 1e-9)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    mesh.to_file(str(dst_p))
    return dst_p


def air_water_node_mask(mesh) -> np.ndarray:
    """结点是否在几何空气/水层，或速度等于文件头 ``v_air`` / ``v_water``。

    几何与 ``smesh.cc`` 的 ``in_air`` / ``in_water`` 一致：绝对深度
    ``z = topo + zpos`` 且 ``z < topo``（即 ``zpos < 0``，海面以上或水柱内）。
    另冻结与头 ``v_air`` / ``v_water`` 相等的结点（如 ``gen_smesh -W`` 写进
    ``vgrid`` 的盖层），避免扰动常数水/气速。
    """
    zpos = np.asarray(mesh.zpos, dtype=float)
    v = np.asarray(mesh.vgrid, dtype=float)
    frozen = np.broadcast_to(zpos[np.newaxis, :] < 0.0, v.shape).copy()
    vw = float(getattr(mesh, "v_water", np.nan))
    va = float(getattr(mesh, "v_air", np.nan))
    if np.isfinite(vw):
        frozen |= np.isclose(v, vw, atol=1e-6, rtol=0.0)
    if np.isfinite(va):
        frozen |= np.isclose(v, va, atol=1e-6, rtol=0.0)
    return frozen


def random_init_velocity_fields(
    src: str | Path,
    *,
    amp_percent: float,
    seed: int,
    smooth_sigma: float = 2.0,
    apply: bool = True,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    """返回 ``(mesh, v_基础, v_扰动后, ΔV)``，不写盘。``mesh.vgrid`` 仍为基础。

    ``apply=False`` 时扰动后与基础相同（未勾选随机初始）。
    水层与空气层结点不扰动（见 ``air_water_node_mask``）；幅度 RMS 只按可扰动结点计。
    """
    mesh = _load_mesh(src)
    v_bg = np.asarray(mesh.vgrid, dtype=float).copy()
    if not apply or float(amp_percent) == 0.0:
        return mesh, v_bg, v_bg.copy(), np.zeros_like(v_bg)
    keep = ~air_water_node_mask(mesh)
    if not np.any(keep):
        return mesh, v_bg, v_bg.copy(), np.zeros_like(v_bg)
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0.0, 1.0, size=v_bg.shape)
    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter

        noise = gaussian_filter(noise, sigma=float(smooth_sigma), mode="nearest")
    rms = float(np.sqrt(np.mean(noise[keep] ** 2))) or 1.0
    noise = noise / rms * (float(amp_percent) * 0.01)
    v_pert = np.where(
        keep, np.maximum(v_bg * (1.0 + noise), 1e-3), v_bg
    )
    return mesh, v_bg, v_pert, v_pert - v_bg


def apply_random_init_to_file(
    src: str | Path,
    dst: str | Path,
    *,
    amp_percent: float,
    seed: int,
    smooth_sigma: float = 2.0,
) -> Path:
    """
    对背景速度做相关随机扰动（白噪声经高斯平滑），写出为新初始模型。

    ``amp_percent``：扰动相对幅度的 RMS 目标量级（百分数量级，再乘 0.01）。
    """
    mesh, _bg, v_pert, _dv = random_init_velocity_fields(
        src, amp_percent=amp_percent, seed=seed, smooth_sigma=smooth_sigma
    )
    mesh.vgrid = v_pert
    mesh.pgrid = 1.0 / np.maximum(mesh.vgrid, 1e-9)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    mesh.to_file(str(dst_p))
    return dst_p


def velocity_percent_anomaly(true_path: str | Path, recovered_path: str | Path) -> np.ndarray:
    """(v_rec - v_bg) / v_bg * 100，此处 true 与 recovered 均为绝对速度网格。"""
    a = _load_mesh(true_path)
    b = _load_mesh(recovered_path)
    if a.vgrid.shape != b.vgrid.shape:
        raise ValueError(
            f"smesh 网格尺寸不一致: {a.vgrid.shape} vs {b.vgrid.shape}"
        )
    return (b.vgrid - a.vgrid) / np.maximum(a.vgrid, 1e-9) * 100.0


def _dws_xyz_to_vgrid(mesh, xyz) -> np.ndarray | None:
    """把 DWS (x,z,w) 落到与 ``vgrid`` 同形的节点阵；对不上则返回 None。"""
    if xyz is None:
        return None
    arr = np.asarray(xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 3:
        return None
    nx, nz = int(mesh.vgrid.shape[0]), int(mesh.vgrid.shape[1])
    if arr.shape[0] != nx * nz:
        return None
    return np.asarray(arr[:, 2], dtype=float).reshape(nx, nz)


def stack_mean_std(
    smesh_paths: list[str | Path],
    *,
    dws_xyz_list: list | None = None,
) -> tuple[object, np.ndarray, np.ndarray]:
    """多模型速度均值与标准差；返回 (模板 mesh, mean_v, std_v)。

    若提供与路径等长的 ``dws_xyz_list``，各点只平均该处 DWS>0 的成员
    （无覆盖不计入；谁都没有覆盖则退回模板速度，σ 为 NaN）。
    """
    if not smesh_paths:
        raise ValueError("没有可汇总的 smesh")
    meshes = [_load_mesh(p) for p in smesh_paths]
    shape0 = meshes[0].vgrid.shape
    for m in meshes[1:]:
        if m.vgrid.shape != shape0:
            raise ValueError("各 smesh 网格尺寸不一致")
    stack = np.stack([m.vgrid for m in meshes], axis=0)
    masked = np.array(stack, dtype=float, copy=True)
    used_cover = False
    if dws_xyz_list is not None:
        if len(dws_xyz_list) != len(meshes):
            raise ValueError("dws_xyz_list 须与 smesh 数量相同")
        for i, xyz in enumerate(dws_xyz_list):
            cov = _dws_xyz_to_vgrid(meshes[i], xyz)
            if cov is None:
                continue
            used_cover = True
            uncovered = ~np.isfinite(cov) | (cov <= 0.0)
            masked[i] = np.where(uncovered, np.nan, stack[i])
    if used_cover:
        n = np.sum(np.isfinite(masked), axis=0)
        s = np.nansum(masked, axis=0)
        mean_v = np.where(n > 0, s / np.maximum(n, 1), stack[0])
        sq = np.nansum((masked - mean_v) ** 2, axis=0)
        std_v = np.where(n > 1, np.sqrt(sq / n), 0.0)
        return meshes[0], mean_v, std_v
    mean_v = np.mean(stack, axis=0)
    std_v = np.std(stack, axis=0, ddof=0)
    return meshes[0], mean_v, std_v


def write_velocity_grid_as_smesh(
    template_path: str | Path,
    vgrid: np.ndarray,
    dst: str | Path,
    *,
    allow_nonpositive: bool = False,
) -> Path:
    mesh = _load_mesh(template_path)
    if mesh.vgrid.shape != vgrid.shape:
        raise ValueError("写出网格与模板尺寸不一致")
    mesh.vgrid = np.asarray(vgrid, dtype=float)
    if not allow_nonpositive:
        mesh.vgrid = np.maximum(mesh.vgrid, 1e-3)
        mesh.pgrid = 1.0 / mesh.vgrid
    else:
        # 百分异常 / 标准差等伪场：保持符号，pgrid 仅占位。
        # np.where 会先算 1/v，零异常仍会 RuntimeWarning；用 where= 跳过零点。
        pgrid = np.zeros_like(mesh.vgrid, dtype=float)
        np.divide(
            1.0, mesh.vgrid, out=pgrid, where=np.abs(mesh.vgrid) > 1e-12
        )
        mesh.pgrid = pgrid
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    mesh.to_file(str(dst_p))
    return dst_p


def stack_reflector_mean_std(
    refl_paths: list[str | Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """多条反射面插到公共 x，返回 (x, mean_z, std_z)。至少两条。"""
    if len(refl_paths) < 2:
        raise ValueError("反射面统计至少需要 2 个界面文件")
    from pyAOBS.model_building.tomoform import load_tomo2d_interface_file

    loaded = [load_tomo2d_interface_file(str(p)) for p in refl_paths]
    x_lo = max(float(np.min(x)) for x, _z in loaded)
    x_hi = min(float(np.max(x)) for x, _z in loaded)
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
        raise ValueError("反射面 x 范围没有交集")
    x0 = loaded[0][0]
    x = np.asarray(x0[(x0 >= x_lo) & (x0 <= x_hi)], dtype=float)
    if x.size < 2:
        x = np.linspace(x_lo, x_hi, 51)
    stack = np.stack(
        [np.interp(x, xi, zi) for xi, zi in loaded],
        axis=0,
    )
    return x, np.mean(stack, axis=0), np.std(stack, axis=0, ddof=0)


def write_interface_xz(
    x,
    z,
    dst: str | Path,
    *,
    header: str = "",
) -> Path:
    """写出 tomo2d ``-F`` 用的 ``x z`` 文本。

    C++ ``Interface2d`` 用 ``countLines`` + ``>>`` 读入，**不跳过 ``#`` 注释**；
    注释行会被当成结点，随后报 ``illegal ordering of x nodes``。
    ``header`` 仅作调用方说明，不写入文件。
    """
    xx = np.asarray(x, dtype=float).ravel()
    zz = np.asarray(z, dtype=float).ravel()
    if xx.size != zz.size or xx.size < 2:
        raise ValueError("反射面至少需要 2 个节点")
    order = np.argsort(xx)
    xx, zz = xx[order], zz[order]
    if np.any(np.diff(xx) <= 0):
        # 合并重合 x（取平均 z），避免网格 xpos 重复导致 -F 读入失败
        keep_x = [float(xx[0])]
        keep_z = [float(zz[0])]
        n_dup = 1
        for xi, zi in zip(xx[1:], zz[1:]):
            if xi <= keep_x[-1]:
                keep_z[-1] = (keep_z[-1] * n_dup + float(zi)) / (n_dup + 1)
                n_dup += 1
                continue
            keep_x.append(float(xi))
            keep_z.append(float(zi))
            n_dup = 1
        xx = np.asarray(keep_x, dtype=float)
        zz = np.asarray(keep_z, dtype=float)
        if xx.size < 2 or np.any(np.diff(xx) <= 0):
            raise ValueError("反射面 x 须严格递增")
    path = Path(dst)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        _ = header
        for xi, zi in zip(xx, zz):
            f.write(f"{xi:.6f}  {zi:.6f}\n")
    return path


def write_interface_mean_std(
    x,
    z_mean,
    z_std,
    dst: str | Path,
    *,
    header: str = "",
) -> Path:
    """写出 ``x  z_mean  z_std``（界面均值与误差）。"""
    xx = np.asarray(x, dtype=float).ravel()
    zm = np.asarray(z_mean, dtype=float).ravel()
    zs = np.asarray(z_std, dtype=float).ravel()
    if xx.size != zm.size or xx.size != zs.size or xx.size < 2:
        raise ValueError("界面均值/误差至少需要 2 个节点")
    path = Path(dst)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {header or 'x_km  z_mean  z_std'}\n")
        if header:
            f.write("# x_km  z_mean  z_std\n")
        for xi, mi, si in zip(xx, zm, zs):
            f.write(f"{xi:.6f}  {mi:.6f}  {si:.6f}\n")
    return path


def load_interface_mean_std(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """读 ``x z_mean z_std``；列不足则返回 None。"""
    p = Path(path)
    if not p.is_file():
        return None
    xs: list[float] = []
    ms: list[float] = []
    ss: list[float] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        try:
            xs.append(float(parts[0]))
            ms.append(float(parts[1]))
            ss.append(float(parts[2]))
        except ValueError:
            continue
    if len(xs) < 2:
        return None
    return (
        np.asarray(xs, dtype=float),
        np.asarray(ms, dtype=float),
        np.asarray(ss, dtype=float),
    )


def realization_refl_path(
    recovered_smesh: str | Path,
    real_dir: str | Path | None = None,
) -> Path | None:
    """反演配套 refl；没有则用该次起始 ``moho.refl``。"""
    hit = companion_inverse_refl(recovered_smesh)
    if hit is not None:
        return hit
    if real_dir is not None:
        init = Path(real_dir) / "moho.refl"
        if init.is_file():
            return init
    return None


def parse_inverse_smesh_name(path: Path | str) -> tuple[int, int] | None:
    """从 ``*.smesh.<iter>.<iset>`` 解析 (iter, iset)；失败返回 None。"""
    name = Path(path).name
    if ".smesh." not in name:
        return None
    parts = name.split(".")
    try:
        return int(parts[-2]), int(parts[-1])
    except (ValueError, IndexError):
        return None


def list_inverse_smesh_files(out_root: str | Path) -> list[tuple[Path, int, int]]:
    """
    列出 ``out_root.smesh.<iter>.<iset>``（扁平或 ``models/``），按 (iter, iset) 升序。

    若 ``out_root`` 本身是目录（旧式一次反演一个文件夹），再扫该目录及其 ``models/``
    下任意 ``*.smesh.<iter>.<iset>``（不要求前缀必须是 ``out``）。
    同 (iter, iset) 优先保留 mtime 较新者。
    """
    root = Path(out_root)
    parent = root.parent if root.parent.as_posix() not in ("", ".") else Path(".")
    stem = root.name
    matches = list(parent.glob(f"{stem}.smesh.*"))
    models_dir = parent / "models"
    if models_dir.is_dir():
        matches.extend(models_dir.glob(f"{stem}.smesh.*"))
    if not matches and parent == Path("."):
        matches = list(Path(".").glob(f"{stem}.smesh.*"))
        m2 = Path("models")
        if m2.is_dir():
            matches.extend(m2.glob(f"{stem}.smesh.*"))
    try:
        if root.is_dir():
            matches.extend(root.glob("*.smesh.*"))
            inner_models = root / "models"
            if inner_models.is_dir():
                matches.extend(inner_models.glob("*.smesh.*"))
    except OSError:
        pass

    best: dict[tuple[int, int], Path] = {}
    for p in matches:
        key = parse_inverse_smesh_name(p)
        if key is None:
            continue
        prev = best.get(key)
        if prev is None:
            best[key] = p
            continue
        try:
            if p.stat().st_mtime >= prev.stat().st_mtime:
                best[key] = p
        except OSError:
            best[key] = p
    return [(best[k], k[0], k[1]) for k in sorted(best.keys())]


def find_latest_inverse_smesh(out_root: str | Path) -> Path:
    """
    查找 tt_inverse ``-O out_root`` 生成的 ``out_root.smesh.<iter>.<iset>``。

    取 ``(iter, iset)`` 最大者——仅表示**最后写出**的网格，不是 χ²/光滑度最优。
    兼容扁平 ``outputs/`` 与归类后的 ``outputs/models/``。
    """
    entries = list_inverse_smesh_files(out_root)
    if not entries:
        root = Path(out_root)
        parent = root.parent if root.parent.as_posix() not in ("", ".") else Path(".")
        models_dir = parent / "models"
        raise FileNotFoundError(
            f"未找到反演输出网格：期望形如 {root.name}.smesh.<iter>.<iset> "
            f"于 {parent.resolve()} 或 {models_dir}"
        )
    return entries[-1][0]


def companion_inverse_refl(smesh_path: str | Path) -> Path | None:
    """同轮 ``*.smesh.<iter>.<iset>`` 对应的 ``*.refl.<iter>.<iset>``（models/ 或扁平）。"""
    p = Path(smesh_path)
    if parse_inverse_smesh_name(p) is None:
        return None
    refl_name = p.name.replace(".smesh.", ".refl.", 1)
    if refl_name == p.name:
        return None
    parent = p.parent
    grand = parent.parent if parent.parent.as_posix() not in ("", ".") else parent
    candidates = (
        parent / refl_name,
        grand / refl_name,
        grand / "models" / refl_name,
        parent / "models" / refl_name,
    )
    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    return None
