"""xarray 打开 GMT / CF NetCDF（``.grd`` / ``.nc``）；与 GMT ``grdinfo`` 可读文件对齐。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

import xarray as xr


def open_netcdf_like_dataset(path: str | Path) -> xr.Dataset:
    """优先 ``engine=netcdf4``（老式 GMT ``cf`` 经典网格）；再试 ``h5netcdf``、默认引擎。

    先 ``decode_cf=False`` 再退化：老式 GMT/classic `.grd` 在 CF 解码后常把网格搞丢或只剩 coords。
    ``decode_times=False``：避免非标准 TIME 解码失败。
    """
    fp = Path(path).expanduser()
    if not fp.is_file():
        raise FileNotFoundError(str(fp))

    errs: list[str] = []
    variants: Iterable[Optional[dict[str, str]]] = (
        {"engine": "netcdf4"},
        {"engine": "h5netcdf"},
        None,
    )
    common_kwargs: Iterable[Mapping[str, object]] = (
        {"decode_times": False, "decode_cf": False},
        {"decode_times": False},
    )

    last_exc: Optional[BaseException] = None
    for extras in variants:
        for kwargs_common in common_kwargs:
            kw = dict(kwargs_common)
            if extras:
                kw.update(extras)
            try:
                return xr.open_dataset(str(fp), **kw)
            except TypeError as te:
                if "decode_cf" in str(te).lower():
                    continue
                tag = repr(extras) if extras else "default-engine"
                errs.append(f"{tag} kw={kw!r}: {te}")
                last_exc = te
            except Exception as exc:  # noqa: BLE001
                tag = repr(extras) if extras else "default-engine"
                errs.append(f"{tag} kw={list(kw)!r}: {exc}")
                last_exc = exc

    # 对齐 zplotpy：部分 GMT .grd 仅用默认解码即可打开（与 decode_cf=False 互不重复成功时无关紧要）
    try:
        return xr.open_dataset(str(fp))
    except Exception as ex:  # noqa: BLE001
        errs.append(f"fallback bare open_dataset: {ex}")
        last_exc = last_exc or ex

    hint = (
        "\n提示：老式 GMT `.grd` 通常需要在 Python 环境中安装：`pip install netCDF4`"
        "\n或通过 `gmt grdconvert file.grd file.nc=nc4` 转为 NetCDF4。"
        if errs
        else ""
    )
    suffix = ("\n尝试记录:\n" + "\n".join(errs) + hint) if errs else ""
    raise IOError(f"xarray 无法打开 {fp.name}: {last_exc}{suffix}") from last_exc
