"""
工区磁盘预览缓存：重开 GUI 后复用道集拼图 / 速度预览体。

目录：``{workdir}/cache/.obs_rtm_preview_cache/``（兼容旧根上同名目录）
键与内存缓存一致；mtime / 参数变化则自动失效。
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CACHE_DIRNAME = os.path.join("cache", ".obs_rtm_preview_cache")
_LEGACY_CACHE = ".obs_rtm_preview_cache"
VEL_FILE_NPZ = "vel_file.npz"
VEL_FILE_META = "vel_file.json"
VEL_BUILTIN_NPZ = "vel_builtin.npz"
VEL_BUILTIN_META = "vel_builtin.json"
MONT_PREFIX = "mont_"
CACHE_VERSION = 1


def cache_dir(workdir: str) -> str:
    d = os.path.join(workdir, CACHE_DIRNAME)
    legacy = os.path.join(workdir, _LEGACY_CACHE)
    # 尚未迁移时继续用旧目录，避免丢缓存
    if not os.path.isdir(d) and os.path.isdir(legacy):
        d = legacy
    os.makedirs(d, exist_ok=True)
    gi = os.path.join(d, ".gitignore")
    if not os.path.isfile(gi):
        try:
            with open(gi, "w", encoding="utf-8") as f:
                f.write("*\n!.gitignore\n")
        except OSError:
            pass
    return d


def _canonical(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return [_canonical(x) for x in obj]
    if isinstance(obj, list):
        return [_canonical(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def keys_equal(a: Any, b: Any) -> bool:
    return _canonical(a) == _canonical(b)


def key_fingerprint(key: Any) -> str:
    payload = json.dumps(_canonical(key), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _rel(workdir: str, path: str) -> str:
    if not path:
        return ""
    ap = os.path.normpath(os.path.abspath(path))
    aw = os.path.normpath(os.path.abspath(workdir))
    try:
        return os.path.relpath(ap, aw).replace("\\", "/")
    except ValueError:
        return ap.replace("\\", "/")


def _abs(workdir: str, rel: str) -> str:
    if not rel:
        return ""
    if os.path.isabs(rel):
        return os.path.normpath(rel)
    return os.path.normpath(os.path.join(workdir, rel.replace("/", os.sep)))


def _write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ── 速度：用户文件 ──────────────────────────────────────────────


def save_vel_file(
    workdir: str,
    *,
    key: tuple,
    vel: np.ndarray,
    meta: dict,
    kind: str,
    path: str,
    pdx: float,
    pdz: float,
) -> None:
    if not workdir:
        return
    d = cache_dir(workdir)
    np.savez_compressed(
        os.path.join(d, VEL_FILE_NPZ),
        vel=np.ascontiguousarray(vel, dtype=np.float32),
    )
    _write_json(
        os.path.join(d, VEL_FILE_META),
        {
            "version": CACHE_VERSION,
            "kind_slot": "vel_file",
            "key": _canonical(key),
            "meta": {str(k): _canonical(v) for k, v in (meta or {}).items()},
            "kind": str(kind),
            "path": _rel(workdir, path),
            "path_abs": os.path.normpath(os.path.abspath(path)),
            "pdx": float(pdx),
            "pdz": float(pdz),
        },
    )


def load_vel_file(workdir: str, key: tuple) -> Optional[dict]:
    if not workdir:
        return None
    d = os.path.join(workdir, CACHE_DIRNAME)
    meta_p = os.path.join(d, VEL_FILE_META)
    npz_p = os.path.join(d, VEL_FILE_NPZ)
    meta = _read_json(meta_p)
    if not meta or not os.path.isfile(npz_p):
        return None
    if not keys_equal(meta.get("key"), key):
        return None
    try:
        with np.load(npz_p) as z:
            vel = np.ascontiguousarray(z["vel"], dtype=np.float32)
    except (OSError, KeyError, ValueError):
        return None
    path = str(meta.get("path_abs") or "")
    if not path or not os.path.isfile(path):
        path = _abs(workdir, str(meta.get("path") or ""))
    if not path or not os.path.isfile(path):
        return None
    return {
        "key": key,
        "vel": vel,
        "meta": dict(meta.get("meta") or {}),
        "kind": str(meta.get("kind") or "?"),
        "path": path,
        "pdx": float(meta.get("pdx") or 0.0),
        "pdz": float(meta.get("pdz") or 0.0),
        # 不写 zelt 键：由主窗口按 path / zelt_vin_path 再解析
    }


# ── 速度：内置一维 ──────────────────────────────────────────────


def save_vel_builtin(
    workdir: str,
    *,
    key: tuple,
    vel: np.ndarray,
    bath: Optional[np.ndarray],
    title: str,
) -> None:
    if not workdir:
        return
    d = cache_dir(workdir)
    kw: Dict[str, Any] = {"vel": np.ascontiguousarray(vel, dtype=np.float32)}
    if bath is not None:
        kw["bath"] = np.ascontiguousarray(bath, dtype=np.float32)
    np.savez_compressed(os.path.join(d, VEL_BUILTIN_NPZ), **kw)
    _write_json(
        os.path.join(d, VEL_BUILTIN_META),
        {
            "version": CACHE_VERSION,
            "kind_slot": "vel_builtin",
            "key": _canonical(key),
            "title": str(title),
            "has_bath": bath is not None,
        },
    )


def load_vel_builtin(workdir: str, key: tuple) -> Optional[dict]:
    if not workdir:
        return None
    d = os.path.join(workdir, CACHE_DIRNAME)
    meta = _read_json(os.path.join(d, VEL_BUILTIN_META))
    npz_p = os.path.join(d, VEL_BUILTIN_NPZ)
    if not meta or not os.path.isfile(npz_p):
        return None
    if not keys_equal(meta.get("key"), key):
        return None
    try:
        with np.load(npz_p) as z:
            vel = np.ascontiguousarray(z["vel"], dtype=np.float32)
            bath = None
            if bool(meta.get("has_bath")) and "bath" in z.files:
                bath = np.ascontiguousarray(z["bath"], dtype=np.float32)
    except (OSError, KeyError, ValueError):
        return None
    return {
        "key": key,
        "vel": vel,
        "bath": bath,
        "title": str(meta.get("title") or "builtin_1d"),
    }


# ── 道集拼图 ────────────────────────────────────────────────────


def save_mont(
    workdir: str,
    load_key: tuple,
    cache: dict,
    *,
    max_slots: int = 6,
) -> None:
    if not workdir or not isinstance(cache, dict):
        return
    raw = cache.get("raw")
    offs = cache.get("offs")
    if raw is None or offs is None:
        return
    d = cache_dir(workdir)
    fp = key_fingerprint(load_key)
    npz_p = os.path.join(d, "%s%s.npz" % (MONT_PREFIX, fp))
    meta_p = os.path.join(d, "%s%s.json" % (MONT_PREFIX, fp))
    used = cache.get("used") or []
    used_rel = [_rel(workdir, str(p)) for p in used]
    np.savez_compressed(
        npz_p,
        raw=np.ascontiguousarray(raw, dtype=np.float32),
        offs=np.ascontiguousarray(offs, dtype=np.float64),
    )
    _write_json(
        meta_p,
        {
            "version": CACHE_VERSION,
            "kind_slot": "mont",
            "key": _canonical(load_key),
            "fingerprint": fp,
            "d1": float(cache.get("d1") or 0.0),
            "o1": float(cache.get("o1") or 0.0),
            "used": used_rel,
            "title_base": str(cache.get("title_base") or ""),
        },
    )
    _prune_mont(d, keep=int(max_slots))


def load_mont(workdir: str, load_key: tuple) -> Optional[dict]:
    if not workdir:
        return None
    d = os.path.join(workdir, CACHE_DIRNAME)
    if not os.path.isdir(d):
        return None
    fp = key_fingerprint(load_key)
    meta_p = os.path.join(d, "%s%s.json" % (MONT_PREFIX, fp))
    npz_p = os.path.join(d, "%s%s.npz" % (MONT_PREFIX, fp))
    meta = _read_json(meta_p)
    if not meta or not os.path.isfile(npz_p):
        return None
    if not keys_equal(meta.get("key"), _canonical(load_key)):
        return None
    try:
        with np.load(npz_p) as z:
            raw = np.ascontiguousarray(z["raw"], dtype=np.float32)
            offs = np.ascontiguousarray(z["offs"], dtype=np.float64)
    except (OSError, KeyError, ValueError):
        return None
    used_rel: Sequence[str] = meta.get("used") or []
    used = [_abs(workdir, str(p)) for p in used_rel]
    if used and not all(os.path.isfile(p) for p in used):
        return None
    from .montage import used_path_index_map
    from .preprocess import time_axis

    d1 = float(meta.get("d1") or 0.0)
    o1 = float(meta.get("o1") or 0.0)
    times = time_axis(raw.shape[0], d1, o1)
    return {
        "load_key": load_key,
        "raw": raw,
        "offs": offs,
        "d1": d1,
        "o1": o1,
        "used": used,
        "used_index": used_path_index_map(used),
        "times": times,
        "mute_key": None,
        "muted": None,
        "prep_raw_key": None,
        "prep_raw": None,
        "prep_muted_key": None,
        "prep_muted": None,
        "gain_muted_key": None,
        "gain_muted": None,
        "disk_prep_key": None,
        "disk_prep": None,
        "disk_gain_key": None,
        "disk_gain": None,
        "title_base": str(meta.get("title_base") or ""),
        "from_disk": True,
    }


def invalidate_mont_sources(workdir: str, sources: Sequence[str]) -> None:
    """删除指定 montage 数据源对应的磁盘槽（load_key[1]=source）。"""
    if not workdir or not sources:
        return
    d = os.path.join(workdir, CACHE_DIRNAME)
    if not os.path.isdir(d):
        return
    want = {str(s) for s in sources}
    for name in os.listdir(d):
        if not (name.startswith(MONT_PREFIX) and name.endswith(".json")):
            continue
        meta = _read_json(os.path.join(d, name))
        if not meta:
            continue
        key = meta.get("key")
        src = None
        if isinstance(key, (list, tuple)) and len(key) > 1:
            src = str(key[1])
        if src not in want:
            continue
        stem = name[: -len(".json")]
        for ext in (".json", ".npz"):
            p = os.path.join(d, stem + ext)
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass


def _prune_mont(d: str, *, keep: int = 6) -> None:
    slots: List[Tuple[float, str]] = []
    for name in os.listdir(d):
        if not (name.startswith(MONT_PREFIX) and name.endswith(".json")):
            continue
        stem = name[: -len(".json")]
        jp = os.path.join(d, name)
        try:
            mtime = float(os.path.getmtime(jp))
        except OSError:
            mtime = 0.0
        slots.append((mtime, stem))
    if len(slots) <= keep:
        return
    slots.sort(key=lambda x: x[0])  # oldest first
    for _mt, stem in slots[: max(0, len(slots) - keep)]:
        for ext in (".json", ".npz"):
            p = os.path.join(d, stem + ext)
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass
