"""无 UI 的表单状态：键名与 profile JSON / Qt 表单绑定一致。"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping


class FormState:
    """字符串/布尔字段字典；供 collectors / profile_io / Qt GUI 共用。"""

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = {}
        if data:
            self.apply_mapping(data)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def has(self, key: str) -> bool:
        return key in self._data

    def get_str(self, key: str, default: str = "") -> str:
        v = self._data.get(key, default)
        if v is None:
            return default
        if isinstance(v, bool):
            return "1" if v else ""
        return str(v).strip()

    def get_bool(self, key: str, default: bool = False) -> bool:
        v = self._data.get(key, default)
        if isinstance(v, bool):
            return v
        if v is None:
            return default
        s = str(v).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off", ""):
            return False
        return bool(v)

    def keys(self) -> Iterable[str]:
        return self._data.keys()

    def items(self) -> Iterator[tuple[str, Any]]:
        return iter(self._data.items())

    def to_profile_dict(self) -> dict[str, Any]:
        return dict(self._data)

    def apply_mapping(self, data: Mapping[str, Any]) -> None:
        """合并映射；兼容旧键 ``fwd.tol1`` / ``fwd.tol2``。"""
        d = dict(data)
        if "fwd.bend_cg_tol" not in d and "fwd.tol1" in d:
            d["fwd.bend_cg_tol"] = d["fwd.tol1"]
        if "fwd.bend_br_tol" not in d and "fwd.tol2" in d:
            d["fwd.bend_br_tol"] = d["fwd.tol2"]
        if "inv.apply_filter" not in d and str(d.get("inv.filter_bound_file") or "").strip():
            d["inv.apply_filter"] = True
        for k, v in d.items():
            self._data[k] = v

    def update(self, updates: Mapping[str, Any]) -> None:
        self._data.update(updates)
