"""V 波形选取状态管理（对齐 zplotpy V / Shift+V 语义）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import SessionState, WaveformSelection


DEFAULT_PRE_SEC = 0.30
DEFAULT_POST_SEC = 0.70


class WaveformSelectionStore:
    """管理 waveform_selections 与 corrected_ttrue。"""

    def __init__(self, state: Optional[SessionState] = None):
        self.state = state or SessionState()

    @property
    def selections(self) -> List[WaveformSelection]:
        return self.state.waveform_selections

    def current_apick_selections(self, apick: Optional[int] = None) -> List[WaveformSelection]:
        pw = int(self.state.current_apick if apick is None else apick)
        return [s for s in self.selections if int(s.pick_word) == pw]

    def upsert(
        self,
        *,
        trace_idx: int,
        offset: float,
        t_display: float,
        t_true: float,
        pick_word: Optional[int] = None,
    ) -> Tuple[WaveformSelection, bool]:
        """添加或更新同一 (trace, pick_word) 的 V 段。返回 (selection, replaced)。"""
        pw = int(self.state.current_apick if pick_word is None else pick_word)
        new_sel = WaveformSelection(
            trace_idx=int(trace_idx),
            offset=float(offset),
            t_display=float(t_display),
            t_true=float(t_true),
            pick_word=pw,
        )
        key = new_sel.key()
        for i, old in enumerate(self.selections):
            if old.key() == key:
                self.selections[i] = new_sel
                self.state.corrected_ttrue[key] = float(t_true)
                return new_sel, True
        self.selections.append(new_sel)
        self.state.corrected_ttrue[key] = float(t_true)
        return new_sel, False

    def remove_last_for_apick(self, apick: Optional[int] = None) -> Optional[WaveformSelection]:
        pw = int(self.state.current_apick if apick is None else apick)
        for i in range(len(self.selections) - 1, -1, -1):
            if int(self.selections[i].pick_word) == pw:
                removed = self.selections.pop(i)
                self.state.corrected_ttrue.pop(removed.key(), None)
                return removed
        return None

    def clear(self) -> None:
        self.state.waveform_selections = []
        self.state.corrected_ttrue = {}

    def t_ref_for(self, sel: WaveformSelection) -> float:
        """姿态/截窗用的真实时间中心：优先叠加校正值。"""
        return float(self.state.corrected_ttrue.get(sel.key(), sel.t_true))

    def update_t_true(self, sel: WaveformSelection, t_true: float, t_display: Optional[float] = None) -> None:
        sel.t_true = float(t_true)
        if t_display is not None:
            sel.t_display = float(t_display)
        self.state.corrected_ttrue[sel.key()] = float(t_true)

    def save_waveop_json(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "current_apick": int(self.state.current_apick),
            "waveform_selections": [s.to_dict() for s in self.selections],
            "corrected_ttrue": {
                f"{k[0]},{k[1]}": float(v) for k, v in self.state.corrected_ttrue.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_waveop_json(self, path: str | Path) -> None:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.state.current_apick = int(data.get("current_apick", 1))
        self.state.waveform_selections = [
            WaveformSelection.from_dict(d) for d in data.get("waveform_selections", [])
        ]
        corrected: Dict[Tuple[int, int], float] = {}
        for k, v in (data.get("corrected_ttrue") or {}).items():
            parts = str(k).split(",")
            if len(parts) == 2:
                corrected[(int(parts[0]), int(parts[1]))] = float(v)
        self.state.corrected_ttrue = corrected

    @staticmethod
    def window_bounds(t_center: float, pre: float = DEFAULT_PRE_SEC, post: float = DEFAULT_POST_SEC):
        return float(t_center) - float(pre), float(t_center) + float(post)
