"""将 GUI 运行摘要写入 work_dir/tomo2d_gui.log（无 UI）。"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

_lock = threading.Lock()
_banner_paths: set[str] = set()


def append_gui_file_log(work_dir: str | Path, text: str, *, enabled: bool = True) -> None:
    """追加一行到 ``work_dir/tomo2d_gui.log``；目录须已存在。"""
    if not enabled:
        return
    wd = Path(work_dir)
    try:
        if not wd.is_dir():
            return
    except OSError:
        return
    path = wd / "tomo2d_gui.log"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        wd_key = str(wd.resolve())
    except OSError:
        wd_key = str(wd)
    with _lock:
        try:
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                if wd_key not in _banner_paths:
                    _banner_paths.add(wd_key)
                    f.write(
                        f"\n======== TOMO2D GUI | 工作目录 {wd} | 开始记录 {ts} ========\n"
                    )
                f.write(f"[{ts}] {text}\n")
        except OSError:
            pass
