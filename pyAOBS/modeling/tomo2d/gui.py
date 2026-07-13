#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TOMO2D GUI 启动别名模块。

用法:
    python -m pyAOBS.modeling.tomo2d.gui
"""

import json
import os
from pathlib import Path
from datetime import datetime

from ...workbench.gui_audit_launchers._common import AuditRuntimeHooks
from .tomo2d_gui import launch_tomo2d_gui


def _audit(event: str, **payload: object) -> None:
    audit_path = os.environ.get("PYAOBS_AUDIT_LOG", "").strip()
    if not audit_path:
        return
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "run_id": os.environ.get("PYAOBS_RUN_ID", "").strip(),
        "payload": payload,
    }
    try:
        p = Path(audit_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def main() -> None:
    _audit("tomo2d_gui_started")
    hooks = AuditRuntimeHooks()
    try:
        hooks.install()
        launch_tomo2d_gui()
    except Exception as exc:
        _audit("tomo2d_gui_error", error=str(exc))
        raise
    finally:
        hooks.restore()
        _audit("tomo2d_gui_closed")


if __name__ == "__main__":
    main()
