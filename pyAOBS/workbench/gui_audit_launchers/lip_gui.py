"""审计包装启动：LIP Petrology GUI（PySide6）。"""

from __future__ import annotations

import runpy
import sys

from ._common import AuditRuntimeHooks, write_audit


def main() -> int:
    target = "pyAOBS.petrology.gui"
    old_argv = list(sys.argv)
    hooks = AuditRuntimeHooks()
    try:
        write_audit("lip_gui_started", argv=old_argv[1:], module=target)
        hooks.install()
        sys.argv = [target] + old_argv[1:]
        runpy.run_module(target, run_name="__main__")
        write_audit("lip_gui_closed")
        return 0
    except Exception as exc:
        write_audit("lip_gui_error", error=str(exc))
        raise
    finally:
        hooks.restore()
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
