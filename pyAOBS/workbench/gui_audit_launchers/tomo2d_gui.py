"""审计包装启动：工作台中的 TOMO2D GUI 入口（仅 Qt / PySide6）。"""

from __future__ import annotations

import runpy
import sys

from ._common import AuditRuntimeHooks, write_audit


def main() -> int:
    target = "pyAOBS.modeling.tomo2d.gui"
    frontend = "qt"

    old_argv = list(sys.argv)
    hooks = AuditRuntimeHooks()
    try:
        write_audit(
            "tomo2d_gui_started",
            argv=old_argv[1:],
            frontend=frontend,
            module=target,
        )
        hooks.install()
        sys.argv = [target] + old_argv[1:]
        runpy.run_module(target, run_name="__main__")
        write_audit("tomo2d_gui_closed", frontend=frontend)
        return 0
    except Exception as exc:
        write_audit("tomo2d_gui_error", error=str(exc), frontend=frontend)
        raise
    finally:
        hooks.restore()
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
