"""审计包装启动：工作台中的 vedit 入口（Qt / PySide6）。"""

from __future__ import annotations

import runpy
import sys

from ._common import AuditRuntimeHooks, write_audit


def main() -> int:
    target = "pyAOBS.modeling.vedit.gui"
    old_argv = list(sys.argv)
    hooks = AuditRuntimeHooks()
    try:
        write_audit("vedit_gui_started", argv=old_argv[1:], frontend="qt", module=target)
        hooks.install()
        sys.argv = [target] + old_argv[1:]
        runpy.run_module(target, run_name="__main__")
        write_audit("vedit_gui_closed", frontend="qt")
        return 0
    except Exception as exc:
        write_audit("vedit_gui_error", error=str(exc), frontend="qt")
        raise
    finally:
        hooks.restore()
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
