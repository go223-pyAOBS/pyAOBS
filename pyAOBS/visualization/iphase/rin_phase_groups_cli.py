"""
无 GUI 的 r.in 参数解译工具。

功能：
1) 读取并解析 ray/nrbnd/ncbnd/nray/ivray/rbnd/cbnd；
2) 输出数组摘要与一致性检查结果；
3) 可批量验证多个 r.in 文件。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _load_parser():
    # 优先包内导入；失败时退回同目录脚本导入
    try:
        from .rin_phase_groups import (
            describe_all_phase_groups,
            parse_rin_phase_arrays_from_file,
            parse_phase_groups_from_rin_file,
        )

        return describe_all_phase_groups, parse_rin_phase_arrays_from_file, parse_phase_groups_from_rin_file
    except Exception:
        cur = Path(__file__).resolve().parent
        if str(cur) not in sys.path:
            sys.path.insert(0, str(cur))
        from rin_phase_groups import (  # type: ignore
            describe_all_phase_groups,
            parse_rin_phase_arrays_from_file,
            parse_phase_groups_from_rin_file,
        )

        return describe_all_phase_groups, parse_rin_phase_arrays_from_file, parse_phase_groups_from_rin_file


def _fmt_float_arr(vals: list[float], n: int = 12) -> str:
    if not vals:
        return "[]"
    body = ", ".join(f"{v:.1f}" for v in vals[:n])
    if len(vals) > n:
        body += f", ... (total={len(vals)})"
    return "[" + body + "]"


def _fmt_int_arr(vals: list[int], n: int = 20) -> str:
    if not vals:
        return "[]"
    body = ", ".join(str(v) for v in vals[:n])
    if len(vals) > n:
        body += f", ... (total={len(vals)})"
    return "[" + body + "]"


def inspect_one(
    rin_path: Path,
    *,
    safe: bool = False,
    json_mode: bool = False,
    describe: bool = False,
) -> tuple[int, dict]:
    describe_all, parse_arrays, parse_groups = _load_parser()
    parse_path_arrays = not safe
    try:
        arr = parse_arrays(rin_path, parse_path_arrays=parse_path_arrays)
        groups = parse_groups(rin_path, parse_path_arrays=parse_path_arrays)
    except Exception as e:
        payload = {
            "path": str(rin_path),
            "ok": False,
            "error": str(e),
        }
        if not json_mode:
            print(f"[FAIL] {rin_path}: {e}")
        return 1, payload

    need_rbnd = sum(arr.nrbnd)
    need_cbnd = sum(arr.ncbnd)
    payload = {
        "path": str(rin_path),
        "ok": True,
        "groups": len(groups.groups),
        "safe_mode": safe,
        "arrays": {
            "ray": arr.ray,
            "nrbnd": arr.nrbnd,
            "rbnd": arr.rbnd,
            "ncbnd": arr.ncbnd,
            "cbnd": arr.cbnd,
            "nray": arr.nray,
            "ivray": arr.ivray,
        },
        "checks": {
            "rbnd_need": need_rbnd,
            "rbnd_have": len(arr.rbnd),
            "cbnd_need": need_cbnd,
            "cbnd_have": len(arr.cbnd),
            "rbnd_ok": (True if safe else len(arr.rbnd) >= need_rbnd),
            "cbnd_ok": (True if safe else len(arr.cbnd) >= need_cbnd),
        },
        "diagnostics": groups.diagnostics,
    }

    if json_mode:
        return 0, payload

    if describe:
        print(f"\n=== {rin_path} 文字说明 ===")
        for line in describe_all(groups.groups):
            print(line)
            print()
        return 0, payload

    print(f"\n=== {rin_path} ===")
    print(f"groups={len(groups.groups)}")
    print(f"ray   : {_fmt_float_arr(arr.ray)}")
    print(f"nrbnd : {_fmt_int_arr(arr.nrbnd)}")
    print(f"rbnd  : {_fmt_int_arr(arr.rbnd)}")
    print(f"ncbnd : {_fmt_int_arr(arr.ncbnd)}")
    print(f"cbnd  : {_fmt_int_arr(arr.cbnd)}")
    print(f"nray  : {_fmt_int_arr(arr.nray)}")
    print(f"ivray : {_fmt_int_arr(arr.ivray)}")

    if parse_path_arrays:
        ok_rb = len(arr.rbnd) >= need_rbnd
        ok_cb = len(arr.cbnd) >= need_cbnd
        print(f"check : rbnd_need={need_rbnd}, rbnd_have={len(arr.rbnd)}, ok={ok_rb}")
        print(f"check : cbnd_need={need_cbnd}, cbnd_have={len(arr.cbnd)}, ok={ok_cb}")
    else:
        print("check : safe mode（未解析 rbnd/cbnd 细节）")

    if groups.diagnostics:
        print("diagnostics:")
        for d in groups.diagnostics:
            print(f"  - {d}")
    else:
        print("diagnostics: none")
    return 0, payload


def main() -> int:
    ap = argparse.ArgumentParser(description="RAYINVR r.in 参数解译（无 GUI）")
    ap.add_argument("rin", nargs="*", help="r.in 文件路径（可多个）")
    ap.add_argument("--glob", dest="glob_pat", default="", help="按 glob 批量扫描（例如 D:/proj/**/r.in）")
    ap.add_argument("--safe", action="store_true", help="安全模式：不展开 rbnd/cbnd")
    ap.add_argument("--json", dest="json_mode", action="store_true", help="以 JSON 输出，便于自动比对")
    ap.add_argument("--json-out", dest="json_out", default="", help="将 JSON 结果写入文件（如 result.json）")
    ap.add_argument(
        "--describe",
        action="store_true",
        help="输出每组 ray/rbnd/cbnd/ivray 的中文释义（与 GUI「文字说明」一致）",
    )
    args = ap.parse_args()

    paths: list[Path] = []
    for p in args.rin:
        paths.append(Path(p))
    if args.glob_pat:
        for p in Path().glob(args.glob_pat):
            if p.is_file():
                paths.append(p)

    if not paths:
        print("请提供 r.in 路径，或使用 --glob 扫描。")
        return 2

    fail = 0
    outputs: list[dict] = []
    uniq: list[Path] = []
    seen = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)

    for p in uniq:
        code, payload = inspect_one(
            p,
            safe=args.safe,
            json_mode=args.json_mode,
            describe=args.describe,
        )
        fail += code
        outputs.append(payload)

    if args.json_mode:
        payload = outputs[0] if len(outputs) == 1 else outputs
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        print(text)
        if args.json_out:
            out_path = Path(args.json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
            print(f"[JSON] 已写入: {out_path.resolve()}")
    elif args.json_out:
        # 用户指定输出文件但未加 --json 时，自动按 JSON 输出处理
        payload = outputs[0] if len(outputs) == 1 else outputs
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"[JSON] 已写入: {out_path.resolve()}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

