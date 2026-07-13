"""Run history scanning and filtering."""

from __future__ import annotations

import json
from pathlib import Path


def scan_run_history(
    project_root: Path,
    *,
    scan_limit: int = 1200,
) -> tuple[list[dict[str, str | Path]], set[str], set[str], bool]:
    runs_dir = project_root / "runs"
    records: list[dict[str, str | Path]] = []
    statuses: set[str] = set()
    nodes: set[str] = set()
    truncated = False
    if not runs_dir.exists():
        return records, statuses, nodes, truncated

    scanned = 0
    try:
        run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.name, reverse=True)
    except OSError:
        run_dirs = []

    for run_dir in run_dirs:
        if scanned >= scan_limit:
            truncated = True
            break
        if not run_dir.is_dir():
            continue
        scanned += 1
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_id = str(m.get("run_id", run_dir.name))
        node_id = str(m.get("node_id", ""))
        status = str(m.get("status", ""))
        finished_at = str(m.get("finished_at", "") or "")
        elapsed = m.get("elapsed_s", "")
        elapsed_str = f"{elapsed:.2f}" if isinstance(elapsed, (int, float)) else str(elapsed)
        command_text = " ".join([str(x) for x in m.get("command", [])]) or ""
        error_text = str(m.get("error", "") or "")
        search_blob = " ".join(
            [run_id, node_id, status, finished_at, elapsed_str, command_text, error_text]
        ).lower()
        records.append(
            {
                "run_id": run_id,
                "node_id": node_id,
                "status": status,
                "finished_at": finished_at,
                "elapsed_s": elapsed_str,
                "return_code": str(m.get("return_code", "")),
                "created_at": str(m.get("created_at", "") or ""),
                "command": command_text,
                "error": error_text,
                "manifest_file": manifest_path,
                "run_dir": run_dir,
                "search_blob": search_blob,
            }
        )
        if status:
            statuses.add(status)
        if node_id:
            nodes.add(node_id)
    return records, statuses, nodes, truncated


def filter_run_records(
    records: list[dict[str, str | Path]],
    *,
    status: str = "全部",
    node: str = "全部",
    keyword: str = "",
    failed_only: bool = False,
) -> list[dict[str, str | Path]]:
    kw = keyword.strip().lower()
    out: list[dict[str, str | Path]] = []
    for rec in records:
        st = str(rec.get("status", ""))
        nd = str(rec.get("node_id", ""))
        if status and status != "全部" and st != status:
            continue
        if node and node != "全部" and nd != node:
            continue
        if failed_only and st == "success":
            continue
        if kw and kw not in str(rec.get("search_blob", "")):
            continue
        out.append(rec)
    return out
