"""Project tree scanning."""

from __future__ import annotations

from pathlib import Path

DEFAULT_NO_RECURSE = {
    "outputs",
    "logs",
    "__pycache__",
    "node_modules",
    ".git",
    ".venv",
    "venv",
}


def scan_tree_nodes(
    parent_path: Path,
    *,
    depth: int = 0,
    max_depth: int = 5,
    max_children_per_dir: int = 400,
    no_recurse_names: set[str] | None = None,
) -> list[dict[str, object]]:
    if no_recurse_names is None:
        no_recurse_names = DEFAULT_NO_RECURSE
    if depth >= max_depth:
        return []
    try:
        children: list[Path] = []
        truncated = False
        for path in parent_path.iterdir():
            children.append(path)
            if len(children) >= max_children_per_dir:
                truncated = True
                break
        children.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError:
        return []

    nodes: list[dict[str, object]] = []
    for path in children:
        if path.name.startswith("."):
            continue
        is_dir = path.is_dir()
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        nodes.append({"path": resolved, "is_dir": is_dir, "depth": depth + 1})
        no_recurse = is_dir and path.name.lower() in no_recurse_names
        if is_dir and not no_recurse:
            nodes.extend(
                scan_tree_nodes(
                    path,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_children_per_dir=max_children_per_dir,
                    no_recurse_names=no_recurse_names,
                )
            )
    if truncated:
        nodes.append(
            {
                "path": None,
                "is_dir": False,
                "depth": depth + 1,
                "placeholder": f"... (仅显示前 {max_children_per_dir} 项)",
            }
        )
    return nodes
