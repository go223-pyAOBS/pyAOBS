"""Qt file-dialog helpers."""

from __future__ import annotations

import os


_FIGURE_FILTER_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("PNG", ".png"),
    ("PDF", ".pdf"),
    ("POSTSCRIPT", ".ps"),
    ("EPS", ".eps"),
    ("JPEG", ".jpg"),
    ("JPG", ".jpg"),
    ("TIFF", ".tif"),
    ("TIF", ".tif"),
)


def normalize_save_path_for_filter(
    path: str,
    selected_filter: str,
    *,
    default_ext: str = ".png",
) -> str:
    """Append extension from QFileDialog selected filter when path has none."""
    _base, ext = os.path.splitext(path)
    if ext:
        return path
    filt = (selected_filter or "").upper()
    for token, suffix in _FIGURE_FILTER_SUFFIXES:
        if token in filt:
            return path + suffix
    return path + default_ext
