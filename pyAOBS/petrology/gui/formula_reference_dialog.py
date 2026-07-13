"""Browse KKHS02 key formulas and default parameters."""

from __future__ import annotations

import html
import re
from pathlib import Path

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from petrology.reference.kkhs02_workflow import (
    HVP_SHARED_INTERFACE_NOTE,
    format_workflow_summary_short,
)

from .app import UI_FONT_PT
from .dialog_utils import show_modeless_dialog
from .qt_styles import (
    COLOR_ACCENT,
    COLOR_BORDER_LIGHT,
    COLOR_BORDER_MED,
    COLOR_PANEL,
    COLOR_PANEL_ALT,
    COLOR_STATUS_BG,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    apply_dialog_style,
    primary_button,
)

_FORMULAS_PATH = Path(__file__).resolve().parents[1] / "reference" / "formulas.yaml"

_FORMULA_EXTRA_QSS = f"""
QTreeWidget {{
    background-color: {COLOR_PANEL};
    border: 2px solid {COLOR_BORDER_MED};
    border-radius: 4px;
    color: {COLOR_TEXT};
    font-size: {UI_FONT_PT}pt;
}}
QTreeWidget::item:selected {{
    background-color: {COLOR_STATUS_BG};
    color: {COLOR_TEXT};
}}
QTextBrowser#LipFormulaBrowser {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_BORDER_LIGHT};
    border-radius: 4px;
    padding: 4px;
}}
QLabel#LipFormulaHint {{
    color: {COLOR_TEXT_MUTED};
    font-size: {UI_FONT_PT - 1}pt;
}}
"""


def _load_formulas() -> dict:
    if yaml is None or not _FORMULAS_PATH.is_file():
        return {"steps": [], "parameters": []}
    with _FORMULAS_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _md_bold_to_html(text: str) -> str:
    """Escape text, then restore **bold** as <b>."""
    parts: list[str] = []
    last = 0
    for m in re.finditer(r"\*\*([^*]+)\*\*", text):
        parts.append(_esc(text[last : m.start()]))
        parts.append(f"<b>{_esc(m.group(1))}</b>")
        last = m.end()
    parts.append(_esc(text[last:]))
    return "".join(parts).replace("\n", "<br/>")


def _strip_md_bold(text: str) -> str:
    return re.sub(r"\*\*([^*]+)\*\*", r"\1", text.strip())


def _latex_to_display(latex: str) -> str:
    s = latex.strip()
    if not s:
        return ""
    reps = (
        (r"\\mathrm\{([^}]*)\}", r"\1"),
        (r"\\bar\s*", ""),
        (r"\\le", "≤"),
        (r"\\ge", "≥"),
        (r"\\Delta", "Δ"),
        (r"\\sum", "Σ"),
        (r"\\partial", "∂"),
        (r"\\cdot", "·"),
        (r"\\,", " "),
        (r"\\circ", "°"),
        (r"\^\{?\\circ\\?\}?", "°"),
        (r"_\{([^}]+)\}", r"_\1"),
        (r"\{([^}]+)\}", r"\1"),
        (r"\\\\", ""),
    )
    for pat, repl in reps:
        s = re.sub(pat, repl, s)
    return re.sub(r"\s+", " ", s).strip()


def _formula_display_text(payload: dict) -> str:
    display = str(payload.get("display") or "").strip()
    if display:
        return display
    latex = str(payload.get("latex") or "").strip()
    if latex:
        return _latex_to_display(latex)
    value = str(payload.get("value") or "").strip()
    if value:
        return f"{payload.get('name', '参数')} = {value}"
    notes = str(payload.get("notes") or "").strip()
    if notes:
        return notes
    return "(无公式)"


def _link_entry_id(entry: object) -> str:
    if isinstance(entry, dict):
        return str(entry.get("id") or "").strip()
    return str(entry).strip()


def _link_entry_blurb(entry: object) -> str:
    if isinstance(entry, dict):
        return _strip_md_bold(str(entry.get("blurb") or ""))
    return ""


def _format_workflow_links(
    entries: list,
    *,
    id_to_name: dict[str, str],
) -> str:
    if not entries:
        return ""
    lines: list[str] = []
    for entry in entries:
        iid = _link_entry_id(entry)
        if not iid:
            continue
        name = id_to_name.get(iid, iid)
        blurb = _link_entry_blurb(entry)
        if blurb:
            lines.append(f"• {name} — {blurb}")
        else:
            lines.append(f"• {name}")
    return "\n".join(lines)


def _link_icon_kind(payload: dict) -> str | None:
    has_prev = bool(payload.get("prev"))
    has_next = bool(payload.get("next"))
    if has_prev and has_next:
        return "both"
    if has_prev:
        return "prev"
    if has_next:
        return "next"
    return None


def _make_workflow_link_icon(kind: str) -> QIcon:
    """Small tree badge: prev / next / both workflow links."""
    size = 16
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    color = QColor(COLOR_ACCENT)
    pen = QPen(color)
    pen.setWidthF(1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)

    mid_y = size / 2
    if kind == "prev":
        painter.drawLine(int(size * 0.72), int(mid_y), int(size * 0.28), int(mid_y))
        painter.drawLine(int(size * 0.28), int(mid_y), int(size * 0.46), int(size * 0.32))
        painter.drawLine(int(size * 0.28), int(mid_y), int(size * 0.46), int(size * 0.68))
    elif kind == "next":
        painter.drawLine(int(size * 0.28), int(mid_y), int(size * 0.72), int(mid_y))
        painter.drawLine(int(size * 0.72), int(mid_y), int(size * 0.54), int(size * 0.32))
        painter.drawLine(int(size * 0.72), int(mid_y), int(size * 0.54), int(size * 0.68))
    else:
        painter.drawLine(int(size * 0.22), int(mid_y), int(size * 0.78), int(mid_y))
        painter.drawLine(int(size * 0.22), int(mid_y), int(size * 0.38), int(size * 0.34))
        painter.drawLine(int(size * 0.22), int(mid_y), int(size * 0.38), int(size * 0.66))
        painter.drawLine(int(size * 0.78), int(mid_y), int(size * 0.62), int(size * 0.34))
        painter.drawLine(int(size * 0.78), int(mid_y), int(size * 0.62), int(size * 0.66))
    painter.end()
    return QIcon(pm)


def _tree_link_tooltip(payload: dict, *, id_to_name: dict[str, str]) -> str:
    parts: list[str] = []
    prev_text = _format_workflow_links(payload.get("prev") or [], id_to_name=id_to_name)
    next_text = _format_workflow_links(payload.get("next") or [], id_to_name=id_to_name)
    if prev_text:
        parts.append("上一步：\n" + prev_text)
    if next_text:
        parts.append("下一步：\n" + next_text)
    return "\n\n".join(parts)


def _apply_tree_link_decor(
    leaf: QTreeWidgetItem,
    payload: dict,
    *,
    id_to_name: dict[str, str],
    icons: dict[str, QIcon],
) -> None:
    kind = _link_icon_kind(payload)
    if kind is None:
        return
    leaf.setIcon(0, icons[kind])
    tip = _tree_link_tooltip(payload, id_to_name=id_to_name)
    if tip:
        leaf.setToolTip(0, tip)


def _detail_css() -> str:
    return f"""
    body {{
      font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
      font-size: 13px;
      line-height: 1.55;
      color: {COLOR_TEXT};
      margin: 0;
      padding: 6px 10px 14px 10px;
    }}
    h1 {{
      font-size: 16px;
      font-weight: 700;
      margin: 2px 0 10px 0;
      padding-bottom: 6px;
      border-bottom: 2px solid #2c3e50;
    }}
    h2 {{
      font-size: 13px;
      font-weight: 700;
      margin: 16px 0 6px 0;
      padding: 3px 0 3px 8px;
      border-left: 4px solid {COLOR_ACCENT};
      background: #f4f8fb;
    }}
    p {{ margin: 6px 0; }}
    ul {{ margin: 4px 0 8px 1.2em; padding: 0; }}
    li {{ margin: 3px 0; }}
    .formula-box {{
      font-family: Consolas, "Courier New", monospace;
      font-size: 13px;
      font-weight: 600;
      white-space: pre-wrap;
      background: {COLOR_PANEL_ALT};
      border: 1px solid {COLOR_BORDER_MED};
      border-left: 4px solid {COLOR_ACCENT};
      border-radius: 4px;
      padding: 10px 12px;
      margin: 4px 0 12px 0;
    }}
    .callout {{
      background: #eef6fc;
      border: 1px solid #a8c8e0;
      border-radius: 4px;
      padding: 8px 10px;
      margin: 0 0 10px 0;
      font-size: 12px;
    }}
    .note {{
      background: #fff8e8;
      border: 1px solid #e0c080;
      border-radius: 4px;
      padding: 8px 10px;
      margin: 10px 0;
      font-size: 12px;
    }}
    .links {{
      background: #fafbfc;
      border: 1px solid #dde3ea;
      border-radius: 4px;
      padding: 8px 10px;
      margin: 0 0 10px 0;
    }}
    .links h2 {{
      margin-top: 0;
      background: transparent;
      border-left-color: #7f8c8d;
    }}
    a.formula-link {{
      color: {COLOR_ACCENT};
      text-decoration: none;
      font-weight: 600;
    }}
    a.formula-link:hover {{ text-decoration: underline; }}
    .blurb {{ color: {COLOR_TEXT_MUTED}; font-weight: 400; }}
    table.defaults {{
      border-collapse: collapse;
      width: 100%;
      margin: 6px 0 10px 0;
      font-size: 12px;
    }}
    table.defaults th, table.defaults td {{
      border: 1px solid #c5cdd6;
      padding: 5px 8px;
      text-align: left;
      vertical-align: top;
    }}
    table.defaults thead th {{
      background: #2c3e50;
      color: #fff;
      font-weight: 600;
    }}
    table.defaults tbody th {{
      background: #f0f3f6;
      font-weight: 600;
      white-space: nowrap;
      width: 22%;
    }}
    table.defaults td.meaning {{
      color: {COLOR_TEXT_MUTED};
      width: 48%;
    }}
    table.defaults tbody tr:nth-child(even) td {{ background: #fafbfc; }}
    .ref {{
      font-size: 12px;
      color: {COLOR_TEXT_MUTED};
      font-style: italic;
    }}
    .empty {{ color: {COLOR_TEXT_MUTED}; padding: 24px 8px; }}
    """


def _link_list_html(
    entries: list,
    *,
    id_to_name: dict[str, str],
    known_ids: set[str],
) -> str:
    if not entries:
        return ""
    items: list[str] = []
    for entry in entries:
        iid = _link_entry_id(entry)
        if not iid:
            continue
        name = id_to_name.get(iid, iid)
        blurb = _link_entry_blurb(entry)
        if iid in known_ids:
            label = f'<a class="formula-link" href="formula:{_esc(iid)}">{_esc(name)}</a>'
        else:
            label = _esc(name)
        if blurb:
            items.append(f"<li>{label} <span class='blurb'>— {_esc(blurb)}</span></li>")
        else:
            items.append(f"<li>{label}</li>")
    if not items:
        return ""
    return "<ul>" + "".join(items) + "</ul>"


def _parse_param_meta(raw: object) -> dict[str, dict[str, str]]:
    """Load default_param_meta (or legacy default_param_meanings) → {key: {label, meaning}}."""
    out: dict[str, dict[str, str]] = {}
    if not isinstance(raw, dict):
        return out
    for key, val in raw.items():
        k = str(key)
        if isinstance(val, dict):
            label = str(val.get("label") or k).strip()
            meaning = str(val.get("meaning") or val.get("note") or "").strip()
        else:
            label = k
            meaning = str(val).strip()
        out[k] = {"label": label, "meaning": meaning}
    return out


def _normalize_defaults(
    defaults: object,
    *,
    param_meta: dict[str, dict[str, str]] | None = None,
) -> list[tuple[str, str, str]]:
    """Return rows (display_label, value, meaning) for the defaults table."""
    meta = param_meta or {}
    if not isinstance(defaults, dict) or not defaults:
        return []
    rows: list[tuple[str, str, str]] = []
    for key, raw in defaults.items():
        key_s = str(key)
        info = meta.get(key_s, {})
        label = str(info.get("label") or key_s)
        meaning = str(info.get("meaning") or "")
        if isinstance(raw, dict):
            value = str(raw.get("value", ""))
            override_label = str(raw.get("label") or raw.get("symbol") or "").strip()
            override_meaning = str(raw.get("meaning") or raw.get("note") or "").strip()
            if override_label:
                label = override_label
            if override_meaning:
                meaning = override_meaning
        else:
            value = str(raw)
        rows.append((label, value, meaning))
    return rows


def _param_table_html(
    rows: list[tuple[str, str, str]],
    *,
    headers: tuple[str, str, str] = ("参数", "取值", "说明（物理含义）"),
) -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        f"<th>{_esc(name)}</th>"
        f"<td>{_esc(value)}</td>"
        f"<td class='meaning'>{_esc(meaning) if meaning else '—'}</td>"
        "</tr>"
        for name, value, meaning in rows
    )
    return (
        f'<table class="defaults"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def _defaults_table_html(
    defaults: object,
    *,
    param_meta: dict[str, dict[str, str]] | None = None,
) -> str:
    return _param_table_html(_normalize_defaults(defaults, param_meta=param_meta))


def format_formula_detail_html(
    payload: dict,
    *,
    id_to_name: dict[str, str],
    known_ids: set[str],
    param_meta: dict[str, dict[str, str]] | None = None,
    title_fallback: str = "",
) -> str:
    """HTML detail pane for one formula / parameter entry."""
    title = str(payload.get("name") or payload.get("title") or title_fallback or "条目")
    formula = _formula_display_text(payload)
    meta = param_meta or {}
    is_global_param = (
        payload.get("value") is not None
        and not payload.get("display")
        and not payload.get("latex")
        and not payload.get("defaults")
    )

    parts: list[str] = [f"<h1>{_esc(title)}</h1>"]

    if is_global_param:
        # Prefer explicit label on the parameter entry if present
        display_name = str(payload.get("label") or title)
        meaning = str(payload.get("purpose") or "").strip()
        parts.append("<h2>默认参数</h2>")
        parts.append(
            _param_table_html([(display_name, str(payload.get("value")), meaning)])
        )
    else:
        parts.append(f'<div class="formula-box">{_esc(formula)}</div>')

    prev_html = _link_list_html(
        payload.get("prev") or [], id_to_name=id_to_name, known_ids=known_ids
    )
    next_html = _link_list_html(
        payload.get("next") or [], id_to_name=id_to_name, known_ids=known_ids
    )
    if prev_html or next_html:
        parts.append('<div class="links">')
        if prev_html:
            parts.append("<h2>← 上一步（来自）</h2>")
            parts.append(prev_html)
        if next_html:
            parts.append("<h2>下一步 →</h2>")
            parts.append(next_html)
        parts.append("</div>")

    for key, heading in (
        ("purpose", "目的"),
        ("principle", "原理"),
        ("usage", "用法"),
        ("gui", "GUI 操作"),
    ):
        if key == "purpose" and is_global_param:
            continue  # already shown as table meaning
        raw = str(payload.get(key) or "").strip()
        if raw:
            parts.append(f"<h2>{_esc(heading)}</h2>")
            parts.append(f"<p>{_md_bold_to_html(raw)}</p>")

    if payload.get("defaults"):
        parts.append("<h2>默认参数</h2>")
        parts.append(_defaults_table_html(payload["defaults"], param_meta=meta))

    notes = str(payload.get("notes") or "").strip()
    if notes:
        parts.append(
            '<div class="note"><h2 style="margin-top:0;border-left-color:#d4a017;">补充</h2>'
        )
        parts.append(f"<p>{_md_bold_to_html(notes)}</p></div>")

    if payload.get("reference"):
        parts.append(f'<p class="ref">文献：{_esc(str(payload["reference"]))}</p>')

    body = "".join(parts)
    return (
        f"<html><head><meta charset='utf-8'><style>{_detail_css()}</style></head>"
        f"<body>{body}</body></html>"
    )


def format_global_params_table_html(data: dict) -> str:
    """Overview table for the「全局参数」folder."""
    params = list(data.get("parameters") or [])
    rows = [
        (
            str(p.get("name") or ""),
            str(p.get("value") or ""),
            _strip_md_bold(str(p.get("purpose") or "")),
        )
        for p in params
        if p.get("name")
    ]
    parts = [
        "<h1>全局参数</h1>",
        "<p>GUI / imodel 常用默认值；点击左侧条目查看用法与上下游链接。</p>",
        "<h2>默认参数</h2>",
        _param_table_html(rows),
    ]
    body = "".join(parts)
    return (
        f"<html><head><meta charset='utf-8'><style>{_detail_css()}</style></head>"
        f"<body>{body}</body></html>"
    )


def format_formula_overview_html(data: dict) -> str:
    """Landing page when a step folder (not a leaf) is selected."""
    overview = str(data.get("workflow_overview") or "").strip()
    summary = format_workflow_summary_short()
    parts = [
        "<h1>公式与参数查阅</h1>",
        '<div class="callout">',
        f"<p><b>正演链摘要</b></p><p>{_esc(summary)}</p>",
        "</div>",
        '<div class="note">',
        f"<p><b>H–Vp 两条轨</b></p><p>{_esc(HVP_SHARED_INTERFACE_NOTE)}</p>",
        "</div>",
    ]
    if overview:
        parts.append("<h2>工作流总览</h2>")
        parts.append(f"<p>{_md_bold_to_html(overview)}</p>")
    parts.append(
        "<p>请在左侧目录选择公式条目。带箭头图标的条目有上下游链接，"
        "详情中可点击跳转。</p>"
    )
    body = "".join(parts)
    return (
        f"<html><head><meta charset='utf-8'><style>{_detail_css()}</style></head>"
        f"<body>{body}</body></html>"
    )


class FormulaReferenceDialog(QDialog):
    def __init__(self, *, focus_id: str | None = None) -> None:
        super().__init__(None)
        self.setModal(False)
        self.setWindowTitle("公式与参数查阅")
        self.resize(920, 680)
        self.setMinimumSize(700, 520)
        apply_dialog_style(self)
        self.setStyleSheet(self.styleSheet() + _FORMULA_EXTRA_QSS)

        data = _load_formulas()
        self._data = data
        self._param_meta = _parse_param_meta(
            data.get("default_param_meta") or data.get("default_param_meanings")
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        body = QHBoxLayout()
        body.setSpacing(10)
        tree_col = QVBoxLayout()
        tree_col.setSpacing(4)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabel("目录")
        self._tree.setMinimumWidth(240)
        tree_col.addWidget(self._tree, stretch=1)
        legend = QLabel("图标：← 仅有上一步　→ 仅有下一步　↔ 双向链接")
        legend.setObjectName("LipFormulaHint")
        legend.setWordWrap(True)
        tree_col.addWidget(legend)
        body.addLayout(tree_col, stretch=1)

        self._link_icons = {
            "prev": _make_workflow_link_icon("prev"),
            "next": _make_workflow_link_icon("next"),
            "both": _make_workflow_link_icon("both"),
        }

        self._browser = QTextBrowser()
        self._browser.setObjectName("LipFormulaBrowser")
        self._browser.setOpenExternalLinks(False)
        self._browser.setOpenLinks(False)
        self._browser.anchorClicked.connect(self._on_anchor)
        body.addWidget(self._browser, stretch=2)
        root.addLayout(body, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = primary_button("关闭")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        self._items_by_id: dict[str, QTreeWidgetItem] = {}
        self._payload: dict[str, dict] = {}
        self._id_to_name: dict[str, str] = {}
        self._populate(data)
        self._tree.currentItemChanged.connect(self._on_select)
        self._browser.setHtml(format_formula_overview_html(data))
        if self._tree.topLevelItemCount():
            self._tree.setCurrentItem(self._tree.topLevelItem(0))

        if focus_id and focus_id in self._items_by_id:
            self._tree.setCurrentItem(self._items_by_id[focus_id])

    def _populate(self, data: dict) -> None:
        self._id_to_name.clear()
        step_blocks: list[tuple[str, list[dict]]] = []
        for step in data.get("steps", []):
            title = step.get("title", step.get("id", ""))
            items = list(step.get("items", []))
            step_blocks.append((str(title), items))
            for item in items:
                iid = str(item.get("id", ""))
                if iid:
                    self._id_to_name[iid] = str(item.get("name") or iid)

        params = list(data.get("parameters", []))
        for p in params:
            name = str(p.get("name", ""))
            if name:
                self._id_to_name[f"param_{name}"] = name

        for title, items in step_blocks:
            step_item = QTreeWidgetItem([title])
            self._tree.addTopLevelItem(step_item)
            for item in items:
                iid = str(item.get("id", ""))
                leaf = QTreeWidgetItem([item.get("name", iid)])
                step_item.addChild(leaf)
                self._items_by_id[iid] = leaf
                self._payload[iid] = item
                _apply_tree_link_decor(
                    leaf, item, id_to_name=self._id_to_name, icons=self._link_icons
                )
            step_item.setExpanded(True)

        if params:
            p_top = QTreeWidgetItem(["全局参数"])
            self._tree.addTopLevelItem(p_top)
            for p in params:
                name = p.get("name", "")
                leaf = QTreeWidgetItem([name])
                p_top.addChild(leaf)
                iid = f"param_{name}"
                self._items_by_id[iid] = leaf
                self._payload[iid] = p
                _apply_tree_link_decor(
                    leaf, p, id_to_name=self._id_to_name, icons=self._link_icons
                )
            p_top.setExpanded(True)

    def _on_anchor(self, url: QUrl) -> None:
        href = url.toString()
        if href.startswith("formula:"):
            self._go_to_id(href[len("formula:") :])

    def _go_to_id(self, iid: str) -> None:
        item = self._items_by_id.get(iid)
        if item is not None:
            self._tree.setCurrentItem(item)

    def _on_select(self, current: QTreeWidgetItem | None, _prev) -> None:
        if current is None:
            return
        iid = None
        for key, item in self._items_by_id.items():
            if item is current:
                iid = key
                break
        if iid is None:
            if current.text(0) == "全局参数":
                self._browser.setHtml(format_global_params_table_html(self._data))
            else:
                self._browser.setHtml(format_formula_overview_html(self._data))
            return
        payload = self._payload.get(iid, {})
        self._browser.setHtml(
            format_formula_detail_html(
                payload,
                id_to_name=self._id_to_name,
                known_ids=set(self._items_by_id),
                param_meta=self._param_meta,
                title_fallback=current.text(0),
            )
        )


def show_formula_reference(
    _parent: QWidget | None = None, *, focus_id: str | None = None
) -> None:
    show_modeless_dialog(FormulaReferenceDialog(focus_id=focus_id), singleton=True)
