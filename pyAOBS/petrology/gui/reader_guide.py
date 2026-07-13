"""读者指南：工作流与各功能说明（仅「帮助 → 读者指南」非模态窗）。"""

from __future__ import annotations

import html

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout

from petrology.reference.kkhs02_workflow import (
    HVP_SHARED_INTERFACE_NOTE,
    HVP_TRACK_COMPARE_ROWS,
    OBS_ANCHOR_NOTE,
    TRACK_FIG12_LINEAR_LABEL,
    TRACK_MODERN_LABEL,
    TRACK_TERMINOLOGY,
    format_kkhs02_steps_text,
)

from .app import UI_FONT_PT
from .dialog_utils import show_modeless_dialog


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _p(text: str) -> str:
    return f"<p>{_esc(text)}</p>"


def _ul(items: list[str]) -> str:
    lis = "".join(f"<li>{item}</li>" for item in items)
    return f"<ul>{lis}</ul>"


def _section(title: str, body_html: str) -> str:
    return (
        f'<section class="sec">'
        f'<h2>{_esc(title)}</h2>'
        f"{body_html}"
        f"</section>"
    )


def _core_concept_html() -> str:
    rows = "".join(
        "<tr>"
        f"<th>{_esc(item)}</th>"
        f"<td>{_esc(linear)}</td>"
        f"<td>{_esc(modern)}</td>"
        "</tr>"
        for item, linear, modern in HVP_TRACK_COMPARE_ROWS
    )
    tracks = "".join(
        f"<li><b>{_esc(title)}</b><br/>{_esc(body)}</li>"
        for title, body in TRACK_TERMINOLOGY
    )
    return (
        '<div class="callout">'
        f"<p><b>先读这一段</b></p>"
        f"<p>{_esc(HVP_SHARED_INTERFACE_NOTE)}</p>"
        "</div>"
        "<h3>共用接口（两条轨都一样）</h3>"
        + _ul(
            [
                "Step&nbsp;4 算出壳厚 <b>H</b> 与柱平均 <b>(P̄,&nbsp;F̄)</b>",
                "Step&nbsp;1 <b>方程&nbsp;(1)</b>：V<sub>bulk</sub> = V(P̄,&nbsp;F̄) → H–Vp 图的纵轴",
                "图面约定对齐 Fig.12：χ 实线族、χ=1 且 b=10/20/30 虚线、灰色 T<sub>p</sub> 等温线、论文轴窗",
            ]
        )
        + "<h3>不同之处（只换熔融 / 厚度，不换 Vp 公式）</h3>"
        '<table class="cmp">'
        "<thead><tr>"
        "<th>项目</th><th>默认「H–Vp 图」（线性）</th><th>Modern（Katz）</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
        '<div class="note">'
        "<p><b>易混淆</b></p>"
        "<ul>"
        "<li>「和 Fig.12 一致」≠ Modern 曲线与印刷 Fig.12 <b>数值重合</b>；"
        "而是同一坐标系 + 同一方程&nbsp;(1)。</li>"
        "<li>Modern 的 F̄ 常大于线性 12%/GPa，故 Vp 往往偏高 — <b>预期行为</b>，不是方程&nbsp;(1) 算错。</li>"
        "<li>要对齐印刷图：看默认「H–Vp 图」或 validation 中的 digitized 12a，不要用 Modern 硬套。</li>"
        "</ul>"
        f"<p>{_esc(OBS_ANCHOR_NOTE)}</p>"
        "</div>"
        "<h3>三种曲线名称</h3>"
        f"<ol>{tracks}</ol>"
    )


def _steps_html() -> str:
    # Keep detailed text from single source; preserve line breaks.
    raw = format_kkhs02_steps_text()
    blocks: list[str] = []
    for para in raw.split("\n\n"):
        lines = [ln for ln in para.split("\n") if ln.strip()]
        if not lines:
            continue
        head, *rest = lines
        if head.startswith("Step "):
            body = "".join(f"<div class='step-line'>{_esc(ln)}</div>" for ln in rest)
            blocks.append(f"<div class='step'><h3>{_esc(head)}</h3>{body}</div>")
        else:
            blocks.append(_p("\n".join(lines)))
    return "".join(blocks)


def format_reader_guide_html() -> str:
    """HTML 正文（读者指南窗）。"""
    css = """
    body {
      font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
      font-size: 13px;
      line-height: 1.55;
      color: #1a1a1a;
      margin: 0;
      padding: 4px 8px 16px 8px;
    }
    h1 {
      font-size: 18px;
      font-weight: 700;
      margin: 4px 0 12px 0;
      padding-bottom: 8px;
      border-bottom: 2px solid #2c3e50;
    }
    h2 {
      font-size: 15px;
      font-weight: 700;
      margin: 22px 0 8px 0;
      padding: 4px 0 4px 8px;
      border-left: 4px solid #2980b9;
      background: #f4f8fb;
    }
    h3 {
      font-size: 13px;
      font-weight: 700;
      margin: 14px 0 6px 0;
      color: #2c3e50;
    }
    p { margin: 6px 0; }
    ul, ol { margin: 6px 0 8px 1.2em; padding: 0; }
    li { margin: 4px 0; }
    .callout {
      background: #eef6fc;
      border: 1px solid #a8c8e0;
      border-radius: 4px;
      padding: 10px 12px;
      margin: 8px 0 12px 0;
    }
    .note {
      background: #fff8e8;
      border: 1px solid #e0c080;
      border-radius: 4px;
      padding: 10px 12px;
      margin: 12px 0;
    }
    table.cmp {
      border-collapse: collapse;
      width: 100%;
      margin: 8px 0 12px 0;
      font-size: 12px;
    }
    table.cmp th, table.cmp td {
      border: 1px solid #c5cdd6;
      padding: 6px 8px;
      vertical-align: top;
      text-align: left;
    }
    table.cmp thead th {
      background: #2c3e50;
      color: #fff;
      font-weight: 600;
    }
    table.cmp tbody th {
      background: #f0f3f6;
      font-weight: 600;
      white-space: nowrap;
      width: 18%;
    }
    table.cmp tbody tr:nth-child(even) td { background: #fafbfc; }
    .step {
      border: 1px solid #dde3ea;
      border-radius: 4px;
      padding: 8px 10px;
      margin: 8px 0;
      background: #fafafa;
    }
    .step h3 { margin-top: 0; }
    .step-line {
      font-size: 12px;
      margin: 2px 0;
      padding-left: 0.5em;
      color: #333;
    }
    .sec { margin-bottom: 4px; }
    code, .mono {
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
      background: #f0f0f0;
      padding: 0 3px;
      border-radius: 2px;
    }
    """
    body = (
        "<h1>LIP Petrology — 读者指南</h1>"
        + _section("1. Modern、Fig.12 与方程 (1)", _core_concept_html())
        + _section(
            "2. 本程序做什么",
            _p(
                "按 KKHS02 四步正演链，把源区参数（Tp、χ、Φ、岩性）正演为壳厚 H "
                "与 bulk 速度 V_bulk（方程 (1)），并与观测锚点 H_obs、V_LC 比较。"
                "典型场景：Greenland 类 LIP，检验「多热的地幔、多宽的柱、多少辉石岩」"
                "能否同时闭合厚度与下地壳速度。"
            ),
        )
        + _section("3. KKHS02 正演四步（Step 1→4）", _steps_html())
        + _section(
            "4. 推荐工作流",
            "<ol>"
            "<li>「观测锚点」填入 H、V_LC（或「导入观测」）。</li>"
            "<li>「正演」— 单组 Tp、χ、Φ 试算数量级。</li>"
            "<li>「Tp–χ 扫描」— Φ 已定，找可行区（左 H–Vp + 右可行热图）。</li>"
            "<li>「Φ 扫描」— Φ 不确定时，多档 Φ 各做 Tp–χ。</li>"
            "<li>「预设对比」— 比较不同 pymelt 岩性预设。</li>"
            "<li>「轨对比」— 需先有扫描；对比 Modern 与默认 H–Vp 线性轨。</li>"
            "<li>「验证」— 模式图、Katz 图件、经典曲线；「公式」查逐步说明。</li>"
            "</ol>"
            "<p>悬停参数标签与工具栏按钮可看简短 toolTip。</p>",
        )
        + _section(
            "5. 观测锚点（左列）",
            _ul(
                [
                    "<b>H</b> — 观测壳厚 (km)；|ΔH| 与 §4 方程 (7)(10)；H–Vp <b>横轴</b>。"
                    "（不是论文 Step&nbsp;1；Step&nbsp;1 = 方程&nbsp;(1) 的 V<sub>bulk</sub>。）",
                    "<b>V_LC</b> — 下地壳观测 Vp；Step&nbsp;2 上界；H–Vp <b>纵轴</b>。",
                    "<b>b</b> — Langmuir 被动厚度 (km)；b=0 为纯主动柱。"
                    "kinzler_linear 要求 P₀&nbsp;&gt;&nbsp;b/30（约 Tp&nbsp;≳&nbsp;1150+3.33·b）。",
                    "<b>Φ</b> — 辉石岩体积分数；Tp–χ 扫描时固定。",
                    "<b>|ΔH|</b> — 可行判据 |H<sub>model</sub>−H<sub>obs</sub>| ≤ 容差。",
                ]
            ),
        )
        + _section(
            "6. 模型参数（中列）",
            _ul(
                [
                    "<b>Tp、χ</b> — 正演单点潜在温度与柱形状参数。",
                    "<b>engine</b> — <span class='mono'>reebox</span>（Katz/pyMelt+REEBOX，推荐 LIP）"
                    "或 <span class='mono'>kinzler_linear</span>（§4 线性几何，较快）。",
                    "<b>Tp scan / step / χ list</b> — 扫描网格。",
                    "<b>快速扫描</b> — 跳过 BurnMan；关则更准更慢。",
                    "<b>bulk 界精修</b> — 同时要求 H 与 Step-2 V<sub>bulk</sub> 界。",
                    "<b>H–Vp 显示</b> — 默认「H–Vp 图」= §4 线性 + 方程&nbsp;(1)；"
                    "可选 Modern / 双轨（见第&nbsp;1&nbsp;节）。",
                    "<b>ΔVp、厚壳 H&gt;</b> — Step-2 读图竖段规则。",
                ]
            ),
        )
        + _section(
            "7. 岩性 reebox（右列）",
            _p("engine=reebox 且 backend=pymelt 时生效。")
            + _ul(
                [
                    "<b>preset</b> — 如 greenland_kg1，填充橄榄石/辉石岩端元与 H₂O。",
                    "<b>per / pyr</b> — pMELTS 岩性键；Φ 控制混合比。",
                    "「岩性说明…」— preset/端元组成与选用时机。",
                    "native backend — Kinzler 线性端元，忽略 preset。",
                ]
            ),
        )
        + _section(
            "8. 主功能",
            "<h3>正演</h3>"
            + _p(
                "对当前 Tp、χ、Φ、b 算一条熔融柱。"
                "主图：H、P₀、Pf、Fmax、F̄；下方 T–P、F–T 与模式图同源。"
            )
            + "<h3>Tp–χ 扫描</h3>"
            + _p(
                "在 Tp×χ 网格上正演，标记满足 |ΔH| 与（可选）bulk Vp 界的点。"
                "左 H–Vp + 右可行热图；「扫描结果」Tab 有表。"
            )
            + "<h3>Φ 扫描</h3>"
            + _p(
                "对 Φ=0…0.25 各跑 Tp–χ；横轴 Φ、纵轴 Vp excess。"
                "Φ 已有估计时通常直接做 Tp–χ 即可。"
            )
            + "<h3>预设对比</h3>"
            + _p("多 pymelt 预设（含 native）各跑 Tp–χ，比可行点数与最佳 |ΔH|。")
            + "<h3>验证菜单</h3>"
            + _ul(
                [
                    "主动/被动熔融模式图 — 含 H–Vp（线性轨 vs Modern）等面板。",
                    "分离结晶 Fig.2 / Fig.5 — Step&nbsp;3：结晶路径与 ΔVp(F)"
                    "（生产默认裸算 ≈0.13–0.15 km/s → GUI「ΔVp」与 Step-2 竖段）。",
                    "Katz (2003) Fig.1–6 — 橄榄石端元熔融验证。",
                    "经典熔融理论曲线 — 固相线、绝热线、F(P) 等。",
                ]
            )
            + _p("以上窗口均为非模态，打开后仍可操作主界面。")
            + "<h3>轨对比 / 导入 / 公式</h3>"
            + _ul(
                [
                    f"<b>轨对比</b> — {_esc(TRACK_MODERN_LABEL)} vs "
                    f"{_esc(TRACK_FIG12_LINEAR_LABEL)}（需先有扫描）。",
                    "<b>导入观测</b> — imodel JSON 或 transect CSV。",
                    "<b>公式</b> — 左目录 + 右 HTML 详情（与读者指南同风格排版；"
                    "可点上下游链接跳转）。",
                ]
            ),
        )
        + _section(
            "9. 输出区 Tab",
            _ul(
                [
                    "<b>扫描结果</b> — 最近一次可行点列表。",
                    "<b>观测读图</b> — Step-2 bulk 界与导入摘要。",
                    "<b>日志</b> — 正演、扫描、导入记录。",
                ]
            ),
        )
    )
    return f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"


def format_reader_guide() -> str:
    """纯文本回退（测试 / 复制用）。"""
    from petrology.reference.kkhs02_workflow import format_track_terminology_text

    parts = [
        "【1. Modern、Fig.12 与方程 (1)】",
        format_track_terminology_text(),
        "",
        "【2. 本程序做什么】",
        "LIP Petrology 按 KKHS02 四步正演链，把源区参数 "
        "（Tp、χ、Φ、岩性）正演为壳厚 H 与 bulk 速度 V_bulk（方程 (1)），"
        "并与观测锚点 H_obs、V_LC 比较。",
        "",
        "【3. KKHS02 正演四步（Step 1→4）】",
        format_kkhs02_steps_text(),
    ]
    return "\n".join(parts)


class ReaderGuideDialog(QDialog):
    """非模态读者指南（菜单「帮助 → 读者指南」/ F1）。"""

    def __init__(self) -> None:
        super().__init__(None)
        self.setWindowTitle("读者指南 — LIP Petrology")
        self.resize(780, 620)
        self.setMinimumSize(560, 420)
        self.setModal(False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        body = QTextBrowser(self)
        body.setOpenExternalLinks(False)
        body.setFont(QFont("Microsoft YaHei UI", UI_FONT_PT))
        body.setHtml(format_reader_guide_html())
        lay.addWidget(body)


def show_reader_guide(_parent=None) -> None:
    show_modeless_dialog(ReaderGuideDialog(), singleton=True)
