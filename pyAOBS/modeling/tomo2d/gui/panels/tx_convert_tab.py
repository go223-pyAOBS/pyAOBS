"""tx.in → tomo2d 转换页（常用展开，震相可折叠）。"""

from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from ...param_hints import apply_param_tooltip
from ..services.obs_stations import parse_station_lis
from ..services.paths import resolve_work_dir
from ..services.tt_plot_data import build_obs_catalog, load_tx_in_picks, picks_to_arrays
from ..services.workflow import resolve_tx_convert_paths
from ..state.form_state import FormState
from ..widgets.field_form import CollapsibleSection, FieldFormTab
from ..widgets.obs_check_list import ObsCheckList

_CORE = [
    ("tx.station_lis", "station.lis", "station.lis", "open"),
    ("tx.tx_in", "tx.in（可多个）", "tx.in", "multi_open"),
    ("tx.data_out", "输出 ttimes.dat", "ttimes.dat", "save"),
    ("tx.geom_out", "输出 geom.dat", "geom.dat", "save"),
]

_PHASE = [
    (
        "_grid2",
        [
            ("tx.refr_phases", "折射震相 (→0)", "1", "text"),
            ("tx.refr_mult_phases", "折射台侧多次 (→4)", "", "text"),
            ("tx.refl_phases", "反射震相 (→1)", "11,12", "text"),
            ("tx.refl_mult_phases", "反射台侧多次 (→5)", "", "text"),
            ("tx.water_phases", "直达水波 (→2)", "", "text"),
            ("tx.mult_phases", "水柱多次 (→3)", "", "text"),
            ("tx.psp_phases", "折合 PSP (→6)", "", "text"),
        ],
    ),
]

_SECTIONS = [
    ("常用（输入 / 输出）", _CORE, True),
    ("震相筛选", _PHASE, False),
]


class TxConvertTab(FieldFormTab):
    plot_tx_in_requested = Signal()
    plot_ttimes_requested = Signal()

    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__(
            state,
            sections=_SECTIONS,
            preview_text="预览 tx 转换",
            run_text="运行 tx 转换",
            parent=parent,
        )
        if not state.has("tx.obs_ids"):
            state.set("tx.obs_ids", "")

        self._obs_timer = QTimer(self)
        self._obs_timer.setSingleShot(True)
        self._obs_timer.setInterval(400)
        self._obs_timer.timeout.connect(self.refresh_obs_list)

        obs_sec = CollapsibleSection("选择 OBS（与预览 tx.in 同一列表）", expanded=True)
        self.obs_list = ObsCheckList(
            state=state,
            persist_key="tx.obs_ids",
            heading="station.lis 号 · 与预览窗左侧相同；勾选写入转换",
        )
        self.obs_list.list_obs.setMinimumHeight(120)
        self.obs_list.list_obs.setMaximumHeight(220)
        apply_param_tooltip(self.obs_list, "tx.obs_ids")
        obs_sec.body_layout.addWidget(self.obs_list)
        btn_refresh = QPushButton("刷新列表")
        btn_refresh.clicked.connect(self.refresh_obs_list)
        apply_param_tooltip(btn_refresh, "tx.obs_ids")
        obs_sec.body_layout.addWidget(btn_refresh)
        self.insert_widget_before_actions(obs_sec)

        bar = QWidget()
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(0, 0, 0, 0)
        btn_tx = QPushButton("预览 tx.in…")
        btn_tx.clicked.connect(self.plot_tx_in_requested.emit)
        apply_param_tooltip(btn_tx, "btn.plot_tx_in")
        btn_tt = QPushButton("预览 ttimes.dat…")
        btn_tt.clicked.connect(self.plot_ttimes_requested.emit)
        apply_param_tooltip(btn_tt, "btn.plot_ttimes")
        hl.addWidget(btn_tx)
        hl.addWidget(btn_tt)
        hl.addStretch(1)
        self.insert_widget_before_actions(bar)

        self._wire_path_watchers()
        QTimer.singleShot(0, self.refresh_obs_list)

    def _wire_path_watchers(self) -> None:
        station = self._path_rows.get("tx.station_lis")
        if station is not None:
            station.edit.editingFinished.connect(self.refresh_obs_list)
            station.edit.textChanged.connect(self._obs_timer.start)
        tx_row = self._multi_path_rows.get("tx.tx_in")
        if tx_row is not None:
            tx_row.edit.textChanged.connect(self._obs_timer.start)

    def on_state_pushed(self) -> None:
        self.refresh_obs_list()

    def pull(self) -> None:
        super().pull()

    def refresh_obs_list(self) -> None:
        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
            station, txins, _d, _g = resolve_tx_convert_paths(self.state, work)
        except Exception as e:
            self.obs_list.set_records([], status=f"无法解析路径：{e}")
            return
        if not station.is_file():
            self.obs_list.set_records(
                [], status=f"station.lis 不存在：{station.name}"
            )
            return
        missing = [p for p in txins if not p.is_file()]
        if not txins or missing:
            names = ", ".join(p.name for p in (missing or txins)[:3]) or "tx.in"
            self.obs_list.set_records([], status=f"tx.in 不存在：{names}")
            return
        try:
            stations = parse_station_lis(station)
            picks = load_tx_in_picks(txins)
            catalog = build_obs_catalog(picks_to_arrays(picks), stations)
        except Exception as e:
            self.obs_list.set_records([], status=f"读取失败：{e}")
            return
        n_id = sum(1 for r in catalog if r.get("obs_id") is not None)
        n_miss = len(catalog) - n_id
        status = f"tx.in {len(catalog)} 个 OBS（已匹配 station.lis {n_id}"
        if n_miss:
            status += f" · 未匹配 {n_miss}"
        status += "）· 与预览 tx.in 同一列表"
        self.obs_list.set_records(catalog, status=status)
