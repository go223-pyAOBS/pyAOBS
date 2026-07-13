"""
iphase 命令行入口

用法:
  python -m pyAOBS.visualization.iphase select tx.in -o tx.out --phases 1 2 3
  python -m pyAOBS.visualization.iphase combine tx1.in tx2.in tx3.in -o1 tx1.out -o2 tx2.out --ip1 1 --ip2 2 --ip3 3 --ip4 10 --ip5 11
  python -m pyAOBS.visualization.iphase info tx.in
"""

from __future__ import annotations

import argparse
import sys

try:
    from .io_tx import read_tx, write_tx
    from .phase_filter import select_phases
    from .phase_combine import combine_ppp_pps_pss
    from .qc_metrics import summary
except ImportError:
    from pyAOBS.visualization.iphase.io_tx import read_tx, write_tx
    from pyAOBS.visualization.iphase.phase_filter import select_phases
    from pyAOBS.visualization.iphase.phase_combine import combine_ppp_pps_pss
    from pyAOBS.visualization.iphase.qc_metrics import summary


def cmd_select(args: argparse.Namespace) -> int:
    tx_in = args.input
    tx_out = args.output
    phases = [int(x) for x in args.phases]
    ds = read_tx(tx_in)
    out, stats = select_phases(ds, phases)
    write_tx(out, tx_out)
    print(f"筛选相位 {phases} 完成: {stats['npick']} 条拾取, {stats['nshot']} 个炮点 -> {tx_out}")
    return 0


def cmd_combine(args: argparse.Namespace) -> int:
    tx1, tx2, tx3 = args.inputs
    o1, o2 = args.out1, args.out2
    res = combine_ppp_pps_pss(
        tx1, tx2, tx3,
        ip1=args.ip1, ip2=args.ip2, ip3=args.ip3,
        ip4=args.ip4, ip5=args.ip5,
    )
    write_tx(res.tx1_out, o1)
    write_tx(res.tx2_out, o2)
    print(f"匹配完成: npick_matched={res.npick_matched}, npick_unmatched={res.npick_unmatched}, nshot={res.nshot}")
    print(f"  tx1.out -> {o1}")
    print(f"  tx2.out -> {o2}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    ds = read_tx(args.input)
    s = summary(ds)
    print(f"炮点数: {s['n_shots']}")
    print(f"拾取数: {s['n_picks']}")
    print("相位统计:", s["phase_counts"])
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .iphase_gui import main as gui_main
    except ImportError:
        from pyAOBS.visualization.iphase.iphase_gui import main as gui_main
    return gui_main()


def cmd_rin_gui(args: argparse.Namespace) -> int:
    try:
        from .rin_phase_groups_gui import main as rin_gui_main
    except ImportError:
        from pyAOBS.visualization.iphase.rin_phase_groups_gui import main as rin_gui_main
    # 调用时传空 argv，避免外层子命令名被 rin_gui 的 argparse 当成 r.in 路径
    return rin_gui_main([])


def main() -> int:
    parser = argparse.ArgumentParser(prog="iphase", description="tx.in 拾取震相处理工具")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # select
    p_sel = sub.add_parser("select", help="筛选指定相位（对应 txphase.f）")
    p_sel.add_argument("input", help="输入 tx.in 路径")
    p_sel.add_argument("-o", "--output", default="tx.out", help="输出路径")
    p_sel.add_argument("--phases", nargs="+", required=True, help="要保留的相位号")
    p_sel.set_defaults(func=cmd_select)

    # combine
    p_comb = sub.add_parser("combine", help="PPP/PPS/PSS 组合（对应 txconv.f）")
    p_comb.add_argument("inputs", nargs=3, metavar=("tx1", "tx2", "tx3"), help="tx1.in tx2.in tx3.in")
    p_comb.add_argument("-o1", "--out1", default="tx1.out", help="输出1路径")
    p_comb.add_argument("-o2", "--out2", default="tx2.out", help="输出2路径")
    p_comb.add_argument("--ip1", type=int, default=1, help="PPP 相位号")
    p_comb.add_argument("--ip2", type=int, default=2, help="PPS 相位号")
    p_comb.add_argument("--ip3", type=int, default=3, help="PSS 相位号")
    p_comb.add_argument("--ip4", type=int, default=10, help="输出1相位号 (PPS-PPP)")
    p_comb.add_argument("--ip5", type=int, default=11, help="输出2相位号 (修正 PSP)")
    p_comb.set_defaults(func=cmd_combine)

    # info
    p_info = sub.add_parser("info", help="显示 tx.in 统计信息")
    p_info.add_argument("input", help="tx.in 路径")
    p_info.set_defaults(func=cmd_info)

    # gui
    p_gui = sub.add_parser("gui", help="启动 iphase 图形界面")
    p_gui.set_defaults(func=cmd_gui)

    # rin-gui
    p_rin_gui = sub.add_parser("rin-gui", help="启动 r.in 相位组专用编辑器")
    p_rin_gui.set_defaults(func=cmd_rin_gui)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
