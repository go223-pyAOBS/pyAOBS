# -*- coding: utf-8 -*-
"""
TOMO2D GUI 参数说明（中文简述）。

依据：
- modeling/tomo2d/readme.pdf（Korenaga, 2003；第 4.3 节 tt_inverse）
- modeling/tomo2d/src/gen_smesh.cc、tt_forward.cc、tt_inverse.cc、graph.cc、
  inverse.cc、bend.cc 等源文件中的用法与实现
- pyAOBS/modeling/tomo2d/tomand.py、help_docs.py（Python 封装与完整原生选项列表）

单位与坐标约定与 tomo2d 一致：水平距离与深度一般为 km；z 为海底以下深度。

各条目前缀标记（与 TomoAnd 及本 GUI 收集 kwargs 的逻辑一致）：
- 【必选】运行该工具时 tomand 要求必须提供，缺则报错或仅打印帮助。
- 【可选】留空则不传对应参数；可执行程序使用其内置默认。
- 【条件必选】仅当所选 vel_opt / grid_opt / mode / cmd_type 等分支成立时才必填。
- 【成组可选】一组参数要么全部留空（不传 -N 等），要么全部合法填写；填了其中任一项则 tomand
  会要求同组其余项齐全（见 tt_forward / tt_inverse 的六项数值）。
"""

from __future__ import annotations

# 键与 FormState / Qt 表单控件对应
TOMO2D_GUI_HINTS: dict[str, str] = {
    "bin_path": (
        "【含义】tomo2d 可执行文件所在目录（gen_smesh、tt_forward、tt_inverse 等）。\n"
        "【作用】决定 GUI/TomoAnd 从哪里启动外部程序。\n"
        "【默认】环境变量 PYAOBS_TOMO2D_BIN / TOMO2D_BIN，否则为本仓库 "
        "modeling/tomo2d/src/build-tomo2d。"
    ),
    "work_dir": (
        "【含义】运行时的工作目录（子进程 cwd；工区打开后通常等于工区根）。\n"
        "【作用】相对路径的输入/输出均相对此目录解析；运行摘要 log、runs/ 等也落在此树下。"
    ),
    "gen.vel_opt": (
        "【必选】速度场构造方式（与 grid_opt 联动）。"
        "uniform：V(z)=v0 + gradient × z；可搭配 uniform/variable 网格。"
        "zelt：从 Zelt v.in 构建（-C），须填 ilayer；"
        "**选择 zelt 时 GUI 会自动将 grid_opt 设为 zelt 并锁定**（源码要求两者同为 zelt）。"
    ),
    "gen.grid_opt": (
        "【必选】网格类型。uniform：-N/-D；variable：-X/-Z/-T；zelt：-E dx 与 -Z。"
        "vel_opt=zelt 时本项被锁定为 zelt；vel_opt=uniform 时可改。"
    ),
    "gen.v0": (
        "【条件必选：vel_opt=uniform】[-A] z=0（海底面）参考速度 v0（km/s）。"
        "速度随深度：V(z)=v0 + gradient × z（见 readme 4.1）。"
    ),
    "gen.gradient": (
        "【条件必选：vel_opt=uniform】[-B] dV/dz（km/s 每 km），即 V(z)=v0 + gradient × z。"
    ),
    "gen.v_in": (
        "【条件必选：vel_opt=zelt】[-C 前半] Zelt v.in。"
        "可选子目录（如 inputs/v.in）；表单保留相对路径。"
        "命令行 -C 只能带文件名，运行前会自动复制到 work_dir 根目录。"
    ),
    "gen.ilayer": (
        "【条件必选：vel_opt=zelt】[-C 后半] v.in 层号。"
        "默认（不勾 -S）：该层深度写入 smesh 的 topo，网格挂在该面上（OBS 作业通常是海底）。"
        "勾选挂海面 (-S) 时：该层当作海底界面，topo 写全 0，界面以上填水速、以下取 v.in 壳幔。"
    ),
    "gen.hang_sea_surface": (
        "【可选】[-S] 仅 zelt。把网格挂在海面：smesh 的 topo 全 0；"
        "ilayer 指定的 v.in 界面当海底；其上填 v_water（-Q，默认 1.5），其下按绝对深度取样 v.in。"
        "z_file 须是海面起算的绝对深度（第一点约 0），不要再用「海底以下」坐标。"
        "默认不勾：zelt 行为与原来完全相同（topo=海底、z 为海底以下）。"
    ),
    "gen.seafloor_out": (
        "【可选】[-G] **输出**海底界面（x z），须同时勾选 -S。"
        "给正演 -B / 反演 -Y 用。不必预先存在。命令行只用文件名，含子目录时写完后挪过去。"
        "莫霍仍用 refl_layer / refl_file (-F)。"
    ),
    "gen.refl_layer": (
        "【可选成对】[-F 前半] 与 refl_file 成对；二者都填时才传 -F。"
        "表示从 Zelt 模型导出该层界面到 refl_file。"
    ),
    "gen.refl_file": (
        "【可选成对】[-F 后半] **输出**反射/界面坐标文件（程序写出，**不必预先存在**）。"
        "与 refl_layer 同时填写。命令行只用文件名；若填子目录，写完后会挪到该路径。"
    ),
    "gen.nx": (
        "【条件必选：grid_opt=uniform】[-N 前半] x 方向节点数（与 nz、xmax、zmax 同组必填）。"
    ),
    "gen.nz": (
        "【条件必选：grid_opt=uniform】[-N 后半] z 方向节点数。"
    ),
    "gen.xmax": (
        "【条件必选：grid_opt=uniform】[-D 前半] 水平范围 0~xmax（km）。"
    ),
    "gen.zmax": (
        "【条件必选：grid_opt=uniform】[-D 后半] 垂向 0~zmax（km），海底以下深度。"
    ),
    "gen.x_file": (
        "【条件必选：grid_opt=variable】[-X] x 节点坐标文件（km，递增）。"
    ),
    "gen.z_file": (
        "【条件必选：grid_opt=variable 或 zelt】[-Z] z 节点深度文件（km，递增）。"
        "variable 与 x_file 同组；zelt 与 dx、v.in 联用。"
        "不勾 -S 时 z 是海底以下深度（第一点通常 0）。"
        "勾选挂海面 (-S) 时改成海面以下的绝对深度，第一点须约 0，zmax 要盖住壳底（比原来的「海底以下 zmax」大约多一个水深）。"
        "程序在 **work_dir** 下打开该路径（相对路径均相对 work_dir）；若报错 ``countLines::can't open``，"
        "多为文件不在该目录、路径写错或与界面「工作目录」不一致；**与 v.in 不同，z_file 路径可含子目录**（如 mesh/z.txt）。"
    ),
    "gen.topo_file": (
        "【可选】[-T] 变间距网格海底地形/水深；点数须与 xfile 一致（readme 示例 2）。"
    ),
    "gen.dx": (
        "【条件必选：grid_opt=zelt】[-E] 近似水平间距（km）；readme 4.1。"
    ),
    "gen.water_col": (
        "【可选】[-W] 水层厚度（gen_smesh.cc 中 wcol），默认 0。"
    ),
    "gen.v_water": (
        "【可选】[-Q] 水速，默认 1.5 km/s；写入网格头。"
    ),
    "gen.v_air": (
        "【可选】[-R] 空气层速度，默认 0.33 km/s；写入网格头。"
    ),
    "gen.zelt_dump": (
        "【可选】【条件：vel_opt=zelt】[-d] Zelt dumpNodes 输出路径前缀（生成 .dnodes/.vnodes/.cells，见 zeltform.cc）。"
        "可只填文件名（相对 work_dir），与「浏览…」等价于在工作目录下保存；浏览用于写到其它目录。"
        "uniform/variable 网格下勿填。"
    ),
    "gen.smesh_out": (
        "【GUI 运行必选】将 gen_smesh 的 stdout 慢度网格落盘（TomoAnd 的 out_file，相对 work_dir）。"
        "可手写文件名不必浏览；浏览用于写到其它目录。"
        "不填则「运行 gen_smesh」与含 gen_smesh 的 pipeline 将被禁止；预览仍可查看参数。"
        "命令行等价于 gen_smesh … > file；纯 Python 脚本调用 tomo.gen_smesh 时 out_file 仍为可选。"
    ),
    "fwd.smesh": (
        "【必选】[-M] 慢度/速度网格（sheared grid），常为 gen_smesh 输出；readme 3.1。"
    ),
    "fwd.geom": (
        "【可选】[-G] 与 syngen.cc ``read_file`` 同构的文本：首行炮点数 nsrc；每炮一行 ``s x y nrcv``，"
        "再紧跟 nrcv 行 ``r x y code t u``（geom 中 t、u 常为 0；**ttimes.dat** 同形但 t、u 为观测）。"
        "``nrcv`` 必须与 r 行数严格一致，且文件末尾不得再多 ``s``/乱行，否则 C 端易未定义行为。"
        "程序 stdout 里打印的 ``r … 非零走时`` 是 **正演结果**（``cout<<syn``），不是多读了一行输入。"
        "段错误还常见于 **接收点坐标超出 -M smesh 模型范围**。"
        "不传 -G 时仅做与网格有关的输出，不做震源-接收正演。"
    ),
    "fwd.refl_file": (
        "【可选】[-F] 反射面文件（壳内反射 / 莫霍，raytype 1）。"
        "水层 2/3 与台侧多次 4/5 的水柱底请填下方 seafloor_file (-B)；"
        "缺 -B 时 2/3/4 仍可沿用本项当海底（与旧 water_fwd 兼容）。"
        "raytype 5 必须同时有 -B 与 -F。"
    ),
    "fwd.seafloor_file": (
        "【可选】[-B] 海底界面。正演里这是水柱底（raytype 2/3/4/5），"
        "不是反演的 -B（反演 -B 仍是转换波）。"
        "只做 0/1 时留空，行为与原来完全相同。"
    ),
    "fwd.conv_file": (
        "【可选】[-X] 转换界面。raytype 6（折合 PSP）、7（PPS）、8（PSS）使用；"
        "0/1 仍是 Fermat 初至，不会被本项劫持。"
        "7/8 还要 -B 海底；单场 Vp→Vs 用 -k。反演 6 对应 -B。"
    ),
    "fwd.kappa": (
        "【可选】[-k] Vp/Vs。-k1.73 或 -k1.73/1.80（面上/面下）。"
        "单场 -M 是 Vp，内部 Vs=Vp/k。填了 vsmesh (-U) 时可不传。"
    ),
    "fwd.vsmesh": (
        "【可选】[-U] 独立 Vs 网格，须与 -M 的 Vp 同维。"
        "6/7/8 真双场：P 段读 -M，S 段读 -U。有此项时不必再靠 -k 从 Vp 造 Vs。"
    ),
    "inv.kappa": (
        "【可选】[-k] 与正演相同。单场用它从 Vp 生成 Vs；填了 vsmesh (-U) 可不传。"
        "反演 -Q 仍是 LSQR 容差。"
    ),
    "inv.vsmesh": (
        "【可选】[-U] 双场初值 Vs，须与 -M 同维。"
        "有则不再用 Vp/κ 覆盖 Vs。盖层 P 仍读 -M。"
    ),
    "fwd.xorder": (
        "【成组可选：与 zorder、clen、nintp、bend_cg_tol、bend_br_tol 同组】[-N 第1项] "
        "图论走时（GraphSolver2d）前向星在网格索引 i（水平）方向的阶数；"
        "与 zorder 一起决定从当前节点尝试松弛的邻域范围（见 graph.cc 中 ForwardStar2d）。"
        "tt_forward 源程序默认 4。"
    ),
    "fwd.zorder": (
        "【成组可选】[-N 第2项] 前向星在网格索引 k（垂向）方向的阶数；"
        "除 |Δi|≤xorder、|Δk|≤zorder 外，源码还对斜向连接做了限制，避免步长过大的对角跳点。"
    ),
    "fwd.clen": (
        "【成组可选】[-N 第3项] crit_len（km）。须严格 >0，否则整条 -N 会被可执行文件忽略。"
        "若合法：在由图论得到的折线路径上，凡线段长度超过 clen 会线性插值细分，再交给弯曲法（refine_if_long）。"
    ),
    "fwd.nintp": (
        "【成组可选】[-N 第4项] β 样条控制点数（BetaSpline2d / BendingSolver2d）；"
        "tt_forward 默认 8。"
    ),
    "fwd.bend_cg_tol": (
        "【成组可选】[-N 第5项] 弯曲 CG 容差（bend.cc）；与 inv.bend_cg_tol 同义，"
        "收集为 TomoAnd.tt_forward 的 tol1。"
    ),
    "fwd.bend_br_tol": (
        "【成组可选】[-N 第6项] Brent 容差系数（bend_brent.cc）；与 inv.bend_br_tol 同义，"
        "收集为 TomoAnd.tt_forward 的 tol2。"
    ),
    "fwd.vred": (
        "【可选】[-r] 折合速度 v0。"
    ),
    "fwd.out_ttime": (
        "【可选】把 tt_forward **标准输出**写成与 tt_inverse -G 同构的走时文件"
        "（首行 nsrc，随后 s/r，末两列为合成走时与 0.01 s 误差）。\n"
        "这不是原生 ``-T``：那个是 ``>`` + x + 折合走时的绘图格式，"
        "拿去反演会报 ``invalid nsrc``。"
    ),
    "fwd.out_ray": (
        "【可选】[-R] 输出射线路径文件。默认留空（文件大、I/O 重）；需要时再填。"
    ),
    "fwd.out_elements": (
        "【可选】[-E] 输出网格单元文件。"
    ),
    "fwd.out_obs_ttime": (
        "【可选】[-O] 写出输入观测走时。"
    ),
    "fwd.out_source": (
        "【可选】[-S] 输出震源位置文件。"
    ),
    "fwd.out_vgrid": (
        "【可选】[-I] 输出速度网格。"
        "子区输出用下方六项 west…dz 拼 -i（须六项齐全）；-n 抑制全网格输出时的空气/水层附加点。"
    ),
    "fwd.out_diff": (
        "【可选】[-D] 输出差分走时。"
    ),
    "fwd.verbose_level": (
        "【可选】留空或 0：不传 -V；填正数：自动 verbose 并传 -V[level]（readme 4.2）。"
    ),
    "fwd.do_full_refl": (
        "【可选】[-A] 反射沿界面贴面走（远偏移常是界面首波）。"
        "不勾时远偏移反射可以穿进地幔，变成初至。"
        "改的是路径，不是计算精度；勾上更耗时，且 OpenMP 正演会退回串行。"
    ),
    "fwd.verbose": "【界面已隐藏】由 verbose_level 决定：>0 时传 -V[level]。",
    "fwd.clock_file": "【可选】[-C] 时钟/used_time 文件路径（须配合 -G）。",
    "fwd.sub_west": "【成组可选：与 sub_east、south、north、dx、dz 同组】[-i 第1项] 子区西界 (km)。",
    "fwd.sub_east": "【成组可选】[-i 第2项] 东界。",
    "fwd.sub_south": "【成组可选】[-i 第3项] 南界。",
    "fwd.sub_north": "【成组可选】[-i 第4项] 北界。",
    "fwd.sub_dx": "【成组可选】[-i 第5项] 输出水平间距。",
    "fwd.sub_dz": "【成组可选】[-i 第6项] 输出垂向间距。",
    "fwd.graph_only": "【可选】[-g] 仅用图论、关闭弯曲（须配合 -G）。",
    "fwd.omit_air_water": "【可选】[-n] 全网格 -I 输出时不写空气/水层附加点（子区 -i 路径下源码不用 printAW）。",
    "damp.vel_opt": (
        "【必选】与 grid_opt 一起；gen_damp 无二者则仅打印帮助。"
        "-A 异常/正常阻尼始终必填；zelt 另加 -C v.in/ilayer，可选 -F 层界"
        "（zelt 时 grid_opt 须为 zelt）。"
    ),
    "damp.grid_opt": "【必选】uniform / variable / zelt；zelt 配套规则同 gen_smesh（vel=zelt ⇔ grid=zelt）。",
    "damp.abnormal_damp": (
        "【必选】[-A 前半] 异常区阻尼值；vel_opt=zelt 时同样需要"
        "（-C/-F 只划区，数值仍来自 -A）。"
    ),
    "damp.normal_damp": (
        "【必选】[-A 后半] 正常区阻尼值；vel_opt=zelt 时同样需要。"
    ),
    "damp.v_in": (
        "【条件必选：vel_opt=zelt】[-C 前半] v.in；规则同 gen.v_in：可含子目录，运行前 stage 到 work_dir 根。"
    ),
    "damp.ilayer": "【条件必选：vel_opt=zelt】[-C 后半] 海面层号。",
    "damp.top_layer": "【条件必选：填写层界时】[-F 前半] 与 bot_layer 成对。",
    "damp.bot_layer": "【条件必选：填写层界时】[-F 后半] 与 top_layer 成对。",
    "vcorr.mode": (
        "【必选】simple_2x2：写出 2×2 CorrelationLength2d，水平/垂直相关长度只随深度从顶到变到底"
        "（沿 x 不变），对应手工 vcorr heredoc（Lht/Lhb/Lvt/Lvb）。"
        "program：调用 gen_vcorr 二进制，用 -A 异常区/正常区划区（不是顶/底）。"
    ),
    "vcorr.Lht": (
        "【条件必选：mode=simple_2x2】水平相关长度顶部 Lh(zmin)。沿 x 两列相同，深度上线性插值到 Lhb。"
    ),
    "vcorr.Lhb": "【条件必选：mode=simple_2x2】水平相关长度底部 Lh(zmax)。",
    "vcorr.Lvt": "【条件必选：mode=simple_2x2】垂直相关长度顶部 Lv(zmin)。",
    "vcorr.Lvb": "【条件必选：mode=simple_2x2】垂直相关长度底部 Lv(zmax)。",
    "vcorr.xmin": "【可选：mode=simple_2x2】测线起点 x（km），默认 0。",
    "vcorr.xmax": (
        "【必选】simple_2x2：测线终点 x（km）。program + uniform：-D 的 xmax。"
    ),
    "vcorr.zmin": "【可选：mode=simple_2x2】深度起点（km），默认 0。",
    "vcorr.zmax": (
        "【必选】simple_2x2：深度终点（km）。program + uniform：-D 的 zmax。"
    ),
    "vcorr.out_file": (
        "【GUI 运行：simple_2x2 必选】写出的二维相关长度文件，供 tt_inverse -CV。"
        "program 模式下将 gen_vcorr 的 stdout 落入该路径。"
    ),
    "vcorr.vel_opt": (
        "【条件必选：mode=program】与 grid_opt 一起；gen_vcorr 无二者则仅打印帮助。"
        "-A 四段相关长度始终必填；zelt 另加 -C，可选 -F（zelt 时 grid_opt 须为 zelt）。"
        "这是异常区/正常区，不要把 Lht/Lhb 填进 -A。"
    ),
    "vcorr.grid_opt": "【必选】uniform / variable / zelt；zelt 配套规则同 gen_smesh。",
    "vcorr.abnormal_h": (
        "【必选】[-A 第1项] 异常区水平相关长度；zelt 时同样需要（-C/-F 只划区）。"
    ),
    "vcorr.abnormal_v": "【必选】[-A 第2项] 异常区垂直相关长度；zelt 时同样需要。",
    "vcorr.normal_h": "【必选】[-A 第3项] 正常区水平相关长度；zelt 时同样需要。",
    "vcorr.normal_v": "【必选】[-A 第4项] 正常区垂直相关长度；zelt 时同样需要。",
    "vcorr.v_in": (
        "【条件必选：vel_opt=zelt】[-C 前半] v.in；规则同 gen.v_in：可含子目录，运行前 stage 到 work_dir 根。"
    ),
    "vcorr.ilayer": "【条件必选：vel_opt=zelt】[-C 后半] 海面层号。",
    "vcorr.top_layer": "【条件必选：填写层界时】[-F 前半] 与 bot_layer 成对。",
    "vcorr.bot_layer": "【条件必选：填写层界时】[-F 后半] 与 top_layer 成对。",
    "dcorr.mode": (
        "【必选】uniform：沿 x 均匀 Lh；zelt：Zelt 层划区（与 gen_vcorr 水平相关同一套几何）；"
        "from_vcorr：在反射节点上取二维 vcorr 的水平分量。"
    ),
    "dcorr.lh": "【条件必选：mode=uniform】[-A] 沿测线的界面相关长度 Lh（km）。",
    "dcorr.xmin": "【条件必选：mode=uniform】[-D 前半] 起点 x（km）。",
    "dcorr.xmax": "【条件必选：mode=uniform】[-D 后半] 终点 x（km）。",
    "dcorr.nx": "【可选】[-N] 沿 x 的点数；空则只写两端点。",
    "dcorr.out_file": (
        "【GUI 运行必选】将 stdout 的 x Lh 行写入该路径，供 tt_inverse -CD。"
    ),
    "dcorr.abnormal_d": "【条件必选：mode=zelt】[-A 前半] 异常区 Lh。",
    "dcorr.normal_d": "【条件必选：mode=zelt】[-A 后半] 正常区 Lh。",
    "dcorr.v_in": (
        "【条件必选：mode=zelt】[-C 前半] v.in；规则同 gen.v_in。"
    ),
    "dcorr.ilayer": "【条件必选：mode=zelt】[-C 后半] 取样用的界面层号。",
    "dcorr.top_layer": "【条件必选：填写层界时】[-F 前半] 异常区顶界面。",
    "dcorr.bot_layer": "【条件必选：填写层界时】[-F 后半] 异常区底界面。",
    "dcorr.dx": "【条件必选：mode=zelt】[-E] 沿界面的水平取样间距。",
    "dcorr.vcorr_file": "【条件必选：mode=from_vcorr】[-V] gen_vcorr 写出的二维相关长度文件。",
    "dcorr.refl_file": (
        "【条件必选：mode=from_vcorr；zelt 下可选】[-R] 反射面文件（x z）。"
        "from_vcorr 时在各节点取 Lh；zelt 时用其 x（及深度）代替层地形节点。"
    ),
    "inv.mesh": "【必选】[-M] 反演网格文件。",
    "inv.data": "【必选】[-G] 走时数据文件。",
    "inv.xorder": (
        "【成组可选：与 zorder、clen、nintp、bend_cg_tol、bend_br_tol 同组】[-N 第1项] "
        "图论前向星水平阶数（graph.cc / ForwardStar2d）；"
        "与 tt_forward 的 -N 六项一一对应；GUI 默认与正演页一致（4/4/0.8/8/1e-4/1e-5）。"
        "若六项全清空不传 -N，则 tt_inverse 内建正演用程序内置默认（与 tt_forward 默认可能不同）。"
    ),
    "inv.zorder": (
        "【成组可选】[-N 第2项] 垂向前向星阶数；不传 -N 时程序默认 3。"
    ),
    "inv.clen": (
        "【成组可选】[-N 第3项] crit_len（km），须严格 >0，否则整条 -N 被忽略。"
        "不传 -N 时程序默认不做按长度细分。"
    ),
    "inv.nintp": (
        "【成组可选】[-N 第4项] β 样条控制点数（inverse.cc）；不传 -N 默认 8。"
    ),
    "inv.bend_cg_tol": (
        "【成组可选】[-N 第5项] 弯曲 CG 容差（bend.cc）；不传 -N 默认 1e-4。"
    ),
    "inv.bend_br_tol": (
        "【成组可选】[-N 第6项] Brent 容差系数（bend_brent.cc）；不传 -N 默认 1e-7。"
    ),
    "inv.refl_file": (
        "【可选】[-F] 反射面文件（联合重力时源码要求必须提供）。"
        "raytype 1 / 5 的莫霍或壳内反射面。海底请填 seafloor_file (-Y)，"
        "不要把 -F 当海底（无 -Y 时 2/3/4 仍可沿用 -F，与旧流程兼容）。"
    ),
    "inv.seafloor_file": (
        "【可选】[-Y] 海底界面（水层 2/3、台侧多次 4/5 的水柱底）。"
        "反演 **-B 仍是转换波**，不能当海底。"
        "只做 0/1 且不冻水/冻壳时留空，核与原来完全相同。"
        "raytype 5 必须同时有 -Y 与 -F。"
    ),
    "inv.conv_file": (
        "【可选】[-B] 转换界面（折合 PSP raytype 6；PPS/PSS 7/8 的钉点）。"
        "6：盖层结点不进核。7/8：配合 -k 反盖层+面下 Vs，-B 不整层冻盖层。"
        "0/1 仍走初至。正演对应字段拼 -X。"
    ),
    "inv.invert_water_only": (
        "【可选】[-y] 只反水：海底以下速度结点不进核。"
        "须同时有 -Y 或 -F。与 -w 互斥。默认不勾。"
    ),
    "inv.invert_crust_only": (
        "【可选】[-w] 只反壳：海底以上速度结点不进核（冻水）。"
        "须同时有 -Y 或 -F。海底结点划到壳侧。与 -y 互斥。"
        "默认不勾：0/1 路径与核完全不变。"
    ),
    "inv.freeze_refl": (
        "【可选】[-u] 冻结 -F 界面几何：LSQR 算出的深度增量不写回。"
        "默认不勾：-F 行为与原来完全相同（莫霍作业不要勾）。"
        "水相用海底当钉点、只反水速时才勾。"
        "也可用很大 -DD / 很小 -TD 软锁界面，不必勾此项。"
        "勾选时必须同时指定 refl_file (-F)。"
    ),
    "inv.refl_weight": "【可选】[-W] 反射/界面项权重（refl_weight，默认 1；readme 称 depth kernel weighting）。",
    "inv.refl_stride": (
        "【可选，GUI 预处理】对 refl_file 抽稀后再作为 -F。"
        "空或 1 不抽稀；N=2 即点距约变为 2×dx（保留首末点，相当于 "
        "awk '(NR)%2==0{print}' 一类每隔一点取样）。"
        "步长只记在运行参数里，不另存 *_sN.refl；真正给 C++ -F 时才写到"
        "运行包 inputs/（未勾选运行包则写到工区 .tomo2d_tmp/）。"
    ),
    "inv.do_full_refl": (
        "【可选】[-A] 与正演同义：反射贴界面走。"
        "不勾时远偏移可以穿幔成初至。改路径，不是把反射算得更准。"
    ),
    "inv.log_file": (
        "【可选】[-L] 迭代日志**输出**路径。"
        "**不勾选**「可复现运行包」：留空则不传 -L（程序不写该日志文件）。"
        "**勾选**运行包：留空时运行包内默认为 **outputs/tt_inverse.log**（相对本次 runs/…/run_dir）。"
        "若填写路径，运行包内会保留你填写的**文件名**并改写到 **outputs/<该文件名>**。"
        "非 ``#`` 开头的数据行共 **26** 列（走时/χ²/CPU/平滑阻尼/LSQR/粗糙度等）；联合重力时行末多 **1** 列。"
        "点「**-L 日志列说明**」见与 ``inverse.cc`` 对照的逐列说明。"
    ),
    "inv.out_root": (
        "【可选】[-O] 反演结果等输出的**根名**（程序会生成多文件， basename 由此决定）。"
        "**不勾选**运行包：留空则不传 -O（由程序内置默认）。"
        "**勾选**运行包：留空时运行包内默认为 **outputs/out**（即根名为 out，相对 run_dir）。"
        "若填写，运行包内为 **outputs/<你填写的 basename>**。"
    ),
    "inv.out_level": (
        "【可选】[-o] 输出级别（数值；非路径）。默认留空=不写 *.tres/* .ray（推荐）。"
        "1：写走时残差；2：再写射线。监视「抽样射线」需 ≥2，生产慎开。"
    ),
    "inv.dws_file": (
        "【可选】[-K] DWS **输出**路径（程序写入；浏览用「另存为」）。**不必**事先存在。"
        "**不勾选**运行包：留空则不传 -K。"
        "**勾选**运行包：留空时默认为 **outputs/dws.dat**；若填写则 **outputs/<basename>**。"
    ),
    "inv.crit_chi": (
        "【可选】[-R] 稳健反演：第一次 LSQR 后 |A·δm−d| 大于该值的拾取本轮不进方程；"
        "下一轮加回再判。>0 才启用。开 -O 时写出 {out}.outliers.<iter>.<iset> 与末轮 .outliers.final。"
    ),
    "inv.lsqr_tol": "【可选】[-Q] LSQR 容差；GUI 默认 1e-3，清空则不传。",
    "inv.niter": "【可选】[-I] 迭代次数；GUI 默认 5，清空则不传。",
    "inv.target_chi2": "【可选】[-J] 目标卡方；GUI 默认 1.0，清空则不传。",
    "inv.smooth_vel": (
        "【可选】[-SV] 类型 1：填单值；类型 2：填 wmin/wmax/dw（两斜杠三段）。"
        "启用速度平滑时须配合 -CV 相关长度文件（源码校验）。"
    ),
    "inv.smooth_dep": (
        "【可选】[-SD] 同 smooth_vel；可与 -CD 或仅用速度相关长度（见 readme 3.3 Note）。"
    ),
    "inv.smooth_vel_log10": "【可选】[-XV] 与类型 2 -SV 联用：平滑权重按 10 的幂扫描。",
    "inv.smooth_dep_log10": "【可选】[-XD] 与类型 2 -SD 联用。",
    "inv.smooth_corr_v_fn": "【可选】[-CV] 速度相关长度文件。",
    "inv.smooth_corr_d_fn": "【可选】[-CD] 深度相关长度文件。",
    "inv.damp_vel": (
        "【可选】[-DV] 固定速度阻尼权重。与自动阻尼 -TV/-TD 互斥："
        "填了本栏或 -DD/-DQ 后，自动阻尼栏会锁定（灰显，不参与运行）。"
    ),
    "inv.damp_dep": (
        "【可选】[-DD] 固定深度阻尼权重。与 -TV/-TD 互斥（填一侧锁定另一侧）。"
    ),
    "inv.damp_v_fn": (
        "【可选】[-DQ] 空间可变速度阻尼（sheared grid，readme 3.4）。"
        "属固定阻尼 -D 一族，需同时设置 `-DV` 且权重 > 0；与 -TV/-TD 互斥。"
    ),
    "inv.auto_damp_max_dv": (
        "【可选】[-TV] 自动速度阻尼：最大速度扰动（百分数，如 20 表示 20%，日志写成 frac=0.2）。"
        "与固定阻尼 -DV/-DD/-DQ 互斥：填了本栏或 -TD 后，固定阻尼栏会锁定。"
    ),
    "inv.auto_damp_max_dd": (
        "【可选】[-TD] 自动深度阻尼：最大深度扰动（百分数）。与 -DV/-DD/-DQ 互斥。"
    ),
    "inv.apply_filter": (
        "【可选】[-s] 每次迭代后对速度扰动做 2D 滤波（须同时开 -SV）。"
        "勾选且不填边界文件：命令行为裸 ``-s``，上边界用 mesh 海底/地形。"
        "勾选并填写文件：``-s文件``。取消勾选则不传 -s。"
    ),
    "inv.filter_bound_file": (
        "【可选】[-s 的文件名] 滤波上边界（Interface2d）。可空："
        "空且已勾选「开 2D 滤波」时用 mesh 地形。日志头写 ``filter_-s: ON|OFF``。"
    ),
    "inv.jumping": "【可选】[-P] 纯跳跃策略（pure jumping）。",
    "inv.print_final_only": (
        "【可选】[-l] 仅输出最终模型。GUI 默认开启以大幅减少中间 smesh 写盘；"
        "曲线监视仍可用 -L / status.jsonl。需要监视窗刷中间模型时再关闭。"
    ),
    "inv.grav_section": (
        "联合重力参数区默认折叠，点击「展开」后填写 -ZG～-ZT；折叠后已填写的值仍会参与预览与运行。"
    ),
    "inv.grav_file": "【条件必选：启用重力时】[-ZG] 重力异常数据文件（x km, mGal）。",
    "inv.grav_grid": "【条件必选：启用重力时】[-ZX] xmin/xmax/zmin/zmax/dx/dz（六段斜杠分隔）。",
    "inv.grav_refrange": "【条件必选：启用重力时】[-ZR] 参考密度柱水平范围 x0/x1。",
    "inv.grav_cont_file": "【可选】[-ZC] 大陆地壳顶界文件（与 grav_ZC iconv 成对）。路径勿含「/」以免 sscanf 截断，建议相对路径。",
    "inv.grav_cont_iconv": "【可选】[-ZC] 速度-密度换算类型 iconv（整数）。",
    "inv.grav_oceanU_up": "【可选】[-ZU] 洋壳上层顶界；与 oceanU_lo、iconv 三项同填。",
    "inv.grav_oceanU_lo": "【可选】[-ZU] 洋壳上层底界文件。",
    "inv.grav_oceanU_iconv": "【可选】[-ZU] iconv。",
    "inv.grav_oceanL_up": "【可选】[-ZL] 洋壳下层顶界；与 iconv 成对。",
    "inv.grav_oceanL_iconv": "【可选】[-ZL] iconv。",
    "inv.grav_sed_up": "【可选】[-ZS] 沉积层顶界；与 sed_lo、iconv 三项同填。",
    "inv.grav_sed_lo": "【可选】[-ZS] 沉积层底界。",
    "inv.grav_sed_iconv": "【可选】[-ZS] iconv。",
    "inv.grav_deriv": "【可选】[-ZD] dvdp/dvdt/drdp/drdt/dTdz（五段斜杠）。",
    "inv.grav_weight": "【可选】[-ZW] 重力相对走时的权重（默认 1）。",
    "inv.grav_z0": "【可选】[-ZZ] 参考深度 z0（km，向下为正）。",
    "inv.grav_dws": (
        "【可选】[-ZK] 重力 DWS **输出**路径。"
        "**不勾选**运行包：留空则不传 -ZK。"
        "**勾选**运行包且已启用联合重力（填了重力相关项）时：留空默认为 **outputs/grav_dws.dat**；若填写则 **outputs/<basename>**。"
    ),
    "inv.grav_cutoff": "【可选】[-ZT] 灵敏度截断 range/val（两段斜杠）。",
    "inv.verbose": "【界面已隐藏】由 verbose_level 决定：>0 时传 -V[level]。",
    "inv.verbose_level": (
        "【可选】默认留空：不传 -V（少终端 I/O）。填正数：传 -V[level]，调试用。"
    ),
    "inv.cli_extras": (
        "tt_inverse 的上述开关已在表单与 TomoAnd.tt_inverse 中映射；完整语义仍以 readme.pdf 4.3 与 "
        "TomoHelp.tt_inverse_help() 为准。"
        "注意：-ZC/-ZU 等路径在 C 中用 sscanf 按「/」分段，路径内勿含额外「/」（可用相对路径）。"
        "并行/策略见顶栏「并行 / 策略环境变量」（env.* → OMP_NUM_THREADS、TOMO2D_INV_*、TOMO2D_FWD_OMP）。"
    ),
    "tab.parallel_env": (
        "顶栏「并行 / 策略」：常显 OMP 线程、正/反演并行与图论FS枚举，可用预设一键组合。\n"
        "「加速策略」与「开发者/对拍」并排折叠：复用/C2F/LSQR列预条件/灵敏度加权/线搜索/LM；Legacy/DIAG——日常生产勿开。\n"
        "这些是环境变量（写入子进程），不是命令行 -X；布尔未勾选显式写 0。\n"
        "前提：二进制需 OpenMP 编译（见 src/README_OMP_BUILD.md）。"
    ),
    "env.omp_num_threads": (
        "【含义】OpenMP 使用的线程数（环境变量 OMP_NUM_THREADS）。\n"
        "【作用】限制/设定 tt_forward、tt_inverse 等 OMP 并行时的线程池大小；"
        "并行粒度是按炮点（source），线程再多也不会超过 source 数带来收益。\n"
        "【填写】正整数，例如 4 或 8。留空=不覆盖系统环境（沿用 shell/系统已有设置）。\n"
        "【注意】不要写成空串再强行导出，否则 libgomp 可能报 Invalid value。"
    ),
    "env.inv_omp": (
        "【含义】是否启用 tt_inverse 的 OpenMP 并行路径（TOMO2D_INV_OMP）。\n"
        "【作用】按炮点并行做射线追踪/核矩阵组装，可明显缩短大炮数反演墙钟时间。\n"
        "【勾选】写 1；未勾选写 0（明确关闭，避免继承 shell 里旧的 =1）。\n"
        "【说明】需 OpenMP 编译。并行时不再刷逐炮 `*` `.`（会交错）；"
        "执行日志会显示 threads/nsrc，并按约 10% 炮点报告 `ray tracing k/N`。"
    ),
    "env.fwd_omp": (
        "【含义】是否启用 tt_forward 的 OpenMP 并行路径（TOMO2D_FWD_OMP）。\n"
        "【作用】按炮点并行正演走时，加速大批炮合成/检验。\n"
        "【勾选】写 1；未勾选写 0。\n"
        "【限制】开启 -A（full reflection）时程序会自动回退串行；射线路径输出仍按炮序写入。"
    ),
    "env.graph_fs_enum": (
        "【含义】图论最短路是否按 forward-star 下标枚举邻居（TOMO2D_GRAPH_FS_ENUM）。\n"
        "【勾选】写 1：弹出节点后只 relax 窗口内几十个邻居（快）。\n"
        "【不勾选】写 0：恢复原策略，对剩余 C/B 逐点 isIn（慢，可对拍回退）。\n"
        "【命令行】未设置该变量时 C++ 走原扫表；GUI 默认勾选并显式写 1 或 0。\n"
        "【独立】与 Legacy / 前向复用 / C2F 无关。对拍预设会关掉此项以便整包旧行为。"
    ),
    "env.inv_legacy_baseline": (
        "【含义】反演「回退基线」模式（TOMO2D_INV_LEGACY_BASELINE），用于与优化版做 A/B 对比。\n"
        "【作用】强制回到较旧行为：关闭前向复用、kernel 归并用旧 sort/merge、"
        "关闭 LSQR 列预条件、关闭 coarse-to-fine。\n"
        "【优先级】最高——开启后 REUSE_* / COARSE2FINE 相关设置一律被忽略；LSQR 列预条件也被强制关闭。\n"
        "【建议】日常生产勿开；仅在核对数值一致性或排查优化引入差异时使用。\n"
        "若只想关掉列预条件、保留 hash 核，请用「LSQR列预条件」而不是整包 Legacy。"
    ),
    "env.inv_lsqr_precond": (
        "【含义】LSQR 列预条件开关（TOMO2D_INV_LSQR_PRECOND）。\n"
        "【作用】勾选后按列范数中位数做 Jacobi 缩放 D_j=clip(s_med/‖A列‖, 1/κ, κ)，"
        "空列 D=0；并强制最少迭代，避免缩放后 ATOL 在 iter=1 假停、模型不动。\n"
        "【默认】关闭。勾选写 1；未勾选写 0（只关预条件，不回退 kernel 归并 / 复用 / C2F）。\n"
        "【优先级】Legacy 开启时强制关闭，此项灰显。无阻尼探步（-TV 的 wdv=0）内部也不缩放。\n"
        "【配合】「上限」现为相对中位数的 κ（TOMO2D_INV_LSQR_PRECOND_MAX）；留空=10。"
    ),
    "env.inv_lsqr_precond_max": (
        "【含义】LSQR 列预条件相对中位数的夹逼 κ（TOMO2D_INV_LSQR_PRECOND_MAX）。\n"
        "【作用】D_j 落在 [1/κ, κ]；范数远小于中位数的列 D=0，不再放大暗结点。\n"
        "【填写】正数，如 10。留空=程序默认 10。填 0=相对不封顶（空列仍为 0）。\n"
        "【条件】须勾选「LSQR列预条件」，且未开 Legacy。"
    ),
    "env.inv_sens_weight": (
        "【含义】灵敏度加权正则开关（TOMO2D_INV_SENS_WEIGHT）。\n"
        "【作用】按数据核列和（DWS）对每个未知量加权：暗结点加大阻尼 T，亮→暗平滑耦合减弱"
        "（R 非对角 × 2/(w_i+w_j)）。Vp、面上 Vs、面下 Vs、莫霍深度各自用本块 DWS 中位数，"
        "用来减弱射线路径拖曳，不针对某一层。\n"
        "【默认】关闭。勾选写 1。与 LSQR 列预条件都按照明缩放，减拖曳时优先只开本项。\n"
        "【配合】κ 为夹逼上限（TOMO2D_INV_SENS_KAPPA，默认 10）。"
    ),
    "env.inv_sens_kappa": (
        "【含义】灵敏度权相对本块 DWS 中位数的夹逼 κ（TOMO2D_INV_SENS_KAPPA）。\n"
        "【作用】w_j=clip(s_med/(s_j+ε s_med), 1/κ, κ)；空照明结点取 κ（最强阻尼）。\n"
        "【填写】正数，如 10。留空=程序默认 10。\n"
        "【条件】须勾选「灵敏度加权」。"
    ),
    "env.inv_linesearch": (
        "【含义】Gauss–Newton 步长线搜索（TOMO2D_INV_LINESEARCH）。\n"
        "【作用】LSQR 给出方向后不整步加上去，按重追后的真实 χ² 做 Armijo 回退"
        "（α=1, 1/2, …）。Vp / Vs / 面上 / 面下 / 莫霍同一套外迭代。\n"
        "【默认】关闭。勾选写 1。扫描多组 -SV/-SD 时自动跳过。\n"
        "【可选】TOMO2D_INV_LS_C（Armijo c，默认 1e-4）、"
        "TOMO2D_INV_LS_RHO（回退因子，默认 0.5）、TOMO2D_INV_LS_AMIN（最小 α，默认 1/32）。"
        "与 LM 同时开时线搜索被忽略。"
    ),
    "env.inv_lm": (
        "【含义】Levenberg–Marquardt 信赖域（TOMO2D_INV_LM）。\n"
        "【作用】用重追真实 χ² 与线性预测比 ρ。ρ 差则加大阻尼并重新 LSQR，"
        "不是沿原方向缩步长。Vp / Vs / 面上 / 面下 / 莫霍同一套外迭代。\n"
        "【默认】关闭。勾选写 1。无 -D/-T 时自动关掉。与线搜索同时开时以本项为准。\n"
        "【可选】TOMO2D_INV_LM_LAMBDA（初值，默认 1）、TOMO2D_INV_LM_UP（默认 4）、"
        "TOMO2D_INV_LM_DOWN（默认 0.5）、TOMO2D_INV_LM_LMAX（默认 256）、"
        "TOMO2D_INV_LM_RHO_ACCEPT（默认 0.1）、TOMO2D_INV_LM_RHO_GOOD（默认 0.5）。"
    ),
    "env.inv_reuse_forward": (
        "【含义】是否启用反演中的前向复用（TOMO2D_INV_REUSE_FORWARD）。\n"
        "【作用】当相邻迭代模型变化足够小时，复用已有正演结果（A/path/残差走时），"
        "减少重复射线追踪，加快迭代。\n"
        "【默认】关闭。仅在未开 Legacy baseline 时生效。\n"
        "【配合】可填 REUSE_THRESH 控制「多小的模型变化才复用」。"
    ),
    "env.inv_reuse_thresh": (
        "【含义】前向复用的模型变化阈值（TOMO2D_INV_REUSE_THRESH）。\n"
        "【作用】迭代间模型扰动低于该阈值时才复用正演；阈值越大越容易复用（更快但更“敢省”）。\n"
        "【条件】需勾选「前向复用」，且未开 Legacy baseline。\n"
        "【填写】正数，如 1e-3。留空=不覆盖系统环境（程序侧用其默认）。"
    ),
    "env.inv_coarse2fine": (
        "【含义】分阶段（coarse-to-fine）反演开关（TOMO2D_INV_COARSE2FINE）。\n"
        "【作用】前期用更强的平滑/阻尼稳住大尺度结构，后期逐步放松到目标强度，"
        "有利于先粗后细、减轻早期伪影。\n"
        "【条件】未开 Legacy baseline 时生效。\n"
        "【配合】下方 SMOOTH/DAMP 的 START→END 控制首轮到末轮的系数（中间对数插值）。"
    ),
    "env.inv_c2f_smooth_start": (
        "【含义】C2F 平滑系数起点（TOMO2D_INV_C2F_SMOOTH_START）。\n"
        "【作用】第 1 轮迭代相对表单平滑权重的倍率；默认约 3.0 表示前期平滑约为设定值的 3 倍。\n"
        "【条件】勾选 Coarse-to-fine。留空用程序默认。"
    ),
    "env.inv_c2f_smooth_end": (
        "【含义】C2F 平滑系数终点（TOMO2D_INV_C2F_SMOOTH_END）。\n"
        "【作用】最后一轮的平滑倍率；默认约 1.0 表示回到表单设定的平滑强度。\n"
        "【条件】勾选 Coarse-to-fine。中间轮次在 START→END 间对数过渡。"
    ),
    "env.inv_c2f_damp_start": (
        "【含义】C2F 阻尼系数起点（TOMO2D_INV_C2F_DAMP_START）。\n"
        "【作用】第 1 轮相对表单阻尼的倍率（默认约 3.0），前期更“稳”、更新更保守。\n"
        "【条件】勾选 Coarse-to-fine。留空用程序默认。"
    ),
    "env.inv_c2f_damp_end": (
        "【含义】C2F 阻尼系数终点（TOMO2D_INV_C2F_DAMP_END）。\n"
        "【作用】末轮阻尼倍率（默认约 1.0），后期允许更大更新以拟合细节。\n"
        "【条件】勾选 Coarse-to-fine。"
    ),
    "env.inv_diag": (
        "【含义】反演诊断哈希开关（TOMO2D_INV_DIAG）。\n"
        "【作用】每迭代/每个 iset 输出 hash_res、hash_A、hash_dmodel、hash_modelv、hash_modeld 等，"
        "便于对比串行 vs 并行、或优化前后从哪一轮开始数值分叉。\n"
        "【建议】仅调试时开启；会增加日志量，不影响正常生产结果路径。"
    ),
    "env.inv_status_jsonl_path": (
        "【含义】反演监视 NDJSON 路径（环境变量 TOMO2D_INV_STATUS_JSONL）。\n"
        "【作用】tt_inverse 每完成一轮 iter×iset 向该文件追加一行 "
        "{iter,iset,chi2,rms,pred_chi,…,smesh}，供监视窗准实时读取。\n"
        "【默认】outputs/status.jsonl（相对子进程 cwd；运行包内即 runs/ttinv_*/outputs/）。\n"
        "【关闭】清空此项则不写。需较新编译的 tt_inverse。"
    ),
    "tab.gen_smesh": (
        "gen_smesh：生成 sheared grid。"
        "【必选】vel_opt + grid_opt；vel_opt=zelt 时 grid_opt 必为 zelt（与 gen_smesh.cc 一致）。"
        "运行前还须填写「smesh 输出文件」；refl、水/空气、地形、-d dump 等见各字段提示。"
    ),
    "tab.tt_forward": (
        "独立正演工具：算走时 / 射线等。"
        "日常实测反演请直接用「6) tt_inverse」（内部已含正演），无需先跑本页。"
        "默认少写盘：out_ray / 多余输出 / verbose 留空。\n"
        "tt_forward：图论 + 弯曲走时（readme 4.2）。"
        "【必选】smesh(-M)。【可选】geom(-G)、反射、输出、clock、graph_only、vgrid 子区六项、-n、verbose_level 等。"
        "六项 -N 为【成组可选】（见 fwd.xorder 等）；GUI 默认填 4/4/0.8/8/1e-4/1e-5，可清空六项则不传 -N。"
        "各输出文件（-T/-R/-E 等）可只填文件名（相对 work_dir），不必浏览。"
        "并行：顶栏勾选「tt_forward OMP」或设 TOMO2D_FWD_OMP（``-A`` full reflection 会自动回退串行）。"
    ),
    "tab.gen_damp": (
        "gen_damp：阻尼文件。【必选】vel_opt + grid_opt 及分支必填项（与 gen_smesh 相同，含 zelt 配套）。"
        "zelt 下 -F 层界为【条件必选】（填则成对）。API 可选 out_file 写 stdout。"
    ),
    "tab.gen_vcorr": (
        "gen_vcorr：速度相关长度（tt_inverse -CV）。"
        "默认 simple_2x2：填 Lht/Lhb/Lvt/Lvb 与 xmax/zmax，写出 2×2 文件（顶/底随深度插值，沿 x 不变）。"
        "program：调用 gen_vcorr 二进制，-A 为异常区/正常区（不是顶/底）。"
        "运行须填写「vcorr 输出文件」。"
    ),
    "tab.gen_dcorr": (
        "gen_dcorr：写出 tt_inverse -CD 的 1D 反射点相关长度（每行 x Lh）。"
        "【必选】mode + 对应分支；运行须填输出文件。"
        "uniform：沿 x 均匀 Lh；zelt：与 gen_vcorr 同一套层划区；"
        "from_vcorr：在反射节点上取二维 vcorr 的水平相关长度（readme 3.3）。"
    ),
    "tab.tt_inverse": (
        "主路径：填实测 mesh/data 后直接反演（内部已含正演）。"
        "默认少写盘：out_level/verbose 留空，print_final_only 开启；"
        "曲线监视用 -L / status.jsonl。要刷中间 smesh 或抽样射线时再关 -l、提高 out_level。"
        "合成走时见正演页「合成数据」。\n"
        "tt_inverse：走时反演（readme 4.3）。"
        "【必选】mesh(-M)、data(-G)。"
        "六项 -N 与 tt_forward 同义【成组可选】；GUI 默认 4/4/0.8/8/1e-4/1e-5，六项全空则不传（内层正演用程序内置默认）。"
        "本页含：反射 -F/-A/-W；策略 -P/-l；输出 -L/-O/-o/-K/-ZK；稳健 -R/-Q；迭代 -I/-J（GUI 默认 I=5、J=1.0、Q=1e-3，清空则不传）；"
        "平滑 -SV/-SD（单值或 wmin/wmax/dw）与 -XV/-XD；相关 -CV/-CD；"
        "固定阻尼 -DV/-DD/-DQ 与自动阻尼 -TV/-TD（二者勿同填）；滤波 -s；"
        "联合重力 -ZG～-ZT 在可折叠「联合重力」分组内；verbose_level。"
        "此外可通过顶栏「并行 / 策略环境变量」启用 OMP、前向复用与 coarse-to-fine（见程序帮助 tt_inverse 章节）。"
        "输出路径（-L/-O/-K/-ZK）在**不勾选**运行包时可只填相对 work_dir 的文件名；留空则多数不传（见各字段）。"
        "**勾选「可复现运行包」**时，输出均落在该次 runs/…/ **outputs/** 下；留空时的默认文件名为："
        "**-L → tt_inverse.log**，**-O 根名 → out**，**-K → dws.dat**，"
        "（启用联合重力时）**-ZK → grav_dws.dat**；若你自行填写则保留所给 basename。"
        "并写 manifest.json（含 gui_profile）。细节见各字段与 TomoHelp.tt_inverse_help()。"
    ),
    "inv.use_repro_bundle": (
        "勾选：运行 tt_inverse（及含该步的 pipeline）时，在 work_dir/runs/ 下新建目录，"
        "命名形如 **ttinv[_备注]_<mesh 主干>_<data 主干>_<本地时间戳>_<6位hex>**（过长会自动缩短备注与主干）。"
        "末尾 **6 位 hex** 为随机后缀（``secrets.token_hex(3)``）：与秒级时间戳一起避免同一秒内、同名 mesh/data/备注时目录重名导致创建失败。"
        "输入快照到 **inputs/**；**outputs/** 下默认（表单对应项留空时）为："
        "**tt_inverse.log**（-L）、根名 **out**（-O）、**dws.dat**（-K）；"
        "若启用联合重力且 -ZK 留空则为 **grav_dws.dat**。已填路径则 outputs/<basename>。"
        "运行结束后将 outputs/ 归入 models/ residuals/ rays/ dws/ logs/ other/；"
        "**不**自动写 final.smesh，选用哪次模型由你结合日志与参数自行决定。"
        "运行前写入 manifest.json（argv、inputs 哈希、post_run、gui_profile）。子进程 cwd 为该 run_dir。"
        "不勾选：仍在 work_dir 下按表单路径运行，留空规则见各输出字段（多数不传）。"
    ),
    "inv.bundle_run_label": (
        "仅在使用可复现运行包时生效：可选短备注，经净化后插入 runs 子目录名（在 ttinv 之后），"
        "便于区分试验（如 line01、smooth_test）。留空则仅用 mesh/data 主干与时间戳。"
    ),
    "tab.stat_smesh": (
        "stat_smesh：网格统计。【必选】mode；"
        "list：list_file + cmd_type，cmd_type=r 时另【必选】ave_file；可选 refl_nnodes(-R)。"
        "mesh：mesh_file + cmd_type；a/b 窗口参数见各字段；-Db 时须填顶/底/中界。"
        "扩展：-P/-U/-X、剔除带 -x/-t/-b（四项齐）。"
    ),
    "tab.edit_smesh": (
        "edit_smesh_HHB：编辑速度网格，结果走 stdout（通常重定向到新文件）。"
        "【必选】smesh_file、cmd_type；各子命令附加字段见 tomand.edit_smesh。"
        "坐标、kstart、-Cr 加密早退；cmd_type=B + remove_bg_file 对应 -CB；cmd_type=b 与默认源码不一致等见 TomoHelp.edit_smesh_help()。"
    ),
    "tab.pipeline": (
        "流程复用各页签当前参数。日常反演请直接用 tt_inverse 页。"
        "仅合成数据可用「gen_smesh → tt_forward → tt_inverse」"
        "（需 fwd.out_ttime → inv.data）。"
        "Pipeline 内 tt_inverse 暂不挂运行包监视。\n"
        "pipeline：顺序执行多步；每步沿用该命令在对应页签的【必选】规则。"
        "例如含 tt_inverse 时须事先填好 inv.mesh、inv.data；"
        "gen_smesh 步须满足 vel_opt/grid_opt 分支必填项。"
        "桥接路径为【可选】辅助填表。"
        "tt_inverse 步是否使用「可复现运行包」与 tt_inverse 页同一复选框一致。"
    ),
    "tab.checkerboard": (
        "棋盘格分辨率测试（一键）：背景模型叠加棋盘扰动 → tt_forward 合成走时 → "
        "从背景模型 tt_inverse → 输出真/恢复百分异常场。"
        "页签「棋盘预览图…」只画扰动（上 ΔV、中棋盘后 Vp、下棋盘前 Vp），不跑正演/反演。"
        "「棋盘结果图…」画测试包：**上**真异常%、**中**恢复异常%、**下**残差%；"
        "图上标注格统计（恢复振幅=RMS比、格中位/均值=逐格 恢复/真、相关 r）。"
        "勾选 DWS 时只统计有覆盖的格。窗口内可下拉切换不同 ``runs/checkerboard_*``。"
        "「预览棋盘格测试」列出本页 A/h/v 与实际传给 tt_forward / tt_inverse 的参数。"
        "反射面默认用背景 smesh 同轮 ``*.refl.<iter>.<iset>``；没有同轮面时请在本页自选，"
        "不会回退 inv.refl_file。"
        "填了 -F 且 geom 含反射震相时，正演/反演都用折射+反射走时，但**界面锁定**（不改深度）。"
        "反演的速度阻尼/光滑/迭代取自 tt_inverse 页；输出在 runs/checkerboard_<时间戳>/。"
    ),
    "cb.bg_smesh": "【必选】背景/起始 smesh；空则回退 inv.mesh 或 fwd.smesh。",
    "cb.geom": "【必选】正演几何 geom（-G）；空则回退 fwd.geom。",
    "cb.refl_file": (
        "【条件】背景 smesh 若是反演输出 ``*.smesh.<iter>.<iset>`` 且旁边有同轮 "
        "``*.refl.<iter>.<iset>``，会自动填入本栏（该模型自己的界面）。"
        "没有同轮界面时请在此自选，**不会**回退 inv.refl_file / fwd.refl_file"
        "（那是反演起始面，不是该背景模型的界面）。"
        "空且无同轮面则不传 -F，合成走时只有折射。"
    ),
    "cb.amp": "【必选】棋盘振幅 A（百分数），对应 edit_smesh -Cc 的 A。",
    "cb.h_len": "【必选】水平波长 h（km）。",
    "cb.v_len": "【必选】垂向波长 v（km）。",
    "tab.monte_carlo": (
        "蒙特卡洛不确定性：N 次反演，可选随机起始模型与走时噪声；"
        "输出均值速度、标准差、相对不确定度% 与水平平均剖面。"
        "模型方式：**smesh** = 用该网格做分段随机 1D（沉积/上地壳/下地壳抽厚度与顶底速度，"
        "地幔接到网格底；每次 -F 的 Moho = 沉积+上地壳+下地壳）；"
        "**v.in** = 指定海底/基底/Conrad/莫霍，勾选地质层后用当前速度区间填入界面向，-F 用莫霍。"
        "选中一种方式后，另一种模型路径锁定。"
        "页签「蒙特卡洛预览图…」：左三幅为第 1 次实现与基础模型（smesh 时叠各次 Moho），"
        "右侧叠 N 条 1D（smesh 从海底起；v.in 从最浅勾选层的顶界面起：海底/基底/Conrad/莫霍），不写盘。"
        "「蒙特卡洛结果图…」：上均值 Vp、下误差 σ，叠界面均值 ±σ（色带）；"
        "均值/σ 只计入 pred χ² 低于阈值（默认 1.8）的实现，且各点只平均有 DWS 覆盖的成员。"
        "反演参数取自 tt_inverse 页；结果在 runs/montecarlo_<时间戳>/。"
    ),
    "mc.base_mesh": (
        "【条件：smesh】分段 1D 的网格/地形/水层。空则回退 inv.mesh。"
        "选 v.in 时此栏锁定；若仍有路径则只当铺 v.in 的网格，速度来自 v.in。"
    ),
    "mc.data": "【必选】走时数据；空则回退 inv.data。",
    "mc.n_runs": "【必选】实现次数 N（≥2）。",
    "mc.seed": "【可选】随机种子基数；第 i 次用 seed+i。",
    "mc.chi_max": (
        "【可选】叠均值/σ 前先按各次反演日志末行 **pred χ²** 筛选，只保留 "
        "pred χ² < 该阈值的实现。默认 1.8。筛完须至少 2 个。"
    ),
    "mc.init_mode": (
        "模型方式：**smesh** = 默认分段随机 1D，用该 smesh 的网格/地形/水层，"
        "沉积 / 上地壳 / 下地壳各抽厚度与顶底速度（交界面速度连续），"
        "地幔接到网格底、只抽顶底速度；-F 为该次 Moho（沉积+上地壳+下地壳）；"
        "**v.in** = 指定海底 / 基底 / Conrad（可选）/ 莫霍，勾选地质层后用速度与厚度区间扰动对应界面向，"
        "未选层保持 v.in；-F 用选定莫霍。"
        "选中一项后锁定另一模型路径。"
    ),
    "mc.random_init": (
        "旧开关。现已不用；未设模型方式时按 smesh（分段 1D）。"
    ),
    "mc.init_amp": (
        "【旧字段】相关百分数扰动已从模型方式中去掉。smesh 现为分段 1D。"
    ),
    "mc.v_in": (
        "【条件：v.in】Zelt v.in。空则回退 gen_smesh 的 v.in。"
        "打开后按界面数刷新海底/基底/Conrad/莫霍下拉（与 vedit 同一套「界面N」）。"
        "选 smesh 时此栏锁定。"
    ),
    "mc.vin_seafloor": "【条件：v.in】海底面。默认界面2（第1层底）。",
    "mc.vin_basement": "【条件：v.in】沉积基底。选「—」则没有沉积层可勾。",
    "mc.vin_conrad": "【条件：v.in】Conrad。选「—」则没有单独下地壳；上地壳接到莫霍。未选时预览不画 Conrad。",
    "mc.vin_moho": "【条件：v.in】莫霍。-F 写出此界面。",
    "mc.vin_units": (
        "【条件：v.in】参与扰动的地质层：sed / uc / lc / mantle。"
        "空=当前界面划分出的全部层；none=不扰动。"
        "选中层用 mc.sed_h/v、uc_h/v、lc_h/v、mantle_v 扰动对应界面向；-F 用（可随厚度移动的）莫霍。"
        "点「选层…」在速度底图上勾选界面与层位。"
    ),
    "mc.sed_h": (
        "【条件：smesh 分段 1D / v.in 勾选沉积】沉积厚度 min ~ max（km）。"
        "smesh：相对海底。v.in：把基底面改成与海底平行、厚度落在此区间。"
    ),
    "mc.sed_v": (
        "【条件：smesh 分段 1D / v.in 勾选沉积】沉积顶底速度 min ~ max（km/s）。"
        "两点都在此区间内独立抽取，底≥顶≥水速。"
    ),
    "mc.uc_h": (
        "【条件：smesh 分段 1D / v.in 勾选上地壳】上地壳厚度 min ~ max（km）。"
        "v.in：移动 Conrad（若已选）或莫霍，使该层厚度落在此区间。"
    ),
    "mc.uc_v": (
        "【条件：smesh 分段 1D / v.in 勾选上地壳】上地壳顶底速度 min ~ max（km/s）。"
        "两点都在此区间内独立抽取，底≥顶≥沉积底。"
    ),
    "mc.lc_h": (
        "【条件：smesh 分段 1D / v.in 勾选下地壳】下地壳厚度 min ~ max（km）。"
        "v.in：移动莫霍使 Conrad–莫霍厚度落在此区间；-F 用移动后的莫霍。"
    ),
    "mc.lc_v": (
        "【条件：smesh 分段 1D / v.in 勾选下地壳】下地壳顶底速度 min ~ max（km/s）。"
        "顶与上地壳底共用同一速度（交界面连续），底在此区间内抽取且 ≥ 顶。"
    ),
    "mc.mantle_v": (
        "【条件：smesh 分段 1D / v.in 勾选地幔】地幔顶底速度 min ~ max（km/s）。"
        "分段 1D 时厚度 = 网格底 − 沉积 − 上地壳 − 下地壳，不随机。"
        "两点都在此区间内独立抽取，底≥顶≥下地壳底。"
    ),
    "mc.sed_g": "【旧字段】沉积改为顶底速度独立抽取，不再用梯度范围。",
    "mc.crust_h": "【旧字段】未填上/下地壳厚度时，按 40%/60% 拆成 mc.uc_h / mc.lc_h。",
    "mc.crust_v": "【旧字段】未填 mc.uc_v 时回退为此范围。",
    "mc.uc_vt": "【旧字段】已并入 mc.uc_v。",
    "mc.uc_vb": "【旧字段】已并入 mc.uc_v。",
    "mc.lc_vb": "【旧字段】已由 mc.lc_v 取代。",
    "mc.mantle_dv": "【旧字段】已由 mc.mantle_v 取代。",
    "mc.crust_g": "【旧字段】上/下地壳改为顶底速度区间抽样。",
    "mc.mantle_g": "【旧字段】地幔改为顶底速度区间抽样。",
    "mc.tt_noise": "勾选：对走时 t 加高斯噪声。",
    "mc.noise_sigma": "噪声标准差（秒）；若勾选「相对 u」则为 σ×u。",
    "mc.noise_relative_u": "勾选：噪声幅度 = sigma × 该行误差列 u。",
    "gui.plot_smesh_cmap": (
        "绘制模型的速度色标，下拉四项（各绘图窗与顶栏共用）："
        "**vp** = 内置 scale_p.cpt，**vs** = scale_s.cpt，**vpvs** = scale_vpvs.cpt（Haiti vpvs1，jet 1.65–2.00），"
        "**water** = scale_water.cpt（水层 Vp 1.35–1.65 km/s：浅层暖色、深层冷色；"
        "海面以上空气不着色；沉积落到色标上端棕红）。"
        "（与 examples/inputs 相同）。色标范围取 CPT 内最小/最大 z。"
        "旧配置里的同名文件路径会认成对应项；找不到外部 .cpt 时退回 vp。"
    ),
    "gui.plot_smesh_contours": (
        "勾选后按**当前色标**叠对应等值线：vp→contour_p，vs→contour_s，"
        "vpvs→contour_vpvs，water→contour_water（间隔 0.01；A 粗实线+标注，C 细虚线）。"
        "不勾选则只画色块、地形与界面。"
    ),
    "gui.plot_smesh_dws_mask": (
        "是否用 DWS 把 **没有射线覆盖**（DWS≤0）的格子留白；有覆盖按 log(DWS) 透明。"
        "**绘制 smesh**：须在图窗「指定 DWS…」（或拖入/粘贴），**不**自动查找；文件名不限。"
        "**反演监视 / 模型挑选 / 模型对比 / 棋盘预览**：勾选后按 **每个** smesh 就近查找，**不**扫整个 runs/——"
        "同目录（含同目录 dws/）→ 该次运行包 ``outputs/dws/``（及旧式 ``outputs/dws.dat``）"
        "→ 运行包目录 / ``-O`` → **最后**才用表单 ``inv.dws_file``（避免集合共用一份）。"
        "差值遮罩为 A∩B（权 min）；统计时各点只平均 DWS>0 的成员。"
        "对比窗「DWS 列表…」可核对每个 smesh 实际用了哪份。"
        "``tt_inverse -K`` 在反演**结束**才写出，监视会在文件出现后自动叠上。"
        "水层与地形线仍画。"
    ),
    "gui.plot_smesh_dws_file": (
        "绘制 smesh 时使用的 DWS 文件（tt_inverse **-K**，每行 ``x z 覆盖权重``）。"
        "仅该图窗使用；反演监视与模型挑选仍自动就近查找。"
    ),
    "gui.plot_smesh_rays": (
        "绘制 smesh 时是否叠加抽样射线（与反演监视同套：按 OBS/炮着色）。"
        "**绘制 smesh**：须在图窗「指定射线…」（或拖入/粘贴），**不**自动查找。"
        "选任一 ``stem.ray.<iter>.<isrc>`` 即可加载同套；也可选正演 **-R** 单文件。"
        "反演射线需 ``out_level (-o)≥2``。"
    ),
    "gui.plot_smesh_ray_root": (
        "绘制 smesh 时使用的射线路径：任一 ``.ray.<iter>.<isrc>``、正演 ``.ray``，"
        "或 ``rays/`` 目录。仅该图窗使用；反演监视仍按 ``-O`` 自动查找。"
    ),
    "btn.tt_inverse_analysis": (
        "读取 **tt_inverse -L** 输出的**数值行**日志（与 inverse.cc 每行一致），做反演质量诊断。"
        "**单日志**：绘制折射 RMS (Pg)、反射 RMS (PmP)、initial χ²、pred χ² 随**迭代次数**曲线。"
        "**多日志**（每个日志单独一行，相对路径相对 **work_dir**；可一次拖入多个 .log）："
        "多 run 叠画折射/反射 RMS 与 pred χ²、参数影响散点（无图例）；"
        "pred χ²–**粗糙度** Pareto 与综合得分 ``score = pred_χ² × (1 + w×R)``（**越小越好**；w 可填）。"
        "Pareto / 参数影响 / 叠画 / 汇总表：**左击**点、条、曲线或表行，多窗口同步高亮（点空白取消）；"
        "**Ctrl+单击**加减选，**Shift+单击**连选，**Shift+左拖**框选；"
        "**右击点一下**已选（右拖仍缩放）：**绘制迭代曲线**、添加至模型对比、反演参数列表（多选对照表）；"
        "单选才可绘制末轮模型+走时拟合、打开目录。"
        "参数影响横轴为对数，刻度用十进制（如 150、600），并标出图上实际参数值。"
        "曲线/散点/条形为 **pyqtgraph**；末步汇总为表格。"
        "依赖 pyqtgraph、numpy。"
    ),
    "btn.program_help": (
        "打开帮助（与 **F1** 相同，非模态）。默认是 **GUI 快速说明**（``docs/HELP.md``：工区、监视、"
        "模型挑选/对比、DWS、分析图交互等）；也可切换 ``python_wrapper``、gen_smesh、tt_forward、"
        "tt_inverse、tt_inverse 日志列说明、stat_smesh、edit_smesh 等 TomoHelp 章节。"
        "tt_inverse 章节含 TOMO2D_INV_REUSE_FORWARD / COARSE2FINE / LEGACY_BASELINE。"
    ),
    "btn.plot_smesh": (
        "打开 smesh 图窗。默认绘制**当前命令页签**里指定的网格："
        "gen_smesh 用 **smesh 输出文件**，tt_forward 用 **smesh (-M)**，"
        "tt_inverse 用 **mesh (-M)**，edit_smesh 用 **smesh_file**，等等。"
        "界面同样只用本页的 **-F / refl**（有配套 ``*.refl.<iter>.<iset>`` 则优先）。"
        "字段为空或文件不存在时打开空图窗，再在窗内「打开…」/拖入/粘贴，**不会**立刻弹出选文件框，也**不会**跨页签去找 ``model.smesh``。"
        "**每次点击都从磁盘重新读取**，改过网格后再点即可看到新图（也可在图窗点「刷新」）。"
        "可将 **.smesh**（含 ``out.smesh.1.1``）、**v.in**、**.grd/.nc** "
        "**拖到主窗口或该图窗**直接绘制（Vp/Vs 常用 grd，没有 smesh）；"
        "界面/反射文件（``*.refl.*``、文件名含 refl，或每行 x z 的文本）同样可拖入叠加，也可点「叠加界面…」。"
        "色标见上方「smesh 色标」。等值线随色标切换（vp / vs / vpvs 各一张内置表），"
        "图窗上可勾选「叠加等值线」、「DWS 遮罩」与「叠加射线」；"
        "在速度图上 **右键**「写入表单」把当前 smesh / 配套反射面写回参数，"
        "或「添加到对比模型」。「清除对比」只在模型对比图上。"
        "「打开所在目录」打开当前 smesh 所在文件夹。"
        "遮罩与射线均须在图窗指定文件（或拖入/粘贴），**不**自动查找。"
        "无射线覆盖处不着色；射线按 OBS/炮抽样着色（需 out_level≥2）。"
        "依赖 matplotlib、xarray、numpy 等。"
    ),
    "btn.model_picker": (
        "从工区 runs/ 选一次反演运行包，再挑各轮 smesh 预览。"
        "写回表单、打开所在目录、加入对比均在预览图 **右键**。"
        "也可用「浏览…」自选其它目录的运行包。不自动判定最优。"
        "两套模型的差值请用「模型对比…」。"
    ),
    "btn.model_compare": (
        "打开模型对比窗：集合可统计多个 smesh 的 **均值 Vp** 与 **误差 σ**"
        "（各点只平均该处 DWS>0 的成员）；指定 A/B 绘制 **B−A**。"
        "色标 ±km/s /「随数据」只作用于 ΔV 与 σ，不改均值 Vp；改完立即重绘。"
        "差值 DWS 为 A∩B（权 min）。「DWS 列表…」可核对每个 smesh 用了哪份。"
        "统计后在均值图/误差图上 **右键点一下** 保存速度或反射面，均值图还可写入表单；"
        "其它速度图右键同样可把当前模型写入表单"
        "（右键拖仍缩放）。速度图上 **右键「添加到对比模型」** 记入同一集合。"
    ),
    "btn.save_profile": "将当前表单保存为 JSON，便于下次加载复现。",
    "gui.write_file_log": (
        "勾选时，将下方「执行日志」中的**摘要**同步追加到工作目录下的 tomo2d_gui.log（UTF-8）："
        "开始/结束、命令行、失败信息等。**子进程 stdout/stderr 正文**体积大，仅显示在界面，不写入该文件。"
        "换工作目录后写入对应目录下的同名文件。"
    ),
    "btn.load_profile": (
        "从 JSON 恢复表单。"
        "若所选为运行包内 manifest.json：含 **gui_profile** 时恢复**整界面**（与「保存配置」同类的路径规范化记录；"
        "与 manifest 内 argv / python_replay 的运行包路径可并存对照）；"
        "否则（旧 manifest）仅用 python_replay 恢复 tt_inverse 页。"
        "若选普通全界面配置 JSON，行为同前。"
    ),
    "btn.preview_gen_smesh": (
        "根据当前选项生成 Python 调用预览，不运行外部程序；"
        "预览中含相对 work_dir 解析后的输入/输出路径及 ✓/✗（与 tt_forward 预览风格一致）。"
    ),
    "btn.run_gen_smesh": "在工作目录下执行 gen_smesh；日志在右侧。",
    "btn.preview_tt_forward": (
        "根据当前选项生成 tt_forward 调用预览；在「解析后命令行」之上会列出 smesh/geom/refl/clock "
        "等输入相对 work_dir 的绝对路径及是否存在（✓/✗）。"
    ),
    "btn.run_tt_forward": "在工作目录下执行 tt_forward。",
    "btn.preview_gen_damp": "预览 gen_damp 的 Python 调用参数；含相对 work_dir 的输入路径检查（✓/✗）。",
    "btn.run_gen_damp": "在工作目录下执行 gen_damp。",
    "btn.preview_gen_vcorr": (
        "预览 gen_vcorr：simple_2x2 显示将写出的 2×2 文本；program 显示二进制 argv。"
    ),
    "btn.run_gen_vcorr": "写出 vcorr 文件（simple_2x2 直接写盘；program 调用 gen_vcorr）。",
    "btn.preview_gen_dcorr": "预览 gen_dcorr 的 Python 调用参数；含相对 work_dir 的输入路径检查（✓/✗）。",
    "btn.run_gen_dcorr": "在工作目录下执行 gen_dcorr，把 x Lh 写入输出文件。",
    "btn.preview_tt_inverse": (
        "预览 tt_inverse 的 Python 调用参数；"
        "含 mesh/data、滤波/平滑/重力等相关路径相对 work_dir 的检查。"
        "勾选可复现运行包时，「解析后命令行」会按 run_dir 拼出 -M/-G/-L/-O/-K（含 inputs/、outputs/），与真实子进程一致。"
    ),
    "btn.run_tt_inverse": (
        "在工作目录下执行 tt_inverse（可能较耗时）。"
        "若勾选「可复现运行包」，则在 work_dir/runs/… 中快照输入并写 manifest.json，输出进该目录下的 outputs/。"
    ),
    "btn.tt_inverse_log_help": (
        "打开 **tt_inverse -L 日志**说明：数据行 26 列含义（迭代号、参数组号、剔除数、Pg/PmP RMS 与 χ²、"
        "图论/弯曲 CPU 时、平滑与阻尼权、LSQR 次数与 χ² 预测、扰动与粗糙度等）；联合重力时多一列重力 RMS。"
        "内容与 ``help_docs.TomoHelp.tt_inverse_logfile_format_help()``、``inverse.cc`` 写入格式一致。"
    ),
    "btn.preview_stat_smesh": "预览 stat_smesh 的 Python 调用参数；含表单中 *_file / *_bound 等路径检查。",
    "btn.run_stat_smesh": "在工作目录下执行 stat_smesh。",
    "btn.preview_edit_smesh": "预览 edit_smesh_HHB 的 Python 调用参数；含 smesh 与可选输入文件路径检查。",
    "btn.run_edit_smesh": "在工作目录下执行 edit_smesh_HHB。",
    "btn.preview_pipeline": (
        "预览当前流程将按什么顺序、用哪些参数执行；"
        "正文含各步输入路径检查块，且运行前仍会做一次输入文件存在性校验。"
    ),
    "btn.run_pipeline": "运行前会做输入文件存在性检查，任一步失败即停止。",
    "tab.tx_convert": (
        "将一个或多个 Zelt 系 tx.in 按震相筛选合并为同一份 ttimes.dat（tt_inverse -G）"
        "与 geom.dat（tt_forward -G）。多文件共用 station.lis，按炮号累积拾取。"
        "OBS 列表与「预览 tx.in」为同一套：勾选既用于显示也用于转换。\n"
        "tx.in→tomo2d：纯 Python 实现（等价 Fortran tx2tomo2d.f），无需编译。"
        "读 station.lis 与**一个或多个** tx.in，合并写出同一份 ttimes.dat（tt_inverse -G）与 geom.dat（tt_forward -g）。"
        "多文件共用台站表，按炮号累积拾取。"
        "折射与反射拾取均写成 **r** 行：第三列整数 **0=折射、1=反射**；"
        "**ttimes.dat** 中该行后两列为 **走时 t 与误差 u**（来自 tx.in），**geom.dat** 中同几何下 t、u 固定为 0。"
        "震相列表可分别留空。"
        "转换页与「预览 tx.in」共用同一套 OBS 列表（station.lis 号、n、左右支）；勾选同时用于显示与转换。"
        "「预览 tx.in / ttimes.dat」左侧列表第一列为 station.lis OBS 号（X=拾取点模型距离，折合 |x−xobs|/vred，同 iphase）。"
    ),
    "tx.station_lis": "【必选】台站/炮点表：每行 ishot x z（自由格式），与 tx.in 中 phase=0 分隔行的 x 匹配（容差 0.001）。",
    "tx.tx_in": (
        "【必选】一个或多个 tx.in：每行一个路径，或「添加多个…」多选。"
        "每文件格式：x t u phase；phase<0 结束；phase=0 为炮头；phase>0 为拾取震相号。"
        "多文件合并到同一对 ttimes.dat / geom.dat。"
    ),
    "tx.data_out": "【必选】可走「浏览…」选目录并取名，或手填路径。仅文件名（无目录）时相对当前「工作目录」；默认 ttimes.dat。",
    "tx.geom_out": "【必选】可走「浏览…」选目录并取名，或手填路径。仅文件名时相对 work_dir；默认 geom.dat（tt_forward -g）。",
    "tx.obs_ids": (
        "【可选】要转换的 OBS 号（station.lis 第一列）。"
        "空或 all=全部；none=不转换；否则逗号分隔如 12,30。"
        "与「预览 tx.in」左侧为同一列表，两处勾选同步。"
    ),
    "tx.refr_phases": (
        "折射震相编号（逗号或空格）。命中拾取写入 **r** 行，第三列整数 **0**（折射）。"
        "**ttimes.dat**：该 r 行写 **走时 t、误差 u**（与 tx.in 一致）；**geom.dat**：同位置与类型，t、u 写 0。"
        "台侧多次请填右侧「折射台侧多次」，不要写进本框。"
    ),
    "tx.refl_phases": (
        "反射震相编号（逗号或空格）；可留空。命中拾取写入 **r** 行，第三列整数 **1**（反射）。"
        "**ttimes.dat**：同样写 **走时 t、误差 u**；**geom.dat**：t、u 写 0。"
        "台侧多次请填右侧「反射台侧多次」，不要写进本框。"
    ),
    "tx.water_phases": (
        "【可选】直达水波在 tx.in 中的震相号（逗号或空格）。默认空：不转换。"
        "命中拾取写入 **r** 行，第三列整数 **2**。不要把水波填进「折射震相」。"
        "正反演时 data 含 2 必须提供 -Y/-F 海底。"
    ),
    "tx.mult_phases": (
        "【可选】一阶水柱多次在 tx.in 中的震相号。默认空：不转换。"
        "命中拾取写入 **r** 行，第三列整数 **3**。正反演时 data 含 3 必须提供 -Y/-F 海底。"
    ),
    "tx.refr_mult_phases": (
        "【可选】折射台侧一阶多次在 tx.in 中的震相号。默认空：不转换。"
        "命中拾取写入 **r** 行，第三列整数 **4**。不要填进「折射震相」。"
        "正反演时 data 含 4 必须提供 -Y 海底（无 -Y 时可沿用 -F）。"
    ),
    "tx.refl_mult_phases": (
        "【可选】反射台侧一阶多次在 tx.in 中的震相号。默认空：不转换。"
        "命中拾取写入 **r** 行，第三列整数 **5**。不要填进「反射震相」。"
        "正反演时 data 含 5 必须同时有 -Y 海底和 -F 莫霍。"
    ),
    "tx.psp_phases": (
        "【可选】折合 PSP（校正后 P–S–P）在 tx.in 中的震相号。默认空：不转换。"
        "命中拾取写入 **r** 行，第三列整数 **6**。不要填进「折射震相」。"
        "正演须 -X 转换面，反演须 -B 转换面。"
    ),
    "btn.preview_tx_convert": (
        "预览 convert_tx_in_to_tomo2d 等价调用与绝对路径；"
        "含台站/各 tx.in 是否存在及输出路径父目录是否存在的检查。"
    ),
    "btn.run_tx_convert": "在工作目录下执行转换（多 tx.in 合并，仅勾选的 OBS）；运行前检查输入文件存在。",
    "btn.plot_tx_in": (
        "绘制表单中的 tx.in（可多文件叠绘）。X=模型距离（拾取 x）；"
        "左侧勾选 OBS（与转换页共用）和震相；可调 vred 折合。导航为原生 pyqtgraph。"
    ),
    "btn.plot_ttimes": (
        "绘制输出路径上的 ttimes.dat。X=模型距离；"
        "左侧勾选 OBS 与震相（0 折射 / 1 反射 / 2 直达水波 / 3 水柱多次 / 4 折射台侧 / 5 反射台侧）；可调 vred。导航为原生 pyqtgraph。"
    ),
    "pipe.recipe": (
        "【必选】选流程模板；各步仍沿用对应页签的必填规则（如 tt_inverse 须 mesh/data）。"
        "合成数据可用「gen_smesh → tt_forward → tt_inverse」（需 fwd.out_ttime→inv.data）。"
        "Pipeline 内 tt_inverse 暂不走运行包监视。"
    ),
    "gui.auto_fwd_to_inv": (
        "【可选·合成数据】默认关闭。仅在用 tt_forward 造合成走时再反演时勾选："
        "成功后把空的 inv.mesh / inv.data（及 -N/-F）从正演页填入。"
        "日常实测反演不必开（反演内部已含正演）。"
    ),
    "pipe.link_smesh": "【可选】桥接：上游 smesh 路径，空则下游不自动填。",
    "pipe.link_damp": "【可选】桥接：gen_damp 产物 -> inv.damp_v_fn。",
    "pipe.link_vcorr_v": "【可选】桥接：gen_vcorr -> inv.smooth_corr_v_fn。",
    "pipe.link_vcorr_d": "【可选】桥接：已有 dcorr 文件 -> inv.smooth_corr_d_fn。",
    "pipe.link_dcorr": "【可选】桥接：gen_dcorr 产物 -> inv.smooth_corr_d_fn。",
    "pipe.auto_wire": "【可选】仅当下游对应框为空时才写入桥接路径。",
    "stat.mode": "【必选】list 或 mesh。",
    "stat.cmd_type": "【必选】list：a 或 r；mesh：a 或 b。",
    "stat.list_file": "【条件必选：mode=list】[-L] 文件列表。",
    "stat.mesh_file": "【条件必选：mode=mesh】[-M] 网格文件。",
    "stat.ave_file": "【条件必选：mode=list 且 cmd_type=r】[-Cr] 均值文件。",
    "stat.ave_x": "【条件必选：mesh + cmd_type=a】[-Da 第1项] 水平平均位置。",
    "stat.xmin": "【条件必选：mesh + cmd_type=b】[-Db 第1项] 水平窗 xmin。",
    "stat.xmax": "【条件必选：mesh + cmd_type=b】[-Db 第2项] 水平窗 xmax。",
    "stat.dx": "【条件必选：mesh + cmd_type=b】[-Db 第3项] 水平步长。",
    "stat.window_len": (
        "【条件必选：mesh+a 或 mesh+b】[-D 末项] 窗口长度。"
    ),
    "stat.top_bound": "【可选】[-T] 顶界。",
    "stat.bot_bound": "【可选】[-B] 底界。",
    "stat.mid_bound": "【可选】[-m] 中界。",
    "stat.verbose": "【可选】[-V] 详细输出。",
    "stat.refl_nnodes": "【可选】【条件：mode=list】[-R] 列表中每条反射线节点数（与 -Cr 联用读反射均值时）。",
    "stat.pt_corr": (
        "【可选】【条件：mode=mesh】[-P] 六段斜杠：Tref/Pref/dVdT/dVdP/a/b（见 stat_smesh.cc）。"
    ),
    "stat.vrepl": "【可选】【mesh】[-U] 低于该速度的值钳制为 vrepl。",
    "stat.abs_xmin": "【成组可选：与 abs_xmax 同组】【mesh】[-X] 参与统计的水平范围下限。",
    "stat.abs_xmax": "【成组可选】【mesh】[-X] 上限。",
    "stat.exclude_cxmin": (
        "【成组可选：与 exclude_cxmax、exclude_top_bound、exclude_bot_bound 同组】"
        "【mesh】[-x] 剔除大陆柱的水平范围下限。"
    ),
    "stat.exclude_cxmax": "【成组可选】【mesh】[-x] 上限。",
    "stat.exclude_top_bound": "【成组可选】【mesh】[-t] 剔除带上界文件。",
    "stat.exclude_bot_bound": "【成组可选】【mesh】[-b] 剔除带下界文件（小写 -b，勿与 -B 底界混淆）。",
    "edit.cmd_type": (
        "【必选】-C 子类型：a/p/P/B/s/rm/c/d/g/l/R/S/G/m/b（B→-CB removeBG；rm→-Cr；b 见 edit_smesh_help）。"
        "除 a、l 外大多需额外字段（见 tomand.edit_smesh）。"
    ),
    "edit.smesh_file": "【必选】待编辑 smesh 路径（位置参数）。",
    "edit.corr_file": "【可选】[-L] 相关文件。",
    "edit.upper_bound": "【可选】[-U] 上边界文件。",
    "edit.paste_file": "【条件必选：cmd_type=p】粘贴用 smesh 路径（-Cp…）。",
    "edit.prof_file": "【条件必选：cmd_type=P】1D 剖面文件（-CP…）。",
    "edit.remove_bg_file": (
        "【条件必选：cmd_type=B】1D 背景剖面文件（-CB…，edit_smesh_HHB removeBG；输出 v_water/v_air 会被置 999）。"
    ),
    "edit.h_len": "【条件必选：cmd_type=s/c】高斯平滑或棋盘格水平尺度（与 v_len 同组）。",
    "edit.v_len": "【条件必选：cmd_type=s/c】垂直尺度。",
    "edit.mx": "【条件必选：cmd_type=rm】水平加密因子（与 mz 同组）。",
    "edit.mz": "【条件必选：cmd_type=rm】垂向加密因子。",
    "edit.amp": "【条件必选：cmd_type=c/d/g/R/S/G】振幅或扰动强度（与各子命令字段同组）。",
    "edit.xmin": "【条件必选：cmd_type=d/S/G】矩形/随机区 xmin（与 xmax、zmin、zmax 等成组）。",
    "edit.xmax": "【条件必选：cmd_type=d/S/G】xmax。",
    "edit.zmin": "【条件必选：cmd_type=d/S/G】zmin。",
    "edit.zmax": "【条件必选：cmd_type=d/S/G】zmax。",
    "edit.x0": "【条件必选：cmd_type=g】高斯中心 x0。",
    "edit.z0": "【条件必选：cmd_type=g】高斯中心 z0。",
    "edit.Lh": "【条件必选：cmd_type=g】高斯水平半宽 Lh。",
    "edit.Lv": "【条件必选：cmd_type=g】高斯垂直半宽 Lv。",
    "edit.seed": "【条件必选：cmd_type=R/S/G】随机种子。",
    "edit.nrand": "【条件必选：cmd_type=R】随机扰动点数。",
    "edit.N": "【条件必选：cmd_type=G】高斯块个数。",
    "edit.dx": "【条件必选：cmd_type=S】子区水平步长（与 dz 及 xmin/xmax/zmin/zmax 同组）。",
    "edit.dz": "【条件必选：cmd_type=S】垂向步长。",
    "edit.vel": "【条件必选：cmd_type=m】莫霍面下速度（与 moho_file 同组）。",
    "edit.moho_file": "【条件必选：cmd_type=m】莫霍面文件。",
    "edit.k": "【条件必选：cmd_type=b】基底参数 k（与 base_file 同组）。",
    "edit.base_file": "【条件必选：cmd_type=b】基底界面文件。",
    "ui.scale": (
        "【可选】界面字号/缩放：状态区显示比例；「重置」恢复基准。"
        "Ctrl+滚轮调节（Windows/macOS：MouseWheel；Linux X11 可映射 Button 4/5）。"
    ),
    "ui.sizegrip": "【可选】右下角尺寸手柄，拖动改变窗口大小（位于底部状态条右侧）。",
    "ui.status_bar": (
        "底部状态条：显示「就绪」、命令执行进度（已运行 N 秒）及完成/失败与用时。"
        "青线以上为工作区，以下为固定状态区。"
    ),
}


def get_param_hint(key: str) -> str:
    return TOMO2D_GUI_HINTS.get(
        key,
        "暂无专门说明。请参阅本文件开头的【必选】/【可选】约定、"
        "modeling/tomo2d/readme.pdf、help_docs.py 或 tomand.py / src/*.cc。",
    )


def soft_wrap_tooltip(text: str, width: int = 48) -> str:
    """把长说明折成适合悬停气泡的多行文本（保留已有换行）。"""
    text = (text or "").strip()
    if not text:
        return ""
    out: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            out.append("")
            continue
        line = ""
        for ch in para:
            line += ch
            if len(line) >= width and ch in "，。；、：；,.!;?）)】」":
                out.append(line)
                line = ""
            elif len(line) >= width + 8:
                out.append(line)
                line = ""
        if line:
            out.append(line)
    return "\n".join(out)


def get_param_tooltip(key: str) -> str:
    """悬停用说明；无专门条目时返回空（不显示笼统兜底文案）。"""
    if key not in TOMO2D_GUI_HINTS:
        return ""
    return soft_wrap_tooltip(TOMO2D_GUI_HINTS[key])


def apply_param_tooltip(widget, key: str) -> None:
    """给 Qt 控件设置 param_hints 悬停说明。"""
    tip = get_param_tooltip(key)
    if tip and hasattr(widget, "setToolTip"):
        widget.setToolTip(tip)
