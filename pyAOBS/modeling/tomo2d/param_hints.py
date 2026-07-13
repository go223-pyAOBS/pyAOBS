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

# 键与 tomo2d_gui 中 self.vars / 控件对应
TOMO2D_GUI_HINTS: dict[str, str] = {
    "bin_path": (
        "【可选】tomo2d 各可执行文件所在目录（例如含 gen_smesh、tt_forward）。"
        "留空则依次使用环境变量 PYAOBS_TOMO2D_BIN / TOMO2D_BIN，或在 PATH 中查找。"
    ),
    "work_dir": (
        "【必选】运行命令时的当前工作目录。相对路径的输入/输出文件均相对于此目录解析。"
    ),
    "gen.vel_opt": (
        "【必选】速度场构造方式（二选一，与 grid_opt 一起决定 gen_smesh 能否运行）。"
        "uniform：V(z)=v0 + gradient × z（平坦海底、由 -A/-B 给定）。"
        "zelt：从 Zelt RAYINVR 的 v.in 构建速度场（-C），须指定海底层号 ilayer；"
        "此时 grid_opt 必须为 zelt（源码中 -C 同时计入速度与网格选项）。"
    ),
    "gen.grid_opt": (
        "【必选】网格类型（三选一）。"
        "uniform：均匀节点 -Nnx/nz -Dxmax/zmax（仅与 vel_opt=uniform）。"
        "variable：由 xfile、zfile 定义节点坐标，可选地形文件 tfile（-T）（仅与 vel_opt=uniform）。"
        "zelt：-E dx 与 -Z zfile（仅与 vel_opt=zelt）；选 zelt 速度时 GUI 会锁定此项。"
    ),
    "gen.v0": (
        "【条件必选：vel_opt=uniform】[-A] z=0（海底面）参考速度 v0（km/s）。"
        "速度随深度：V(z)=v0 + gradient × z（见 readme 4.1）。"
    ),
    "gen.gradient": (
        "【条件必选：vel_opt=uniform】[-B] dV/dz（km/s 每 km），即 V(z)=v0 + gradient × z。"
    ),
    "gen.v_in": (
        "【条件必选：vel_opt=zelt】[-C 前半] Zelt v.in（RAYINVR 风格）。"
        "因 gen_smesh.cc 用 sscanf 解析 ``-C路径/层号``，路径内不能有 ``/``；"
        "GUI 与 TomoAnd 只传**文件名**（无目录），文件须放在 **work_dir 根目录**（与运行 cwd 一致）。"
    ),
    "gen.ilayer": (
        "【条件必选：vel_opt=zelt】[-C 后半] v.in 中海面层号（readme 4.1）。"
    ),
    "gen.refl_layer": (
        "【条件必选：填写反射输出时】[-F 前半] 与 refl_file 成对；"
        "仅当二者都提供时 tomand 才传 -F（gen_smesh）。"
    ),
    "gen.refl_file": (
        "【条件必选：填写反射输出时】[-F 后半] 反射界面相关文件名；须与 refl_layer 同时填写。"
        "与 -C 同理：``-F层号/路径`` 的路径段不能含 ``/``，只填**文件名**且放在 **work_dir 根目录**。"
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
        "【可选】[-W] 水层厚度（gen_smesh.cc 中 wcol）；简单示例常不用。"
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
        "【可选】[-F] 反射面文件。"
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
        "【可选】[-T] 输出计算走时文件。"
    ),
    "fwd.out_ray": (
        "【可选】[-R] 输出射线路径文件。"
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
        "【可选】[-A] 反射震相更细处理，更耗时。"
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
        "uniform：-A 异常/正常阻尼；zelt：-C v.in/ilayer，可选 -F 层界（zelt 时 grid_opt 须为 zelt）。"
    ),
    "damp.grid_opt": "【必选】uniform / variable / zelt；zelt 配套规则同 gen_smesh（vel=zelt ⇔ grid=zelt）。",
    "damp.abnormal_damp": "【条件必选：vel_opt=uniform】[-A 前半] 异常区阻尼。",
    "damp.normal_damp": "【条件必选：vel_opt=uniform】[-A 后半] 正常区阻尼。",
    "damp.v_in": (
        "【条件必选：vel_opt=zelt】[-C 前半] v.in；规则同 gen.v_in：仅**文件名**、须在 **work_dir 根目录**。"
    ),
    "damp.ilayer": "【条件必选：vel_opt=zelt】[-C 后半] 海面层号。",
    "damp.top_layer": "【条件必选：填写层界时】[-F 前半] 与 bot_layer 成对。",
    "damp.bot_layer": "【条件必选：填写层界时】[-F 后半] 与 top_layer 成对。",
    "vcorr.vel_opt": (
        "【必选】与 grid_opt 一起；gen_vcorr 无二者则仅打印帮助。"
        "uniform：-A 四个相关长度；zelt：-C，可选 -F（zelt 时 grid_opt 须为 zelt）。"
    ),
    "vcorr.grid_opt": "【必选】uniform / variable / zelt；zelt 配套规则同 gen_smesh。",
    "vcorr.abnormal_h": "【条件必选：vel_opt=uniform】[-A 第1项] 异常区水平相关长度。",
    "vcorr.abnormal_v": "【条件必选：vel_opt=uniform】[-A 第2项] 异常区垂直相关长度。",
    "vcorr.normal_h": "【条件必选：vel_opt=uniform】[-A 第3项] 正常区水平相关长度。",
    "vcorr.normal_v": "【条件必选：vel_opt=uniform】[-A 第4项] 正常区垂直相关长度。",
    "vcorr.v_in": (
        "【条件必选：vel_opt=zelt】[-C 前半] v.in；规则同 gen.v_in：仅**文件名**、须在 **work_dir 根目录**。"
    ),
    "vcorr.ilayer": "【条件必选：vel_opt=zelt】[-C 后半] 海面层号。",
    "vcorr.top_layer": "【条件必选：填写层界时】[-F 前半] 与 bot_layer 成对。",
    "vcorr.bot_layer": "【条件必选：填写层界时】[-F 后半] 与 top_layer 成对。",
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
    "inv.refl_file": "【可选】[-F] 反射面文件。联合重力反演时原生程序要求必须提供 -F。",
    "inv.refl_weight": "【可选】[-W] 反射/界面项权重（refl_weight，默认 1；readme 称 depth kernel weighting）。",
    "inv.do_full_refl": "【可选】[-A] 完整反射计算。",
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
    "inv.out_level": "【可选】[-o] 输出级别（数值；非路径）。",
    "inv.dws_file": (
        "【可选】[-K] DWS **输出**路径（程序写入；浏览用「另存为」）。**不必**事先存在。"
        "**不勾选**运行包：留空则不传 -K。"
        "**勾选**运行包：留空时默认为 **outputs/dws.dat**；若填写则 **outputs/<basename>**。"
    ),
    "inv.crit_chi": "【可选】[-R] 稳健反演临界卡方。",
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
    "inv.damp_vel": "【可选】[-DV] 速度阻尼权重。",
    "inv.damp_dep": "【可选】[-DD] 深度阻尼权重。",
    "inv.damp_v_fn": (
        "【可选】[-DQ] 空间可变速度阻尼（sheared grid，readme 3.4）。"
        "仅对速度阻尼项生效，需同时设置 `-DV` 且权重 > 0。"
    ),
    "inv.auto_damp_max_dv": (
        "【可选】[-TV] 自动速度阻尼：最大速度扰动（%）。与 damp_vel/damp_dep/damp_v_fn 互斥，不可同填。"
    ),
    "inv.auto_damp_max_dd": "【可选】[-TD] 自动深度阻尼：最大深度扰动（%）。与固定阻尼互斥。",
    "inv.filter_bound_file": "【可选】[-s] 每次迭代后 2D 滤波边界文件（与平滑等联用，见 readme 4.3）。",
    "inv.jumping": "【可选】[-P] 纯跳跃策略（pure jumping）。",
    "inv.print_final_only": "【可选】[-l] 仅输出最终模型（否则逐步输出）。",
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
        "【可选】留空或 0：不传 -V；填正数：自动 verbose 并传 -V[level]。"
    ),
    "inv.cli_extras": (
        "tt_inverse 的上述开关已在表单与 TomoAnd.tt_inverse 中映射；完整语义仍以 readme.pdf 4.3 与 "
        "TomoHelp.tt_inverse_help() 为准。"
        "注意：-ZC/-ZU 等路径在 C 中用 sscanf 按「/」分段，路径内勿含额外「/」（可用相对路径）。"
        "性能/策略相关环境变量（不在表单中）："
        "TOMO2D_INV_OMP、TOMO2D_INV_REUSE_FORWARD + TOMO2D_INV_REUSE_THRESH、"
        "TOMO2D_INV_COARSE2FINE + TOMO2D_INV_C2F_*、TOMO2D_INV_LEGACY_BASELINE。"
    ),
    "tab.gen_smesh": (
        "gen_smesh：生成 sheared grid。"
        "【必选】vel_opt + grid_opt；vel_opt=zelt 时 grid_opt 必为 zelt（与 gen_smesh.cc 一致）。"
        "运行前还须填写「smesh 输出文件」；refl、水/空气、地形、-d dump 等见各字段提示。"
    ),
    "tab.tt_forward": (
        "tt_forward：图论 + 弯曲走时（readme 4.2）。"
        "【必选】smesh(-M)。【可选】geom(-G)、反射、输出、clock、graph_only、vgrid 子区六项、-n、verbose_level 等。"
        "六项 -N 为【成组可选】（见 fwd.xorder 等）；GUI 默认填 4/4/0.8/8/1e-4/1e-5，可清空六项则不传 -N。"
        "各输出文件（-T/-R/-E 等）可只填文件名（相对 work_dir），不必浏览。"
        "可通过环境变量 ``TOMO2D_FWD_OMP=1`` 启用 source 级并行（``-A`` full reflection 会自动回退串行）。"
    ),
    "tab.gen_damp": (
        "gen_damp：阻尼文件。【必选】vel_opt + grid_opt 及分支必填项（与 gen_smesh 相同，含 zelt 配套）。"
        "zelt 下 -F 层界为【条件必选】（填则成对）。API 可选 out_file 写 stdout。"
    ),
    "tab.gen_vcorr": (
        "gen_vcorr：相关长度等文件。【必选】vel_opt + grid_opt 及分支必填项（zelt 配套同 gen_smesh）；"
        "zelt 下 -F 同 gen_damp。API 可选 out_file 写 stdout。"
    ),
    "tab.tt_inverse": (
        "tt_inverse：走时反演（readme 4.3）。"
        "【必选】mesh(-M)、data(-G)。"
        "六项 -N 与 tt_forward 同义【成组可选】；GUI 默认 4/4/0.8/8/1e-4/1e-5，六项全空则不传（内层正演用程序内置默认）。"
        "本页含：反射 -F/-A/-W；策略 -P/-l；输出 -L/-O/-o/-K/-ZK；稳健 -R/-Q；迭代 -I/-J（GUI 默认 I=5、J=1.0、Q=1e-3，清空则不传）；"
        "平滑 -SV/-SD（单值或 wmin/wmax/dw）与 -XV/-XD；相关 -CV/-CD；"
        "固定阻尼 -DV/-DD/-DQ 与自动阻尼 -TV/-TD（二者勿同填）；滤波 -s；"
        "联合重力 -ZG～-ZT 在可折叠「联合重力」分组内；verbose_level。"
        "此外可通过环境变量启用前向复用与 coarse-to-fine（见程序帮助中的 tt_inverse 章节）。"
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
        "pipeline：顺序执行多步；每步沿用该命令在对应页签的【必选】规则。"
        "例如含 tt_inverse 时须事先填好 inv.mesh、inv.data；"
        "gen_smesh 步须满足 vel_opt/grid_opt 分支必填项。"
        "桥接路径为【可选】辅助填表。"
        "tt_inverse 步是否使用「可复现运行包」与 tt_inverse 页同一复选框一致。"
    ),
    "gui.plot_smesh_cmap": (
        "绘制 smesh 时使用的色标：留空为 **seismic**；可填 matplotlib 内置名（如 **viridis**）；"
        "或填 **GMT .cpt**（相对 **work_dir**）。**.cpt** 时色标范围默认取 CPT 文件内最小/最大 z，"
        "与 GMT 一致；若仍用数据分位数会导致低速水体颜色错位。"
    ),
    "btn.tt_inverse_analysis": (
        "读取 **tt_inverse -L** 输出的**数值行**日志（与 inverse.cc 每行一致），做反演质量诊断。"
        "**单日志**：绘制 RMS、initial χ²、pred χ² 随**迭代次数**曲线。"
        "**多日志**（每行一个路径，相对路径相对 **work_dir**）："
        "末步 RMS/χ² 柱状图（横轴附 **平滑/阻尼** sv/sd/dv/dd，对应日志列 14–17）；"
        "多 run 叠画迭代曲线；"
        "pred χ²–**粗糙度** Pareto 与综合得分 ``score = pred_χ² × (1 + w×R)``（w 可填，条形与左图散点同色）；"
        "各权重与 pred χ² / R 的散点（**反演参数与指标**）、**末步汇总表**。"
        "依赖 matplotlib、numpy。"
    ),
    "btn.program_help": (
        "打开程序帮助文档窗口，可切换查看 ``python_wrapper``、gen_smesh、tt_forward、tt_inverse、"
        "tt_inverse 日志列说明、stat_smesh、edit_smesh 等章节；内容来自 ``help_docs.TomoHelp``。"
        "其中 tt_inverse 章节已包含 TOMO2D_INV_REUSE_FORWARD / COARSE2FINE / LEGACY_BASELINE 的说明。"
    ),
    "btn.plot_smesh": (
        "选择 smesh 后，会询问反射界面：**是**=自选 tomo2d ``-F`` 文本（每行 x z），文件框默认打开**与当前 smesh 同目录**；**否**=若 "
        "**fwd.refl_file** / **inv.refl_file** 已填且文件在 work_dir 下存在则自动叠加；**取消**=不画反射线。"
        "随后弹出**一个**内嵌 Matplotlib 图窗（含缩放/移动、**保存**等工具栏）；**不再**自动写入 work_dir 下的 PNG。"
        "色标见上方「smesh 色标」；脚本侧可 ``plot_velocity_model(..., cmap='/path/to.cpt')``。"
        "依赖 matplotlib、xarray、numpy 等。"
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
    "btn.preview_gen_vcorr": "预览 gen_vcorr 的 Python 调用参数；含相对 work_dir 的输入路径检查（✓/✗）。",
    "btn.run_gen_vcorr": "在工作目录下执行 gen_vcorr。",
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
        "tx.in→tomo2d：纯 Python 实现（等价 Fortran tx2tomo2d.f），无需编译。"
        "读 station.lis 与 tx.in，写出 ttimes.dat（tt_inverse -G）与 geom.dat（tt_forward -g）。"
        "折射与反射拾取均写成 **r** 行：第三列整数 **0=折射、1=反射**；"
        "**ttimes.dat** 中该行后两列为 **走时 t 与误差 u**（来自 tx.in），**geom.dat** 中同几何下 t、u 固定为 0。"
        "震相列表可分别留空。"
    ),
    "tx.station_lis": "【必选】台站/炮点表：每行 ishot x z（自由格式），与 tx.in 中 phase=0 分隔行的 x 匹配（容差 0.001）。",
    "tx.tx_in": "【必选】tx.in：每行 x t u phase；phase<0 结束；phase=0 为炮头；phase>0 为拾取震相号。",
    "tx.data_out": "【必选】可走「浏览…」选目录并取名，或手填路径。仅文件名（无目录）时相对当前「工作目录」；默认 ttimes.dat。",
    "tx.geom_out": "【必选】可走「浏览…」选目录并取名，或手填路径。仅文件名时相对 work_dir；默认 geom.dat（tt_forward -g）。",
    "tx.refr_phases": (
        "折射震相编号（逗号或空格）。命中拾取写入 **r** 行，第三列整数 **0**（折射）。"
        "**ttimes.dat**：该 r 行写 **走时 t、误差 u**（与 tx.in 一致）；**geom.dat**：同位置与类型，t、u 写 0。"
    ),
    "tx.refl_phases": (
        "反射震相编号（逗号或空格）；可留空。命中拾取写入 **r** 行，第三列整数 **1**（反射）。"
        "**ttimes.dat**：同样写 **走时 t、误差 u**；**geom.dat**：t、u 写 0。"
    ),
    "btn.preview_tx_convert": (
        "预览 convert_tx_in_to_tomo2d 等价调用与绝对路径；"
        "含台站/tx.in 是否存在及输出路径父目录是否存在的检查。"
    ),
    "btn.run_tx_convert": "在工作目录下执行转换；运行前检查输入文件存在。",
    "pipe.recipe": (
        "【必选】选流程模板；各步仍沿用对应页签的必填规则（如 tt_inverse 须 mesh/data）。"
        "不会自动改写输出文件名。"
    ),
    "pipe.link_smesh": "【可选】桥接：上游 smesh 路径，空则下游不自动填。",
    "pipe.link_damp": "【可选】桥接：gen_damp 产物 -> inv.damp_v_fn。",
    "pipe.link_vcorr_v": "【可选】桥接：gen_vcorr -> inv.smooth_corr_v_fn。",
    "pipe.link_vcorr_d": "【可选】桥接：gen_vcorr -> inv.smooth_corr_d_fn。",
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
