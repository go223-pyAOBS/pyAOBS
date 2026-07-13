#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tomo2d 程序帮助文档

下文各 xxx_help() 方法描述的是 **原生可执行文件** 的命令行选项（与 TomoAnd 最终拼出的参数一致）。

Python 封装（可执行路径、缺参校验、subprocess 调用）见 TomoHelp.python_wrapper_help()。

GUI 侧逐字段说明（含【必选】/【可选】/【条件必选】/【成组可选】）见同目录 param_hints.py 中的 TOMO2D_GUI_HINTS。
"""

class TomoHelp:
    """
    tomo2d 原生命令帮助文档类；另提供 python_wrapper_help() 说明 TomoAnd 封装层。
    """

    @staticmethod
    def python_wrapper_help():
        """TomoAnd（Python 封装）的配置与行为说明。"""
        return """
TomoAnd — Python 封装说明

可执行文件目录（按优先级）:
  1) 构造参数: TomoAnd(bin_path=".../tomo2d/bin")
  2) 环境变量: PYAOBS_TOMO2D_BIN
  3) 环境变量: TOMO2D_BIN
  4) 默认: 当前 modeling/tomo2d 包目录；也可将程序加入 PATH 由系统解析

调用习惯:
  - 缺省关键参数时（如 gen_smesh 未给 vel_opt/grid_opt），会打印对应命令的帮助文本（与各 xxx_help() 一致），不启动外部程序。
  - 完整参数时通过 subprocess 列表方式调用，失败时抛出 TomoCommandError（含命令行与 stderr）。
  - 默认捕获子进程输出，结束后才能在返回值/异常里看到日志。要在终端实时查看进展，构造
    ``TomoAnd(..., capture_subprocess_output=False)``；对 tt_forward / tt_inverse 等还可传
    ``verbose=True``（拼 ``-V``，程序向 stderr 打印炮点/接收点等进度）。

标记含义（与 GUI param_hints.py 一致）:
  【必选】    调用 TomoAnd 时必须传入，否则仅打印帮助或抛出 ValueError（缺少必需参数）。
  【可选】    不传则不拼对应开关，由可执行文件使用其内置默认。
  【条件必选】仅当所选 vel_opt / grid_opt / mode / cmd_type 等分支成立时才需要。
  【成组可选】一组关键字要么全不传（不生成 -N），要么全传且合法；任一键出现则同组其余键必填。

各方法的 TomoAnd 必填规则摘要
------------------------------
edit_smesh(smesh_file, cmd_type, ...)
  【必选】smesh_file, cmd_type
  【条件必选】按 cmd_type 见下（a、l 无附加字段）；完整语义与坐标约定见 TomoHelp.edit_smesh_help()。
    p→paste_file; P→prof_file; B→remove_bg_file（-CB，HHB removeBG）;
    s→h_len,v_len; rm→mx,mz（生成 -Cr，见下文）;
    c→amp,h_len,v_len; d→amp,xmin,xmax,zmin,zmax; g→amp,x0,z0,Lh,Lv;
    R→seed,amp,nrand; S→…; G→…; m→vel,moho_file;
    b→k,base_file（TomoAnd 会拼 -Cb，但随包 edit_smesh_HHB.cc 无此分支，默认二进制将报错）
  【可选】corr_file（-L）, upper_bound（-U）

gen_smesh(**kwargs)
  【必选】vel_opt, grid_opt
  【配套】vel_opt=zelt 与 grid_opt=zelt 必须同时成立（gen_smesh.cc 中 -C 同时计入速度与网格计数）。
  【条件必选】vel_opt=uniform → v0, gradient
             vel_opt=zelt   → v_in, ilayer；若传 refl_layer 或 refl_file 则二者须成对齐全
  【条件必选】grid_opt=uniform  → nx, nz, xmax, zmax
             grid_opt=variable → x_file, z_file
             grid_opt=zelt     → dx, z_file（仅与 vel_opt=zelt）
  【可选】topo_file（仅 variable）, water_col, v_water, v_air, zelt_dump_file（-d，仅 vel_opt=zelt）,
         out_file（可选，将 stdout 慢度网格写入该路径；等价于 shell 下 gen_smesh ... > file）

gen_damp / gen_vcorr(**kwargs)
  【必选】vel_opt, grid_opt（网格分支同 gen_smesh；zelt 配套规则同 gen_smesh）
  【可选】out_file（将 stdout 写入路径，规则同 gen_smesh）
  gen_damp 【条件必选】uniform → abnormal_damp, normal_damp
           zelt → v_in, ilayer；若出现 top_layer 或 bot_layer 则须成对
  gen_vcorr【条件必选】uniform → abnormal_h, abnormal_v, normal_h, normal_v
           zelt → 同 gen_damp

tt_forward(smesh, geom=None, **kwargs)
  【必选】smesh（对应 -M）
  【可选】geom, refl_file, do_full_refl, out_opts 各项, vred, verbose, verbose_level,
         graph_only（-g）, clock_file（-C）, omit_air_water（-n）
  【成组可选】-N 六项: xorder, zorder, clen, nintp, tol1, tol2
             （与 tt_inverse 的 bend_cg_tol、bend_br_tol 同位；GUI 两页字段名与 inverse 一致，forward 收集为 tol1/tol2。）
             须同时传入；clen 与 tol 须 >0，否则原生程序会丢弃整条 -N。
  【成组可选】vgrid_subregion: (west,east,south,north,dx,dz)，须同时给出且配合 out_opts['vgrid']（-i）。

tt_inverse(mesh, data, **kwargs)
  【必选】mesh, data（-M / -G）
  【成组可选】-N 六项（规则同 tt_forward）
  【可选】refl_file, do_full_refl, refl_weight, jumping, print_final_only, filter_bound_file,
         log_file, out_root, out_level, dws_file, crit_chi, lsqr_tol, niter, target_chi2,
         auto_damp_max_dv, auto_damp_max_dd（与 damp_opts 互斥）,
         smooth_opts: vel/dep（数值或 "wmin/wmax/dw" 字符串）, corr_v_fn, corr_d_fn, vel_log10, dep_log10,
         damp_opts, gravity_opts（见 tt_inverse_help 联合重力段）, verbose, verbose_level

stat_smesh(**kwargs)
  【必选】mode（'list' | 'mesh'）
  mode=list  【必选】list_file, cmd_type（'a'|'r'）；cmd_type=r 时另【必选】ave_file
  mode=mesh 【必选】mesh_file, cmd_type（'a'|'b'）
             cmd_type=a → ave_x, window_len
             cmd_type=b → xmin, xmax, dx, window_len
  【可选】top_bound, bot_bound, mid_bound, verbose
  【可选】list 模式: refl_nnodes（-R）
  【可选】mesh 模式: pt_corr（-P 六段斜杠字符串）, vrepl（-U）, abs_xmin/abs_xmax（-X，须成对）,
         exclude_cxmin/exclude_cxmax/exclude_top_bound/exclude_bot_bound（-x/-t/-b，四项须齐）
  【条件必选】mesh 且 cmd_type=b 时须同时提供 top_bound, bot_bound, mid_bound（原生 -Db 要求）。

与本文档的关系:
  - 各 edit_smesh_help、gen_smesh_help 等描述的是 **命令行** 语义。
  - TomoAnd 的方法名将选项映射为关键字参数，实现见 tomand.py。
  - GUI 逐控件说明见 param_hints.TOMO2D_GUI_HINTS。
"""

    @staticmethod
    def edit_smesh_help():
        """编辑慢度网格文件的帮助文档（整理自 readme.pdf「edit smesh」与 src/edit_smesh_HHB.cc）。"""
        return """
edit_smesh_HHB — 编辑速度网格（sheared grid 文本格式）

手册（readme.pdf）中程序名为 edit_smesh；本包调用可执行文件 **edit_smesh_HHB**，对应源码
**edit_smesh_HHB.cc**（在 Korenaga 原版基础上增加 prof1d / removeBG 等，文件头注释有说明）。
程序从 stdin 无交互，**新网格写到 stdout**，典型用法：`edit_smesh_HHB old.dat -C... > new.dat`。

SYNOPSIS（与手册一致，可执行名替换为 edit_smesh_HHB）:
  edit_smesh_HHB <grid_file> -C<cmd> [ -L<vcorr_file> -U<upper_file> ]

位置参数:
  <grid_file>   输入网格，格式与 gen_smesh 输出一致：
                首行 nx nz v_water v_air；接着三行 x(i)、topo(i)、z(k)；再 nx×nz 个速度值。

坐标约定（读源码时的要点）:
  - 列水平位置为 xpos(i)；**绝对深度**常用 z_abs = topo(i) + zpos(k)（海底面以下 zpos 与 topo 相加）。
  - **-Cc** 棋盘格：在 k≥kstart(i) 的结点上用 x=xpos(i)、z=z_abs 计算扰动。
  - **-Cd** 矩形异常：循环 k 从 1 到 nz（**含浅部/水柱侧格点**），判据为 x 与 z_abs 落在矩形内。
  - **-Cg** 高斯异常：对全部 i,k 用 x、z_abs；振幅 A 在源码中先乘 0.01（百分数→小数）再乘到速度上。

kstart(i) 与 -U、-Cm:
  若给出 **-U<upper>** 或子命令为 **-Cm**（莫霍界面），程序读入同一类 **Interface2d** 文件，按界面深度得到每列起始有效垂向索引 kstart(i)：
  多种操作只对 k≥kstart 的结点施加以下的「壳内」编辑（**-Cd 除外**，见上）。
  **-Cm** 解析用 `sscanf(..., "%lf/%[^/]", ...)`：**Moho 文件路径中不能含字符 '/'**（否则与 C 分段冲突）。
  若同时使用 -U 与 -Cm，源码共用缓冲区读界面文件，实际以**最后一次**解析到的路径为准，应避免混用或需自行核对二进制行为。

全局可选:
  -L<corr_file>   相关长度场（CorrelationLength2d）。若存在：
                  • **-Cs** 平滑：在每个结点按 corr 改写高斯窗 Lh、Lv；
                  • **-CR** 随机：对扰动场再做一次与 -Cs 相同权重的空间平滑（pur → new_pur）。
  -U<upper_file>  上界深度（Interface2d），用于计算 kstart，限制「编辑深度」。

-C 子命令一览（argv 中紧跟 -C 的**第一个字符**为子类型；大小写敏感）:

  -Ca            每层垂向：对该层所有 i 求水平平均，再写回该层各点。
  -Cp<smesh>     将另一 sheared 网格的速度「贴入」：若物理点不在水/空（SlownessMesh2d::inWater/inAir），
                 则用另一网格插值速度覆盖（慢度网格内仍存速度）。
  -CP<1dprof>    粘贴 1D 剖面：以 Interface2d 读剖面；自 kstart 起，相对深度 z_rel = zpos(k)-zpos(kstart(i))，
                 用 prof1d.z(z_rel) 写入速度（手册称 ad hoc，剖面文件格式须与该用法一致）。
  -CB<1dprof>    **【edit_smesh_HHB.cc 扩展，readme 未列】** removeBG：读 1D 背景剖面；
                 k<kstart 的速度置 0；k≥kstart 处 v ← v − 背景(z_rel)；输出头 v_water、v_air 被置为 999。
                 TomoAnd：`cmd_type='B'`, `remove_bg_file=<1dprof>`。
  -CsLh/Lv       高斯平滑：|Δx|≤Lh 且 |Δz|≤Lv 范围内的加权平均（指数核）；默认 Lh,Lv 为常数，配合 -L 可空间变化。
                 仅 k≥kstart 与邻域 kk≥kstart(ii) 参与。
  -Crmx/mz       **加密网格**：注意命令行是字母 **r**，即 `-C` + `r` + `mx/mz`，例如 `-Cr2/2`。
                 在 x、z 方向分别将每个单元分为 mx、mz 段，双线性插值速度；**打印新网格后直接 exit(0)**，
                 不与其他子命令链式执行。TomoAnd 中 cmd_type=`rm` 会生成 `-Cr...`。
  -CcA/h/v       棋盘格：v ← v × (1 + 0.01×A×sin(2πx/h)×sin(2πz/v))，h、v 为水平/垂向「波长」尺度 (km)。
  -CdA/xmin/xmax/zmin/zmax
                 矩形域：域内 v ← v × (1 + 0.01×A)，A 为百分数；x、z_abs 为判据边界。
  -CgA/x0/z0/Lh/Lv
                 高斯型扰动：v ← v × (1 + A_frac × exp(-((x-x0)/Lh)²-((z_abs-z0)/Lv)²))，A 输入为百分数。
  -Cl            去低速区（LVZ）：对每列 i，自上而下扫描，使低于当前列最大速度的结点被抬升到该最大值。
  -CRseed/A/nrand
                 随机扰动：srand(seed)；A 为百分数；约每 nrand 个壳内结点施加随机振幅；可与 -L 联用平滑扰动场。
  -CSseed/A/xmin/xmax/dx/zmin/zmax/dz
                 另一随机化：在 x、z 方向离散网格上随机 sin 相位与振幅，再乘到壳内速度上（细节见源码随机系数）。
  -CGseed/A/N/xmin/xmax/zmin/zmax
                 再一种随机：在矩形区内随机 N 个高斯鼓包，各自随机中心、半宽与振幅，依次乘到 vgrid 上。
  -Cmv/mohofile  将 **Moho 界面以下**（k≥kstart）的速度一律设为常数 v；mohofile 为 Moho 界面（Interface2d）。

  **-Cb**（TomoAnd cmd_type=`b`）:
  TomoAnd 会拼接 `-Cb<k>/<basefile>`，但 **edit_smesh_HHB.cc 与 edit_smesh.cc 均无 `case 'b':`**，
  用本仓库源码编译的 edit_smesh_HHB 会报 **invalid -C option**。若你手头的二进制支持 -Cb，应来自其它分支/补丁。

手册 EXAMPLES（命令名可写作 edit_smesh 或 edit_smesh_HHB）:
  矩形 −5% 异常，x∈[10,20] km，z∈[5,8] km:
    edit_smesh vgrid.orig.dat -Cd-5/10/20/5/8 > vgrid.new.dat
  棋盘格 3%，水平波长 10 km、垂向 5 km:
    edit_smesh vgrid.orig.dat -Cc3/10/5 > vgrid.new.dat

TomoAnd.edit_smesh: 将 cmd_type 与字段映射为上述 -C 串，并可选附加 -L / -U；cmd_type='B' 对应 -CB。
"""

    @staticmethod
    def gen_smesh_help():
        """生成慢度网格文件的帮助文档"""
        return """
gen_smesh - 生成慢度网格文件

用法: gen_smesh [选项]

速度设置选项:
  [均匀梯度]:
    -A<v0>        初始速度
    -B<gradient>  速度梯度
  
  [Zelt格式输入]:
    -C<v.in>/<ilayer>  
                  从Zelt格式文件读取；须指定海面层号 ilayer
    -F<jlayer>/<refl_file>
                  可选。将 jlayer 层顶作为反射界面写入 refl_file（如 Moho）
    -d<dump_file>
                  可选，与 Zelt 流程联用：将节点信息等到 dump_file（gen_smesh.cc -d）

网格生成选项:
  [均匀网格]:
    -N<nx>/<nz>   网格点数
    -D<xmax>/<zmax>
                  最大范围
  
  [变间距网格]:
    -X<xfile>     x坐标文件
    -Z<zfile>     z坐标文件
    -T<tfile>     地形文件
  
  [基于Zelt的网格]:
    -E<dx>        x方向间距
    -Z<zfile>     z坐标文件

其他选项:
  -W<wcol>       水层厚度（显式水柱）
  -Q<v_water>    水层速度 [默认1.5 km/s]
  -R<v_air>      空气层速度 [默认0.33 km/s]

示例（readme 4.1）:
  均匀梯度+均匀网格: -A3.0 -B0.5 -N51/51 -D25/10
  均匀梯度+变间距:  -A3.0 -B0.5 -X<xfile> -Z<zfile> [-T<tfile>，tfile 点数须与 xfile 一致]
  Zelt+近似水平距:  -C<v.in>/<ilayer> -E<dx> -Z<zfile> [ -F<j>/<refl> ]；程序可在原地形节点间加密

TomoAnd: vel_opt+grid_opt 为【必选】；zelt_dump_file（-d）见 python_wrapper_help()（仅 zelt）。
"""

    @staticmethod
    def gen_damp_help():
        """生成阻尼文件的帮助文档"""
        return """
gen_damp - 生成阻尼文件

用法: gen_damp [选项]

说明: readme 正文以 gen_smesh 为主；阻尼网格的构造方式与 gen_smesh 的速度+网格选项一一对应。

速度设置选项:
  [均匀梯度]:
    -A<abnormal>/<normal>
                  异常和正常区域的阻尼值
  
  [Zelt格式输入]:
    -C<v.in>/<ilayer>
                  从Zelt格式文件读取
    -F<top_layer>/<bot_layer>
                  指定上下边界层

网格生成选项:
  与gen_smesh相同:
    [均匀网格]: -N<nx>/<nz> -D<xmax>/<zmax>
    [变间距网格]: -X<xfile> -Z<zfile> -T<tfile>
    [Zelt网格]: -E<dx> -Z<zfile>

TomoAnd: 规则同 gen_smesh；uniform 须 abnormal_damp/normal_damp；zelt 须 v_in/ilayer。
"""

    @staticmethod
    def gen_vcorr_help():
        """生成速度相关文件的帮助文档"""
        return """
gen_vcorr - 生成速度相关文件

用法: gen_vcorr [选项]

说明: 与 gen_smesh 共用同一套网格生成语义；相关长度分区与 Zelt/uniform 的对应关系见 readme 及源码。

速度设置选项:
  [均匀梯度]:
    -A<abnormal_h>/<abnormal_v>/<normal_h>/<normal_v>
                  异常和正常区域的水平/垂直相关长度
  
  [Zelt格式输入]:
    -C<v.in>/<ilayer>
                  从Zelt格式文件读取
    -F<top_layer>/<bot_layer>
                  指定上下边界层

网格生成选项:
  与gen_smesh相同:
    [均匀网格]: -N<nx>/<nz> -D<xmax>/<zmax>
    [变间距网格]: -X<xfile> -Z<zfile> -T<tfile>
    [Zelt网格]: -E<dx> -Z<zfile>

TomoAnd: 规则同 gen_smesh；uniform 须四段相关长度；zelt 须 v_in/ilayer。
"""

    @staticmethod
    def tt_forward_help():
        """正演走时计算的帮助文档"""
        return """
tt_forward - 正演走时计算

用法（readme 4.2 / tt_forward.cc）:
  tt_forward -M<mesh> [ -G<geom> -F<refl> -A ]
    [ -N<x>/<z>/<clen>/<nintp>/<tot1>/<tot2> -E<elem> -g
      -T<ttime> -O<obs> -r<v0> -D<diff> -R<ray> -S<src> -I<vel>
      -i<w>/<e>/<s>/<n>/<dx>/<dz> -n -C<clock> -V[level] ]

  亦支持将路径单独作为下一参数，例如 ``-M /mnt/e/mesh.dat``（与 ``-M/mnt/e/mesh.dat`` 等价；
  TomoAnd 对 POSIX 绝对路径、Windows 盘符/UNC、以 ``-`` 开头的文件名采用拆参形式；需使用已含该解析的 tt_forward 可执行文件）。

必需参数:
  -M<smesh>      慢度/速度网格文件

几何与反射（无 -G 时仅做与网格有关的输出，不进行震源-接收正演）:
  -G<geom>       几何文件（格式同走时数据文件，走时与误差可填 0）
  -F<refl>       反射界面文件
  -A             反射震相额外精细处理（更耗时）

数值参数（可选；不传则用程序内置默认。TomoAnd：六项须同时给出且合法，否则不要传 -N）:
  -N<xorder>/<zorder>/<clen>/<nintp>/<tot1>/<tot2>
                  图论前向星阶数 xorder、zorder；弯曲段最大长度 clen；每段插值点数 nintp；
                  弯曲迭代容差 tot1（共轭梯度）、tot2（Brent）

仅网格相关输出（可不配 -G）:
  -E<elem>       输出网格单元列表
  -I<vel>        输出速度网格到 vel 文件
  -i<west>/<east>/<south>/<north>/<dx>/<dz>
                  与 -I 联用：仅输出该矩形范围、以 dx/dz 为间距的速度网格（西/东/南/北边界）
  -n             与 -I 联用：抑制输出空气层与水层节点（tt_forward.cc printAW=false）

正演与算法开关（须配合 -G）:
  -g             仅用图论法、关闭弯曲细化（graphOnly）
  -C<clock_file> 使用时钟文件

走时与射线输出（须配合 -G）:
  -T<ttime>      计算走时输出文件
  -O<obs_ttime>  将输入观测走时抄出到文件
  -r<v0>         折合速度 v0（折合走时输出）
  -D<diff>       差分走时及 misfit/chisq 等
  -R<ray>        射线路径
  -S<src>        震源位置

其他:
  -V[level]      verbose（level 0 或 1）

OpenMP 并行（源码 ``syngen.cc`` 的 source 级并行）:
  需在编译时启用 OpenMP，并在运行前设置：
    export OMP_NUM_THREADS=3
    export TOMO2D_FWD_OMP=1
  说明：
    • 并行粒度是 source（``for isrc``），线程数建议不超过 source 数。
    • ``-A``（full reflection）模式下会自动回退串行（需临时改写共享网格）。
    • 并行时逐 source 进度符号（* . + #）会被抑制；射线路径输出仍按 source 顺序写入。
    • ``-V-1`` 已支持，等效静默级别。

TomoAnd / GUI：已映射 -N/-r/-V、graph_only/clock_file/omit_air_water、vgrid_subregion（-i）等，见 tomand.tt_forward。
"""

    @staticmethod
    def tt_inverse_help():
        """走时反演的帮助文档（整理自 readme.pdf 4.3 节与 tt_inverse.cc；选项以可执行为准）。"""
        return """
tt_inverse - 走时反演（折射 + 反射联合走时层析；可选联合重力）

用法（readme 4.3 SYNOPSIS 概括）:
  tt_inverse -M<grid> -G<data>
    [ -N<x>/<z>/<clen>/<nintp>/<tol1>/<tol2> ]
    [ -F<refl> -A -W<weight> -L<log> -O<root> [ -o<level> ] -l -K<dws> ]
    [ -P -R<crit_chi> -Q<lsqr_tol> -s<bound_file> -V[level] ]
    [ -CV<vcorr> -CD<dcorr> ]
    [ 迭代选项 ] [ 平滑选项 ] [ 阻尼选项 ] [ 联合重力选项 ]

说明: 程序内含正演；-N 六项含义同 tt_forward（readme 与 tt_forward 均记为 tol1/tol2）。

必需参数:
  -M<mesh>       速度网格文件（sheared grid）
  -G<data>       走时数据文件（格式见 readme 3.2）

数值参数（可选；六项须同时合法，clen 与各 tol 须 >0，否则整条 -N 被忽略）:
  -N<xorder>/<zorder>/<clen>/<nintp>/<bend_cg_tol>/<bend_br_tol>
                  图论阶数、路径细分长度、样条控制点数、弯曲 CG/Brent 容差（同 tt_forward）

反射与权重:
  -F<refl>       反射界面文件（联合重力反演时源码要求必须提供反射面）
  -A             反射走时更精细计算（与 tt_forward -A 同义）
  -W<weight>     readme 称为 depth kernel weighting factor；源码中为反射权重 refl_weight（默认 1）

输出与日志:
  -L<logfile>    迭代日志（详见 ``TomoHelp.tt_inverse_logfile_format_help()``：非 ``#`` 行共 26 列数值；
                 联合重力时行末多 1 列 RMS gravity misfit）。文件首若干 ``#`` 行为配置摘要。
  -O<outroot>    输出文件根名
  -o<level>      输出级别：1 可打出走时残差；2 可打出射线路径（readme 4.3）
  -l             仅输出最终模型（否则按步输出中间结果）
  -K<dws_file>   输出 DWS 至指定文件

稳健性与求解控制:
  -P             纯跳跃策略（pure jumping）
  -R<crit_chi>   稳健反演临界 chi（>0 时启用）
  -Q<lsqr_tol>   LSQR 容差（注意：空间可变阻尼**文件**在源码中为 -DQ；readme 3.4 英文版若写作 -Q 易与 LSQR 混淆）
  -s<bound_file> 每次迭代后做 2-D 滤波；边界由 bound 文件给出（与速度平滑同时启用时由 inverse 调用）

详细输出:
  -V[level]      verbose；可接数字级别

OpenMP 并行（源码 ``inverse.cc`` 的 source 级并行）:
  需在编译时启用 OpenMP（如 CMake: ``-DTOMO2D_ENABLE_OPENMP=ON``），并在运行前设置环境变量：
    export OMP_NUM_THREADS=3
    export OMP_PLACES=threads
    export OMP_PROC_BIND=spread
    export TOMO2D_INV_OMP=1
    export TOMO2D_INV_REUSE_FORWARD=1
    export TOMO2D_INV_REUSE_THRESH=1e-3
    export TOMO2D_INV_COARSE2FINE=1
    export TOMO2D_INV_C2F_SMOOTH_START=3
    export TOMO2D_INV_C2F_SMOOTH_END=1
    export TOMO2D_INV_C2F_DAMP_START=3
    export TOMO2D_INV_C2F_DAMP_END=1
    # A/B 对比时可一键回退优化开关
    export TOMO2D_INV_LEGACY_BASELINE=1
  说明：
    • 并行粒度是 source（``for isrc``），线程数建议不超过 source 数。
    • ``-V-1`` 已支持并可正确解析为 ``verbose_level=-1``；并行时逐 source 的细粒度进度符号（* . + #）会被抑制，以避免多线程输出交错。
    • 若使用 ``-A``（full reflection）则会退回串行（共享网格写入路径）。
    • ``TOMO2D_INV_REUSE_FORWARD`` 仅在 ``TOMO2D_INV_REUSE_THRESH>0`` 时生效。
    • ``TOMO2D_INV_COARSE2FINE`` 启用分阶段反演：前期增强平滑/阻尼，后期逐步回落到目标参数；四个 C2F 参数应为正值。
    • ``TOMO2D_INV_LEGACY_BASELINE=1`` 会同时回退四项：关闭前向复用、kernel 归并回退 ``vector<pair>+sort+merge``、关闭 LSQR 列预条件、关闭 coarse-to-fine。

相关长度（平滑前须满足源码校验）:
  -CV<vcorr>     速度节点相关长度文件（格式 readme 3.3；启用速度平滑 -SV 时**必须**）
  -CD<dcorr>     反射点相关长度文件；若不提供，程序可从速度水平相关长度采样（readme 3.3 Note）

迭代类型 1（多迭代、单组参数）:
  -I<niter>      最大迭代次数
  -J<target_chi2>目标卡方 chi^2
  -SV<wsv>       速度平滑权重（单值形式）
  -SD<wsd>       界面深度平滑权重（单值形式）

迭代类型 2（单次迭代、多组平滑权重扫描）:
  -SV<wsv_min>/<wsv_max>/<dw>   速度平滑权重从 min 到 max 步进 dw；可配合 -XV
  -SD<wsd_min>/<wsd_max>/<dw>   深度平滑权重扫描；可配合 -XD
  -XV            速度平滑权按 10 的幂缩放（与 readme / 源码 -X 子选项 V 一致）
  -XD            深度平滑权按 10 的幂缩放

阻尼（自动与固定二者互斥，源码 -T 与 -D 不能同时开）:
  -TV<max_dv>    自动速度阻尼：最大速度扰动（%）
  -TD<max_dd>    自动深度阻尼：最大深度扰动（%）
  -DV<wdv>       固定速度阻尼权重
  -DD<wdd>       固定深度阻尼权重
  -DQ<damp_file> 空间可变速度阻尼（sheared grid 格式，readme 3.4；挤压试验 squeezing）
                 注意：``-DQ`` 仅作用于速度阻尼项，需同时给出 ``-DV`` 且 ``wdv>0``；
                 否则程序会报错退出（避免静默无效）。
                 作用机制（源码）：先在 ``calc_damping_matrix()`` 中对局部速度阻尼核乘空间权重 ``w(x,z)``，
                 再在 ``_solve()`` 中整体乘固定阻尼系数 ``wdv``；因此局部等效强度可近似理解为
                 ``wdv × w(x,z)``（严格上为耦合矩阵项，不是逐点独立惩罚）。

联合重力反演（readme「Joint gravity inversion options」；须 -F 反射面，且至少一种域 + -ZX/-ZR）:
  -ZG<grav_file> 重力数据：每行 x(km)、自由空气异常(mGal)；点数宜少（核矩阵大）
  -ZX<xmin>/<xmax>/<zmin>/<zmax>/<dx>/<dz>  重力正演用细网格
  -ZC<cont_up>/<iconv>           大陆地壳顶界 + 速度-密度换算类型
  -ZU<ocean_U_up>/<ocean_U_lo>/<iconv>     洋壳上层顶/底界 + iconv
  -ZL<ocean_L_up>/<iconv>        洋壳下层顶界 + iconv
  -ZS<sed_up>/<sed_lo>/<iconv>    沉积层顶/底 + iconv
  -ZD<dvdp>/<dvdt>/<drdp>/<drdt>/<dTdz>    原位 dV/dP、dV/dT、dρ/dP、dρ/dT 与地温梯度 dT/dz
  -ZR<x0>/<x1>    参考密度柱的水平范围 (km)
  -ZW<weight>    重力相对走时的权重（默认 1.0）
  -ZZ<z0>        参考深度 (km，向下为正；默认 0)
  -ZK<grav_dws>  输出重力 DWS 文件
  -ZT<cut_range>/<cut_val>      灵敏度截断：距观测点距离 > cut_range 且 |灵敏度| < cut_val 的节点置零

边界文件格式与反射界面相同：每行一个水平坐标与边界深度（readme 文末 Note）。

TomoAnd（Python）/ GUI：上述选项均已映射为关键字参数（见 python_wrapper_help() 中 tt_inverse 条）。
  gravity_opts 字典键：grav_file, grid_spec, refrange, continent (path,iconv), ocean_upper (up,lo,iconv),
  ocean_lower (up,iconv), sediment (up,lo,iconv), deriv, weight_grav, z0, grav_dws, cutoff。
  路径中含「/」的界面文件可能与 sscanf 分段冲突，宜用无斜杠的相对路径。

环境变量优化开关（非命令行参数）摘要:
  TOMO2D_INV_OMP=1                 启用 source 级 OpenMP 并行路径。
  TOMO2D_INV_REUSE_FORWARD=1       启用前向复用（需 TOMO2D_INV_REUSE_THRESH>0）。
  TOMO2D_INV_REUSE_THRESH=1e-3     前向复用阈值（相邻迭代模型相对变化阈值）。
  TOMO2D_INV_COARSE2FINE=1         启用 coarse-to-fine 分阶段反演。
  TOMO2D_INV_C2F_SMOOTH_START/END  平滑阶段因子起止（默认 3→1）。
  TOMO2D_INV_C2F_DAMP_START/END    阻尼阶段因子起止（默认 3→1）。
  TOMO2D_INV_LEGACY_BASELINE=1     一键回退到优化前基线（前向复用/归并优化/LSQR 预条件/coarse-to-fine 全关闭）。
"""

    @staticmethod
    def tt_inverse_logfile_format_help() -> str:
        """
        ``-L`` 日志正文格式说明，与 ``inverse.cc`` 中 ``TomographicInversion2d::solve`` 写入一致
        （非 ``#`` 开头的数据行；联合重力时末尾追加一列）。
        """
        return """
tt_inverse -L 日志文件格式（数据行）

【文件结构】
  • 开头多行以 ``#`` 开头的注释：策略、射线参数、平滑/阻尼、数据量、节点数、LSQR 容差等
    （``inverse.cc`` 约 296–317 行）。
    若启用 ``-DQ`` 且 ``-DV>0``，还会出现 ``# squeezing enabled wdv=...`` 以确认空间可变速度阻尼已生效。
  • 若启用滤波，在相应迭代可能出现一行 ``# a posteriori filter check: …``（约 720–723 行）。
  • 正文为若干行空格分隔数值；**每一行对应一次「迭代 iter × 参数组 iset」** 结束后的统计
    （``inverse.cc`` 约 764–779 行）。

【数据行列号与物理量】（与 readme 4.3 中 -L 说明一致；震相按数据 ``r`` 行第三列 code 分为两类：
  code=0 记入列 6–8，code=1 记入列 9–11，通常对应折射 Pg 与反射 PmP。）

  1.  iteration（迭代序号）
  2.  set number（本迭代内平滑/阻尼参数组序号 iset）
  3.  剔除数据条数（ndata - ndata_valid，稳健/滤波后未参与 LSQR 的拾取）
  4.  走时 RMS 残差（Pg+PmP 合并，对全部有效数据）
  5.  初始 χ²（Pg+PmP 合并，对有效数据平均，见源码 init_chi_total/ndata_valid）
  6.  有效 Pg 类数据条数（raytype code 0，ndata_in[0]）
  7.  Pg 类走时 RMS 残差
  8.  Pg 类初始 χ²（按该类数据平均）
  9.  有效 PmP 类数据条数（raytype code 1，ndata_in[1]）
 10.  PmP 类走时 RMS 残差
 11.  PmP 类初始 χ²（按该类数据平均）
 12.  图论射线追踪 CPU 时间（秒，graph_time）
 13.  弯曲法射线追踪 CPU 时间（秒，bend_time）
 14.  速度节点平滑权重 weight_s_v
 15.  深度节点平滑权重 weight_s_d
 16.  速度阻尼权重 wdv
 17.  深度阻尼权重 wdd
 18.  LSQR 调用次数 nlsqr
 19.  LSQR 累计迭代次数 lsqr_iter
 20.  LSQR 所用 CPU 时间（秒，lsqr_time）
 21.  由 LSQR 解预测的 χ² pred_chi（calc_chi()）
 22.  平均速度扰动度量 dv_norm（calc_ave_dmv）
 23.  平均深度扰动度量 dd_norm（calc_ave_dmd）
 24.  速度节点水平粗糙度 Lmvh
 25.  速度节点垂向粗糙度 Lmvv
 26.  深度节点粗糙度 Lmd

【联合重力】
  若启用 ``addonGravity``，在上述 26 列之后同一行再输出一列：重力 RMS misfit（rms_grav）。

【前向复用说明】
  若启用 ``TOMO2D_INV_REUSE_FORWARD=1`` 且达到 ``TOMO2D_INV_REUSE_THRESH`` 阈值，
  该迭代会复用上一轮前向结果，此时本行 ``graph_time`` / ``bend_time`` 可能显著下降甚至接近 0，属于预期行为。

【参考源码】
  ``modeling/tomo2d/src/inverse.cc``：``solve`` 内 ``if (printLog) { *log_os_p << iter << … }``。
"""

    @staticmethod
    def stat_smesh_help():
        """慢度网格统计的帮助文档"""
        return """
stat_smesh - 慢度网格统计（readme；stat_smesh.cc）

用法:
  stat_smesh -L<list_file> -C<cmd> [ -R<n> ]
  stat_smesh -M<mesh> -D<cmd>
    [ -T<topb> -B<botb> -m<midb>
      -P<Tref>/<Pref>/<dVdT>/<dVdP>/<a>/<b>
      -U<vrepl> -X<xmin>/<xmax>
      -x<cxmin>/<cxmax> -t<ctopb> -b<cbotb> ]
  [ -V ]

文件列表模式:
  -L<list>       网格或反射界面文件列表
  -Ca            集合平均
  -Cr<ave_file>  相对 ave_file 求标准差（RMS）
  -R<n>          列表中为反射界面且每条约 n 个节点（非速度网格）

单网格模式:
  -M<mesh>       单个速度网格
  -Da<avex>/<wlen>
                 在 x=avex 处取水平平均，窗口半宽 wlen (km)
  -Db<xmin>/<xmax>/<dx>/<wlen>
                 自 xmin 到 xmax、步长 dx 做水平+垂向块平均，窗口 wlen (km)
                 （源码要求同时提供 -T -B -m 顶/底/中界文件）

温压校正（可选）:
  -P<Tref>/<Pref>/<dVdT>/<dVdP>/<a>/<b>
                 Tref(C)、Pref(MPa)、dV/dT、dV/dP；地温 T=a*z+b（z 为海底以下深度）

速度与范围（可选）:
  -U<vrepl>      将低于 vrepl 的速度钳制为 vrepl
  -X<xmin>/<xmax> 运算有效水平范围

剔除大陆等区域（可选，须与 -x 同时给 -t -b）:
  -x<cxmin>/<cxmax>  剔除带水平范围
  -t<ctopb>          剔除带上边界文件
  -b<cbotb>          剔除带下边界文件

边界（mesh 模式常用）:
  -T<topb>  -B<botb>  -m<midb>   顶/底/中界（与反射界面相同格式：x、深度）

其他:
  -V             详细输出

TomoAnd / GUI：list/mesh 主流程及 -R/-P/-U/-X/-x/-t/-b 等均已映射（见 tomand.stat_smesh）。
""" 