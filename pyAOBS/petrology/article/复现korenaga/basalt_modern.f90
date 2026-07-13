module basalt_types
    implicit none

    integer, parameter :: MAX_PHAS = 10
    integer, parameter :: MAX_COMP = 20
    integer, parameter :: MAX_SYS = 5

    type :: phase_eq_state
        integer :: ns
        character(len=4) :: snames(MAX_SYS)
        real(8) :: sparm(MAX_SYS)
        
        integer :: nphas, ncomp, ncimp, ncompt, npdim, ncdim
        character(len=4) :: pnamea(MAX_PHAS)
        character(len=4) :: cnamej(MAX_COMP)
        
        real(8) :: fl
        real(8) :: fa(MAX_PHAS)
        real(8) :: csj(MAX_COMP)
        real(8) :: clj(MAX_COMP)
        real(8) :: caj(MAX_PHAS, MAX_COMP)
        real(8) :: ta(MAX_PHAS)
        real(8) :: uaj(MAX_PHAS, MAX_COMP)
        real(8) :: fkdaj(MAX_PHAS, MAX_COMP)
        
        integer :: impdim
        integer :: impl(MAX_PHAS)
        integer :: imcl(MAX_PHAS)
        real(8) :: da0(MAX_PHAS)
        real(8) :: daj(MAX_PHAS, MAX_COMP)
        
        real(8) :: rj(MAX_COMP)
        real(8) :: qa(MAX_PHAS)
        real(8) :: pab(MAX_PHAS, MAX_PHAS)
    end type phase_eq_state

    type :: calc_params
        real(4) :: ti, tf, dt, dp, temp_offset, flr
        real(4) :: p_high, p_low, p_step  ! 压力参数(kbar), 用于多压力模式
    end type calc_params

    type :: calc_modes
        integer :: model_type
        logical :: init_on
        logical :: printer_on
        logical :: summary_only
        logical :: changes_only
        logical :: polybaric  ! .true.=多压力模式, .false.=单压力模式
    end type calc_modes

end module basalt_types

module basalt_constants
    use basalt_types
    implicit none

    contains

    subroutine initialize_state(state)
        type(phase_eq_state), intent(out) :: state
        
        state%ns = 2
        state%snames = ['T K ', 'P KB', '    ', '    ', '    ']
        state%sparm = [1273.16d0, 0.d0, 0.d0, 0.d0, 0.d0]
        
        state%nphas = 3
        state%ncomp = 6
        state%ncimp = 1
        state%ncompt = 7
        state%npdim = 10
        state%ncdim = 20
        
        state%pnamea = ['PLAG', 'OL  ', 'CPX ', '    ', '    ', &
                        '    ', '    ', '    ', '    ', '    ']
        state%cnamej = ['CAA ', 'NAAL', 'MGO ', 'FEO ', 'CAWO', &
                        'TIO2', 'SIO2', '    ', '    ', '    ', &
                        '    ', '    ', '    ', '    ', '    ', &
                        '    ', '    ', '    ', '    ', '    ']
        
        state%fl = 1.0d0
        state%fa = 0.0d0
        state%csj = 0.0d0
        state%clj = 0.0d0
        state%caj = 0.0d0
        state%fkdaj = 0.0d0
        state%rj = 1.0d0
        state%qa = 0.0d0
        state%pab = 0.0d0
        
        state%ta = [1.0d0, 0.666667d0, 1.0d0, 0.d0, 0.d0, &
                    0.d0, 0.d0, 0.d0, 0.d0, 0.d0]
        
        state%uaj = 0.0d0
        state%uaj(1, 1) = 1.666667d0
        state%uaj(1, 2) = 2.5d0
        state%uaj(2, 3) = 1.0d0
        state%uaj(2, 4) = 1.0d0
        state%uaj(3, 3) = 2.0d0
        state%uaj(3, 4) = 2.0d0
        state%uaj(3, 5) = 1.0d0
        state%uaj(3, 1) = 1.333333d0
        state%uaj(3, 2) = 2.0d0
        state%uaj(3, 6) = 1.0d0
        
        state%impdim = 10
        state%impl = [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
        state%imcl = [7, 7, 7, 0, 0, 0, 0, 0, 0, 0]
        state%da0 = 0.0d0
        
        state%daj = 0.0d0
        ! DAJ(IMP, COMP): implicit equation IMP for component COMP
        ! Fortran column-major DATA order from original:
        ! col 1: (0.666667, 0, 0.3333, 0...)
        ! col 2: (1.5, 0, 1.0, 0...)
        ! col 3: (0, 0.5, 1.0, 0...)
        ! col 4: (0, 0.5, 1.0, 0...)
        ! col 5: (1.6, 0, 1.0, 0...)
        state%daj(1, 1) = 0.666667d0
        state%daj(3, 1) = 0.3333d0
        state%daj(1, 2) = 1.5d0
        state%daj(3, 2) = 1.0d0
        state%daj(2, 3) = 0.5d0
        state%daj(3, 3) = 1.0d0
        state%daj(2, 4) = 0.5d0
        state%daj(3, 4) = 1.0d0
        state%daj(1, 5) = 1.6d0
        state%daj(3, 5) = 1.0d0
        
    end subroutine initialize_state

end module basalt_constants

module basalt_math
    implicit none

    contains

    function arhenf(a, b, t) result(kd)
        real(8), intent(in) :: a, b, t
        real(8) :: kd
        ! Standard Arrhenius form used by the high-pressure extension
        ! (BASALT+lamiuar.FOR): Kd = 10^(A/T + B).  This keeps Kd positive
        ! and in a physically reasonable range (order 1-10).
        kd = 10.0d0**(a / t + b)
        if (kd < 1.0d-6) kd = 1.0d-6
    end function arhenf

    function arhenf_p(a, b, c, t, p) result(kd)
        real(8), intent(in) :: a, b, c, t, p
        real(8) :: kd
        kd = 10.0d0**(a / t + b + c * p)
    end function arhenf_p

    subroutine mateq(a, n, y, det, ndim)
        integer, intent(in) :: n, ndim
        real(8), intent(inout) :: a(ndim, ndim)
        real(8), intent(inout) :: y(ndim)
        real(8), intent(out) :: det
        
        integer :: i, j, k, m, imx
        real(8) :: amx, temp, akk, aki
        integer :: ichg(40)
        
        det = 1.0d0
        
        do k = 1, n
            amx = abs(a(k, k))
            imx = k
            do i = k, n
                if (abs(a(i, k)) > amx) then
                    amx = abs(a(i, k))
                    imx = i
                end if
            end do
            
            if (amx < 1.0d-8) then
                det = 0.0d0
                return
            end if
            
            if (imx /= k) then
                do j = 1, n
                    temp = a(k, j)
                    a(k, j) = a(imx, j)
                    a(imx, j) = temp
                end do
                temp = y(k)
                y(k) = y(imx)
                y(imx) = temp
                ichg(k) = imx
                det = -det
            else
                ichg(k) = k
            end if
            
            akk = a(k, k)
            det = det * akk
            do j = 1, n
                a(k, j) = a(k, j) / akk
            end do
            y(k) = y(k) / akk
            
            do i = 1, n
                if (i /= k) then
                    aki = a(i, k)
                    do j = 1, n
                        a(i, j) = a(i, j) - aki * a(k, j)
                    end do
                    y(i) = y(i) - aki * y(k)
                end if
            end do
        end do
        
        do k = 1, n
            i = n + 1 - k
            j = ichg(i)
            if (i /= j) then
                do m = 1, n
                    temp = a(m, i)
                    a(m, i) = a(m, j)
                    a(m, j) = temp
                end do
            end if
        end do
        
    end subroutine mateq

end module basalt_math

module basalt_core
    use basalt_types
    use basalt_math
    implicit none

    contains

    subroutine kdcalc(state, kdmode)
        type(phase_eq_state), intent(inout) :: state
        integer, intent(in) :: kdmode
        
        real(8) :: tk, p, an
        
        tk = state%sparm(1)
        p = state%sparm(2)
        
        if (kdmode == 1) then
            state%fkdaj = 0.0d0
        end if
        
        ! KDMODE=2: 仅温度依赖; KDMODE=4: 温度+压力依赖
        if (kdmode >= 2) then
            if (kdmode == 4) then
                ! 压力相关参数 (Langmuir 1992 扩展)
                state%fkdaj(2, 3) = arhenf_p(3740.d0, -1.87d0, 0.0008d0, tk, p)
                state%fkdaj(2, 4) = arhenf_p(3911.d0, -2.50d0, 0.0006d0, tk, p)

                state%fkdaj(3, 3) = arhenf_p(3798.d0, -2.28d0, 0.004d0, tk, p)
                state%fkdaj(3, 4) = 0.24d0 * state%fkdaj(3, 3)
                state%fkdaj(3, 5) = arhenf_p(1738.d0, -0.753d0, 0.009d0, tk, p)
                state%fkdaj(3, 6) = arhenf_p(1034.d0, -1.27d0, 0.005d0, tk, p)
                state%fkdaj(3, 1) = arhenf_p(2418.d0, -2.30d0, 0.006d0, tk, p)
                state%fkdaj(3, 2) = arhenf_p(5087.d0, -4.48d0, 0.003d0, tk, p)
            else
                ! 仅温度依赖 (Weaver & Langmuir 1990 原始参数)
                state%fkdaj(2, 3) = arhenf(2715.d0, -1.158d0, tk)
                state%fkdaj(2, 4) = arhenf(4230.d0, -2.741d0, tk)

                state%fkdaj(3, 3) = arhenf(3798.d0, -2.28d0, tk)
                state%fkdaj(3, 4) = 0.24d0 * state%fkdaj(3, 3)
                state%fkdaj(3, 5) = arhenf(1738.d0, -0.753d0, tk)
                state%fkdaj(3, 6) = arhenf(1034.d0, -1.27d0, tk)
                state%fkdaj(3, 1) = arhenf(2418.d0, -2.30d0, tk)
                state%fkdaj(3, 2) = arhenf(5087.d0, -4.48d0, tk)
            end if
        end if
        
        ! KDMODE=3/4: 斜长石组成+压力依赖
        if (kdmode == 3 .or. kdmode == 4) then
            an = state%clj(1) / (state%clj(1) + 1.5d0 * state%clj(2))
            if (kdmode == 4) then
                state%fkdaj(1, 1) = 10.0d0**(2446.d0 / tk - (1.122d0 + 0.2562d0 * an) + 0.012d0 * p)
                state%fkdaj(1, 2) = 10.0d0**((3195.d0 + 3283.d0 * an) / tk - &
                                           (2.318d0 + 1.885d0 * an) + 0.007d0 * p)
            else
                state%fkdaj(1, 1) = arhenf(2446.d0, -(1.122d0 + 0.2562d0 * an), tk)
                state%fkdaj(1, 2) = arhenf((3195.d0 + 3283.d0 * an), &
                                           -(2.318d0 + 1.885d0 * an), tk)
            end if
        end if
        
    end subroutine kdcalc

    function stoich(state, jp) result(istabl)
        type(phase_eq_state), intent(in) :: state
        integer, intent(in) :: jp
        logical :: istabl
        
        real(8) :: t, p
        
        istabl = .true.
        
        if (jp == 3) then
            p = state%sparm(2)
            t = 2.0d0 * (state%caj(3, 3) + state%caj(3, 4)) + state%caj(3, 5)
            ! 高压下(P>4kbar) CPX 饱和阈值更高 (Langmuir 1992)
            if (p > 4.0d0) then
                if (t < 1.1d0) istabl = .false.
            else
                if (t < 1.0d0) istabl = .false.
            end if
        end if
        
    end function stoich

    subroutine cimpl(state)
        type(phase_eq_state), intent(inout) :: state
        
        integer :: i, j, k, imp, imc
        real(8) :: s, fl
        
        fl = state%fl
        if (fl <= 0.0d0) fl = 1.0d-12
        
        if (state%ncimp > 0) then
            do j = state%ncomp + 1, state%ncompt
                state%clj(j) = 0.0d0
                do k = 1, state%nphas
                    state%caj(k, j) = 0.0d0
                end do
            end do
        end if
        
        do i = 1, state%impdim
            imp = state%impl(i)
            if (imp == 0) exit
            
            imc = state%imcl(i)
            s = state%da0(i)
            do j = 1, state%ncomp
                s = s + state%daj(i, j) * state%caj(imp, j)
            end do
            state%caj(imp, imc) = s
            ! 隐式组分的液相组成由质量守恒得到
            state%clj(imc) = (state%csj(imc) - s * state%fa(imp)) / fl
        end do
        
    end subroutine cimpl

    function phase_residual(state, l, f) result(q)
        type(phase_eq_state), intent(in) :: state
        integer, intent(in) :: l
        real(8), intent(in) :: f
        real(8) :: q
        integer :: j
        q = -state%ta(l)
        do j = 1, state%ncomp
            if (state%uaj(l, j) == 0.0d0) cycle
            q = q + state%uaj(l, j) * state%fkdaj(l, j) * state%csj(j) / &
                (1.0d0 + f * (state%fkdaj(l, j) - 1.0d0))
        end do
    end function phase_residual

    subroutine solve_one_phase(state, l, nerr)
        type(phase_eq_state), intent(inout) :: state
        integer, intent(in) :: l
        integer, intent(out) :: nerr

        real(8) :: f_lo, f_hi, f_mid, q_lo, q_hi, q_mid
        integer :: bisect_iter

        integer :: j

        nerr = 0
        ! 清零其他相分数，确保仅当前相活动
        do j = 1, state%nphas
            if (j /= l) state%fa(j) = 0.0d0
        end do

        f_lo = 0.0d0
        f_hi = 0.9999d0
        q_lo = phase_residual(state, l, f_lo)
        q_hi = phase_residual(state, l, f_hi)

        if (q_lo >= 0.0d0) then
            ! 未饱和，无相结晶
            state%fa(l) = 0.0d0
            state%fl = 1.0d0
            return
        end if
        if (q_hi < 0.0d0) then
            ! 始终过饱和，取最大允许相分数
            state%fa(l) = f_hi
            state%fl = 1.0d0 - f_hi
            return
        end if

        do bisect_iter = 1, 80
            f_mid = 0.5d0 * (f_lo + f_hi)
            q_mid = phase_residual(state, l, f_mid)
            if (q_mid < 0.0d0) then
                f_lo = f_mid
                q_lo = q_mid
            else
                f_hi = f_mid
                q_hi = q_mid
            end if
            if (abs(q_mid) < 1.0d-10 .or. f_hi - f_lo < 1.0d-14) exit
        end do

        state%fa(l) = f_mid
        state%fl = 1.0d0 - f_mid
    end subroutine solve_one_phase

    subroutine state_solve(state, nl, list, nerr)
        type(phase_eq_state), intent(inout) :: state
        integer, intent(out) :: nl, list(MAX_PHAS), nerr

        integer, parameter :: max_iter = 500
        real(8), parameter :: tol = 1.0d-5
        real(8), parameter :: alpha_min = 1.0d-4

        integer :: iter, i, j, l, k, m, ls_iter, nl_seed, ngrid, i1, i2
        integer :: retry_count, nl_full, list_full(MAX_PHAS), kd_mode
        real(8) :: t, tq, tst, tu, det, qmax, sum_dfa, residual, new_residual
        real(8) :: dfa(MAX_PHAS), fa_old(MAX_PHAS)
        real(8) :: denom(MAX_COMP), ddenom, alpha, sum_active, sum_total
        real(8) :: f1, f2, best_norm, current_norm, best_f1, best_f2
        logical :: is_active, all_zero, converged, use_grid
        logical :: keep(MAX_PHAS)

        nerr = 0
        if (state%sparm(2) > 0.0d0) then
            kd_mode = 4   ! 高压：温度+压力相关 Kd
        else
            kd_mode = 3   ! 1 atm：温度+斜长石组成相关 Kd
        end if
        call kdcalc(state, 1)
        call kdcalc(state, kd_mode)

        converged = .false.
        use_grid = .false.

        do retry_count = 0, 1
            do iter = 1, max_iter
            ! 计算 RJ, CLJ, CAJ 以及当前分母（用于步长控制）
            do j = 1, state%ncomp
                t = 1.0d0
                do l = 1, state%nphas
                    t = t + state%fa(l) * (state%fkdaj(l, j) - 1.0d0)
                end do
                denom(j) = t
                state%rj(j) = 1.0d0 / t
                state%clj(j) = state%csj(j) * state%rj(j)
                do l = 1, state%nphas
                    state%caj(l, j) = state%fkdaj(l, j) * state%clj(j)
                end do
            end do

            nl = 0
            qmax = 0.0d0
            residual = 0.0d0
            keep = .false.
            do l = 1, state%nphas
                tq = -state%ta(l)
                do j = 1, state%ncomp
                    tq = tq + state%uaj(l, j) * state%caj(l, j)
                end do
                state%qa(l) = tq

                if (tq <= 0.0d0 .and. stoich(state, l)) then
                    keep(l) = .true.
                    nl = nl + 1
                    list(nl) = l
                    residual = residual + tq * tq
                    if (abs(tq) > qmax) qmax = abs(tq)
                end if
            end do

            if (nl == 0) then
                state%fa = 0.0d0
                state%fl = 1.0d0
                ! 重新计算液相/矿物组成，确保质量平衡与当前状态一致
                do j = 1, state%ncomp
                    state%rj(j) = 1.0d0
                    state%clj(j) = state%csj(j)
                    do l = 1, state%nphas
                        state%caj(l, j) = state%fkdaj(l, j) * state%clj(j)
                    end do
                end do
                call cimpl(state)
                nerr = 0
                return
            end if

            ! 清理当前非活动相的相分数，防止上一迭代遗留值破坏质量守恒
            do l = 1, state%nphas
                is_active = .false.
                do j = 1, nl
                    if (list(j) == l) is_active = .true.
                end do
                if (.not. is_active) state%fa(l) = 0.0d0
            end do

            ! 单相情况：使用稳健的一维二分求根，避免牛顿方向错误导致发散
            if (nl == 1) then
                l = list(1)
                call solve_one_phase(state, l, nerr)
                if (nerr /= 0) return
                if (state%fa(l) > 1.0d-12) then
                    converged = .true.
                else
                    state%fa(l) = 0.0d0
                    state%fl = 1.0d0
                    nl = 0
                    list = 0
                    converged = .true.
                end if
                exit
            end if

            ! 保存完整过饱和相列表，供网格搜索重试使用
            nl_full = nl
            do j = 1, nl
                list_full(j) = list(j)
            end do

            ! 构建与原始 BASALT.FOR 一致的近似 Jacobian 矩阵
            do j = 1, nl
                l = list(j)
                dfa(j) = state%qa(l)
                do k = 1, nl
                    m = list(k)
                    t = 0.0d0
                    do i = 1, state%ncomp
                        tu = state%uaj(l, i)
                        if (tu == 0.0d0) cycle
                        t = t + tu * state%caj(m, i) * state%rj(i) * state%rj(i) * &
                              (state%fkdaj(m, i) - 1.0d0)
                    end do
                    state%pab(j, k) = -t
                end do
            end do

            call mateq(state%pab, nl, dfa, det, state%npdim)
            if (abs(det) < 1.0d-10) then
                nerr = 1
                return
            end if

            ! 保存旧值，检查是否所有活动相分数均为 0（首次结晶）
            all_zero = .true.
            do j = 1, nl
                l = list(j)
                fa_old(j) = state%fa(l)
                if (state%fa(l) > 1.0d-12) all_zero = .false.
            end do

            ! 首次结晶：只给 Newton 方向为正的相提供小正初值；
            ! 方向非正的相在当前状态下无法真正结晶，暂时排除。
            if (all_zero) then
                nl_seed = 0
                sum_dfa = 0.0d0
                do j = 1, nl
                    l = list(j)
                    if (dfa(j) > 0.0d0) then
                        nl_seed = nl_seed + 1
                        list(nl_seed) = l
                        sum_dfa = sum_dfa + dfa(j)
                    else
                        keep(l) = .false.
                        state%fa(l) = 0.0d0
                    end if
                end do
                nl = nl_seed
                if (nl == 0) then
                    state%fa = 0.0d0
                    state%fl = 1.0d0
                    do j = 1, state%ncomp
                        state%rj(j) = 1.0d0
                        state%clj(j) = state%csj(j)
                        do l = 1, state%nphas
                            state%caj(l, j) = state%fkdaj(l, j) * state%clj(j)
                        end do
                    end do
                    call cimpl(state)
                    nerr = 0
                    return
                end if

                if (use_grid) then
                    ! 重试时使用网格搜索：在完整过饱和相空间中寻找稳健初值
                    nl = nl_full
                    do j = 1, nl_full
                        list(j) = list_full(j)
                    end do
                    if (nl == 1) then
                        ! 单相：一维搜索
                        l = list(1)
                        best_norm = huge(1.0d0)
                        best_f1 = 0.0d0
                        do i1 = 1, 50
                            f1 = 0.99d0 * dble(i1) / 50.0d0
                            tq = -state%ta(l)
                            do j = 1, state%ncomp
                                tq = tq + state%uaj(l, j) * state%fkdaj(l, j) * state%csj(j) / &
                                     (1.0d0 + f1 * (state%fkdaj(l, j) - 1.0d0))
                            end do
                            current_norm = tq * tq
                            if (current_norm < best_norm) then
                                best_norm = current_norm
                                best_f1 = f1
                            end if
                        end do
                        state%fa(l) = best_f1
                    else if (nl == 2) then
                        ! 两相：二维搜索
                        ngrid = 40
                        best_norm = huge(1.0d0)
                        best_f1 = 0.0d0
                        best_f2 = 0.0d0
                        do i1 = 0, ngrid
                            f1 = 0.99d0 * dble(i1) / dble(ngrid)
                            do i2 = 0, ngrid - i1
                                f2 = 0.99d0 * dble(i2) / dble(ngrid)
                                current_norm = 0.0d0
                                do j = 1, 2
                                    l = list(j)
                                    tq = -state%ta(l)
                                    do k = 1, state%ncomp
                                        tq = tq + state%uaj(l, k) * state%fkdaj(l, k) * state%csj(k) / &
                                             (1.0d0 + f1 * (state%fkdaj(list(1), k) - 1.0d0) + &
                                              f2 * (state%fkdaj(list(2), k) - 1.0d0))
                                    end do
                                    current_norm = current_norm + tq * tq
                                end do
                                if (current_norm < best_norm) then
                                    best_norm = current_norm
                                    best_f1 = f1
                                    best_f2 = f2
                                end if
                            end do
                        end do
                        state%fa(list(1)) = best_f1
                        state%fa(list(2)) = best_f2
                    else
                        ! 三相及以上：仅对前两个相做二维搜索，其余置零
                        nl = 2
                        list(1) = list_full(1)
                        list(2) = list_full(2)
                        ngrid = 40
                        best_norm = huge(1.0d0)
                        best_f1 = 0.0d0
                        best_f2 = 0.0d0
                        do i1 = 0, ngrid
                            f1 = 0.99d0 * dble(i1) / dble(ngrid)
                            do i2 = 0, ngrid - i1
                                f2 = 0.99d0 * dble(i2) / dble(ngrid)
                                current_norm = 0.0d0
                                do j = 1, 2
                                    l = list(j)
                                    tq = -state%ta(l)
                                    do k = 1, state%ncomp
                                        tq = tq + state%uaj(l, k) * state%fkdaj(l, k) * state%csj(k) / &
                                             (1.0d0 + f1 * (state%fkdaj(list(1), k) - 1.0d0) + &
                                              f2 * (state%fkdaj(list(2), k) - 1.0d0))
                                    end do
                                    current_norm = current_norm + tq * tq
                                end do
                                if (current_norm < best_norm) then
                                    best_norm = current_norm
                                    best_f1 = f1
                                    best_f2 = f2
                                end if
                            end do
                        end do
                        state%fa(list(1)) = best_f1
                        state%fa(list(2)) = best_f2
                    end if
                else
                    ! 首次结晶：只给 Newton 方向为正的相提供小正初值
                    if (sum_dfa > 0.0d0) then
                        do j = 1, nl
                            l = list(j)
                            state%fa(l) = dfa(j) / sum_dfa * 1.0d-3
                        end do
                    else
                        do j = 1, nl
                            state%fa(list(j)) = 1.0d-3 / dble(nl)
                        end do
                    end if
                end if
                call kdcalc(state, kd_mode)
                cycle
            end if

            ! 计算可行步长 alpha 的约束上限
            alpha = 1.0d0

            ! (a) 保证活动相分数非负
            do j = 1, nl
                l = list(j)
                if (dfa(j) < -1.0d-14) then
                    alpha = min(alpha, -0.99d0 * state%fa(l) / dfa(j))
                end if
            end do

            ! (b) 保证液相分数非负（总相分数 <= 0.9999）
            sum_active = 0.0d0
            sum_dfa = 0.0d0
            do j = 1, nl
                sum_active = sum_active + state%fa(list(j))
                sum_dfa = sum_dfa + dfa(j)
            end do
            sum_total = 0.0d0
            do l = 1, state%nphas
                sum_total = sum_total + state%fa(l)
            end do
            if (sum_dfa > 1.0d-14 .and. sum_total + alpha * sum_dfa > 0.9999d0) then
                alpha = (0.9999d0 - sum_total) / sum_dfa * 0.99d0
            end if

            ! (c) 保证液相组成非负（分母 > 0）
            do j = 1, state%ncomp
                ddenom = 0.0d0
                do k = 1, nl
                    m = list(k)
                    ddenom = ddenom + dfa(k) * (state%fkdaj(m, j) - 1.0d0)
                end do
                if (abs(ddenom) > 1.0d-14 .and. denom(j) + alpha * ddenom <= 0.0d0) then
                    alpha = min(alpha, -0.99d0 * denom(j) / ddenom)
                end if
            end do

            ! 回溯线搜索：确保活动相残差下降
            do ls_iter = 1, 12
                tst = 0.0d0
                do j = 1, nl
                    l = list(j)
                    t = fa_old(j) + alpha * dfa(j)
                    if (t < 0.0d0) t = 0.0d0
                    if (t > 0.9999d0) t = 0.9999d0
                    tst = tst + abs(state%fa(l) - t)
                    state%fa(l) = t
                end do

                ! 重新计算总相分数与液相分数
                sum_total = 0.0d0
                do l = 1, state%nphas
                    sum_total = sum_total + state%fa(l)
                end do
                if (sum_total > 0.9999d0) then
                    t = 0.9999d0 / sum_total
                    do l = 1, state%nphas
                        state%fa(l) = state%fa(l) * t
                    end do
                    sum_total = 0.9999d0
                end if
                state%fl = 1.0d0 - sum_total

                call kdcalc(state, kd_mode)

                ! 计算新残差
                new_residual = 0.0d0
                do j = 1, nl
                    l = list(j)
                    tq = -state%ta(l)
                    do i = 1, state%ncomp
                        tq = tq + state%uaj(l, i) * state%fkdaj(l, i) * state%csj(i) / &
                             (1.0d0 + sum(state%fa(1:state%nphas) * &
                             (state%fkdaj(1:state%nphas, i) - 1.0d0)))
                    end do
                    new_residual = new_residual + tq * tq
                end do

                ! 若残差下降或步长已经很小则接受
                if (new_residual < residual * 0.99d0 .or. alpha <= alpha_min) exit

                ! 否则回退并折半步长
                alpha = alpha * 0.5d0
                do j = 1, nl
                    state%fa(list(j)) = fa_old(j)
                end do
                if (alpha < alpha_min) exit
            end do

            ! 收敛判断：相分数变化足够小且活动相饱和残差足够小
            if (tst <= tol .and. qmax <= tol) then
                converged = .true.
                exit
            end if
        end do

        if (converged) exit

        if (retry_count == 1) exit

        ! 准备使用网格搜索重试：重置相分数
        use_grid = .true.
        state%fa = 0.0d0
        state%fl = 1.0d0
        do j = 1, state%ncomp
            state%rj(j) = 1.0d0
            state%clj(j) = state%csj(j)
            do l = 1, state%nphas
                state%caj(l, j) = state%fkdaj(l, j) * state%clj(j)
            end do
        end do
        call kdcalc(state, kd_mode)
    end do

    if (.not. converged) nerr = 2

100     continue

        ! 清理收敛后实际非活动的相分数，保持输出物理一致
        do l = 1, state%nphas
            is_active = .false.
            do j = 1, nl
                if (list(j) == l) is_active = .true.
            end do
            if (.not. is_active) state%fa(l) = 0.0d0
        end do
        sum_total = 0.0d0
        do l = 1, state%nphas
            sum_total = sum_total + state%fa(l)
        end do
        if (sum_total > 0.9999d0) then
                t = 0.9999d0 / sum_total
                do l = 1, state%nphas
                    state%fa(l) = state%fa(l) * t
                end do
                sum_total = 0.9999d0
            end if
            state%fl = 1.0d0 - sum_total

            ! 将相分数中极小的数值置零，避免亚正规数输出
            do l = 1, state%nphas
                if (state%fa(l) < 1.0d-12) state%fa(l) = 0.0d0
            end do
            if (state%fl < 1.0d-12) state%fl = 0.0d0

            ! 重新计算液相/矿物组成，确保与清理后的相分数质量平衡
            call kdcalc(state, kd_mode)
            do j = 1, state%ncomp
                t = 1.0d0
                do l = 1, state%nphas
                    t = t + state%fa(l) * (state%fkdaj(l, j) - 1.0d0)
                end do
                state%rj(j) = 1.0d0 / t
                state%clj(j) = state%csj(j) * state%rj(j)
                do l = 1, state%nphas
                    state%caj(l, j) = state%fkdaj(l, j) * state%clj(j)
                end do
            end do
            call cimpl(state)

    end subroutine state_solve

end module basalt_core

module basalt_io
    use basalt_types
    use basalt_core, only: cimpl
    implicit none

    contains

    subroutine prnter(state, modes, temp, flr, nerr, nupage)
        type(phase_eq_state), intent(inout) :: state
        type(calc_modes), intent(in) :: modes
        real(4), intent(in) :: temp, flr
        integer, intent(in) :: nerr, nupage
        
        integer :: nwrt, k, j
        character(len=80) :: fmt_str
        
        call cimpl(state)
        nwrt = 6
        
        if (.not. modes%summary_only) then
            if (nupage /= 0) then
                write(fmt_str, "('(A4,2X,A12,', I0, '(2X,A12))')") state%nphas
                write(nwrt, fmt_str) 'FRAC', 'LIQ', (trim(state%pnamea(j)), j=1,state%nphas)
            end if
            write(fmt_str, "('(A4,2X,', I0, '(ES14.4,2X))')") state%nphas + 1
            write(nwrt, fmt_str) 'FRAC', state%fl, (state%fa(j), j=1,state%nphas)
            do k = 1, state%ncompt
                write(fmt_str, "('(A4,2X,', I0, '(ES14.4,2X))')") state%nphas + 2
                write(nwrt, fmt_str) trim(state%cnamej(k)), state%csj(k), state%clj(k), &
                    (state%caj(j, k), j=1,state%nphas)
            end do
            write(nwrt, '(A,F10.4,3X,A,F6.2,5X,A,F10.4)') &
                'TEMP=', temp, 'P(kbar)=', real(state%sparm(2)), 'FLR =', flr
            write(nwrt, '(4(A4,A,F10.4,5X))') &
                (trim(state%snames(k)), '=', state%sparm(k), k=1,state%ns)
        else
            if (nupage /= 0) then
                write(fmt_str, "('(A12,2X,A12,2X,A12,', I0, '(2X,A12),', I0, '(2X,A12))')") &
                    state%nphas, state%ncompt
                write(nwrt, fmt_str) 'TEMP', 'P(kbar)', 'FLR', &
                    (trim(state%pnamea(k)), k=1,state%nphas), &
                    (trim(state%cnamej(j)), j=1,state%ncompt)
            end if
            write(fmt_str, "('(F12.2,2X,F6.2,2X,F12.4,', I0, '(2X,ES14.4),', I0, '(2X,ES14.4))')") &
                state%nphas, state%ncompt
            write(nwrt, fmt_str) temp, real(state%sparm(2)), flr, (state%fa(j), j=1,state%nphas), &
                (state%clj(j), j=1,state%ncompt)
        end if
        
        if (nerr == 1) write(nwrt, '(A)') 'MATRIX INVERSION PROBLEM IN STATE'
        if (nerr == 2) write(nwrt, '(A)') 'MAXIMUM ITERATIONS REACHED IN STATE'
        
        if (modes%printer_on) then
            nwrt = 2
            call cimpl(state)
            if (.not. modes%summary_only) then
                write(fmt_str, "('(A4,2X,A12,', I0, '(2X,A12))')") state%nphas
                write(nwrt, fmt_str) 'FRAC', 'LIQ', (trim(state%pnamea(j)), j=1,state%nphas)
                write(fmt_str, "('(A4,2X,', I0, '(ES14.4,2X))')") state%nphas + 1
                write(nwrt, fmt_str) 'FRAC', state%fl, (state%fa(j), j=1,state%nphas)
                do k = 1, state%ncompt
                    write(fmt_str, "('(A4,2X,', I0, '(ES14.4,2X))')") state%nphas + 2
                    write(nwrt, fmt_str) trim(state%cnamej(k)), state%csj(k), state%clj(k), &
                        (state%caj(j, k), j=1,state%nphas)
                end do
                write(nwrt, '(A,F10.4,3X,A,F6.2,5X,A,F10.4)') &
                    'TEMP=', temp, 'P(kbar)=', real(state%sparm(2)), 'FLR =', flr
                write(nwrt, '(4(A4,A,F10.4,5X))') &
                    (trim(state%snames(k)), '=', state%sparm(k), k=1,state%ns)
            else
                write(fmt_str, "('(F12.2,2X,F6.2,2X,F12.4,', I0, '(2X,ES14.4),', I0, '(2X,ES14.4))')") &
                    state%nphas, state%ncompt
                write(nwrt, fmt_str) temp, real(state%sparm(2)), flr, (state%fa(j), j=1,state%nphas), &
                    (state%clj(j), j=1,state%ncompt)
            end if
            if (nerr == 1) write(nwrt, '(A)') 'MATRIX INVERSION PROBLEM IN STATE'
            if (nerr == 2) write(nwrt, '(A)') 'MAXIMUM ITERATIONS REACHED IN STATE'
        end if
        
    end subroutine prnter

end module basalt_io

module basalt_driver
    use basalt_types
    use basalt_core
    use basalt_io
    implicit none

    contains

    subroutine driver(state, modes, params)
        type(phase_eq_state), intent(inout) :: state
        type(calc_modes), intent(in) :: modes
        type(calc_params), intent(in) :: params
        
        real(4) :: p
        integer :: nupage, nchang
        logical :: noprnt
        
        nupage = 1
        nchang = -1
        noprnt = .false.
        
        if (modes%init_on) then
            state%fa = 0.0d0
        end if
        
        ! 单压力模式: 固定 P=0 (或指定压力)
        if (.not. modes%polybaric) then
            state%sparm(2) = dble(params%p_high)
            call temprun(state, modes, params, params%ti + params%temp_offset, nupage, nchang, noprnt)
            return
        end if
        
        ! 多压力模式: 从 P_HIGH 到 P_LOW, 步长 P_STEP
        p = params%p_high
        do while (p >= params%p_low - 0.01)
            state%sparm(2) = dble(p)
            call temprun(state, modes, params, params%ti + params%temp_offset, nupage, nchang, noprnt)
            ! 为下一压力段更新熔体组成(从最后液相组成开始)
            state%csj(1:state%ncomp) = state%clj(1:state%ncomp)
            p = p - params%p_step
        end do
        
    end subroutine driver

    subroutine temprun(state, modes, params, temp_start, nupage, nchang, noprnt)
        type(phase_eq_state), intent(inout) :: state
        type(calc_modes), intent(in) :: modes
        type(calc_params), intent(in) :: params
        real(4), intent(in) :: temp_start
        integer, intent(inout) :: nupage, nchang
        logical, intent(inout) :: noprnt
        
        real(4) :: temp, flr
        integer :: nl, list(MAX_PHAS), nerr
        logical :: do_print
        integer :: nt, j, k, step_count
        
        temp = temp_start
        flr = 1.0
        step_count = 0
        
        do j = 1, state%ncomp
            state%clj(j) = state%csj(j)
        end do
        
        do while (.true.)
            step_count = step_count + 1
            state%sparm(1) = dble(temp)   ! 更新当前温度
            call state_solve(state, nl, list, nerr)
            
            select case(modes%model_type)
            case(1)
                flr = 1.0
            case(2)
                flr = flr * real(state%fl)
            case(3)
                flr = flr / (1.0 - real(state%fl))
            end select
            
            if (nl /= 0) noprnt = .false.
            
            ! 判断是否需要输出
            do_print = .true.
            if (modes%changes_only) then
                ! 仅当矿物相组成发生变化时输出
                nt = nchang
                nchang = 0
                do k = 1, nl
                    nchang = nchang + list(k)
                end do
                if (nchang == nt) do_print = .false.
            end if
            
            if (do_print .and. params%dp > 0.0) then
                ! 基于温度步数判断输出间隔, 避免浮点误差导致永远不输出
                if (mod(step_count - 1, nint(params%dp / abs(params%dt))) /= 0) then
                    do_print = .false.
                end if
            end if
            
            if (do_print .or. nerr /= 0) then
                call prnter(state, modes, temp, flr, nerr, nupage)
            end if
            
            if (nerr /= 0) return
            
            temp = temp + params%dt
            if (((params%tf + params%temp_offset) - temp) * params%dt <= 0.0) return
            
            select case(modes%model_type)
            case(1)
                flr = 1.0
            case(2)
                do j = 1, state%ncomp
                    state%csj(j) = state%clj(j)
                end do
            case(3)
                do j = 1, state%ncomp
                    state%csj(j) = (state%csj(j) - state%clj(j) * state%fl) / &
                                   (1.0d0 - state%fl)
                end do
            end select
            
        end do
        
    end subroutine temprun

end module basalt_driver

program basalt_modern
    use basalt_types
    use basalt_constants
    use basalt_driver
    implicit none

    type(phase_eq_state) :: state
    type(calc_params) :: params
    type(calc_modes) :: modes
    
    real(4) :: cdat(50, 20)
    character(len=256) :: cmd_file, data_file, line
    integer :: ntask, ncdat, nclo, nchi, k, j, n
    integer :: iunit, iostat_val, line_num
    real(4) :: s
    logical :: nodata, nomodl, notemp
    
    call initialize_state(state)
    
    nodata = .true.
    nomodl = .true.
    notemp = .true.
    
    modes%model_type = 0
    modes%init_on = .false.  ! 默认继承上一温度步相分数, 保证温度序列连续性
    modes%printer_on = .false.
    modes%summary_only = .false.
    modes%changes_only = .false.
    modes%polybaric = .false.
    
    params%ti = 1000.0
    params%tf = 1000.0
    params%dt = 10.0
    params%dp = 10.0
    params%temp_offset = 273.16
    params%flr = 0.5
    params%p_high = 0.0
    params%p_low = 0.0
    params%p_step = 0.5
    
    ncdat = 0
    nclo = 1
    nchi = 1
    
    ! 获取命令行参数：任务脚本文件
    if (command_argument_count() < 1) then
        write(*, '(A)') 'USAGE: basalt_modern <task_script.txt>'
        write(*, '(A)') ''
        write(*, '(A)') 'Task script format (one task per line):'
        write(*, '(A)') '  1 <datafile>        ! load composition data from file'
        write(*, '(A)') '  2 <nclo> <nchi>     ! data range to run'
        write(*, '(A)') '  3 <model>           ! 1=equilibrium, 2=fractional crystallization, 3=fractional melting'
        write(*, '(A)') '  4                   ! toggle initialization'
        write(*, '(A)') '  5                   ! toggle full/summary output'
        write(*, '(A)') '  6                   ! toggle changes-only output'
        write(*, '(A)') '  7                   ! toggle printer'
        write(*, '(A)') '  8 <ti> <tf> <dt> <dp> <flr> [p_high] [p_low] [p_step]'
        write(*, '(A)') '                       ! temperature+pressure parameters and run'
        write(*, '(A)') '  9                   ! run model'
        write(*, '(A)') ' 10                   ! toggle Celsius/Kelvin'
        write(*, '(A)') ' 11                   ! toggle single/polybaric mode'
        write(*, '(A)') ' 99                   ! exit'
        stop
    end if
    
    call get_command_argument(1, cmd_file)
    
    open(newunit=iunit, file=cmd_file, status='old', action='read', iostat=iostat_val)
    if (iostat_val /= 0) then
        write(*, '(A,A,A)') 'ERROR: cannot open task script file: ', trim(cmd_file)
        stop
    end if
    
    line_num = 0
    do
        read(iunit, '(A)', iostat=iostat_val) line
        if (iostat_val /= 0) exit
        line_num = line_num + 1
        line = adjustl(line)
        if (len_trim(line) == 0) cycle
        if (line(1:1) == '!') cycle
        
        read(line, *, iostat=iostat_val) ntask
        if (iostat_val /= 0) then
            write(*, '(A,I0,A)') 'ERROR: invalid task at line ', line_num, ': ' // trim(line)
            cycle
        end if
        
        select case(ntask)
        case(1)
            read(line, *, iostat=iostat_val) ntask, data_file
            if (iostat_val /= 0) then
                write(*, '(A,I0,A)') 'ERROR: task 1 requires a data file at line ', line_num
                cycle
            end if
            call read_data_file(state, trim(data_file), cdat, ncdat, nodata)
            
        case(2)
            read(line, *, iostat=iostat_val) ntask, nclo, nchi
            if (iostat_val /= 0) then
                write(*, '(A,I0,A)') 'ERROR: task 2 requires two integers at line ', line_num
                cycle
            end if
            
        case(3)
            read(line, *, iostat=iostat_val) ntask, modes%model_type
            if (iostat_val /= 0) then
                write(*, '(A,I0,A)') 'ERROR: task 3 requires model type at line ', line_num
                cycle
            end if
            if (modes%model_type >= 1 .and. modes%model_type <= 3) then
                nomodl = .false.
            else
                write(*, '(A)') 'INVALID MODEL SELECTION'
                nomodl = .true.
            end if
            
        case(4)
            modes%init_on = .not. modes%init_on
            
        case(5)
            modes%summary_only = .not. modes%summary_only
            
        case(6)
            modes%changes_only = .not. modes%changes_only
            
        case(7)
            modes%printer_on = .not. modes%printer_on
            
        case(8)
            read(line, *, iostat=iostat_val) ntask, params%ti, params%tf, params%dt, params%dp, params%flr, &
                                              params%p_high, params%p_low, params%p_step
            if (iostat_val /= 0) then
                ! 允许仅提供前5个参数(单压力模式)
                read(line, *, iostat=iostat_val) ntask, params%ti, params%tf, params%dt, params%dp, params%flr
                if (iostat_val /= 0) then
                    write(*, '(A,I0,A)') 'ERROR: task 8 requires 5 or 8 real numbers at line ', line_num
                    cycle
                end if
                params%p_high = 0.0
                params%p_low = 0.0
                params%p_step = 0.5
            end if
            if (params%ti > 0.0) then
                params%dt = abs(params%dt)
                if (params%ti > params%tf) params%dt = -params%dt
                notemp = .false.
            else
                notemp = .true.
                cycle
            end if
            ! task 8 followed by implicit task 9
            call run_model(state, modes, params, cdat, ncdat, nclo, nchi, nodata, nomodl, notemp)
            
        case(9)
            call run_model(state, modes, params, cdat, ncdat, nclo, nchi, nodata, nomodl, notemp)
            
        case(10)
            if (abs(params%temp_offset - 273.0) > 1.0) then
                params%temp_offset = 273.16
                write(*, '(A)') 'CELSIUS TEMPERATURE SCALE ASSUMED'
            else
                params%temp_offset = 0.0
                write(*, '(A)') 'KELVIN TEMPERATURE SCALE ASSUMED'
            end if

        case(11)
            modes%polybaric = .not. modes%polybaric
            if (modes%polybaric) then
                write(*, '(A)') 'POLYBARIC MULTI-PRESSURE MODE (LANGMUIR 1992)'
            else
                write(*, '(A)') 'SINGLE PRESSURE MODE (ORIGINAL 1990)'
            end if
            
        case(99)
            exit
            
        case default
            write(*, '(A,I0,A,I0)') 'WARNING: invalid task number ', ntask, ' at line ', line_num
            
        end select
    end do
    
    close(iunit)
    
contains

    subroutine read_data_file(state, data_file, cdat, ncdat, nodata)
        type(phase_eq_state), intent(in) :: state
        character(len=*), intent(in) :: data_file
        real(4), intent(out) :: cdat(50, 20)
        integer, intent(out) :: ncdat
        logical, intent(out) :: nodata
        
        integer :: iu, ios, j
        character(len=512) :: line
        real(4) :: s
        
        ncdat = 0
        open(newunit=iu, file=data_file, status='old', action='read', iostat=ios)
        if (ios /= 0) then
            write(*, '(A,A,A)') 'ERROR: cannot open data file: ', trim(data_file)
            nodata = .true.
            return
        end if
        
        do
            read(iu, '(A)', iostat=ios) line
            if (ios /= 0) exit
            line = adjustl(line)
            if (len_trim(line) == 0) cycle
            if (line(1:1) == '!') cycle
            
            read(line, *, iostat=ios) (cdat(ncdat+1, j), j=1,state%ncompt)
            if (ios /= 0) then
                write(*, '(A,A)') 'ERROR: invalid data line: ', trim(line)
                cycle
            end if
            
            s = 0.0
            do j = 1, state%ncompt
                s = s + abs(cdat(ncdat+1, j))
            end do
            if (s <= 1e-6) exit
            ncdat = ncdat + 1
            if (ncdat >= 50) exit
        end do
        
        close(iu)
        nodata = (ncdat <= 0)
        if (.not. nodata) then
            write(*, '(A,I0,A,A)') 'Loaded ', ncdat, ' data cases from ', trim(data_file)
        end if
    end subroutine read_data_file
    
    subroutine run_model(state, modes, params, cdat, ncdat, nclo, nchi, nodata, nomodl, notemp)
        type(phase_eq_state), intent(inout) :: state
        type(calc_modes), intent(in) :: modes
        type(calc_params), intent(in) :: params
        real(4), intent(in) :: cdat(50, 20)
        integer, intent(in) :: ncdat
        integer, intent(in) :: nclo, nchi
        logical, intent(in) :: nodata, nomodl, notemp
        
        integer :: k, j, ilo, ihi
        
        if (nodata) then
            write(*, '(A)') 'NO DATA ENTERED - PLEASE ENTER DATA FIRST (TASK 1)'
            return
        end if
        if (nomodl) then
            write(*, '(A)') 'NO MODEL SELECTED - PLEASE SELECT MODEL FIRST (TASK 3)'
            return
        end if
        if (notemp) then
            write(*, '(A)') 'NO TEMPERATURE PARAMETERS - PLEASE ENTER THEM FIRST (TASK 8)'
            return
        end if
        
        ilo = nclo
        ihi = nchi
        if (ilo < 1) ilo = 1
        if (ihi > ncdat) ihi = ncdat
        
        do k = ilo, ihi
            do j = 1, state%ncompt
                state%csj(j) = dble(cdat(k, j))
            end do
            call driver(state, modes, params)
        end do
    end subroutine run_model

end program basalt_modern