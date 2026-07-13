subroutine crscor_kernel(pilot, trace, npts, start_idx, ncrcor, nlag, ratio, best_lag, max_corr)
  implicit none
  integer, intent(in) :: npts, start_idx, ncrcor, nlag
  real(4), intent(in) :: pilot(ncrcor), trace(npts), ratio
  integer, intent(out) :: best_lag
  real(4), intent(out) :: max_corr

  integer :: i, j, trace_start
  real(4) :: enerp, enerx, ccsum, pweght
  real(4) :: tmpp(ncrcor), tmpx(ncrcor), ccmax

  best_lag = 0
  max_corr = 0.0
  ccmax = -1.0e20

  if (npts <= 0 .or. ncrcor <= 0 .or. ncrcor > npts) return

  enerp = 0.0
  do j = 1, ncrcor
    tmpp(j) = pilot(j)
    enerp = enerp + tmpp(j) * tmpp(j)
  end do
  if (enerp <= 0.0) return

  call hilbert_phase(tmpp, ncrcor)

  do i = -nlag, nlag
    trace_start = start_idx + i
    if (trace_start < 0) cycle
    if (trace_start + ncrcor > npts) cycle

    enerx = 0.0
    do j = 1, ncrcor
      tmpx(j) = trace(trace_start + j)
      enerx = enerx + tmpx(j) * tmpx(j)
    end do
    if (enerx <= 0.0) cycle

    call hilbert_phase(tmpx, ncrcor)

    ccsum = 0.0
    do j = 1, ncrcor
      pweght = sqrt(cos(0.5 * (tmpp(j) - tmpx(j))) ** 2) ** ratio
      ccsum = ccsum + pilot(j) * trace(trace_start + j) * pweght
    end do

    ccsum = ccsum / sqrt(enerp * enerx)
    if (ccsum > ccmax) then
      ccmax = ccsum
      best_lag = i
      max_corr = ccsum
    end if
  end do
end subroutine crscor_kernel

subroutine hilbert_phase(x, n)
  implicit none
  integer, intent(in) :: n
  integer :: it, i
  real(4), intent(inout) :: x(n)
  real(4) :: soma, theta, phase, denom
  real(4), parameter :: pi = 3.14159265359
  real(4), allocatable :: buffer(:)

  if (n <= 0) return
  allocate(buffer(-n:n))

  do i = -n, 0
    buffer(i) = 0.0
  end do
  do i = 1, n
    buffer(i) = x(i)
  end do

  do it = 1, n
    soma = 0.0
    do i = -n, n
      if (it == i) cycle
      soma = soma + buffer(i) / real(i - it, kind=4)
    end do
    soma = soma / pi
    denom = x(it)
    if (abs(denom) < 1.0e-12) then
      if (soma >= 0.0) then
        theta = 1.0e12
      else
        theta = -1.0e12
      end if
    else
      theta = soma / denom
    end if
    phase = atan(theta)
    x(it) = phase
  end do

  deallocate(buffer)
end subroutine hilbert_phase

