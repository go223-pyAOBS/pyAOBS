subroutine vg2_kernel(x, npts, dts, tvg, pvg)
  implicit none
  integer, intent(in) :: npts
  real(4), intent(inout) :: x(npts)
  real(4), intent(in) :: dts, tvg, pvg

  integer :: nw, n1, i
  real(4) :: esum
  real(4), allocatable :: e(:), xvg(:)

  if (npts <= 0) then
    return
  end if

  allocate(e(npts), xvg(npts))

  nw = nint(tvg / dts) + 1
  if (mod(nw, 2) == 0) nw = nw + 1
  n1 = nw / 2 + 1
  if (nw < 1) nw = 1
  if (nw > npts) nw = npts

  do i = 1, npts
    e(i) = abs(x(i)) ** pvg
    xvg(i) = 0.0
  end do

  esum = 0.0
  do i = 1, nw
    esum = esum + e(i)
  end do

  do i = 1, n1
    if (esum /= 0.0) xvg(i) = x(i) / esum
  end do

  if (nw < npts) then
    do i = n1 + 1, npts - n1
      esum = esum - e(i - n1) + e(i + n1 - 1)
      if (esum /= 0.0) xvg(i) = x(i) / esum
    end do
    esum = esum - e(npts - nw) + e(npts)
  end if

  do i = npts - n1 + 1, npts
    if (esum /= 0.0) xvg(i) = x(i) / esum
  end do

  do i = 1, npts
    x(i) = xvg(i)
  end do

  deallocate(e, xvg)
end subroutine vg2_kernel

