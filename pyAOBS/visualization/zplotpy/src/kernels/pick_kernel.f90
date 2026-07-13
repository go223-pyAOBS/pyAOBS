subroutine pick_kernel(seis, npts, start_idx, end_idx, nwind, dts, minenratio, &
                       pick_idx, max_ratio, iflag)
  implicit none
  integer, intent(in) :: npts, start_idx, end_idx, nwind
  real(4), intent(in) :: seis(npts), dts, minenratio
  integer, intent(out) :: pick_idx, iflag
  real(4), intent(out) :: max_ratio

  integer :: i, j, nit, nw2, nstpnw, im1, ipos, final_pick_idx
  real(4) :: sum1, sum2, er
  real(4), allocatable :: seis2(:)

  iflag = 0
  pick_idx = -1
  max_ratio = 0.0

  if (npts <= 0) return
  if (nwind < 1) return
  if (start_idx < 0 .or. end_idx > npts .or. start_idx >= end_idx) return

  nw2 = nwind * 2
  if (end_idx - start_idx < nw2 + 2) return

  allocate(seis2(npts))
  do i = 1, npts
    seis2(i) = seis(i) * seis(i)
  end do

  nit = min(npts - nw2, end_idx - nw2)
  if (nit < start_idx) then
    deallocate(seis2)
    return
  end if

  nstpnw = start_idx + nwind
  if (nstpnw - nwind < 0 .or. nstpnw + nwind >= npts) then
    deallocate(seis2)
    return
  end if

  sum1 = 0.0
  sum2 = 0.0
  do j = 1, nwind
    sum1 = sum1 + seis2(nstpnw - j + 1)
    sum2 = sum2 + seis2(nstpnw + j + 1)
  end do

  max_ratio = minenratio
  if (sum1 > 0.0) then
    er = sum2 / sum1
    if (er > max_ratio) then
      ipos = start_idx
      max_ratio = er
      iflag = 1
    end if
  end if

  do i = start_idx + 1, nit
    im1 = i - 1
    if (im1 < 0) cycle
    if (im1 + nwind >= npts) cycle
    if (i + nwind >= npts) cycle
    if (i + nw2 >= npts) cycle

    sum1 = sum1 - seis2(im1 + 1) + seis2(im1 + nwind + 1)
    sum2 = sum2 - seis2(i + nwind + 1) + seis2(i + nw2 + 1)

    if (sum1 > 0.0) then
      er = sum2 / sum1
      if (er > max_ratio) then
        ipos = i
        max_ratio = er
        iflag = 1
      end if
    end if
  end do

  if (iflag == 1) then
    final_pick_idx = ipos + nwind - 1
    if (final_pick_idx >= 0 .and. final_pick_idx < npts) then
      pick_idx = final_pick_idx
    else
      iflag = 0
      pick_idx = -1
      max_ratio = 0.0
    end if
  end if

  deallocate(seis2)
end subroutine pick_kernel

