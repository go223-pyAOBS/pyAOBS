subroutine build_shade_segments(x_values, t_values, npts, baseline_x, fill_positive, row_step, &
                                seg_start_x, seg_start_t, seg_end_x, seg_end_t, nseg)
  implicit none
  integer, intent(in) :: npts, fill_positive, row_step
  real(4), intent(in) :: x_values(npts), t_values(npts), baseline_x
  real(4), intent(out) :: seg_start_x(npts), seg_start_t(npts)
  real(4), intent(out) :: seg_end_x(npts), seg_end_t(npts)
  integer, intent(out) :: nseg

  integer :: i, step_local
  logical :: keep

  nseg = 0
  if (npts <= 0) return

  step_local = row_step
  if (step_local < 1) step_local = 1

  do i = 1, npts, step_local
    if (fill_positive == 1) then
      keep = x_values(i) > baseline_x
    else
      keep = x_values(i) < baseline_x
    end if

    if (keep) then
      nseg = nseg + 1
      seg_start_x(nseg) = baseline_x
      seg_start_t(nseg) = t_values(i)
      seg_end_x(nseg) = x_values(i)
      seg_end_t(nseg) = t_values(i)
    end if
  end do
end subroutine build_shade_segments

