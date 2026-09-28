c     version 1.2  Jul 2026
c
c     common blocks for RAYINVR_INTERFACE
c     ray_ips_stored: +1 = P-wave segment, -1 = S-wave (same test as pltray/irayps)
c     ray_phase_stored: ivray / tx.out phase id for that ray family
      integer ray_npt_stored(prayt),ray_count_stored,
     +        ray_shot_stored(prayt),ray_num_stored(prayt),
     +        ray_phase_stored(prayt),
     +        ray_ips_stored(prayt,ppray)
      real*4 ray_x_stored(prayt,ppray),ray_z_stored(prayt,ppray),
     +       ray_t_stored(prayt,ppray),ray_tt_stored(prayt),
     +       ray_angle_i_stored(prayt),ray_angle_f_stored(prayt)

      common /ray_storage/ ray_x_stored, ray_z_stored, ray_t_stored, 
     +                     ray_tt_stored, ray_npt_stored,
     +                     ray_count_stored, ray_shot_stored,
     +                     ray_num_stored, ray_angle_i_stored,
     +                     ray_angle_f_stored, ray_ips_stored,
     +                     ray_phase_stored