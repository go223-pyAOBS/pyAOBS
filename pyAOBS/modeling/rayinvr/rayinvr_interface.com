c     version 1.0  May 2025
c
c     common blocks for RAYINVR_INTERFACE
      integer ray_npt_stored(prayt),ray_count_stored,
     +        ray_shot_stored(prayt),ray_num_stored(prayt)
      real*4 ray_x_stored(prayt,ppray),ray_z_stored(prayt,ppray),
     +       ray_t_stored(prayt,ppray),ray_tt_stored(prayt),
     +       ray_angle_i_stored(prayt),ray_angle_f_stored(prayt)

      common /ray_storage/ ray_x_stored, ray_z_stored, ray_t_stored, 
     +                     ray_tt_stored, ray_npt_stored,
     +                     ray_count_stored, ray_shot_stored,
     +                     ray_num_stored, ray_angle_i_stored,
     +                     ray_angle_f_stored