c     rayinvr_interface.f - 简化接口文件
c     为rayinvr程序提供Python接口
c     直接利用rayinvr.com中的公共数组

c     初始化所有rayinvr变量
      subroutine init_rayinvr_vars()
      include 'rayinvr.par'
      include 'rayinvr.com'
      include 'rayinvr_interface.com'
      integer i, j, k, l
      
      irkc = 0
      tol = 0.0005
      hdenom = 64.0
      hmin = 0.01
      idump = 0
      itx = 1
      vred = 8.0
      
      ! 初始化nzed数组
      do i=1,pncntr
         nzed(i) = 1
      enddo
      
      ! 初始化nvel数组
      do i=1,player
         do j=1,2
            nvel(i,j) = 1
         enddo
      enddo
      
      step = 0.05
      smin = -1.0
      smax = -1.0
      ntt = 1
      
      ! 初始化ray数组
      do i=1,prayf
         ray(i) = 0.0
      enddo
      
      xmin = 0.0
      xmax = -99999.0
      xmm = 250.0
      ndecix = -2
      ntickx = -1
      
      xmint = -9999.0
      xmaxt = -9999.0
      xmmt = -9999.0
      ndecxt = -2
      ntckxt = -1
      
      zmin = 0.0
      zmax = 50.0
      zmm = 75.0
      ndeciz = -2
      ntickz = -1
      
      tmin = 0.0
      tmax = 10.0
      tmm = 75.0
      ndecit = -2
      ntickt = -1
      
      xtmin = -999999.0
      xtmax = -999999.0
      xtmint = -999999.0
      xtmaxt = -999999.0
      ztmin = -999999.0
      ztmax = -999999.0
      ttmin = -999999.0
      ttmax = -999999.0
      
      symht = 0.5
      albht = 2.5
      ibsmth = 0
      nbsmth = 10
      npbnd = 100
      
      ! 初始化cv数组
      do i=1,player
         do j=1,ptrap
            do k=1,4
               do l=1,5
                  cv(i,j,k,l) = 0.0
               enddo
            enddo
         enddo
      enddo
      
      crit = 1.0
      hws = -1.0
      
      iplot = 1
      iplots = 0
      orig = 12.5
      sep = 7.5
      iseg = 0
      nseg = 0
      
      xwndow = 0.0
      ywndow = 0.0
      
      ! 初始化colour数组
      do i=1,pcol
         colour(i) = -1
      enddo
      
      ! 初始化mcol数组
      do i=1,5
         mcol(i) = -1
      enddo
      
      sf = 1.2
      ibcol = 0
      ifcol = 1
      
      return
      end

c     运行rayinvr主程序
      subroutine run_rayinvr_main()
      include 'rayinvr.par'
      include 'rayinvr.com'
      include 'rayinvr_interface.com'           
      integer i
      
      call init_ray_storage()
      call init_rayinvr_vars()  ! 初始化所有变量
      call main()
      return
      end


c     初始化数组
      subroutine run_ini_setup()
      include 'rayinvr.par'
      include 'rayinvr.com'
      include 'rayinvr_interface.com'      
      call ini_setup()
      return
      end


c     获取射线路径数据
      subroutine get_ray_paths(ray_idx, x_array, z_array, 
     +                        n_points, max_points)
      include 'rayinvr.par'
      include 'rayinvr.com'
      include 'rayinvr_interface.com'      
      integer ray_idx, n_points, max_points, i
      real x_array(*), z_array(*)
      
      n_points = npt_in
      if (n_points .gt. max_points) n_points = max_points
      
      do 20 i=1,n_points
         x_array(i) = xr(i)
         z_array(i) = zr(i)
20    continue
      
      return
      end


c     获取走时数据
      subroutine get_travel_times(n_times, x_array, t_array, 
     +                          x_shot_array, max_times)
      include 'rayinvr.par'
      include 'rayinvr.com'
      include 'rayinvr_interface.com'      
      integer n_times, max_times, i
      real x_array(*), t_array(*), x_shot_array(*)
      
      n_times = ntt
      if (n_times .gt. max_times) n_times = max_times
      
      do 40 i=1,n_times
         x_array(i) = range(i)
         t_array(i) = tt(i)
         x_shot_array(i) = xshtar(i)
40    continue
      
      return
      end


c     检查npt_in值
      subroutine check_npt(n_points)
      include 'rayinvr.par'
      include 'rayinvr.com'
      include 'rayinvr_interface.com'      
      integer n_points
      
      n_points = npt_in
      write(6,*) '当前npt_in值 =', n_points
      return
      end 

c     初始化射线存储
      subroutine init_ray_storage()
      include 'rayinvr.par'
      include 'rayinvr.com'
      include 'rayinvr_interface.com'      
      ray_count_stored = 0
      return
      end

c     存储当前射线
      subroutine store_ray(n_pts, x_array, z_array, t_array, total_time)
      include 'rayinvr.par'
      include 'rayinvr.com'
      include 'rayinvr_interface.com'      
      integer n_pts, i
      real x_array(*), z_array(*), t_array(*), total_time
c      write(6,*) 'Now call store_ray'
      if (ray_count_stored .lt. prayt .and. n_pts .gt. 0) then
        ray_count_stored = ray_count_stored + 1
        ray_npt_stored(ray_count_stored) = n_pts
        ray_tt_stored(ray_count_stored) = total_time
        
        do 10 i=1,n_pts
          if (i .le. ppray) then
            ray_x_stored(ray_count_stored, i) = x_array(i)
            ray_z_stored(ray_count_stored, i) = z_array(i)
            ray_t_stored(ray_count_stored, i) = t_array(i)
          endif
10      continue
      endif
      
      return
      end

c     获取存储的射线数量
      subroutine get_ray_count(n_rays)
      include 'rayinvr.par'
      include 'rayinvr.com'
      include 'rayinvr_interface.com'
      
      integer n_rays
      
      n_rays = ray_count_stored
      return
      end

c     获取指定射线
      subroutine get_stored_ray(ray_idx, x_arr, z_arr, t_arr, 
     +                        n_points, max_points)
      include 'rayinvr.par'
      include 'rayinvr.com'
      include 'rayinvr_interface.com'

      integer ray_idx, n_points, max_points, i
      real x_arr(*), z_arr(*), t_arr(*)
      
      if (ray_idx .gt. 0 .and. ray_idx .le. ray_count_stored) then
        n_points = ray_npt_stored(ray_idx)
        if (n_points .gt. max_points) n_points = max_points
        
        do 20 i=1,n_points
          x_arr(i) = ray_x_stored(ray_idx, i)
          z_arr(i) = ray_z_stored(ray_idx, i)
          t_arr(i) = ray_t_stored(ray_idx, i)
20      continue
      else
        n_points = 0
      endif
      
      return
      end

c     获取指定射线的总走时
      subroutine get_ray_time(ray_idx, travel_time)
      include 'rayinvr.par'
      include 'rayinvr.com'
      include 'rayinvr_interface.com'      
      integer ray_idx
      real travel_time
      
      if (ray_idx .gt. 0 .and. ray_idx .le. ray_count_stored) then
        travel_time = ray_tt_stored(ray_idx)
      else
        travel_time = 0.0
      endif
      
      return
      end

c     获取观测走时数据
      subroutine get_observed_data(nobs, x_array, t_array, u_array, 
     +                             phase_array)
      include 'rayinvr.par'
      include 'rayinvr.com'
      include 'rayinvr_interface.com'      
      integer nobs
      real*4 x_array(*), t_array(*), u_array(*)
      integer phase_array(*)
      
c     获取观测数据数量
      nobs = 0
      do i = 1, prayi+pshot2+1
         if (ipf(i) .gt. 0) then
            nobs = nobs + 1
            x_array(nobs) = xpf(i)
            t_array(nobs) = tpf(i)
            u_array(nobs) = upf(i)
            phase_array(nobs) = ipf(i)
         endif
      enddo
      
      return
      end 

c     获取射线角度信息
      subroutine get_ray_angles(ray_idx, initial_angle, final_angle,
     +                         shot_num, ray_num)
      include 'rayinvr.par'
      include 'rayinvr.com'
      include 'rayinvr_interface.com'      
      integer ray_idx, shot_num, ray_num
      real initial_angle, final_angle
      
      if (ray_idx .gt. 0 .and. ray_idx .le. ray_count_stored) then
        initial_angle = ray_angle_i_stored(ray_idx)
        final_angle = ray_angle_f_stored(ray_idx)
        shot_num = ray_shot_stored(ray_idx)
        ray_num = ray_num_stored(ray_idx)
      else
        initial_angle = 0.0
        final_angle = 0.0
        shot_num = 0
        ray_num = 0
      endif
      
      return
      end 