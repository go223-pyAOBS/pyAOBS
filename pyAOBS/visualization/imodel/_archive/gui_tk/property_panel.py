"""下方物性剖面显示。"""

from __future__ import annotations

from .deps import *  # noqa: F403


class PropertyPanelMixin:
    def plot_property_profile(self, prop: str = 'density'):
        """将整个速度模型转换成密度/温度/压力的二维网格并显示"""
        if self.grid_data is None:
            messagebox.showwarning('Warning', 'Please load model first')
            return
        
        # 获取坐标和速度网格
        z_coord = self.profile_extractor.z_coord
        x_coord = self.profile_extractor.x_coord
        z_vals = np.asarray(self.grid_data[z_coord].values, dtype=float)
        x_vals = np.asarray(self.grid_data[x_coord].values, dtype=float)
        
        # 自动检测速度变量名（与主模型一致）
        if not hasattr(self, 'velocity_var') or self.velocity_var is None:
            self._detect_velocity_var()
        vel_var = self.velocity_var
        velocity_data = np.asarray(self.grid_data[vel_var].values, dtype=float)  # shape: (nz, nx)
        
        nz, nx = velocity_data.shape
        
        # 预分配2D结果网格
        property_grid = np.zeros_like(velocity_data, dtype=float)
        
        # 密度转换需要区分海水：简单使用Vp阈值判断
        water_density = (
            self.property_calculator.seawater_density_g_cm3
            if self.property_calculator is not None
            else 1.03
        )
        water_temp = (
            self.property_calculator.seawater_temperature_c
            if self.property_calculator is not None
            else 4.0
        )
        seafloor_temp = (
            self.property_calculator.seafloor_temperature_c
            if self.property_calculator is not None
            else water_temp
        )
        water_pressure_grad = (
            self.property_calculator.seawater_pressure_gradient_mpa_per_km
            if self.property_calculator is not None
            else 10.1
        )
        geothermal_gradient = (
            self.property_calculator.geothermal_gradient_c_per_km
            if self.property_calculator is not None
            else 30.0
        )
        seawater_note = (
            f"Seawater params: T={water_temp:.2f}°C, "
            f"rho={water_density:.3f} g/cm³, dP/dz={water_pressure_grad:.2f} MPa/km"
        )
        geothermal_note = f"Geothermal gradient: {geothermal_gradient:.2f} °C/km"

        def _is_water(vp_val: float, z_val: float, seafloor_val: float) -> bool:
            if self.property_calculator is not None:
                return self.property_calculator.is_water_zone(
                    vp=vp_val,
                    z=z_val,
                    seafloor_depth=seafloor_val,
                )
            return z_val < seafloor_val and abs(vp_val - 1.5) <= 0.25
        
        if prop == 'density':
            # 从empirical_formulas统一调用密度计算公式
            method = getattr(self, 'density_method_var', None)
            density_method = method.get() if method is not None else 'gardner'
            density_method_library = self._density_method_for_library(density_method)
            
            # 对整个速度网格应用密度公式
            with np.errstate(invalid='ignore'):
                property_grid = calculate_density(velocity_data, method=density_method_library)
            property_grid = np.asarray(property_grid, dtype=float)

            # 海水：使用统一海水判据 + 可配置海水密度
            seafloor_depths = self._get_seafloor_depths(x_vals)
            water_mask = np.zeros((nz, nx), dtype=bool)
            for i in range(nz):
                z = float(z_vals[i])
                for j in range(nx):
                    vp = float(velocity_data[i, j])
                    seafloor_z = seafloor_depths[j] if j < len(seafloor_depths) else 0.0
                    if _is_water(vp, z, seafloor_z):
                        property_grid[i, j] = water_density
                        water_mask[i, j] = True

            if self._uses_tomo2d_sediment_density(density_method):
                basement_depths = np.asarray(
                    [self._get_basement_depth(float(x_val)) for x_val in np.asarray(x_vals, dtype=float)],
                    dtype=float,
                )
                sed_mask = np.zeros((nz, nx), dtype=bool)
                for j in range(nx):
                    seafloor_z = float(seafloor_depths[j]) if j < len(seafloor_depths) else 0.0
                    basement_z = float(basement_depths[j]) if j < len(basement_depths) else np.nan
                    if not np.isfinite(basement_z) or basement_z <= seafloor_z:
                        continue
                    z_col = np.asarray(z_vals, dtype=float)
                    sed_col = (z_col >= seafloor_z) & (z_col < basement_z)
                    sed_mask[:, j] = sed_col
                sed_mask = sed_mask & (~water_mask)
                if np.any(sed_mask):
                    property_grid[sed_mask] = 1.0 + 1.18 * np.power(
                        np.maximum(np.asarray(velocity_data, dtype=float)[sed_mask] - 1.5, 0.0),
                        0.22,
                    )
                    property_grid[sed_mask] = np.minimum(property_grid[sed_mask], 2.6)
            
            x_label = 'Distance (km)'
            y_label = 'Depth (km)'
            title = 'Density Model (g/cm³)'
            cmap = 'viridis'
            vmin = np.nanmin(property_grid)
            vmax = np.nanmax(property_grid)
        
        elif prop == 'temperature':
            # 从empirical_formulas统一调用温度计算公式
            # 温度从海底开始计算
            temp_gradient = geothermal_gradient  # °C/km
            
            # 获取海底面深度映射
            seafloor_depths = self._get_seafloor_depths(x_vals)
            
            # 为每个网格点计算温度（考虑海底面深度，使用统一接口）
            for i in range(nz):
                z = float(z_vals[i])
                for j in range(nx):
                    vp = float(velocity_data[i, j])
                    # 获取该X位置的海底面深度
                    seafloor_z = seafloor_depths[j] if j < len(seafloor_depths) else 0.0
                    if _is_water(vp, z, seafloor_z):
                        temp_val = water_temp
                    else:
                        # 岩石层：使用统一接口（从海底起算）
                        temp_val = calculate_temperature_from_depth(
                            z,
                            temperature_gradient=temp_gradient,
                            surface_temperature=seafloor_temp,
                            seafloor_depth=seafloor_z,
                        )
                    property_grid[i, j] = temp_val
            
            x_label = 'Distance (km)'
            y_label = 'Depth (km)'
            title = 'Temperature Model (°C, from seafloor)'
            cmap = 'hot'
            vmin = np.nanmin(property_grid)
            vmax = np.nanmax(property_grid)
        
        elif prop == 'pressure':
            # 从empirical_formulas统一调用压力计算公式
            # 压力从海底开始计算
            pressure_gradient = 30.0  # MPa/km
            
            # 获取海底面深度映射
            seafloor_depths = self._get_seafloor_depths(x_vals)
            
            # 为每个网格点计算压力（考虑海底面深度，使用统一接口）
            for i in range(nz):
                z = float(z_vals[i])
                for j in range(nx):
                    vp = float(velocity_data[i, j])
                    # 获取该X位置的海底面深度
                    seafloor_z = seafloor_depths[j] if j < len(seafloor_depths) else 0.0
                    if _is_water(vp, z, seafloor_z):
                        pressure_val = max(0.0, z) * water_pressure_grad
                    else:
                        # 岩石层总压力：海水柱静水压 + 海底以下增量压力
                        hydro_at_seafloor = max(0.0, float(seafloor_z)) * water_pressure_grad
                        rock_increment = calculate_pressure_from_depth(
                            z,
                            pressure_gradient=pressure_gradient,
                            seafloor_depth=seafloor_z,
                        )
                        pressure_val = hydro_at_seafloor + float(rock_increment)
                    property_grid[i, j] = pressure_val
            
            x_label = 'Distance (km)'
            y_label = 'Depth (km)'
            title = 'Pressure Model (MPa, from seafloor)'
            cmap = 'plasma'
            vmin = np.nanmin(property_grid)
            vmax = np.nanmax(property_grid)
        else:
            messagebox.showerror('Error', f'Unknown property type: {prop}')
            return
        
        # 创建绘图窗口（类似主模型窗口）
        win = tk.Toplevel(self.root)
        win.title(f"{title} | {geothermal_note} | {seawater_note}")
        win.geometry('1000x700+150+50')
        
        fig = plt.Figure(figsize=(10, 3.5))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        
        # 使用pcolormesh绘制2D网格（与主模型图一致）
        im = ax.pcolormesh(
            x_vals,
            z_vals,
            property_grid,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading='auto'
        )
        
        # 存储等值线对象（用于属性图）
        prop_contour_lines = None
        
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(title.split('(')[0].strip(), fontsize=10)
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f"{title}\n{geothermal_note}; {seawater_note}", fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, linestyle='--', alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = ttk.Frame(win)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        
        # 保存/导出按钮和控制选项
        control_frame = ttk.Frame(win)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=4)
        
        # 存储界面线和速度等值线对象
        prop_interface_lines = []  # 存储所有界面线对象
        prop_velocity_contour_lines = None  # 速度等值线对象
        
        # 绘制界面（如果读取了v.in文件）
        def _plot_prop_interfaces():
            """在属性图中绘制界面"""
            nonlocal prop_interface_lines
            # 清除之前的界面线
            for line in prop_interface_lines:
                line.remove()
            prop_interface_lines.clear()
            
            if not self.zelt_model:
                return
            
            try:
                # 绘制所有层界面
                for i in range(len(self.zelt_model.depth_nodes)):
                    x_coords, z_coords = self.zelt_model.get_layer_geometry(i)
                    
                    # 过滤掉NaN值
                    valid_mask = ~(np.isnan(x_coords) | np.isnan(z_coords))
                    if np.any(valid_mask):
                        x_valid = np.array(x_coords)[valid_mask]
                        z_valid = np.array(z_coords)[valid_mask]
                        
                        # 检查是否是选中的沉积基底界面
                        is_basement = (self.basement_interface_idx is not None and 
                                      i == self.basement_interface_idx)
                        # 检查是否是选中的海底面界面
                        is_seafloor = (self.seafloor_interface_idx is not None and 
                                      i == self.seafloor_interface_idx)
                        
                        # 最后一个界面（底部边界）用不同样式
                        if i == len(self.zelt_model.depth_nodes) - 1:
                            # 底部边界
                            line, = ax.plot(x_valid, z_valid, 'k--', linewidth=2.0, 
                                          alpha=0.8)
                            prop_interface_lines.append(line)
                        else:
                            # 层界面
                            if is_basement:
                                # 选中的沉积基底界面：用红色高亮显示
                                line, = ax.plot(x_valid, z_valid, 'r-', linewidth=2.5, 
                                              alpha=0.9)
                                prop_interface_lines.append(line)
                            elif is_seafloor:
                                # 选中的海底面界面：用蓝色高亮显示
                                line, = ax.plot(x_valid, z_valid, 'b-', linewidth=2.5, 
                                              alpha=0.9)
                                prop_interface_lines.append(line)
                            else:
                                # 普通层界面
                                line, = ax.plot(x_valid, z_valid, 'k-', linewidth=1.5, 
                                              alpha=0.7)
                                prop_interface_lines.append(line)
            except Exception as e:
                self.log_result(f"Warning: Failed to plot interfaces in property plot: {str(e)}")
        
        # 等值线显示选项（属性图）
        prop_show_contours_var = tk.BooleanVar(value=False)
        
        def _toggle_prop_contours():
            """切换属性图等值线显示"""
            nonlocal prop_contour_lines
            if prop_show_contours_var.get():
                # 绘制等值线
                if prop_contour_lines is not None:
                    for coll in prop_contour_lines.collections:
                        coll.remove()
                num_levels = 10
                levels = np.linspace(vmin, vmax, num_levels)
                prop_contour_lines = ax.contour(
                    x_vals,
                    z_vals,
                    property_grid,
                    levels=levels,
                    colors='white',
                    linewidths=0.8,
                    alpha=0.6
                )
                ax.clabel(prop_contour_lines, inline=True, fontsize=8, fmt='%.2f')
            else:
                # 清除等值线
                if prop_contour_lines is not None:
                    for coll in prop_contour_lines.collections:
                        coll.remove()
                    prop_contour_lines = None
            canvas.draw()
        
        # 速度等值线显示选项
        prop_show_velocity_contours_var = tk.BooleanVar(value=False)
        
        # 默认速度等值线值
        default_velocity_levels = [1.5, 1.8, 2.1, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
        
        # 存储速度等值线值的复选框变量
        vel_contour_vars = {}
        for level in default_velocity_levels:
            vel_contour_vars[level] = tk.BooleanVar(value=True)  # 默认全部选中
        
        def _toggle_velocity_contours():
            """切换速度等值线显示"""
            nonlocal prop_velocity_contour_lines
            if prop_show_velocity_contours_var.get():
                # 绘制速度等值线
                if prop_velocity_contour_lines is not None:
                    for coll in prop_velocity_contour_lines.collections:
                        coll.remove()
                
                # 获取选中的速度等值线值
                selected_levels = [level for level in default_velocity_levels 
                                  if vel_contour_vars[level].get()]
                
                if selected_levels:
                    prop_velocity_contour_lines = ax.contour(
                        x_vals,
                        z_vals,
                        velocity_data,
                        levels=selected_levels,
                        colors='yellow',
                        linewidths=1.2,
                        alpha=0.8,
                        linestyles='--'
                    )
                    ax.clabel(prop_velocity_contour_lines, inline=True, fontsize=8, 
                            fmt='%.1f', colors='yellow')
            else:
                # 清除速度等值线
                if prop_velocity_contour_lines is not None:
                    for coll in prop_velocity_contour_lines.collections:
                        coll.remove()
                    prop_velocity_contour_lines = None
            canvas.draw()
        
        # 界面显示复选框
        prop_show_interfaces_var = tk.BooleanVar(value=True)
        if self.zelt_model:
            def _toggle_prop_interfaces():
                """切换界面显示"""
                if prop_show_interfaces_var.get():
                    _plot_prop_interfaces()
                else:
                    # 清除界面线
                    for line in prop_interface_lines:
                        line.remove()
                    prop_interface_lines.clear()
                canvas.draw()
            
            interface_check = ttk.Checkbutton(control_frame, text='Show Interfaces', 
                                            variable=prop_show_interfaces_var,
                                            command=_toggle_prop_interfaces)
            interface_check.pack(side=tk.LEFT, padx=5)
            # 初始绘制界面
            _plot_prop_interfaces()
        
        # 等值线复选框
        contour_check = ttk.Checkbutton(control_frame, text='Show Contours', 
                                       variable=prop_show_contours_var,
                                       command=_toggle_prop_contours)
        contour_check.pack(side=tk.LEFT, padx=5)
        
        # 速度等值线复选框和选项
        vel_contour_frame = ttk.LabelFrame(control_frame, text='Velocity Contours', padding=2)
        vel_contour_frame.pack(side=tk.LEFT, padx=5)
        
        vel_contour_check = ttk.Checkbutton(vel_contour_frame, text='Show', 
                                          variable=prop_show_velocity_contours_var,
                                          command=_toggle_velocity_contours)
        vel_contour_check.pack(side=tk.LEFT, padx=2)
        
        # 为每个速度等值线值创建复选框
        vel_levels_frame = ttk.Frame(vel_contour_frame)
        vel_levels_frame.pack(side=tk.LEFT, padx=2)
        
        # 创建速度等值线值的复选框
        for level in default_velocity_levels:
            cb = ttk.Checkbutton(vel_levels_frame, text=f'{level:.1f}', 
                               variable=vel_contour_vars[level], 
                               command=_toggle_velocity_contours)
            cb.pack(side=tk.LEFT, padx=1)
        
        def _save_prop_figure():
            filename = filedialog.asksaveasfilename(
                title='Save Property Model Figure',
                filetypes=[
                    ('PNG', '*.png'),
                    ('PDF', '*.pdf'),
                    ('PS', '*.ps'),
                    ('EPS', '*.eps'),
                    ('JPG', '*.jpg'),
                    ('TIF', '*.tif'),
                    ('All files', '*.*'),
                ],
            )
            if not filename:
                return
            try:
                import os
                base, ext = os.path.splitext(filename)
                if not ext:
                    ext = '.png'
                    filename = base + ext
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                self.log_result(f"Property model figure saved to: {filename}")
            except Exception as e:
                messagebox.showerror('Error', f'Failed to save figure: {str(e)}')
        
        def _export_prop_data():
            filename = filedialog.asksaveasfilename(
                title='Export Property Model Data',
                filetypes=[('NetCDF files', '*.nc'), ('All files', '*.*')],
            )
            if not filename:
                return
            try:
                # 创建xarray Dataset并保存
                import xarray as xr
                prop_ds = xr.Dataset(
                    {prop: (['z', 'x'], property_grid)},
                    coords={
                        z_coord: z_vals,
                        x_coord: x_vals
                    }
                )
                prop_ds.to_netcdf(filename)
                self.log_result(f"Property model data exported to: {filename}")
            except Exception as e:
                messagebox.showerror('Error', f'Failed to export data: {str(e)}')
        
        ttk.Button(control_frame, text='Save Figure', command=_save_prop_figure).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(control_frame, text='Export Data', command=_export_prop_data).pack(
            side=tk.LEFT, padx=5
        )
        
        self.log_result(f"\n{'='*50}")
        self.log_result(f"{title}")
        self.log_result(f"  Property range: {vmin:.2f} - {vmax:.2f}")
        self.log_result(f"  Grid size: {nx} x {nz}")
    
