"""模型加载、主图绘制、等值线、界面与海底面。"""

from __future__ import annotations

from .deps import *  # noqa: F403


class ModelSurfaceMixin:
    def open_model(self):
        """Open velocity model file (支持 grid 和 v.in 格式)"""
        filename = filedialog.askopenfilename(
            title='Load Vp Model',
            filetypes=[
                ('Grid files', '*.grd *.nc'),
                ('v.in files', 'v.in'),
                ('NetCDF files', '*.nc'),
                ('All files', '*.*')
            ]
        )
        if filename:
            self._load_model_from_file(filename)

    def open_vp_model(self):
        """Alias for loading primary Vp model."""
        self.open_model()

    def open_vs_model(self):
        """Open optional Vs model file (grid format)"""
        filename = filedialog.askopenfilename(
            title='Load Vs Model',
            filetypes=[
                ('Grid files', '*.grd *.nc'),
                ('NetCDF files', '*.nc'),
                ('All files', '*.*')
            ]
        )
        if filename:
            self._load_vs_model_from_file(filename)

    def _load_vs_model_from_file(self, filename: str) -> bool:
        """Load Vs model and bind it to property calculator."""
        if not filename:
            return False
        try:
            processor = GridModelProcessor(grid_file=filename)
            self.vs_grid_data = processor.velocity_grid
            self.vs_model_file = str(Path(filename).resolve())
            if self.grid_data is not None:
                self.property_calculator = PropertyCalculator(self.grid_data, vs_grid_data=self.vs_grid_data)
            self.log_result(f"Vs model loaded: {filename}")
            if self.property_calculator is not None:
                self.log_result(
                    f"Vs velocity variable detected: {self.property_calculator.vs_velocity_var}"
                )
                self._log_vp_vs_coverage_summary()
            self._save_workbench_state()
            return True
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load Vs model:\n{str(e)}')
            self.log_result(f"Error loading Vs model: {str(e)}")
            return False

    def _load_model_from_file(self, filename: str, *, persist_state: bool = True) -> bool:
        """Load model file and refresh plotting state."""
        if not filename:
            return False
        try:
            file_path = Path(filename)
            is_vin_format = (
                file_path.name == 'v.in'
                or file_path.name.endswith('.vin')
                or self._is_vin_format(filename)
            )

            if is_vin_format:
                self.log_result("Detected v.in format, loading...")
                self.grid_data = self._load_vin_model(filename)
                self.processor = GridModelProcessor()
                self.processor.velocity_grid = self.grid_data
                self.log_result("v.in model loaded successfully")
                self.log_result(
                    f"Number of layers: {len(self.zelt_model.depth_nodes) if self.zelt_model else 'N/A'}"
                )
                self._update_basement_interface_options()
                self._update_seafloor_interface_options()
            else:
                self.zelt_model = None
                self.basement_interface_idx = None
                self._update_basement_interface_options()
                self.seafloor_interface_idx = None
                self.seafloor_depth_map = None
                self._update_seafloor_interface_options()
                self.seafloor_depth_map = None
                self.processor = GridModelProcessor(grid_file=filename)
                self.grid_data = self.processor.velocity_grid
                self.log_result("Grid model loaded successfully")

            self.current_model_file = str(file_path.resolve())
            self._full_model_gravity_cache = None
            self._detect_velocity_var()
            self.profile_extractor = ProfileExtractor(self.grid_data)
            self.property_calculator = PropertyCalculator(self.grid_data, vs_grid_data=self.vs_grid_data)
            self._classifier_diag_logged = False
            self.plot_model()
            self.log_result(f"Model loaded: {filename}")
            self.log_result(f"Format: {'v.in' if is_vin_format else 'grid'}")
            self.log_result(f"Velocity variable detected: {self.velocity_var}")
            if self.vs_grid_data is not None:
                self.log_result(
                    f"Vs model active: {self.vs_model_file or 'in-memory dataset'} "
                    f"(var={self.property_calculator.vs_velocity_var})"
                )
            self._log_vp_vs_coverage_summary()
            self._log_classifier_diagnostics()
            if persist_state:
                self._save_workbench_state()
            return True
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load model:\n{str(e)}')
            import traceback
            traceback.print_exc()
            self.log_result(f"Error loading model: {str(e)}")
            return False
    
    def plot_model(self, 
                   cmap: str = 'viridis',
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   show_colorbar: bool = True):
        """绘制速度模型"""
        if self.grid_data is None:
            return
        
        self.ax_main.clear()
        
        # 获取速度数据（使用检测到的速度变量名）
        if not hasattr(self, 'velocity_var'):
            self._detect_velocity_var()
        
        velocity = self.grid_data[self.velocity_var].values
        
        # 获取坐标（使用ProfileExtractor检测到的坐标名）
        if hasattr(self, 'profile_extractor'):
            x_coord = self.profile_extractor.x_coord
            z_coord = self.profile_extractor.z_coord
        else:
            coords = list(self.grid_data.coords)
            x_coord = coords[0] if coords else 'x'
            z_coord = coords[1] if len(coords) > 1 else 'z'
        
        x_coords = self.grid_data.coords[x_coord].values
        z_coords = self.grid_data.coords[z_coord].values
        
        # 确定速度范围
        if vmin is None:
            vmin = np.nanmin(velocity)
        if vmax is None:
            vmax = np.nanmax(velocity)
        
        # 使用pcolormesh代替imshow，可以更好地处理坐标方向
        # pcolormesh会自动根据坐标数组的方向正确显示
        # 参考show_model.py中的实现（304行）：plt.pcolormesh(ds.x.values, ds.z.values, ds.velocity.values, ...)
        # pcolormesh会自动处理坐标方向，无需手动设置extent和origin
        im = self.ax_main.pcolormesh(
            x_coords,
            z_coords,
            velocity,
            shading='auto',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        
        # 绘制等值线（如果启用）
        if hasattr(self, 'show_contours_var') and self.show_contours_var.get():
            self._draw_contours(x_coords, z_coords, velocity, vmin, vmax)
        
        self.ax_main.set_xlabel('Distance (km)')
        self.ax_main.set_ylabel('Depth (km)')
        self.ax_main.set_title('Velocity Model')
        self.ax_main.invert_yaxis()  # 深度向下增加
        
        if show_colorbar:
            if not hasattr(self, 'cbar') or self.cbar is None:
                self.cbar = self.fig.colorbar(im, ax=self.ax_main)
                self.cbar.set_label('Velocity (km/s)')
            else:
                self.cbar.update_normal(im)
        
        # 如果存在 zelt_model，绘制界面
        if self.zelt_model and self.show_interfaces_var.get():
            self._plot_interfaces()
        
        # 绘制加载的界面（网格文件）
        if hasattr(self, 'loaded_interfaces') and self.loaded_interfaces and self.show_interfaces_var.get():
            self._plot_loaded_interfaces()
        
        # 绘制单个basement_interface_data（向后兼容，如果不在loaded_interfaces中）
        if self.basement_interface_data is not None and self.show_interfaces_var.get():
            # 检查是否已经在loaded_interfaces中
            is_in_loaded = False
            if hasattr(self, 'loaded_interfaces') and self.loaded_interfaces:
                for interface in self.loaded_interfaces:
                    if (np.array_equal(interface['x'], self.basement_interface_data['x']) and
                        np.array_equal(interface['z'], self.basement_interface_data['z'])):
                        is_in_loaded = True
                        break
            
            if not is_in_loaded:
                x_coords = self.basement_interface_data['x']
                z_coords = self.basement_interface_data['z']
                self.ax_main.plot(x_coords, z_coords, 'g-', linewidth=2.0, alpha=0.8, label='Loaded Interface')
        
        # 更新图例（如果有标签）
        handles, labels = self.ax_main.get_legend_handles_labels()
        if labels:
            # 过滤掉空标签
            filtered_handles = [h for h, l in zip(handles, labels) if l]
            filtered_labels = [l for l in labels if l]
            if filtered_labels:
                self.ax_main.legend(filtered_handles, filtered_labels, 
                                  loc='upper right', fontsize=8, framealpha=0.8)
        
        self.canvas.draw()
        return im
    
    @staticmethod
    def _is_vin_format_static(filename: str) -> bool:
        """通过读取文件内容判断是否为 v.in 格式（静态方法）"""
        try:
            with open(filename, 'r') as f:
                first_line = f.readline().strip()
                if not first_line:
                    return False
                
                # v.in 格式的第一行通常以层号（1-2位数字）开头
                parts = first_line.split()
                if parts and parts[0].isdigit():
                    # 进一步检查：尝试读取前3行，看是否符合v.in格式
                    # v.in 格式每3行为一组：层号+x坐标, 计数+值, 标志
                    try:
                        layer_num = int(parts[0])
                        # 检查第二行是否存在
                        second_line = f.readline().strip()
                        if second_line:
                            parts2 = second_line.split()
                            if parts2 and parts2[0].isdigit():
                                # 检查第三行是否存在（标志行）
                                third_line = f.readline().strip()
                                if third_line:
                                    # 符合v.in格式的基本特征
                                    return True
                    except (ValueError, IndexError):
                        pass
        except Exception:
            pass
        return False
    
    def _is_vin_format(self, filename: str) -> bool:
        """通过读取文件内容判断是否为 v.in 格式（实例方法，调用静态方法）"""
        return ModelSurfaceMixin._is_vin_format_static(filename)
    
    def _load_vin_model(self, filename: str, dx: float = 2.0, dz: float = 0.5) -> xr.Dataset:
        """加载 v.in 格式模型并转换为 grid 格式
        
        Args:
            filename: v.in 文件路径
            dx: x方向网格间距（km），默认2.0
            dz: z方向网格间距（km），默认0.5
            
        Returns:
            xr.Dataset: 转换后的grid格式数据
        """
        # 使用 ZeltVelocityModel2d 读取 v.in 文件
        self.zelt_model = ZeltVelocityModel2d(model_file=filename)
        
        # 转换为 xarray Dataset
        grid_data = self.zelt_model.to_xarray(dx=dx, dz=dz)
        
        return grid_data
    
    def _draw_contours(self, x_coords, z_coords, data, vmin, vmax, num_levels=10):
        """绘制等值线"""
        # 清除之前的等值线
        if self.contour_lines is not None:
            for coll in self.contour_lines.collections:
                coll.remove()
            self.contour_lines = None
        
        # 计算等值线级别
        levels = np.linspace(vmin, vmax, num_levels)
        
        # 绘制等值线
        self.contour_lines = self.ax_main.contour(
            x_coords,
            z_coords,
            data,
            levels=levels,
            colors='white',
            linewidths=0.8,
            alpha=0.6
        )
        
        # 可选：添加等值线标签
        self.ax_main.clabel(self.contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    def _toggle_contours(self):
        """切换等值线显示"""
        if self.grid_data is None:
            return
        # 重新绘制模型（会自动根据show_contours_var决定是否绘制等值线）
        self.plot_model()
    
    def _toggle_interfaces(self):
        """切换界面显示状态"""
        if self.zelt_model:
            self.plot_model()
        else:
            self.log_result("No v.in model loaded. Interface display only works for v.in format.")
        if not self._restoring_state:
            self._save_workbench_state()
    
    def _plot_interfaces(self):
        """绘制v.in模型的界面"""
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
                    
                    # 绘制界面线
                    # 检查是否是选中的沉积基底界面
                    is_basement = (self.basement_interface_idx is not None and 
                                  i == self.basement_interface_idx)
                    # 检查是否是选中的海底面界面
                    is_seafloor = (self.seafloor_interface_idx is not None and 
                                  i == self.seafloor_interface_idx)
                    
                    # 最后一个界面（底部边界）用不同样式
                    if i == len(self.zelt_model.depth_nodes) - 1:
                        # 底部边界
                        self.ax_main.plot(x_valid, z_valid, 'k--', linewidth=2.0, 
                                         alpha=0.8, label='Bottom boundary' if i == len(self.zelt_model.depth_nodes) - 1 else '')
                    else:
                        # 层界面
                        if is_basement:
                            # 选中的沉积基底界面：用红色高亮显示
                            label = f'Basement Interface {i+1}' if i == 0 else f'Basement Interface {i+1}'
                            self.ax_main.plot(x_valid, z_valid, 'r-', linewidth=2.5, 
                                             alpha=0.9, label=label)
                        elif is_seafloor:
                            # 选中的海底面界面：用蓝色高亮显示
                            label = f'Seafloor Interface {i+1}'
                            self.ax_main.plot(x_valid, z_valid, 'b-', linewidth=2.5, 
                                             alpha=0.9, label=label)
                        else:
                            # 普通层界面
                            label = f'Layer {i+1} interface' if i == 0 else ''
                            self.ax_main.plot(x_valid, z_valid, 'k-', linewidth=1.5, 
                                             alpha=0.7, label=label)
            
            # 添加图例（如果有标签）
            handles, labels = self.ax_main.get_legend_handles_labels()
            if labels:
                # 只显示第一个和最后一个标签，避免图例过长
                if len(labels) > 2:
                    filtered_handles = [handles[0], handles[-1]]
                    filtered_labels = [labels[0], labels[-1]]
                    self.ax_main.legend(filtered_handles, filtered_labels, 
                                       loc='upper right', fontsize=8)
                else:
                    self.ax_main.legend(loc='upper right', fontsize=8)
        except Exception as e:
            # 如果绘制界面失败，记录错误但不中断程序
            self.log_result(f"Warning: Failed to plot interfaces: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _update_basement_interface_options(self):
        """更新沉积基底界面选择选项"""
        # 检查是否有basement_interface_combo（可能在初始化时还未创建）
        if not hasattr(self, 'basement_interface_combo'):
            return
        
        if self.zelt_model and len(self.zelt_model.depth_nodes) > 0:
            # v.in文件：列出所有界面
            options = ['None'] + [f'Interface {i+1}' for i in range(len(self.zelt_model.depth_nodes) - 1)]
            self.basement_interface_combo['values'] = options
            self.basement_interface_combo['state'] = 'readonly'
            # 如果当前选择是None，保持None；否则检查选择是否仍然有效
            current_selection = self.basement_interface_var.get()
            if current_selection not in options:
                self.basement_interface_var.set('None')
                self.basement_interface_idx = None
            self.log_result(f"Basement interface options updated: {len(options)-1} interfaces available")
        else:
            # 网格文件：禁用下拉框
            self.basement_interface_combo['values'] = ['None']
            self.basement_interface_combo['state'] = 'readonly'
            self.basement_interface_var.set('None')
            self.basement_interface_idx = None
    
    def _on_basement_interface_selected(self, event=None):
        """处理沉积基底界面选择"""
        selected = self.basement_interface_var.get()
        if selected == 'None':
            self.basement_interface_idx = None
            self.log_result("Basement interface: None (profiles will start from 0 depth)")
        elif selected.startswith('Interface '):
            idx_str = selected.replace('Interface ', '')
            try:
                self.basement_interface_idx = int(idx_str) - 1  # 转换为0-based索引
                self.log_result(f"Basement interface set to: {selected} (index {self.basement_interface_idx})")
            except ValueError:
                self.basement_interface_idx = None
        
        # 重新绘制模型以高亮选中的界面
        if self.grid_data is not None:
            self.plot_model()
        if not self._restoring_state:
            self._save_workbench_state()
    
    def _load_interface_file(self):
        """加载界面文件（用于网格文件，支持多界面）"""
        filename = filedialog.askopenfilename(
            title='Load Interface File',
            filetypes=[
                ('Text files', '*.txt *.dat'),
                ('CSV files', '*.csv'),
                ('All files', '*.*')
            ]
        )
        if filename:
            try:
                # 初始化loaded_interfaces列表（如果不存在）
                if not hasattr(self, 'loaded_interfaces'):
                    self.loaded_interfaces = []
                
                # 读取界面文件
                # 支持两种格式：
                # 1. 简单格式：两列数据（x, z）
                # 2. 多界面格式：包含注释行标识不同界面
                interfaces_in_file = []
                
                with open(filename, 'r') as f:
                    lines = f.readlines()
                
                current_interface = {'x': [], 'z': []}
                current_name = None
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        # 如果是注释行，检查是否包含界面名称
                        if line.startswith('#') and ('Interface' in line or 'Boundary' in line):
                            # 保存之前的界面（如果有数据）
                            if len(current_interface['x']) > 0:
                                interface_name = current_name or f'Interface {len(interfaces_in_file) + 1}'
                                interfaces_in_file.append({
                                    'x': np.array(current_interface['x']),
                                    'z': np.array(current_interface['z']),
                                    'name': interface_name
                                })
                            # 开始新界面
                            current_interface = {'x': [], 'z': []}
                            # 从注释中提取名称
                            if 'Interface' in line:
                                parts = line.split('Interface')
                                if len(parts) > 1:
                                    current_name = parts[-1].strip()
                                else:
                                    current_name = line.replace('#', '').strip()
                            elif 'Boundary' in line:
                                current_name = line.replace('#', '').strip()
                            else:
                                current_name = None
                        continue
                    
                    # 尝试解析数据行
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            x = float(parts[0])
                            z = float(parts[1])
                            current_interface['x'].append(x)
                            current_interface['z'].append(z)
                    except ValueError:
                        continue
                
                # 保存最后一个界面
                if len(current_interface['x']) > 0:
                    interface_name = current_name or f'Interface {len(interfaces_in_file) + 1}'
                    interfaces_in_file.append({
                        'x': np.array(current_interface['x']),
                        'z': np.array(current_interface['z']),
                        'name': interface_name
                    })
                
                # 如果没有找到多界面格式，尝试简单格式
                if len(interfaces_in_file) == 0:
                    data = np.loadtxt(filename, delimiter=None)
                    if data.ndim == 2 and data.shape[1] >= 2:
                        x_coords = data[:, 0]
                        z_coords = data[:, 1]
                        interfaces_in_file.append({
                            'x': x_coords,
                            'z': z_coords,
                            'name': os.path.basename(filename)
                        })
                    else:
                        messagebox.showerror('Error', 'Invalid interface file format. Expected 2 columns (x, z).')
                        return
                
                # 添加到加载的界面列表
                for interface in interfaces_in_file:
                    interface['file'] = filename
                    self.loaded_interfaces.append(interface)
                
                # 更新界面文件标签（显示加载的界面数量）
                num_interfaces = len(self.loaded_interfaces)
                if num_interfaces == 1:
                    self.interface_file_label.config(text=os.path.basename(filename), 
                                                    foreground='black')
                else:
                    self.interface_file_label.config(text=f'{num_interfaces} interfaces', 
                                                    foreground='black')
                
                # 向后兼容：如果是第一个界面，也设置basement_interface_data
                if len(interfaces_in_file) > 0:
                    first_interface = interfaces_in_file[0]
                    self.basement_interface_data = {'x': first_interface['x'], 'z': first_interface['z']}
                    self.basement_interface_file = filename
                
                self.log_result(f"Interface file loaded: {filename}")
                self.log_result(f"  Number of interfaces: {len(interfaces_in_file)}")
                for i, interface in enumerate(interfaces_in_file):
                    self.log_result(f"    Interface {i+1}: {interface['name']} ({len(interface['x'])} points)")
                
                # 重新绘制模型（显示界面）
                self.plot_model()
                
            except Exception as e:
                messagebox.showerror('Error', f'Failed to load interface file: {str(e)}')
    
    def _clear_loaded_interfaces(self):
        """清除所有加载的界面"""
        if hasattr(self, 'loaded_interfaces'):
            num_interfaces = len(self.loaded_interfaces)
            self.loaded_interfaces = []
            self.basement_interface_data = None
            self.basement_interface_file = None
            self.interface_file_label.config(text='None', foreground='gray')
            self.log_result(f"Cleared {num_interfaces} loaded interface(s)")
            # 重新绘制模型
            self.plot_model()
    
    def _save_interface_file(self):
        """保存界面文件（支持从v.in文件或速度等值线提取）"""
        if self.zelt_model:
            # 如果有v.in文件，可以选择保存所有界面或选中的界面
            self._save_vin_interfaces()
        else:
            # 如果是网格文件，可以保存当前加载的界面或从速度等值线提取
            self._save_grid_interfaces()
    
    def _save_vin_interfaces(self):
        """保存v.in格式的界面"""
        if not self.zelt_model:
            messagebox.showwarning('Warning', 'No v.in model loaded')
            return
        
        # 创建对话框让用户选择要保存的界面
        win = tk.Toplevel(self.root)
        win.title('Save Interfaces')
        win.geometry('400x300+200+200')
        
        ttk.Label(win, text='Select interfaces to save:', font=('TkDefaultFont', 10, 'bold')).pack(pady=10)
        
        # 创建复选框列表
        interface_vars = {}
        interface_frame = ttk.Frame(win)
        interface_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 添加滚动条
        canvas = tk.Canvas(interface_frame)
        scrollbar = ttk.Scrollbar(interface_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 为每个界面创建复选框
        for i in range(len(self.zelt_model.depth_nodes)):
            var = tk.BooleanVar(value=True)
            interface_vars[i] = var
            
            # 确定界面名称
            if i == len(self.zelt_model.depth_nodes) - 1:
                label_text = f'Bottom Boundary (Interface {i+1})'
            elif self.basement_interface_idx is not None and i == self.basement_interface_idx:
                label_text = f'Basement Interface {i+1} (Selected)'
            elif self.seafloor_interface_idx is not None and i == self.seafloor_interface_idx:
                label_text = f'Seafloor Interface {i+1} (Selected)'
            else:
                label_text = f'Layer {i+1} Interface'
            
            ttk.Checkbutton(scrollable_frame, text=label_text, variable=var).pack(anchor='w', pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        def _save_selected():
            """保存选中的界面"""
            selected_indices = [i for i, var in interface_vars.items() if var.get()]
            if not selected_indices:
                messagebox.showwarning('Warning', 'Please select at least one interface')
                return
            
            filename = filedialog.asksaveasfilename(
                title='Save Interfaces',
                defaultextension='.txt',
                filetypes=[
                    ('Text files', '*.txt'),
                    ('All files', '*.*')
                ]
            )
            
            if filename:
                try:
                    with open(filename, 'w') as f:
                        # 写入文件头
                        f.write(f"# Interface file saved from v.in model\n")
                        f.write(f"# Number of interfaces: {len(selected_indices)}\n")
                        f.write(f"# Format: x(km) z(km)\n")
                        f.write(f"# Each interface is separated by a blank line\n\n")
                        
                        # 写入每个选中的界面
                        for idx in selected_indices:
                            x_coords, z_coords = self.zelt_model.get_layer_geometry(idx)
                            
                            # 过滤NaN值
                            valid_pairs = [(x, z) for x, z in zip(x_coords, z_coords) 
                                         if not (np.isnan(x) or np.isnan(z))]
                            
                            if valid_pairs:
                                # 写入界面标识
                                if idx == len(self.zelt_model.depth_nodes) - 1:
                                    f.write(f"# Bottom Boundary\n")
                                else:
                                    f.write(f"# Interface {idx+1}\n")
                                
                                # 写入坐标对
                                for x, z in valid_pairs:
                                    f.write(f"{x:.6f}\t{z:.6f}\n")
                                f.write("\n")  # 界面之间用空行分隔
                    
                    messagebox.showinfo('Success', f'Interfaces saved to: {filename}')
                    self.log_result(f"Interfaces saved: {filename} ({len(selected_indices)} interfaces)")
                    win.destroy()
                except Exception as e:
                    messagebox.showerror('Error', f'Failed to save interfaces: {str(e)}')
        
        button_frame = ttk.Frame(win)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text='Save Selected', command=_save_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text='Cancel', command=win.destroy).pack(side=tk.LEFT, padx=5)
    
    def _save_grid_interfaces(self):
        """保存网格文件的界面（当前加载的界面或从速度等值线提取）"""
        if self.grid_data is None:
            messagebox.showwarning('Warning', 'Please load model first')
            return
        
        # 创建对话框
        win = tk.Toplevel(self.root)
        win.title('Save Interface from Grid Model')
        win.geometry('400x250+200+200')
        
        ttk.Label(win, text='Select interface source:', font=('TkDefaultFont', 10, 'bold')).pack(pady=10)
        
        source_var = tk.StringVar(value='loaded')
        
        # 选项1：保存当前加载的界面
        ttk.Radiobutton(win, text='Save currently loaded interface', 
                       variable=source_var, value='loaded').pack(anchor='w', padx=20, pady=5)
        
        # 选项2：从速度等值线提取
        ttk.Radiobutton(win, text='Extract from velocity contour', 
                       variable=source_var, value='contour').pack(anchor='w', padx=20, pady=5)
        
        # 速度等值线值输入（如果选择从等值线提取）
        contour_frame = ttk.Frame(win)
        contour_frame.pack(pady=10)
        ttk.Label(contour_frame, text='Velocity value (km/s):').pack(side=tk.LEFT, padx=5)
        contour_value_var = tk.StringVar(value='3.0')
        contour_entry = ttk.Entry(contour_frame, textvariable=contour_value_var, width=10)
        contour_entry.pack(side=tk.LEFT, padx=5)
        
        def _save_interface():
            """保存界面"""
            source = source_var.get()
            
            filename = filedialog.asksaveasfilename(
                title='Save Interface',
                defaultextension='.txt',
                filetypes=[
                    ('Text files', '*.txt'),
                    ('All files', '*.*')
                ]
            )
            
            if not filename:
                return
            
            try:
                if source == 'loaded':
                    # 保存当前加载的界面
                    if self.basement_interface_data is None:
                        messagebox.showwarning('Warning', 'No interface loaded')
                        win.destroy()
                        return
                    
                    x_coords = self.basement_interface_data['x']
                    z_coords = self.basement_interface_data['z']
                    
                else:  # source == 'contour'
                    # 从速度等值线提取界面
                    try:
                        contour_value = float(contour_value_var.get())
                    except ValueError:
                        messagebox.showerror('Error', 'Invalid velocity value')
                        return
                    
                    # 获取速度数据
                    z_coord = self.profile_extractor.z_coord
                    x_coord = self.profile_extractor.x_coord
                    z_vals = np.asarray(self.grid_data[z_coord].values, dtype=float)
                    x_vals = np.asarray(self.grid_data[x_coord].values, dtype=float)
                    
                    if not hasattr(self, 'velocity_var') or self.velocity_var is None:
                        self._detect_velocity_var()
                    vel_var = self.velocity_var
                    velocity_data = np.asarray(self.grid_data[vel_var].values, dtype=float)
                    
                    # 提取等值线：对每个X位置，找到速度等于contour_value的Z位置
                    x_coords = []
                    z_coords = []
                    
                    for j, x in enumerate(x_vals):
                        # 获取该X位置的速度列
                        vp_col = velocity_data[:, j]
                        
                        # 找到速度等于或最接近contour_value的位置
                        # 使用插值方法找到精确的Z位置
                        diff = np.abs(vp_col - contour_value)
                        min_idx = np.argmin(diff)
                        
                        if diff[min_idx] < 0.1:  # 如果找到接近的值（容差0.1 km/s）
                            # 在min_idx附近进行线性插值
                            if min_idx > 0 and min_idx < len(vp_col) - 1:
                                # 线性插值
                                v1, v2 = vp_col[min_idx], vp_col[min_idx + 1]
                                z1, z2 = z_vals[min_idx], z_vals[min_idx + 1]
                                if v2 != v1:
                                    z_interp = z1 + (z2 - z1) * (contour_value - v1) / (v2 - v1)
                                else:
                                    z_interp = z1
                                x_coords.append(x)
                                z_coords.append(z_interp)
                            else:
                                x_coords.append(x)
                                z_coords.append(z_vals[min_idx])
                    
                    if len(x_coords) == 0:
                        messagebox.showwarning('Warning', f'No contour found for velocity {contour_value} km/s')
                        win.destroy()
                        return
                    
                    # 转换为numpy数组并排序
                    x_coords = np.array(x_coords)
                    z_coords = np.array(z_coords)
                    sorted_indices = np.argsort(x_coords)
                    x_coords = x_coords[sorted_indices]
                    z_coords = z_coords[sorted_indices]
                
                # 保存到文件
                with open(filename, 'w') as f:
                    f.write(f"# Interface file\n")
                    f.write(f"# Format: x(km) z(km)\n")
                    if source == 'contour':
                        f.write(f"# Extracted from velocity contour: {contour_value_var.get()} km/s\n")
                    f.write(f"# Number of points: {len(x_coords)}\n\n")
                    
                    for x, z in zip(x_coords, z_coords):
                        f.write(f"{x:.6f}\t{z:.6f}\n")
                
                messagebox.showinfo('Success', f'Interface saved to: {filename}')
                self.log_result(f"Interface saved: {filename}")
                win.destroy()
                
            except Exception as e:
                messagebox.showerror('Error', f'Failed to save interface: {str(e)}')
        
        button_frame = ttk.Frame(win)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text='Save', command=_save_interface).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text='Cancel', command=win.destroy).pack(side=tk.LEFT, padx=5)
    
    def _get_basement_depth(self, x: float) -> float:
        """获取指定X位置的沉积基底深度
        
        Args:
            x: X坐标
            
        Returns:
            沉积基底深度（如果未设置，返回0）
        """
        if self.zelt_model and self.basement_interface_idx is not None:
            # v.in文件：从界面获取深度
            try:
                x_coords, z_coords = self.zelt_model.get_layer_geometry(self.basement_interface_idx)
                # 找到最接近的X位置
                x_array = np.array(x_coords)
                z_array = np.array(z_coords)
                # 移除NaN值
                valid_mask = ~(np.isnan(x_array) | np.isnan(z_array))
                if np.any(valid_mask):
                    x_valid = x_array[valid_mask]
                    z_valid = z_array[valid_mask]
                    # 插值获取深度
                    if len(x_valid) > 1:
                        z_depth = np.interp(x, x_valid, z_valid)
                        return float(z_depth)
                    elif len(x_valid) == 1:
                        return float(z_valid[0])
            except Exception as e:
                self.log_result(f"Warning: Failed to get basement depth at x={x:.2f}: {str(e)}")
        elif self.basement_interface_data is not None:
            # 网格文件：从界面文件获取深度
            try:
                x_coords = self.basement_interface_data['x']
                z_coords = self.basement_interface_data['z']
                # 插值获取深度
                if len(x_coords) > 1:
                    z_depth = np.interp(x, x_coords, z_coords)
                    return float(z_depth)
                elif len(x_coords) == 1:
                    return float(z_coords[0])
            except Exception as e:
                self.log_result(f"Warning: Failed to get basement depth from interface file at x={x:.2f}: {str(e)}")
        
        # 默认返回0（从模型顶部开始）
        return 0.0
    
    def _update_seafloor_interface_options(self):
        """更新海底面界面选择选项"""
        # 检查是否有seafloor_interface_combo（可能在初始化时还未创建）
        if not hasattr(self, 'seafloor_interface_combo'):
            return
        
        if self.zelt_model and len(self.zelt_model.depth_nodes) > 0:
            # v.in文件：列出所有界面，加上Auto选项
            options = ['Auto'] + [f'Interface {i+1}' for i in range(len(self.zelt_model.depth_nodes) - 1)]
            self.seafloor_interface_combo['values'] = options
            self.seafloor_interface_combo['state'] = 'readonly'
            # 如果当前选择不在选项中，设置为Auto
            current_selection = self.seafloor_interface_var.get()
            if current_selection not in options:
                self.seafloor_interface_var.set('Auto')
                self.seafloor_interface_idx = None
            self.log_result(f"Seafloor interface options updated: {len(options)-1} interfaces available (plus Auto)")
        else:
            # 网格文件：只有Auto选项
            self.seafloor_interface_combo['values'] = ['Auto']
            self.seafloor_interface_combo['state'] = 'readonly'
            self.seafloor_interface_var.set('Auto')
            self.seafloor_interface_idx = None
    
    def _on_seafloor_interface_selected(self, event=None):
        """处理海底面界面选择"""
        selected = self.seafloor_interface_var.get()
        if selected == 'Auto':
            self.seafloor_interface_idx = None
            self.log_result("Seafloor interface: Auto (will detect from velocity: vp=1.5, vs=0)")
        elif selected.startswith('Interface '):
            idx_str = selected.replace('Interface ', '')
            try:
                self.seafloor_interface_idx = int(idx_str) - 1  # 转换为0-based索引
                self.log_result(f"Seafloor interface set to: {selected} (index {self.seafloor_interface_idx})")
            except ValueError:
                self.seafloor_interface_idx = None
        
        # 清除海底面深度映射，下次访问时会重新计算
        self.seafloor_depth_map = None
        
        # 重新绘制模型以高亮选中的界面
        if self.grid_data is not None:
            self.plot_model()
        if not self._restoring_state:
            self._save_workbench_state()
    def _detect_seafloor_from_velocity(self, x_vals, z_vals, velocity_data):
        """从速度数据自动检测海底面深度
        
        Args:
            x_vals: X坐标数组
            z_vals: Z坐标数组
            velocity_data: 速度数据 (nz, nx)
            
        Returns:
            seafloor_depths: 每个X位置对应的海底面深度数组
        """
        vp_water_value = 1.5  # km/s，海水P波速度值
        vp_water_tolerance = 0.1  # km/s，允许的误差范围（1.4-1.6都认为是海水）
        
        nz, nx = velocity_data.shape
        seafloor_depths = np.zeros(nx, dtype=float)
        
        # 对每个X位置，从浅到深查找海水层和岩石层的分界
        for j in range(nx):
            vp_col = velocity_data[:, j]
            
            # 查找海水层：速度在1.4-1.6之间（严格约束）
            water_mask = np.abs(vp_col - vp_water_value) <= vp_water_tolerance
            
            # 查找第一个非海水点（岩石开始）
            # 岩石：速度不在海水范围内（即 |vp - 1.5| > 0.1）
            rock_mask = ~water_mask
            
            if np.any(rock_mask):
                # 找到第一个岩石点的深度
                first_rock_idx = np.where(rock_mask)[0][0]
                # 海底面深度应该是最后一个海水点的深度，即第一个岩石点之前的深度
                if first_rock_idx > 0:
                    # 使用最后一个海水点的深度
                    seafloor_depths[j] = float(z_vals[first_rock_idx - 1])
                else:
                    # 如果第一个点就是岩石，说明没有海水层，使用最浅深度
                    seafloor_depths[j] = float(z_vals[0])
            else:
                # 如果没有找到岩石，说明全部是海水，使用最浅深度
                seafloor_depths[j] = float(z_vals[0])
        
        return seafloor_depths
    
    def _get_seafloor_depth(self, x: float) -> float:
        """获取指定X位置的海底面深度
        
        Args:
            x: X坐标
            
        Returns:
            海底面深度（如果未设置，自动检测或返回0）
        """
        if self.zelt_model and self.seafloor_interface_idx is not None:
            # v.in文件：从界面获取深度
            try:
                x_coords, z_coords = self.zelt_model.get_layer_geometry(self.seafloor_interface_idx)
                x_array = np.array(x_coords)
                z_array = np.array(z_coords)
                valid_mask = ~(np.isnan(x_array) | np.isnan(z_array))
                if np.any(valid_mask):
                    x_valid = x_array[valid_mask]
                    z_valid = z_array[valid_mask]
                    # 插值获取深度
                    seafloor_z = np.interp(x, x_valid, z_valid)
                    return float(seafloor_z)
            except Exception as e:
                self.log_result(f"Warning: Failed to get seafloor depth from interface: {str(e)}")
        
        # 自动检测或使用默认值
        if self.grid_data is None:
            return 0.0
        
        # 如果已有深度映射，直接使用
        if self.seafloor_depth_map is not None:
            x_vals = np.asarray(self.grid_data.coords[self.profile_extractor.x_coord].values)
            if len(self.seafloor_depth_map) == len(x_vals):
                idx = np.argmin(np.abs(x_vals - x))
                return float(self.seafloor_depth_map[idx])
        
        # 否则自动检测
        return 0.0  # 临时返回0，实际会在_get_seafloor_depths中检测
    
    def _get_seafloor_depths(self, x_vals):
        """获取所有X位置的海底面深度数组
        
        Args:
            x_vals: X坐标数组
            
        Returns:
            seafloor_depths: 每个X位置对应的海底面深度数组
        """
        # 如果已有深度映射，直接返回
        if self.seafloor_depth_map is not None:
            if len(self.seafloor_depth_map) == len(x_vals):
                return self.seafloor_depth_map
        
        # 检查是否指定了界面
        if self.zelt_model and self.seafloor_interface_idx is not None:
            # v.in文件：从界面获取深度
            try:
                x_coords, z_coords = self.zelt_model.get_layer_geometry(self.seafloor_interface_idx)
                x_array = np.array(x_coords)
                z_array = np.array(z_coords)
                valid_mask = ~(np.isnan(x_array) | np.isnan(z_array))
                if np.any(valid_mask):
                    x_valid = x_array[valid_mask]
                    z_valid = z_array[valid_mask]
                    # 插值获取所有X位置的深度
                    seafloor_depths = np.interp(x_vals, x_valid, z_valid)
                    self.seafloor_depth_map = seafloor_depths
                    return seafloor_depths
            except Exception as e:
                self.log_result(f"Warning: Failed to get seafloor depths from interface: {str(e)}")
        
        # 自动检测：从速度数据检测海水层（适用于v.in和grd文件）
        if self.grid_data is not None:
            try:
                z_coord = self.profile_extractor.z_coord
                z_vals = np.asarray(self.grid_data[z_coord].values, dtype=float)
                if not hasattr(self, 'velocity_var') or self.velocity_var is None:
                    self._detect_velocity_var()
                vel_var = self.velocity_var
                velocity_data = np.asarray(self.grid_data[vel_var].values, dtype=float)
                
                seafloor_depths = self._detect_seafloor_from_velocity(x_vals, z_vals, velocity_data)
                self.seafloor_depth_map = seafloor_depths
                
                # 检查是否检测到有效的海底面（不是全部为0）
                if np.any(seafloor_depths > 0):
                    self.log_result(f"Seafloor depths auto-detected from velocity (vp = 1.5 ± 0.1 km/s)")
                    self.log_result(f"  Seafloor depth range: {np.nanmin(seafloor_depths):.2f} - {np.nanmax(seafloor_depths):.2f} km")
                else:
                    self.log_result(f"Warning: No seafloor detected (all depths are 0). Using z=0 as seafloor.")
                
                return seafloor_depths
            except Exception as e:
                self.log_result(f"Warning: Failed to auto-detect seafloor: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 默认：返回0（假设z=0就是海底）
        seafloor_depths = np.zeros_like(x_vals, dtype=float)
        self.seafloor_depth_map = seafloor_depths
        self.log_result(f"Warning: Using default seafloor depth (z=0) for all X positions")
        return seafloor_depths

    def _workbench_interface_files(self) -> list[str]:
        """会话 JSON `imodel_gui.interface_files`（与 Qt workbench 对齐）。"""
        out: list[str] = []
        seen: set[str] = set()
        if not hasattr(self, "loaded_interfaces"):
            return out
        for iface in self.loaded_interfaces or []:
            fp = str(iface.get("file") or "").strip()
            if not fp:
                continue
            try:
                key = str(Path(fp).resolve())
            except Exception:
                key = fp
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out
