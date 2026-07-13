"""剖面抽取、1D 范围、剖面图窗口。"""

from __future__ import annotations

from .deps import *  # noqa: F403


class ProfileMixin:
    def extract_vertical_profile(self):
        """Extract vertical profile for a range of X coordinates (1D model)"""
        if self.grid_data is None or self.profile_extractor is None:
            messagebox.showwarning('Warning', 'Please load model first')
            return
        
        try:
            # 获取X范围
            x_min_str = self.x_min_entry.get().strip()
            x_max_str = self.x_max_entry.get().strip()
            
            if not x_min_str or not x_max_str:
                messagebox.showerror('Error', 'Please enter both X_min and X_max')
                return
            
            x_min = float(x_min_str)
            x_max = float(x_max_str)
            
            if x_min >= x_max:
                messagebox.showerror('Error', 'X_min must be less than X_max')
                return
            
            # 获取采样间隔
            sampling_interval = float(self.sampling_interval_var.get())
            
            # 生成X采样点
            x_samples = np.arange(x_min, x_max + sampling_interval/2, sampling_interval)
            
            # 获取模型的Z坐标范围
            z_coords = self.grid_data[self.profile_extractor.z_coord].values
            z_min = float(z_coords.min())
            z_max = float(z_coords.max())
            
            # 对每个X位置提取垂直剖面，然后平均
            # 优化：使用向量化操作一次性提取所有数据
            self.log_result(f"Extracting profiles for {len(x_samples)} X positions...")
            all_profiles = []
            
            # 获取Z坐标（所有剖面共享）
            z_coords = self.grid_data[self.profile_extractor.z_coord].values
            velocity_var = self.profile_extractor.velocity_var
            x_coord = self.profile_extractor.x_coord
            z_coord = self.profile_extractor.z_coord
            
            # 向量化提取：一次性提取所有X位置的所有Z坐标的速度值
            try:
                # 使用xarray的向量化选择，比循环快得多
                # 创建一个包含所有(x, z)组合的DataArray
                velocity_data = self.grid_data[velocity_var]
                
                # 对每个X位置，提取所有Z坐标的速度（向量化）
                for i, x in enumerate(x_samples):
                    try:
                        # 使用sel一次性选择所有z坐标，比循环快得多
                        v_slice = velocity_data.sel({x_coord: x}, method='nearest')
                        # 如果v_slice是1D数组（z坐标），直接提取值
                        if hasattr(v_slice, 'values'):
                            vp_values = v_slice.values
                            # 确保vp_values是1D数组
                            if vp_values.ndim > 1:
                                # 如果有多维，取第一个维度
                                vp_values = vp_values.flatten()[:len(z_coords)]
                            else:
                                vp_values = vp_values[:len(z_coords)]
                            
                            # 确保长度匹配
                            if len(vp_values) != len(z_coords):
                                # 如果长度不匹配，使用插值或截断
                                if len(vp_values) > len(z_coords):
                                    vp_values = vp_values[:len(z_coords)]
                                else:
                                    # 如果v_slice是标量，扩展到所有z坐标
                                    if np.isscalar(vp_values):
                                        vp_values = np.full(len(z_coords), float(vp_values))
                                    else:
                                        # 填充NaN
                                        vp_values = np.pad(vp_values, (0, len(z_coords) - len(vp_values)), 
                                                          constant_values=np.nan)
                            
                            # 转换为float数组
                            vp_values = np.asarray(vp_values, dtype=float)
                            
                            profile = pd.DataFrame({
                                'depth': z_coords,
                                'vp': vp_values
                            })
                            
                            # 获取当前X位置的沉积基底深度并调整深度
                            basement_depth = self._get_basement_depth(x)
                            if basement_depth > 0:
                                profile_depths = profile['depth'].values
                                adjusted_depths = profile_depths - basement_depth
                                valid_mask = adjusted_depths >= 0
                                if np.any(valid_mask):
                                    profile = pd.DataFrame({
                                        'depth': adjusted_depths[valid_mask],
                                        'vp': profile['vp'].values[valid_mask]
                                    })
                            
                            all_profiles.append(profile)
                        else:
                            # 回退到原始方法
                            profile = self.profile_extractor.extract_vertical_profile(x)
                            
                            # 获取当前X位置的沉积基底深度并调整
                            basement_depth = self._get_basement_depth(x)
                            if basement_depth > 0:
                                profile_depths = profile['depth'].values
                                adjusted_depths = profile_depths - basement_depth
                                valid_mask = adjusted_depths >= 0
                                if np.any(valid_mask):
                                    profile = pd.DataFrame({
                                        'depth': adjusted_depths[valid_mask],
                                        'vp': profile['vp'].values[valid_mask]
                                    })
                            
                            all_profiles.append(profile)
                    except Exception as e:
                        # 如果向量化失败，回退到原始方法
                        try:
                            profile = self.profile_extractor.extract_vertical_profile(x)
                            all_profiles.append(profile)
                        except Exception as e2:
                            self.log_result(f"Warning: Failed to extract profile at x={x:.2f}: {str(e2)}")
            except Exception as e:
                # 如果向量化完全失败，回退到原始循环方法
                self.log_result(f"Warning: Vectorized extraction failed, using fallback method: {str(e)}")
                for x in x_samples:
                    try:
                        profile = self.profile_extractor.extract_vertical_profile(x)
                        
                        # 获取当前X位置的沉积基底深度并调整
                        basement_depth = self._get_basement_depth(x)
                        if basement_depth > 0:
                            profile_depths = profile['depth'].values
                            adjusted_depths = profile_depths - basement_depth
                            valid_mask = adjusted_depths >= 0
                            if np.any(valid_mask):
                                profile = pd.DataFrame({
                                    'depth': adjusted_depths[valid_mask],
                                    'vp': profile['vp'].values[valid_mask]
                                })
                        
                        all_profiles.append(profile)
                    except Exception as e2:
                        self.log_result(f"Warning: Failed to extract profile at x={x:.2f}: {str(e2)}")
            
            # 安全地检查 all_profiles（避免数组布尔比较错误）
            has_profiles = False
            if all_profiles is not None:
                if isinstance(all_profiles, (list, tuple)):
                    has_profiles = len(all_profiles) > 0
                elif isinstance(all_profiles, np.ndarray):
                    has_profiles = all_profiles.size > 0
                elif hasattr(all_profiles, '__len__'):
                    try:
                        has_profiles = len(all_profiles) > 0
                    except (TypeError, ValueError):
                        has_profiles = False
            
            if not has_profiles:
                messagebox.showerror('Error', 'Failed to extract any profiles')
                return
            
            # 对齐所有剖面的深度坐标
            # 由于每个剖面都已经经过界面修正（深度从基底算起），
            # 我们需要找到所有剖面的深度范围，然后创建一个统一的深度网格
            all_depths = []
            for profile in all_profiles:
                profile_depths = profile['depth'].values
                if isinstance(profile_depths, pd.Series):
                    profile_depths = profile_depths.values
                profile_depths = np.asarray(profile_depths, dtype=float)
                # 只保留有效值
                valid_depths = profile_depths[np.isfinite(profile_depths)]
                if len(valid_depths) > 0:
                    all_depths.extend(valid_depths.tolist())
            
            if len(all_depths) == 0:
                messagebox.showerror('Error', 'No valid depth values found in profiles')
                return
            
            # 创建统一的深度网格（使用所有剖面的深度范围）
            all_depths_array = np.array(all_depths)
            depth_min = float(np.nanmin(all_depths_array))
            depth_max = float(np.nanmax(all_depths_array))
            
            # 使用第一个剖面的深度间隔作为参考
            first_profile_depths = all_profiles[0]['depth'].values
            if isinstance(first_profile_depths, pd.Series):
                first_profile_depths = first_profile_depths.values
            first_profile_depths = np.asarray(first_profile_depths, dtype=float)
            valid_first_depths = first_profile_depths[np.isfinite(first_profile_depths)]
            
            if len(valid_first_depths) > 1:
                # 使用第一个剖面的深度间隔
                depth_interval = np.mean(np.diff(np.sort(valid_first_depths)))
            else:
                # 如果只有一个点，使用默认间隔
                depth_interval = 0.1
            
            # 创建统一的深度网格
            reference_depths = np.arange(depth_min, depth_max + depth_interval/2, depth_interval)
            
            # 优化：使用向量化操作计算平均值，而不是嵌套循环
            # 将所有剖面的速度值对齐到相同的深度网格
            self.log_result(f"Averaging {len(all_profiles)} profiles...")
            
            # 由于每个剖面经过界面修正后，深度值可能不同，长度也可能不同
            # 因此必须使用插值对齐方法，不能直接向量化平均
            # 使用scipy插值来对齐所有剖面到相同的深度网格
            from scipy import interpolate
            
            vp_interpolated_list = []
            
            for profile in all_profiles:
                try:
                    profile_depths = profile['depth'].values
                    profile_vp = profile['vp'].values
                    if isinstance(profile_depths, pd.Series):
                        profile_depths = profile_depths.values
                    if isinstance(profile_vp, pd.Series):
                        profile_vp = profile_vp.values
                    
                    profile_depths = np.asarray(profile_depths, dtype=float)
                    profile_vp = np.asarray(profile_vp, dtype=float)
                    
                    # 移除NaN值
                    valid_mask = np.isfinite(profile_vp) & np.isfinite(profile_depths)
                    if np.sum(valid_mask) < 2:
                        # 如果有效点太少，跳过这个剖面
                        self.log_result(f"Warning: Profile has too few valid points, skipping")
                        continue
                    
                    profile_depths_valid = profile_depths[valid_mask]
                    profile_vp_valid = profile_vp[valid_mask]
                    
                    # 确保深度值是单调递增的（插值要求）
                    if len(profile_depths_valid) > 1:
                        # 排序
                        sort_idx = np.argsort(profile_depths_valid)
                        profile_depths_valid = profile_depths_valid[sort_idx]
                        profile_vp_valid = profile_vp_valid[sort_idx]
                        
                        # 使用线性插值
                        f_interp = interpolate.interp1d(
                            profile_depths_valid, profile_vp_valid,
                            kind='linear', bounds_error=False, fill_value=np.nan
                        )
                        vp_interp = f_interp(reference_depths)
                        vp_interpolated_list.append(vp_interp)
                    else:
                        # 如果只有一个点，无法插值
                        self.log_result(f"Warning: Profile has only one valid point, skipping")
                        continue
                except Exception as e:
                    # 如果插值失败，跳过这个剖面
                    self.log_result(f"Warning: Interpolation failed for a profile: {str(e)}")
                    continue
            
            # 计算平均值
            if len(vp_interpolated_list) > 0:
                vp_matrix = np.array(vp_interpolated_list)
                # 只对有效值（非NaN）计算平均
                averaged_vp = np.nanmean(vp_matrix, axis=0)
                # 转换为列表
                averaged_vp = averaged_vp.tolist()
                self.log_result(f"Successfully interpolated {len(vp_interpolated_list)} profiles to unified depth grid")
            else:
                # 如果所有插值都失败，使用NaN填充
                messagebox.showerror('Error', 'Failed to interpolate any profiles. Please check the data.')
                return
            
            # 注意：所有剖面在提取时已经经过了界面修正（深度从基底算起）
            # reference_depths 已经是修正后的深度，不需要再次调整
            # 只保留有效值（非NaN）的点
            valid_mask = ~np.isnan(averaged_vp)
            if np.any(valid_mask):
                final_depths = reference_depths[valid_mask]
                final_vp = [averaged_vp[i] for i in range(len(averaged_vp)) if valid_mask[i]]
            else:
                messagebox.showerror('Error', 'No valid averaged values found.')
                return
            
            # 检查是否有basement interface设置
            has_basement = False
            if self.zelt_model and self.basement_interface_idx is not None:
                has_basement = True
            elif self.basement_interface_data is not None:
                has_basement = True
            
            if has_basement:
                self.log_result(f"Profiles are already adjusted to start from basement (depth >= 0)")
            else:
                self.log_result("No basement interface set, profiles start from model top (depth=0)")
            
            # 创建一维模型DataFrame（平均后的结果）
            # 深度已经是修正后的（从基底算起），不需要再次调整
            one_d_model = pd.DataFrame({
                'depth': final_depths,
                'vp': final_vp
            })
            
            # 保存结果
            profile_id = f'vertical_1d_x{x_min:.2f}_to_{x_max:.2f}_sampling{sampling_interval:.1f}'
            self.profiles[profile_id] = one_d_model
            
            # 记录结果
            self.log_result(f"\n{'='*50}")
            self.log_result(f"1D Vertical Profile (X range: {x_min:.2f} - {x_max:.2f} km)")
            self.log_result(f"  Sampling interval: {sampling_interval:.1f} km")
            self.log_result(f"  Number of X samples: {len(x_samples)}")
            
            # 安全地获取min/max值（避免数组布尔比较错误）
            try:
                depth_min = float(one_d_model['depth'].min())
                depth_max = float(one_d_model['depth'].max())
                vp_min = float(one_d_model['vp'].min())
                vp_max = float(one_d_model['vp'].max())
                self.log_result(f"  Depth range: {depth_min:.2f} - {depth_max:.2f} km")
                self.log_result(f"  Velocity range: {vp_min:.2f} - {vp_max:.2f} km/s")
            except Exception as e:
                self.log_result(f"  Warning: Could not calculate ranges: {str(e)}")
            
            self.log_result(f"  Profile ID: {profile_id}")
            
            # 绘制剖面（传递所有剖面数据）
            title = f'1D Vertical Profile (X: {x_min:.2f}-{x_max:.2f} km, sampling: {sampling_interval:.1f} km)'
            self.plot_profile(one_d_model, title=title, all_profiles=all_profiles, x_samples=x_samples)
            
        except ValueError as e:
            messagebox.showerror('Error', f'Invalid input: {str(e)}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to extract profile: {str(e)}')
            import traceback
            traceback.print_exc()
            self.log_result(f"Error: {str(e)}")
    
    def plot_profile(self, profile: pd.DataFrame, title: str = 'Profile', 
                    horizontal: bool = False, all_profiles: Optional[List[pd.DataFrame]] = None,
                    x_samples: Optional[List[float]] = None):
        """绘制剖面图（在新窗口中，支持多条剖面线和控制面板）"""
        # 如果窗口已存在，先关闭
        if self.profile_window is not None and self.profile_window.winfo_exists():
            self.profile_window.destroy()
        
        # 创建新窗口（调整宽度，使其更窄）
        self.profile_window = tk.Toplevel(self.root)
        self.profile_window.title(title)
        self.profile_window.geometry('600x700+200+100')
        
        # 保存数据（安全地处理，避免数组布尔比较错误）
        if all_profiles is not None:
            if isinstance(all_profiles, (list, tuple)):
                self.profile_all_profiles = list(all_profiles) if len(all_profiles) > 0 else []
            elif isinstance(all_profiles, np.ndarray):
                self.profile_all_profiles = [all_profiles[i] for i in range(all_profiles.size)] if all_profiles.size > 0 else []
            else:
                try:
                    self.profile_all_profiles = list(all_profiles) if len(all_profiles) > 0 else []
                except (TypeError, ValueError):
                    self.profile_all_profiles = []
        else:
            self.profile_all_profiles = []
        
        self.profile_one_d_model = profile
        
        if x_samples is not None:
            if isinstance(x_samples, (list, tuple)):
                self.profile_x_samples = list(x_samples) if len(x_samples) > 0 else []
            elif isinstance(x_samples, np.ndarray):
                self.profile_x_samples = [float(x) for x in x_samples.flat] if x_samples.size > 0 else []
            else:
                try:
                    self.profile_x_samples = list(x_samples) if len(x_samples) > 0 else []
                except (TypeError, ValueError):
                    self.profile_x_samples = []
        else:
            self.profile_x_samples = []
        
        self.profile_title = title
        
        # 创建控制面板
        control_frame = ttk.Frame(self.profile_window)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # 左侧：按钮组
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text='Save Figure', 
                  command=self.save_profile_figure).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text='Export Data', 
                  command=self.export_profile_data).pack(side=tk.LEFT, padx=5)
        
        # 右侧：显示控制复选框（如果有多条剖面）
        # 安全地检查 all_profiles（避免数组布尔比较错误）
        has_multiple_profiles = False
        if all_profiles is not None:
            if isinstance(all_profiles, (list, tuple)):
                has_multiple_profiles = len(all_profiles) > 1
            elif isinstance(all_profiles, np.ndarray):
                has_multiple_profiles = all_profiles.size > 1
            elif hasattr(all_profiles, '__len__'):
                try:
                    has_multiple_profiles = len(all_profiles) > 1
                except (TypeError, ValueError):
                    has_multiple_profiles = False
        
        # 显示控制复选框区域
        checkbox_frame = ttk.LabelFrame(control_frame, text='显示选项', padding=5)
        checkbox_frame.pack(side=tk.RIGHT, padx=5)
        
        # 创建复选框变量（如果不存在）
        if not hasattr(self, 'profile_show_individual_var'):
            self.profile_show_individual_var = tk.BooleanVar(value=True)
        if not hasattr(self, 'profile_show_average_var'):
            self.profile_show_average_var = tk.BooleanVar(value=True)
        
        if has_multiple_profiles:
            ttk.Checkbutton(checkbox_frame, text='显示各X位置剖面', 
                           variable=self.profile_show_individual_var,
                           command=self._refresh_profile_plot).pack(side=tk.LEFT, padx=5)
            ttk.Checkbutton(checkbox_frame, text='显示平均剖面', 
                           variable=self.profile_show_average_var,
                           command=self._refresh_profile_plot).pack(side=tk.LEFT, padx=5)
        
        # 加载1D模型范围数据
        self._load_1d_ranges()
        
        # 1D模型范围选择区域
        if self.profile_1d_ranges:
            range_frame = ttk.LabelFrame(control_frame, text='背景范围', padding=5)
            range_frame.pack(side=tk.RIGHT, padx=5)
            
            # 为每个范围创建复选框
            for filename, data in self.profile_1d_ranges.items():
                # 创建友好的显示名称（去掉.txt后缀）
                display_name = filename.replace('.txt', '').replace('_', ' ').replace('-', ' ')
                if not hasattr(self, 'profile_1d_selected') or filename not in self.profile_1d_selected:
                    if not hasattr(self, 'profile_1d_selected'):
                        self.profile_1d_selected = {}
                    self.profile_1d_selected[filename] = tk.BooleanVar(value=False)
                
                ttk.Checkbutton(range_frame, text=display_name, 
                               variable=self.profile_1d_selected[filename],
                               command=self._refresh_profile_plot).pack(side=tk.TOP, anchor='w', padx=2, pady=1)
        
        # 创建matplotlib图形（调整宽度，使其更窄）
        self.profile_fig = plt.Figure(figsize=(4, 6))
        self.profile_ax = self.profile_fig.add_subplot(111)
        self.profile_fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)
        
        # 创建canvas
        self.profile_canvas = FigureCanvasTkAgg(self.profile_fig, master=self.profile_window)
        self.profile_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建工具栏
        toolbar_frame = ttk.Frame(self.profile_window)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        profile_toolbar = NavigationToolbar2Tk(self.profile_canvas, toolbar_frame)
        profile_toolbar.update()
        
        # 绘制剖面
        self._refresh_profile_plot()
    
    def _refresh_profile_plot(self):
        """刷新剖面图（根据复选框状态）"""
        if self.profile_ax is None or self.profile_one_d_model is None:
            return
        
        self.profile_ax.clear()
        
        # 先绘制选中的1D模型范围作为背景（在剖面线之前绘制）
        if hasattr(self, 'profile_1d_selected') and self.profile_1d_selected:
            for filename, var in self.profile_1d_selected.items():
                if var.get() and filename in self.profile_1d_ranges:
                    data = self.profile_1d_ranges[filename]
                    vp = data['vp']
                    depth = data['depth']
                    # 绘制多边形范围（使用填充区域）
                    display_name = filename.replace('.txt', '').replace('_', ' ').replace('-', ' ')
                    self.profile_ax.fill(vp, depth, alpha=0.15, label=display_name, 
                                       edgecolor='gray', linewidth=0.8, linestyle='--')
        
        # 获取显示选项
        # 安全地检查profile_all_profiles（避免数组布尔比较错误）
        has_profiles = False
        profile_list = []
        if self.profile_all_profiles is not None:
            try:
                if isinstance(self.profile_all_profiles, (list, tuple)):
                    has_profiles = len(self.profile_all_profiles) > 0
                    profile_list = list(self.profile_all_profiles) if has_profiles else []
                elif isinstance(self.profile_all_profiles, np.ndarray):
                    has_profiles = bool(self.profile_all_profiles.size > 0)
                    if has_profiles:
                        profile_list = [self.profile_all_profiles[i] for i in range(self.profile_all_profiles.size)]
                elif hasattr(self.profile_all_profiles, '__len__'):
                    # 尝试获取长度
                    try:
                        length = len(self.profile_all_profiles)
                        has_profiles = length > 0
                        if has_profiles:
                            profile_list = list(self.profile_all_profiles)
                    except (TypeError, ValueError):
                        has_profiles = False
                        profile_list = []
                else:
                    has_profiles = False
                    profile_list = []
            except Exception:
                has_profiles = False
                profile_list = []
        
        if has_profiles:
            show_individual = getattr(self, 'profile_show_individual_var', tk.BooleanVar(value=True)).get()
        else:
            show_individual = False
        show_average = getattr(self, 'profile_show_average_var', tk.BooleanVar(value=True)).get()
        
        # 绘制各X位置的剖面（如果启用）
        if show_individual and has_profiles and len(profile_list) > 0:
            # 确保x_samples也是列表（安全地处理，避免数组布尔比较错误）
            x_samples_list = []
            if self.profile_x_samples is not None:
                try:
                    if isinstance(self.profile_x_samples, np.ndarray):
                        if self.profile_x_samples.size > 0:
                            x_samples_list = [float(x) for x in self.profile_x_samples.flat]
                    elif isinstance(self.profile_x_samples, (list, tuple)):
                        if len(self.profile_x_samples) > 0:
                            x_samples_list = [float(x) for x in self.profile_x_samples]
                    elif hasattr(self.profile_x_samples, '__len__'):
                        try:
                            length = len(self.profile_x_samples)
                            if length > 0:
                                x_samples_list = [float(x) for x in self.profile_x_samples]
                        except (TypeError, ValueError):
                            x_samples_list = []
                except Exception:
                    x_samples_list = []
            
            # 确保 x_samples_list 长度与 profile_list 匹配
            if len(x_samples_list) != len(profile_list):
                # 如果长度不匹配，生成默认的X值
                x_samples_list = [float(i) for i in range(len(profile_list))]
            
            # 使用颜色映射来区分不同的剖面
            num_profiles = len(profile_list)
            colors = plt.cm.viridis(np.linspace(0, 1, num_profiles))
            for i, (profile, x) in enumerate(zip(profile_list, x_samples_list)):
                # 只在前几个和最后一个添加标签，避免图例过长
                if i < 3 or i == num_profiles - 1:
                    label = f'x={x:.2f} km'
                else:
                    label = ''
                self.profile_ax.plot(profile['vp'], profile['depth'], 
                                    color=colors[i], linewidth=1.2, alpha=0.6,
                                    label=label)
        
        # 绘制平均剖面（如果启用）
        if show_average:
            self.profile_ax.plot(self.profile_one_d_model['vp'], self.profile_one_d_model['depth'], 
                               'r-', linewidth=2.5, alpha=0.9, label='Average')
        
        # 根据是否选择了沉积基底界面，设置不同的y轴标签
        has_basement = False
        if self.zelt_model and self.basement_interface_idx is not None:
            has_basement = True
        elif self.basement_interface_data is not None:
            has_basement = True
        
        if has_basement:
            ylabel = 'Depth beneath basement (km)'
        else:
            ylabel = 'Depth (km)'
        
        self.profile_ax.set_ylabel(ylabel, fontsize=12)
        
        # 计算速度范围，设置紧凑的横坐标范围
        all_vp_values = []
        if show_individual and len(profile_list) > 0:
            for profile in profile_list:
                vp_vals = profile['vp'].values
                if isinstance(vp_vals, pd.Series):
                    vp_vals = vp_vals.values
                vp_vals = np.asarray(vp_vals, dtype=float)
                valid_vp = vp_vals[np.isfinite(vp_vals)]
                if len(valid_vp) > 0:
                    all_vp_values.extend(valid_vp.tolist())
        
        if show_average and self.profile_one_d_model is not None:
            avg_vp_vals = self.profile_one_d_model['vp'].values
            if isinstance(avg_vp_vals, pd.Series):
                avg_vp_vals = avg_vp_vals.values
            avg_vp_vals = np.asarray(avg_vp_vals, dtype=float)
            valid_avg_vp = avg_vp_vals[np.isfinite(avg_vp_vals)]
            if len(valid_avg_vp) > 0:
                all_vp_values.extend(valid_avg_vp.tolist())
        
        if len(all_vp_values) > 0:
            vp_min = float(np.nanmin(all_vp_values))
            vp_max = float(np.nanmax(all_vp_values))
            # 添加5%的边距，使图形更紧凑
            vp_range = vp_max - vp_min
            if vp_range > 0:
                margin = vp_range * 0.05
                self.profile_ax.set_xlim(vp_min - margin, vp_max + margin)
            else:
                # 如果范围太小，设置一个小的固定范围
                self.profile_ax.set_xlim(vp_min - 0.1, vp_max + 0.1)
        
        self.profile_ax.set_xlabel('Velocity (km/s)', fontsize=12)
        # y轴标签已在上面根据basement interface状态设置
        self.profile_ax.set_title(self.profile_title, fontsize=14)
        self.profile_ax.invert_yaxis()
        self.profile_ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加图例（如果有多条线或背景范围）
        handles, labels = self.profile_ax.get_legend_handles_labels()
        if labels:
            # 过滤掉空标签
            filtered_handles = [h for h, l in zip(handles, labels) if l]
            filtered_labels = [l for l in labels if l]
            
            if filtered_labels:
                # 检查是否有背景范围
                has_background = hasattr(self, 'profile_1d_selected') and self.profile_1d_selected and \
                               any(var.get() for var in self.profile_1d_selected.values())
                
                # 如果标签太多且没有背景范围，只显示部分
                if len(filtered_labels) > 8 and not has_background:
                    # 显示前3个、最后1个和Average
                    avg_idx = next((i for i, l in enumerate(filtered_labels) if 'Average' in l), None)
                    display_handles = filtered_handles[:3]
                    display_labels = filtered_labels[:3]
                    if len(filtered_handles) > 3:
                        display_handles.append(filtered_handles[-1])
                        display_labels.append(filtered_labels[-1])
                    if avg_idx is not None and avg_idx not in [0, 1, 2, len(filtered_handles)-1]:
                        display_handles.append(filtered_handles[avg_idx])
                        display_labels.append(filtered_labels[avg_idx])
                    self.profile_ax.legend(display_handles, display_labels, 
                                          loc='best', fontsize=8, ncol=1, framealpha=0.8)
                else:
                    # 如果有背景范围，使用较小的字体和两列布局
                    if has_background:
                        self.profile_ax.legend(filtered_handles, filtered_labels, 
                                              loc='best', fontsize=7, ncol=1, framealpha=0.8)
                    else:
                        self.profile_ax.legend(filtered_handles, filtered_labels, 
                                              loc='best', fontsize=8, framealpha=0.7)
        
        if self.profile_canvas:
            self.profile_canvas.draw()
    
    def _load_1d_ranges(self):
        """加载1d文件夹中的一维模型范围数据"""
        import os
        from pathlib import Path
        
        # 如果已经加载过，直接返回
        if hasattr(self, 'profile_1d_ranges') and self.profile_1d_ranges:
            return
        
        # 查找1d文件夹路径
        # 尝试多个可能的位置
        _here = Path(__file__).resolve()
        possible_paths = [
            _here.parents[2] / 'utils' / '1d',
            _here.parents[3] / 'pyAOBS' / 'utils' / '1d',
            Path.cwd() / 'pyAOBS' / 'utils' / '1d',
        ]
        
        # 如果grid_data存在，尝试从pyAOBS包路径查找
        if self.grid_data is not None:
            try:
                import pyAOBS
                pyAOBS_path = Path(pyAOBS.__file__).parent
                possible_paths.insert(0, pyAOBS_path / 'utils' / '1d')
            except:
                pass
        
        self.profile_1d_ranges = {}
        
        for path in possible_paths:
            if path.exists() and path.is_dir():
                # 读取所有txt文件
                for txt_file in sorted(path.glob('*.txt')):
                    try:
                        # 读取文件（两列：速度，深度，使用制表符或空格分隔）
                        data = np.loadtxt(txt_file, delimiter=None)
                        if data.ndim == 2 and data.shape[1] >= 2:
                            vp = data[:, 0]  # 第一列：速度
                            depth = data[:, 1]  # 第二列：深度（从基底面算起）
                            
                            # 确保数据是有效的
                            valid_mask = np.isfinite(vp) & np.isfinite(depth)
                            if np.any(valid_mask):
                                vp = vp[valid_mask]
                                depth = depth[valid_mask]
                                
                                # 存储数据
                                filename = txt_file.name
                                self.profile_1d_ranges[filename] = {
                                    'vp': vp,
                                    'depth': depth
                                }
                                self.log_result(f"Loaded 1D range: {filename} ({len(vp)} points)")
                    except Exception as e:
                        self.log_result(f"Warning: Failed to load {txt_file.name}: {str(e)}")
                
                # 找到第一个有效路径就停止
                if self.profile_1d_ranges:
                    break
        
        if not self.profile_1d_ranges:
            self.log_result("No 1D range files found in utils/1d directory")
    
    def save_profile_figure(self):
        """保存剖面图"""
        if self.profile_fig is None:
            messagebox.showwarning('Warning', 'No profile figure to save')
            return
        
        filename = filedialog.asksaveasfilename(
            title='Save Profile Figure',
            filetypes=[
                ('PNG files', '*.png'),
                ('PDF files', '*.pdf'),
                ('PostScript files', '*.ps'),
                ('EPS files', '*.eps'),
                ('JPEG files', '*.jpg'),
                ('TIFF files', '*.tif'),
                ('All files', '*.*')
            ]
        )
        
        if filename:
            try:
                import os
                base_name, ext = os.path.splitext(filename)
                supported_extensions = ['.png', '.pdf', '.ps', '.eps', '.jpg', '.jpeg', '.tif', '.tiff']
                
                if not ext:
                    filename = filename + '.png'
                elif ext.lower() not in supported_extensions:
                    filename = base_name + '.png'
                
                self.profile_fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo('Success', f'Figure saved to: {filename}')
                self.log_result(f"Profile figure saved: {filename}")
            except Exception as e:
                messagebox.showerror('Error', f'Failed to save figure: {str(e)}')
    
    def export_profile_data(self):
        """导出剖面数据"""
        if self.profile_one_d_model is None:
            messagebox.showwarning('Warning', 'No profile data to export')
            return
        
        filename = filedialog.asksaveasfilename(
            title='Export Profile Data',
            defaultextension='.csv',
            filetypes=[
                ('CSV files', '*.csv'),
                ('Text files', '*.txt'),
                ('All files', '*.*')
            ]
        )
        
        if filename:
            try:
                import os
                ext = os.path.splitext(filename)[1].lower()
                
                if ext == '.csv':
                    # 导出CSV格式
                    self.profile_one_d_model.to_csv(filename, index=False)
                else:
                    # 导出文本格式
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("Depth (km), Velocity (km/s)\n")
                        for _, row in self.profile_one_d_model.iterrows():
                            f.write(f"{row['depth']:.6f}, {row['vp']:.6f}\n")
                
                messagebox.showinfo('Success', f'Data exported to: {filename}')
                self.log_result(f"Profile data exported: {filename}")
            except Exception as e:
                messagebox.showerror('Error', f'Failed to export data: {str(e)}')
