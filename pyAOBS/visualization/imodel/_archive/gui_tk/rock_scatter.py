"""Vp-Vs 散点、Vp/Vs-Vp 图与 DEM 缓存。"""

from __future__ import annotations

from .deps import *  # noqa: F403


class RockScatterMixin:
    def show_vp_vs_scatter(self):
        """Show Vp-Vs scatter plot of rock database"""
        # 检查分类器是否可用
        if self.property_calculator is None or self.property_calculator.rock_classifier is None:
            messagebox.showwarning('Warning', 'Rock classifier not available. Please load model first.')
            return
        
        # 获取分类器的数据库
        try:
            classifier = self.property_calculator.rock_classifier.classifier
            if classifier is None:
                messagebox.showwarning('Warning', 'Rock classifier database not loaded.')
                return
            
            # 获取标准化的数据库数据
            db_data = classifier.rock_database_standard
            
            # 检查是否有vp和vs列
            if 'vp' not in db_data.columns or 'vs' not in db_data.columns:
                messagebox.showerror('Error', 'Database missing vp or vs columns.')
                return
            
            # 创建新窗口
            if self.scatter_window is None or not self.scatter_window.winfo_exists():
                self.scatter_window = tk.Toplevel(self.root)
                self.scatter_window.title('Vp-Vs Scatter Plot - Rock Database')
                self.scatter_window.geometry('800x600+200+100')
                
                # 创建matplotlib图形
                self.scatter_fig = plt.Figure(figsize=(8, 6))
                self.scatter_ax = self.scatter_fig.add_subplot(111)
                self.scatter_fig.subplots_adjust(left=0.1, right=0.7, top=0.95, bottom=0.1)
                
                # 创建canvas
                self.scatter_canvas = FigureCanvasTkAgg(self.scatter_fig, master=self.scatter_window)
                self.scatter_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # 创建工具栏
                toolbar_frame = ttk.Frame(self.scatter_window)
                toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
                scatter_toolbar = NavigationToolbar2Tk(self.scatter_canvas, toolbar_frame)
                scatter_toolbar.update()
                
                # 只添加保存按钮
                control_frame = ttk.Frame(self.scatter_window)
                control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
                ttk.Button(control_frame, text='Save Figure', 
                          command=self.save_scatter_figure).pack(side=tk.LEFT, padx=5)
            
            # 绘制数据库散点图
            self.scatter_ax.clear()
            
            # 统一坐标轴范围
            vs_min, vs_max = 2.0, 5.0  # 横波速度范围
            vp_min, vp_max = 4.0, 8.5  # 纵波速度范围
            
            # 绘制Vp/Vs参考线（放在散点图之下，zorder=1）
            vp_vs_ratios = [1.7, 1.9, 2.1]
            vs_line = np.linspace(vs_min, vs_max, 100)
            line_styles = ['--', '--', '--']
            line_colors = ['gray', 'gray', 'gray']
            line_alphas = [0.5, 0.5, 0.5]
            
            for ratio, linestyle, color, alpha in zip(vp_vs_ratios, line_styles, line_colors, line_alphas):
                vp_line = ratio * vs_line
                # 只绘制在合理范围内的部分
                mask = (vp_line >= vp_min) & (vp_line <= vp_max)
                self.scatter_ax.plot(vs_line[mask], vp_line[mask], 
                                   linestyle=linestyle, color=color, alpha=alpha, 
                                   linewidth=1.5, zorder=1,
                                   label=f'Vp/Vs = {ratio:.1f}')
            
            # 定义要突出显示的岩石类型及其颜色（使用小写作为键，用于大小写不敏感匹配）
            highlighted_rocks_lower = {
                'basalt': 'blue',
                'serpentinite': 'green',
                'gabbro': 'orange',
                'dunite': 'purple',
                'granite': 'brown'
            }
            # 显示名称（原始大小写）
            highlighted_rocks_display = {
                'basalt': 'Basalt',
                'serpentinite': 'Serpentinite',
                'gabbro': 'Gabbro',
                'dunite': 'Dunite',
                'granite': 'Granite'
            }
            
            # 按岩石类型分组绘制数据库点（zorder=5，在参考线之上）
            if 'rock_type' in db_data.columns:
                # 创建大小写不敏感的匹配映射
                rock_type_lower_map = {}
                for rock_type in db_data['rock_type'].unique():
                    rock_type_lower = str(rock_type).lower().strip()
                    rock_type_lower_map[rock_type_lower] = rock_type
                
                # 先绘制其他岩石（统一颜色，不显示在图例中）
                other_rocks_mask = pd.Series([False] * len(db_data), index=db_data.index)
                for rock_type in db_data['rock_type'].unique():
                    rock_type_lower = str(rock_type).lower().strip()
                    if rock_type_lower not in highlighted_rocks_lower:
                        other_rocks_mask |= (db_data['rock_type'] == rock_type)
                
                if other_rocks_mask.any():
                    other_vp = db_data.loc[other_rocks_mask, 'vp']
                    other_vs = db_data.loc[other_rocks_mask, 'vs']
                    self.scatter_ax.scatter(other_vs, other_vp,
                                          c='lightgray', alpha=0.4, s=30,
                                          edgecolors='gray', linewidths=0.3,
                                          zorder=5, label='_nolegend_')  # 不显示在图例中
                
                # 绘制突出显示的岩石类型（大小写不敏感匹配）
                for rock_key_lower, color in highlighted_rocks_lower.items():
                    # 查找匹配的岩石类型（大小写不敏感）
                    matched_rock_type = None
                    for db_rock_type in db_data['rock_type'].unique():
                        if str(db_rock_type).lower().strip() == rock_key_lower:
                            matched_rock_type = db_rock_type
                            break
                    
                    if matched_rock_type is not None:
                        mask = db_data['rock_type'] == matched_rock_type
                        if mask.any():
                            vp_data = db_data.loc[mask, 'vp']
                            vs_data = db_data.loc[mask, 'vs']
                            display_name = highlighted_rocks_display.get(rock_key_lower, matched_rock_type)
                            self.scatter_ax.scatter(vs_data, vp_data,
                                                  c=color, label=display_name,
                                                  alpha=0.7, s=60,
                                                  edgecolors='black', linewidths=0.8,
                                                  zorder=5)
            else:
                # 如果没有岩石类型列，直接绘制所有点
                self.scatter_ax.scatter(db_data['vs'], db_data['vp'],
                                      alpha=0.6, s=50, edgecolors='black', linewidths=0.5,
                                      label='Database', zorder=5)
            
            # 绘制从模型添加的点（zorder=10，在最上层）
            if self.model_points:
                model_vp = [p['vp'] for p in self.model_points]
                model_vs = [p['vs'] for p in self.model_points]
                self.scatter_ax.scatter(model_vs, model_vp,
                                      c='red', marker='*', s=200,
                                      edgecolors='darkred', linewidths=1.5,
                                      label='Model Points', zorder=10)
                
                # 为每个点添加编号标签
                for i, point in enumerate(self.model_points):
                    if point.get('type') == 'point':  # 只对点选择的点添加编号
                        point_number = point.get('point_number', i + 1)
                        self.scatter_ax.annotate(f'{point_number}', 
                                               (point['vs'], point['vp']),
                                               xytext=(5, 5), textcoords='offset points',
                                               fontsize=8, color='white',
                                               bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                                               ha='left', va='bottom', zorder=11)
            
            # 设置坐标轴范围
            self.scatter_ax.set_xlim(vs_min, vs_max)
            self.scatter_ax.set_ylim(vp_min, vp_max)
            
            # 设置标签和图例
            self.scatter_ax.set_xlabel('S-wave Velocity (km/s)', fontsize=12)
            self.scatter_ax.set_ylabel('P-wave Velocity (km/s)', fontsize=12)
            self.scatter_ax.set_title('Vp-Vs Scatter Plot - Rock Database', fontsize=14, fontweight='bold')
            self.scatter_ax.grid(True, alpha=0.3)
            self.scatter_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            # 更新canvas
            self.scatter_canvas.draw()
            
            # 记录信息
            self.log_result(f"\nVp-Vs scatter plot displayed:")
            self.log_result(f"  Database points: {len(db_data)}")
            self.log_result(f"  Model points: {len(self.model_points)}")
            self.log_result(f"  Data status: All velocities corrected to 200 MPa, 25°C")
            self.log_result(f"  Coordinate range: Vs [2.0-5.0 km/s], Vp [4.0-8.5 km/s]")
            if 'rock_type' in db_data.columns:
                unique_rocks = db_data['rock_type'].unique()
                self.log_result(f"  Total rock types: {len(unique_rocks)}")
                highlighted_count = sum(1 for r in unique_rocks 
                                      if str(r).lower().strip() in highlighted_rocks_lower)
                self.log_result(f"  Highlighted rock types: {highlighted_count}")
            
        except Exception as e:
            messagebox.showerror('Error', f'Failed to create scatter plot: {str(e)}')
            import traceback
            traceback.print_exc()
    
    def show_vp_vs_ratio_plot(self):
        """Show Vp/Vs vs Vp scatter plot of rock database"""
        # 检查分类器是否可用
        if self.property_calculator is None or self.property_calculator.rock_classifier is None:
            messagebox.showwarning('Warning', 'Rock classifier not available. Please load model first.')
            return
        
        # 获取分类器的数据库
        try:
            classifier = self.property_calculator.rock_classifier.classifier
            if classifier is None:
                messagebox.showwarning('Warning', 'Rock classifier database not loaded.')
                return
            
            # 获取标准化的数据库数据
            db_data = classifier.rock_database_standard
            
            # 检查是否有vp和vs列
            if 'vp' not in db_data.columns or 'vs' not in db_data.columns:
                messagebox.showerror('Error', 'Database missing vp or vs columns.')
                return
            
            # 计算Vp/Vs比值
            db_data = db_data.copy()
            db_data['vp_vs_ratio'] = db_data['vp'] / db_data['vs']
            
            # 创建新窗口
            if self.vp_vs_ratio_window is None or not self.vp_vs_ratio_window.winfo_exists():
                self.vp_vs_ratio_window = tk.Toplevel(self.root)
                self.vp_vs_ratio_window.title('Vp/Vs vs Vp Plot - Rock Database')
                self.vp_vs_ratio_window.geometry('800x600+250+150')
                
                # 创建matplotlib图形（略窄的窗口）
                self.vp_vs_ratio_fig = plt.Figure(figsize=(7, 6))
                self.vp_vs_ratio_ax = self.vp_vs_ratio_fig.add_subplot(111)
                self.vp_vs_ratio_fig.subplots_adjust(left=0.1, right=0.7, top=0.95, bottom=0.1)
                
                # 创建canvas
                self.vp_vs_ratio_canvas = FigureCanvasTkAgg(self.vp_vs_ratio_fig, master=self.vp_vs_ratio_window)
                self.vp_vs_ratio_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # 创建工具栏
                toolbar_frame = ttk.Frame(self.vp_vs_ratio_window)
                toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
                ratio_toolbar = NavigationToolbar2Tk(self.vp_vs_ratio_canvas, toolbar_frame)
                ratio_toolbar.update()
                
                # 创建控制面板
                control_frame = ttk.Frame(self.vp_vs_ratio_window)
                control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
                
                # 左侧：保存按钮
                button_frame = ttk.Frame(control_frame)
                button_frame.pack(side=tk.LEFT, padx=5)
                ttk.Button(button_frame, text='Save Figure', 
                          command=self.save_vp_vs_ratio_figure).pack(side=tk.LEFT, padx=5)
            
                # 右侧：显示控制复选框
                checkbox_frame = ttk.LabelFrame(control_frame, text='显示选项', padding=5)
                checkbox_frame.pack(side=tk.RIGHT, padx=5)
                
                # 创建复选框
                ttk.Checkbutton(checkbox_frame, text='采样岩石分布', 
                               variable=self.vp_vs_ratio_show_rocks,
                               command=self._refresh_vp_vs_ratio_plot).pack(side=tk.LEFT, padx=5)
                
                ttk.Checkbutton(checkbox_frame, text='含水量(蛇纹石化%)', 
                               variable=self.vp_vs_ratio_show_water,
                               command=self._refresh_vp_vs_ratio_plot).pack(side=tk.LEFT, padx=5)
                
                ttk.Checkbutton(checkbox_frame, text='纵横比曲线', 
                               variable=self.vp_vs_ratio_show_aspect_ratio,
                               command=self._refresh_vp_vs_ratio_plot).pack(side=tk.LEFT, padx=5)
                
                ttk.Checkbutton(checkbox_frame, text='孔隙度曲线', 
                               variable=self.vp_vs_ratio_show_porosity,
                               command=self._refresh_vp_vs_ratio_plot).pack(side=tk.LEFT, padx=5)
            
            # 保存数据到实例变量，以便后续刷新时使用
            self.vp_vs_ratio_db_data = db_data.copy()
            self.vp_vs_ratio_model_points_data = self.vp_vs_ratio_model_points.copy() if self.vp_vs_ratio_model_points else []
            
            # 计算并保存含水量数据
            water_contents = [2, 4, 6, 8, 10, 12, 13]  # wt%
            water_vp_list = []
            water_ratio_list = []
            water_labels = []
            for w in water_contents:
                serpentinization = calculate_serpentinization_from_water_content(w)
                vp_mps = calculate_vp_from_serpentinization(serpentinization)
                vs_mps = calculate_vs_from_serpentinization(serpentinization)
                vp_kms = vp_mps / 1000.0
                vs_kms = vs_mps / 1000.0
                vp_vs_ratio = vp_kms / vs_kms
                water_vp_list.append(vp_kms)
                water_ratio_list.append(vp_vs_ratio)
                water_labels.append(f'{w}% H₂O\n({serpentinization:.1f}% β)')
            self.vp_vs_ratio_water_data = {
                'vp': water_vp_list,
                'ratio': water_ratio_list,
                'labels': water_labels
            }
            
            # 保存DEM参数和曲线数据
            host_vp_dunite = 8.299
            host_vs_dunite = 4.731
            host_density_dunite = 3.310
            inclusion_k = 2.2
            inclusion_mu = 0.0
            inclusion_density = 1.03
            critical_porosity = 0.65
            aspect_ratios = [0.05, 0.03, 0.02, 0.013, 0.01, 0.0067, 0.005, 0.002, 0.001, 0.0001]
            aspect_ratio_labels = ['0.05', '0.03', '0.02', '0.013', '0.01', '0.0067', '0.005', '0.002', '0.001', '0.0001']
            porosity_max = 0.4
            n_points = 500
            porosity_range = np.linspace(0, porosity_max, n_points)
            
            # 检查缓存并加载/计算曲线数据
            cache_file = self._get_dem_cache_filename(
                host_vp_dunite, host_vs_dunite, host_density_dunite,
                inclusion_k, inclusion_mu, inclusion_density,
                critical_porosity, aspect_ratios, porosity_max, n_points,
                rock_name='dunite'
            )
            
            curves_data = self._load_dem_curves_cache(cache_file)
            
            # 验证缓存数据的完整性
            if curves_data is not None:
                missing_aspect_ratios = []
                for aspect_ratio in aspect_ratios:
                    if aspect_ratio not in curves_data:
                        missing_aspect_ratios.append(aspect_ratio)
                    else:
                        vp_array, vp_vs_ratio_array, por_array = curves_data[aspect_ratio]
                        if np.sum(~np.isnan(vp_array)) == 0:
                            missing_aspect_ratios.append(aspect_ratio)
                if missing_aspect_ratios:
                    # 如果有纵横比在缓存中缺失或数据全为NaN，则认为缓存无效
                    curves_data = None
            
            if curves_data is None:
                # 缓存不存在或数据不完整，进行计算
                progress_window = tk.Toplevel(self.root)
                progress_window.title('计算DEM曲线')
                progress_window.geometry('400x120')
                progress_window.transient(self.root)
                progress_window.grab_set()
                
                progress_window.update_idletasks()
                x = (progress_window.winfo_screenwidth() // 2) - (400 // 2)
                y = (progress_window.winfo_screenheight() // 2) - (120 // 2)
                progress_window.geometry(f'400x120+{x}+{y}')
                
                progress_label = ttk.Label(progress_window, text='正在计算DEM曲线...', font=('Arial', 10))
                progress_label.pack(pady=10)
                
                progress_bar = ttk.Progressbar(progress_window, length=350, mode='determinate')
                progress_bar.pack(pady=5)
                
                status_label = ttk.Label(progress_window, text='', font=('Arial', 9))
                status_label.pack(pady=5)
                
                def update_progress(current, total, message):
                    if total > 0:
                        progress = (current / total) * 100
                        progress_bar['value'] = progress
                        status_label['text'] = message
                        progress_window.update_idletasks()
                
                try:
                    curves_data = self._calculate_dem_curves(
                        aspect_ratios, porosity_range, host_vp_dunite,
                        host_vs_dunite, host_density_dunite, inclusion_k,
                        inclusion_mu, inclusion_density, critical_porosity,
                        progress_callback=update_progress
                    )
                    self._save_dem_curves_cache(cache_file, curves_data)
                finally:
                    progress_window.destroy()
            
            # 保存DEM参数和曲线数据
            self.vp_vs_ratio_curves_data = curves_data
            self.vp_vs_ratio_dem_params = {
                'aspect_ratios': aspect_ratios,
                'aspect_ratio_labels': aspect_ratio_labels,
                'porosities_percent': [12.51, 8.08, 5.69, 3.37, 2.22, 0.62, 0.25, 0.02],
                'porosity_range': porosity_range
            }
            
            # 调用统一的绘制函数
            self._refresh_vp_vs_ratio_plot()
            
            # 记录信息
            self.log_result(f"\nVp/Vs vs Vp plot displayed:")
            self.log_result(f"  Database points: {len(self.vp_vs_ratio_db_data)}")
            self.log_result(f"  Model points: {len(self.vp_vs_ratio_model_points_data)}")
            if self.vp_vs_ratio_water_data:
                self.log_result(f"  Water content reference points: {len(self.vp_vs_ratio_water_data['vp'])}")
            if self.vp_vs_ratio_curves_data:
                self.log_result(f"  DEM curves calculated: {len(self.vp_vs_ratio_curves_data)} aspect ratios")
            self.log_result(f"  Display options: Rocks={self.vp_vs_ratio_show_rocks.get()}, "
                          f"Water={self.vp_vs_ratio_show_water.get()}, "
                          f"AspectRatio={self.vp_vs_ratio_show_aspect_ratio.get()}, "
                          f"Porosity={self.vp_vs_ratio_show_porosity.get()}")
            
        except Exception as e:
            messagebox.showerror('Error', f'Failed to create Vp/Vs vs Vp plot: {str(e)}')
            import traceback
            traceback.print_exc()
    
    def _refresh_vp_vs_ratio_plot(self):
        """根据复选框状态刷新Vp/Vs vs Vp图（内部方法）"""
        if self.vp_vs_ratio_ax is None or self.vp_vs_ratio_db_data is None:
            return
        
        # 清空图形
        self.vp_vs_ratio_ax.clear()
        
        # 1. 绘制采样岩石分布（如果选中）
        if self.vp_vs_ratio_show_rocks.get():
            self._draw_rock_database_points()
        
        # 2. 绘制模型点（始终显示，如果存在）
        if self.vp_vs_ratio_model_points_data:
            self._draw_model_points()
        
        # 3. 绘制含水量参考点（如果选中）
        if self.vp_vs_ratio_show_water.get() and self.vp_vs_ratio_water_data:
            self._draw_water_content_points()
        
        # 4. 绘制纵横比曲线（如果选中）
        if self.vp_vs_ratio_show_aspect_ratio.get() and self.vp_vs_ratio_curves_data and self.vp_vs_ratio_dem_params:
            self._draw_aspect_ratio_curves()
        
        # 5. 绘制孔隙度曲线（如果选中）
        if self.vp_vs_ratio_show_porosity.get() and self.vp_vs_ratio_curves_data and self.vp_vs_ratio_dem_params:
            self._draw_porosity_curves()
        
        # 设置坐标轴和标签
        self._setup_vp_vs_ratio_axes()
        
        # 更新canvas
        if self.vp_vs_ratio_canvas:
            self.vp_vs_ratio_canvas.draw()
    
    def _draw_rock_database_points(self):
        """绘制采样岩石分布点"""
        if self.vp_vs_ratio_db_data is None:
            return
        
        db_data = self.vp_vs_ratio_db_data
        
        # 定义要突出显示的岩石类型及其颜色
        highlighted_rocks_lower = {
            'basalt': 'blue',
            'serpentinite': 'green',
            'gabbro': 'orange',
            'dunite': 'purple',
            'granite': 'brown'
        }
        highlighted_rocks_display = {
            'basalt': 'Basalt',
            'serpentinite': 'Serpentinite',
            'gabbro': 'Gabbro',
            'dunite': 'Dunite',
            'granite': 'Granite'
        }
        
        # 按岩石类型分组绘制数据库点
        if 'rock_type' in db_data.columns:
            # 先绘制其他岩石（统一颜色，不显示在图例中）
            other_rocks_mask = pd.Series([False] * len(db_data), index=db_data.index)
            for rock_type in db_data['rock_type'].unique():
                rock_type_lower = str(rock_type).lower().strip()
                if rock_type_lower not in highlighted_rocks_lower:
                    other_rocks_mask |= (db_data['rock_type'] == rock_type)
            
            if other_rocks_mask.any():
                other_vp = db_data.loc[other_rocks_mask, 'vp']
                other_ratio = db_data.loc[other_rocks_mask, 'vp_vs_ratio']
                self.vp_vs_ratio_ax.scatter(
                    other_vp,
                    other_ratio,
                    c='lightgray',
                    alpha=0.4,
                    s=30,
                    edgecolors='gray',
                    linewidths=0.3,
                    zorder=5,
                    label='_nolegend_',
                )
            
            # 绘制突出显示的岩石类型
            for rock_key_lower, color in highlighted_rocks_lower.items():
                matched_rock_type = None
                for db_rock_type in db_data['rock_type'].unique():
                    if str(db_rock_type).lower().strip() == rock_key_lower:
                        matched_rock_type = db_rock_type
                        break
                
                if matched_rock_type is not None:
                    mask = db_data['rock_type'] == matched_rock_type
                    if mask.any():
                        vp_data = db_data.loc[mask, 'vp']
                        ratio_data = db_data.loc[mask, 'vp_vs_ratio']
                        display_name = highlighted_rocks_display.get(rock_key_lower, matched_rock_type)
                        self.vp_vs_ratio_ax.scatter(
                            vp_data,
                            ratio_data,
                            c=color,
                            label=display_name,
                            alpha=0.7,
                            s=60,
                            edgecolors='black',
                            linewidths=0.8,
                            zorder=5,
                        )
        else:
            # 如果没有岩石类型列，直接绘制所有点
            self.vp_vs_ratio_ax.scatter(
                db_data['vp'],
                db_data['vp_vs_ratio'],
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidths=0.5,
                label='Database',
                zorder=5,
            )
            
    def _draw_model_points(self):
        """绘制模型点（带编号）"""
        if not self.vp_vs_ratio_model_points_data:
            return
        
        valid_points = []
        for p in self.vp_vs_ratio_model_points_data:
            vp_val = p.get('vp')
            vs_val = p.get('vs')
            if self._is_valid_vp_vs_pair(vp_val, vs_val):
                valid_points.append(p)
        if not valid_points:
            return

        model_vp = [p['vp'] for p in valid_points]
        model_ratio = [p['vp'] / p['vs'] for p in valid_points]
        self.vp_vs_ratio_ax.scatter(
            model_vp,
            model_ratio,
            c='red',
            marker='*',
            s=200,
            edgecolors='darkred',
            linewidths=1.5,
            label='Model Points',
            zorder=10,
        )
            
        # 为每个点添加编号标签
        for i, point in enumerate(valid_points):
            if point.get('type') == 'point':  # 只对点选择的点添加编号
                point_number = point.get('point_number', i + 1)
                vp = point['vp']
                ratio = point['vp'] / point['vs']
                self.vp_vs_ratio_ax.annotate(f'{point_number}', 
                                           (vp, ratio),
                                           xytext=(5, 5), textcoords='offset points',
                                           fontsize=8, color='white',
                                           bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                                           ha='left', va='bottom', zorder=11)
    
    def _draw_water_content_points(self):
        """绘制含水量(蛇纹石化百分比)参考点"""
        if not self.vp_vs_ratio_water_data:
            return
        
        water_vp_list = self.vp_vs_ratio_water_data['vp']
        water_ratio_list = self.vp_vs_ratio_water_data['ratio']
        water_labels = self.vp_vs_ratio_water_data['labels']
        
        if water_vp_list:
            self.vp_vs_ratio_ax.scatter(
                water_vp_list,
                water_ratio_list,
                c='cyan',
                marker='D',
                s=150,
                edgecolors='darkblue',
                linewidths=1.5,
                alpha=0.8,
                zorder=8,
                label='Water Content Reference',
            )
            
            # 添加标签
            for i, (vp, ratio, label) in enumerate(zip(water_vp_list, water_ratio_list, water_labels)):
                self.vp_vs_ratio_ax.annotate(
                    label,
                    (vp, ratio),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    color='darkblue',
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='white',
                        edgecolor='darkblue',
                        alpha=0.7,
                    ),
                    zorder=9,
                    ha='left',
                    va='bottom',
                )
            
    def _draw_aspect_ratio_curves(self):
        """绘制不同纵横比对应曲线"""
        if not self.vp_vs_ratio_curves_data or not self.vp_vs_ratio_dem_params:
            return
        
        curves_data = self.vp_vs_ratio_curves_data
        aspect_ratios = self.vp_vs_ratio_dem_params['aspect_ratios']
        aspect_ratio_labels = self.vp_vs_ratio_dem_params['aspect_ratio_labels']
        
        # 纵横比曲线使用黑色
        curve_color = 'black'
        
        # 绘制纵横比曲线
        for i, aspect_ratio in enumerate(aspect_ratios):
            if aspect_ratio in curves_data:
                vp_array, vp_vs_ratio_array, por_array = curves_data[aspect_ratio]
                
                # 过滤显示范围
                mask = (vp_array >= 1.0) & (vp_array <= 12.0) & \
                       (vp_vs_ratio_array >= 1.0) & (vp_vs_ratio_array <= 3.0) & \
                       (~np.isnan(vp_array)) & (~np.isnan(vp_vs_ratio_array))
                
                vp_filtered = vp_array[mask]
                vp_vs_ratio_filtered = vp_vs_ratio_array[mask]
                
                if len(vp_filtered) > 1:
                    display_mask = (vp_filtered >= 2.5) & (vp_filtered <= 9.0) & \
                                  (vp_vs_ratio_filtered >= 1.5) & (vp_vs_ratio_filtered <= 2.3)
                    
                    if np.sum(display_mask) > 1:
                        vp_display = vp_filtered[display_mask]
                        vp_vs_ratio_display = vp_vs_ratio_filtered[display_mask]
                        self.vp_vs_ratio_ax.plot(vp_display, vp_vs_ratio_display,
                                                color=curve_color, linewidth=1.5, alpha=0.7,
                                                label=f'Aspect Ratio = {aspect_ratio_labels[i]}',
                                                zorder=6)
                    elif len(vp_filtered) > 1:
                        self.vp_vs_ratio_ax.plot(vp_filtered, vp_vs_ratio_filtered,
                                                color=curve_color, linewidth=1.5, alpha=0.7,
                                                label=f'Aspect Ratio = {aspect_ratio_labels[i]}',
                                                zorder=6)
    
    def _draw_porosity_curves(self):
        """绘制不同孔隙度对应曲线"""
        if not self.vp_vs_ratio_curves_data or not self.vp_vs_ratio_dem_params:
            return
        
        curves_data = self.vp_vs_ratio_curves_data
        aspect_ratios = self.vp_vs_ratio_dem_params['aspect_ratios']
        porosities_percent = self.vp_vs_ratio_dem_params['porosities_percent']
        porosities = [p / 100.0 for p in porosities_percent]
        
        # 颜色渐变：使用viridis colormap（与纵横比曲线相同的颜色方案）
        porosity_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(porosities)))
        
        for i, target_porosity in enumerate(porosities):
            vp_curve = []
            vp_vs_ratio_curve = []
            
            # 对于每个纵横比，找到对应孔隙度的点
            for aspect_ratio in aspect_ratios:
                if aspect_ratio in curves_data:
                    vp_array, vp_vs_ratio_array, por_array = curves_data[aspect_ratio]
                    
                    valid_mask = (
                        (~np.isnan(vp_array))
                        & (~np.isnan(vp_vs_ratio_array))
                        & (vp_array > 0)
                        & (vp_vs_ratio_array > 0)
                    )
                    
                    if np.sum(valid_mask) > 1:
                        from scipy.interpolate import interp1d
                        try:
                            por_valid = por_array[valid_mask]
                            vp_valid = vp_array[valid_mask]
                            vp_vs_ratio_valid = vp_vs_ratio_array[valid_mask]
                            
                            if len(por_valid) > 1 and np.all(np.diff(por_valid) > 0):
                                if por_valid[0] <= target_porosity <= por_valid[-1]:
                                    vp_interp = interp1d(
                                        por_valid,
                                        vp_valid,
                                        kind='linear',
                                        bounds_error=False,
                                        fill_value='extrapolate',
                                    )
                                    vp_vs_ratio_interp = interp1d(
                                        por_valid,
                                        vp_vs_ratio_valid,
                                        kind='linear',
                                        bounds_error=False,
                                        fill_value='extrapolate',
                                    )
                                    
                                    vp_at_por = float(vp_interp(target_porosity))
                                    vp_vs_ratio_at_por = float(vp_vs_ratio_interp(target_porosity))
                                    
                                    if (
                                        not np.isnan(vp_at_por)
                                        and not np.isnan(vp_vs_ratio_at_por)
                                        and vp_at_por > 0
                                        and vp_at_por < 20.0
                                        and vp_vs_ratio_at_por > 0
                                        and vp_vs_ratio_at_por < 10.0
                                    ):
                                        vp_curve.append(vp_at_por)
                                        vp_vs_ratio_curve.append(vp_vs_ratio_at_por)
                                elif target_porosity < por_valid[0] and por_valid[0] - target_porosity < 0.02:
                                    vp_at_por = float(vp_valid[0])
                                    vp_vs_ratio_at_por = float(vp_vs_ratio_valid[0])
                                    if (
                                        not np.isnan(vp_at_por)
                                        and not np.isnan(vp_vs_ratio_at_por)
                                        and vp_at_por > 0
                                        and vp_at_por < 20.0
                                        and vp_vs_ratio_at_por > 0
                                        and vp_vs_ratio_at_por < 10.0
                                    ):
                                        vp_curve.append(vp_at_por)
                                        vp_vs_ratio_curve.append(vp_vs_ratio_at_por)
                                elif target_porosity > por_valid[-1] and target_porosity - por_valid[-1] < 0.02:
                                    vp_at_por = float(vp_valid[-1])
                                    vp_vs_ratio_at_por = float(vp_vs_ratio_valid[-1])
                                    if (
                                        not np.isnan(vp_at_por)
                                        and not np.isnan(vp_vs_ratio_at_por)
                                        and vp_at_por > 0
                                        and vp_at_por < 20.0
                                        and vp_vs_ratio_at_por > 0
                                        and vp_vs_ratio_at_por < 10.0
                                    ):
                                        vp_curve.append(vp_at_por)
                                        vp_vs_ratio_curve.append(vp_vs_ratio_at_por)
                        except Exception:
                            continue
            
            # 如果有足够的点，按Vp排序后绘制曲线
            if len(vp_curve) > 1:
                vp_curve = np.array(vp_curve)
                vp_vs_ratio_curve = np.array(vp_vs_ratio_curve)
                
                sort_indices = np.argsort(vp_curve)
                vp_sorted = vp_curve[sort_indices]
                vp_vs_ratio_sorted = vp_vs_ratio_curve[sort_indices]
                
                display_mask = (vp_sorted >= 2.5) & (vp_sorted <= 9.0) & \
                              (vp_vs_ratio_sorted >= 1.5) & (vp_vs_ratio_sorted <= 2.3)
                
                if np.sum(display_mask) > 1:
                    self.vp_vs_ratio_ax.plot(
                        vp_sorted[display_mask],
                        vp_vs_ratio_sorted[display_mask],
                        color=porosity_colors[i],
                        linewidth=2,
                        alpha=0.8,
                        linestyle='--',
                        label=f'Porosity = {porosities_percent[i]:.2f}%',
                        zorder=7,
                    )
                elif len(vp_sorted) > 1:
                    self.vp_vs_ratio_ax.plot(
                        vp_sorted,
                        vp_vs_ratio_sorted,
                        color=porosity_colors[i],
                        linewidth=2,
                        alpha=0.8,
                        linestyle='--',
                        label=f'Porosity = {porosities_percent[i]:.2f}%',
                        zorder=7,
                    )
            
    def _setup_vp_vs_ratio_axes(self):
        """设置Vp/Vs vs Vp图的坐标轴和标签"""
        self.vp_vs_ratio_ax.set_xlabel('P-wave Velocity (km/s)', fontsize=12)
        self.vp_vs_ratio_ax.set_ylabel('Vp/Vs Ratio', fontsize=12)
        self.vp_vs_ratio_ax.set_title('Vp/Vs vs Vp Plot - Rock Database', fontsize=14, fontweight='bold')
        self.vp_vs_ratio_ax.grid(True, alpha=0.3)
        self.vp_vs_ratio_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        self.vp_vs_ratio_ax.set_xlim(3.0, 8.5)
        self.vp_vs_ratio_ax.set_ylim(1.6, 2.15)
        # 设置Vp/Vs轴坐标间隔为0.05
        self.vp_vs_ratio_ax.set_yticks(np.arange(1.6, 2.16, 0.05))
        
        # 标记Dunite点
        host_vp_dunite = 8.299
        host_vs_dunite = 4.731
        self.vp_vs_ratio_ax.scatter(
            [host_vp_dunite],
            [host_vp_dunite / host_vs_dunite],
            c='yellow',
            marker='o',
            s=200,
            edgecolors='black',
            linewidths=2,
            label='Dunite',
            zorder=10,
        )
        self.vp_vs_ratio_ax.annotate(
            'Dunite',
            (host_vp_dunite, host_vp_dunite / host_vs_dunite),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='yellow',
                edgecolor='black',
                alpha=0.8,
            ),
            zorder=11,
        )
            
    def _update_vp_vs_ratio_plot(self):
        """Update Vp/Vs vs Vp plot with current model points (internal method)"""
        if self.vp_vs_ratio_fig is None or self.vp_vs_ratio_ax is None:
            return
        
        try:
            # 检查分类器是否可用
            if self.property_calculator is None or self.property_calculator.rock_classifier is None:
                return
            
            classifier = self.property_calculator.rock_classifier.classifier
            if classifier is None:
                return
            
            # 获取标准化的数据库数据
            db_data = classifier.rock_database_standard
            
            # 检查是否有vp和vs列
            if 'vp' not in db_data.columns or 'vs' not in db_data.columns:
                return
            
            # 计算Vp/Vs比值
            db_data = db_data.copy()
            db_data['vp_vs_ratio'] = db_data['vp'] / db_data['vs']
            
            # 更新数据
            self.vp_vs_ratio_db_data = db_data
            self.vp_vs_ratio_model_points_data = self.vp_vs_ratio_model_points.copy() if self.vp_vs_ratio_model_points else []
            
            # 使用统一的刷新函数
            self._refresh_vp_vs_ratio_plot()
            
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to update Vp/Vs vs Vp plot: {e}")
    
    def save_vp_vs_ratio_figure(self):
        """Save Vp/Vs vs Vp plot figure"""
        if self.vp_vs_ratio_fig is None:
            messagebox.showwarning('Warning', 'No figure to save.')
            return
        
        # 定义支持的文件类型
        filetypes = [
            ('PNG files', '*.png'),
            ('PDF files', '*.pdf'),
            ('PostScript files', '*.ps'),
            ('EPS files', '*.eps'),
            ('JPEG files', '*.jpg'),
            ('TIFF files', '*.tif'),
            ('All files', '*.*')
        ]
        
        filename = tk.filedialog.asksaveasfilename(
            title='Save Vp/Vs vs Vp Plot',
            # 不设置defaultextension，让文件对话框根据用户选择的文件类型自动添加扩展名
            # 这样当用户选择文件类型时，会自动添加相应的扩展名
            filetypes=filetypes
        )
        
        if filename:
            try:
                import os
                base_name, ext = os.path.splitext(filename)
                # 支持的文件扩展名列表
                supported_extensions = ['.png', '.pdf', '.ps', '.eps', '.jpg', '.jpeg', '.tif', '.tiff']
                
                # 如果文件名没有扩展名，说明用户可能选择了"All files"或没有明确选择类型
                # 在这种情况下，默认使用PNG格式
                if not ext:
                    filename = filename + '.png'
                # 如果扩展名不在支持列表中，替换为PNG
                elif ext.lower() not in supported_extensions:
                    filename = base_name + '.png'
                
                self.vp_vs_ratio_fig.savefig(filename, dpi=300, bbox_inches='tight')
                self.log_result(f"  Vp/Vs vs Vp plot saved to: {filename}")
                messagebox.showinfo('Success', f'Figure saved to {filename}')
            except Exception as e:
                messagebox.showerror('Error', f'Failed to save figure: {str(e)}')
    
    def _get_dem_cache_filename(self, host_vp, host_vs, host_density, inclusion_k, inclusion_mu, 
                                  inclusion_density, critical_porosity, aspect_ratios, porosity_max, n_points,
                                  rock_name='dunite'):
        """生成DEM曲线缓存文件名
        
        Args:
            host_vp, host_vs, host_density: 背景材料参数
            inclusion_k, inclusion_mu, inclusion_density: 包含物参数
            critical_porosity: 临界孔隙度
            aspect_ratios: 纵横比列表
            porosity_max: 最大孔隙度
            n_points: 孔隙度点数
            rock_name: 背景岩石名称，默认'dunite'
            
        Returns:
            缓存文件路径
        """
        # 格式化数值，去除小数点后的尾随零
        def format_number(num):
            """格式化数字，去除不必要的尾随零"""
            if isinstance(num, (int, float)):
                # 如果是整数，返回整数格式
                if abs(num - int(num)) < 1e-10:
                    return str(int(num))
                # 否则保留必要的小数位（最多6位），去除尾随零
                formatted = f"{num:.6f}".rstrip('0').rstrip('.')
                return formatted
            return str(num)
        
        # 生成文件名：dem_curves_cache_{岩石名称}_{vp}_{vs}_{density}_{critical_porosity}.pkl
        vp_str = format_number(host_vp)
        vs_str = format_number(host_vs)
        density_str = format_number(host_density)
        phic_str = format_number(critical_porosity)
        
        cache_filename_str = f'dem_curves_cache_{rock_name}_{vp_str}_{vs_str}_{density_str}_{phic_str}.pkl'
        
        # 获取utils目录路径
        utils_dir = Path(__file__).resolve().parents[2] / 'utils'
        cache_filename = utils_dir / cache_filename_str
        return cache_filename
    
    def _load_dem_curves_cache(self, cache_file):
        """加载DEM曲线缓存
        
        Args:
            cache_file: 缓存文件路径
            
        Returns:
            如果成功，返回字典 {aspect_ratio: (vp_array, vp_vs_ratio_array, porosity_range)}，否则返回None
        """
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                return data
        except Exception as e:
            warnings.warn(f"Failed to load DEM cache: {e}")
        return None
    
    def _save_dem_curves_cache(self, cache_file, curves_data):
        """保存DEM曲线缓存
        
        Args:
            cache_file: 缓存文件路径
            curves_data: 字典 {aspect_ratio: (vp_array, vp_vs_ratio_array, porosity_range)}
        """
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(curves_data, f)
        except Exception as e:
            warnings.warn(f"Failed to save DEM cache: {e}")
    
    def _calculate_dem_curves(self, aspect_ratios, porosity_range, host_vp_dunite, 
                              host_vs_dunite, host_density_dunite, inclusion_k, 
                              inclusion_mu, inclusion_density, critical_porosity,
                              progress_callback=None):
        """计算DEM曲线数据
        
        Args:
            aspect_ratios: 纵横比列表
            porosity_range: 孔隙度数组
            host_vp_dunite, host_vs_dunite, host_density_dunite: 背景材料参数
            inclusion_k, inclusion_mu, inclusion_density: 包含物参数
            critical_porosity: 临界孔隙度
            progress_callback: 进度回调函数 callback(current, total, message)
            
        Returns:
            字典 {aspect_ratio: (vp_array, vp_vs_ratio_array, porosity_range)}
        """
        curves_data = {}
        total_tasks = len(aspect_ratios) * len(porosity_range)
        current_task = 0
        
        for i, aspect_ratio in enumerate(aspect_ratios):
            vp_list = []
            vp_vs_ratio_list = []
            
            # 更新进度：开始计算新的纵横比
            if progress_callback:
                progress_callback(current_task, total_tasks, 
                                f"计算纵横比 {aspect_ratio:.6f} ({i+1}/{len(aspect_ratios)})")
            
            for j, porosity in enumerate(porosity_range):
                try:
                    vp_eff, vs_eff, vp_vs_ratio = calculate_dem_velocity(
                        porosity=porosity,
                        aspect_ratio=aspect_ratio,
                        host_vp=host_vp_dunite,
                        host_vs=host_vs_dunite,
                        host_density=host_density_dunite,
                        inclusion_k=inclusion_k,
                        inclusion_mu=inclusion_mu,
                        inclusion_density=inclusion_density,
                        n_steps=200,
                        critical_porosity=critical_porosity
                    )
                    
                    # 放宽过滤条件：只要值在合理范围内就保存（不限制上限，让后续绘制时再过滤）
                    if (vp_eff > 0 and vs_eff > 0 and vp_vs_ratio > 0 and 
                        not np.isnan(vp_eff) and not np.isnan(vs_eff) and not np.isnan(vp_vs_ratio) and
                        vp_eff < 20.0 and vs_eff < 20.0 and vp_vs_ratio < 10.0):  # 放宽上限
                        vp_list.append(vp_eff)
                        vp_vs_ratio_list.append(vp_vs_ratio)
                    else:
                        # 如果值无效，添加NaN以断开曲线
                        vp_list.append(np.nan)
                        vp_vs_ratio_list.append(np.nan)
                except (ValueError, ZeroDivisionError, Exception) as e:
                    # 如果计算失败，添加NaN以断开曲线
                    vp_list.append(np.nan)
                    vp_vs_ratio_list.append(np.nan)
                    # 不打印错误，避免输出过多
                
                # 更新进度（每10个点或最后一个点更新一次，避免过于频繁）
                current_task += 1
                if progress_callback and (j % 10 == 0 or j == len(porosity_range) - 1):
                    progress_callback(current_task, total_tasks, 
                                    f"纵横比 {aspect_ratio:.6f} ({i+1}/{len(aspect_ratios)}) - "
                                    f"孔隙度 {porosity:.4f} ({j+1}/{len(porosity_range)})")
            
            # 转换为numpy数组
            vp_array = np.array(vp_list)
            vp_vs_ratio_array = np.array(vp_vs_ratio_list)
            
            # 检查是否有有效数据
            valid_count = np.sum(~np.isnan(vp_array))
            if valid_count > 0:
                curves_data[aspect_ratio] = (vp_array, vp_vs_ratio_array, porosity_range)
            else:
                # 即使没有有效数据，也保存（可能后续处理时会用到）
                curves_data[aspect_ratio] = (vp_array, vp_vs_ratio_array, porosity_range)
        
        # 完成所有计算
        if progress_callback:
            progress_callback(total_tasks, total_tasks, "计算完成")
        
        return curves_data
    
    def _update_scatter_plot(self):
        """Update scatter plot with current model points (internal method)"""
        if self.scatter_fig is None or self.scatter_ax is None:
            return
        
        try:
            classifier = self.property_calculator.rock_classifier.classifier
            if classifier is None:
                return
            
            db_data = classifier.rock_database_standard
            
            if 'vp' not in db_data.columns or 'vs' not in db_data.columns:
                return
            
            # 重新绘制
            self.scatter_ax.clear()
            
            # 统一坐标轴范围
            vs_min, vs_max = 2.0, 5.0  # 横波速度范围
            vp_min, vp_max = 4.0, 8.5  # 纵波速度范围
            
            # 绘制Vp/Vs参考线（放在散点图之下，zorder=1）
            vp_vs_ratios = [1.7, 1.9, 2.1]
            vs_line = np.linspace(vs_min, vs_max, 100)
            line_styles = ['--', '--', '--']
            line_colors = ['gray', 'gray', 'gray']
            line_alphas = [0.5, 0.5, 0.5]
            
            for ratio, linestyle, color, alpha in zip(vp_vs_ratios, line_styles, line_colors, line_alphas):
                vp_line = ratio * vs_line
                # 只绘制在合理范围内的部分
                mask = (vp_line >= vp_min) & (vp_line <= vp_max)
                self.scatter_ax.plot(vs_line[mask], vp_line[mask], 
                                   linestyle=linestyle, color=color, alpha=alpha, 
                                   linewidth=1.5, zorder=1,
                                   label=f'Vp/Vs = {ratio:.1f}')
            
            # 定义要突出显示的岩石类型及其颜色（使用小写作为键，用于大小写不敏感匹配）
            highlighted_rocks_lower = {
                'basalt': 'blue',
                'serpentinite': 'green',
                'gabbro': 'orange',
                'dunite': 'purple',
                'granite': 'brown'
            }
            # 显示名称（原始大小写）
            highlighted_rocks_display = {
                'basalt': 'Basalt',
                'serpentinite': 'Serpentinite',
                'gabbro': 'Gabbro',
                'dunite': 'Dunite',
                'granite': 'Granite'
            }
            
            # 按岩石类型分组绘制数据库点（zorder=5，在参考线之上）
            if 'rock_type' in db_data.columns:
                # 创建大小写不敏感的匹配映射
                rock_type_lower_map = {}
                for rock_type in db_data['rock_type'].unique():
                    rock_type_lower = str(rock_type).lower().strip()
                    rock_type_lower_map[rock_type_lower] = rock_type
                
                # 先绘制其他岩石（统一颜色，不显示在图例中）
                other_rocks_mask = pd.Series([False] * len(db_data), index=db_data.index)
                for rock_type in db_data['rock_type'].unique():
                    rock_type_lower = str(rock_type).lower().strip()
                    if rock_type_lower not in highlighted_rocks_lower:
                        other_rocks_mask |= (db_data['rock_type'] == rock_type)
                
                if other_rocks_mask.any():
                    other_vp = db_data.loc[other_rocks_mask, 'vp']
                    other_vs = db_data.loc[other_rocks_mask, 'vs']
                    self.scatter_ax.scatter(other_vs, other_vp,
                                          c='lightgray', alpha=0.4, s=30,
                                          edgecolors='gray', linewidths=0.3,
                                          zorder=5, label='_nolegend_')  # 不显示在图例中
                
                # 绘制突出显示的岩石类型（大小写不敏感匹配）
                for rock_key_lower, color in highlighted_rocks_lower.items():
                    # 查找匹配的岩石类型（大小写不敏感）
                    matched_rock_type = None
                    for db_rock_type in db_data['rock_type'].unique():
                        if str(db_rock_type).lower().strip() == rock_key_lower:
                            matched_rock_type = db_rock_type
                            break
                    
                    if matched_rock_type is not None:
                        mask = db_data['rock_type'] == matched_rock_type
                        if mask.any():
                            vp_data = db_data.loc[mask, 'vp']
                            vs_data = db_data.loc[mask, 'vs']
                            display_name = highlighted_rocks_display.get(rock_key_lower, matched_rock_type)
                            self.scatter_ax.scatter(vs_data, vp_data,
                                                  c=color, label=display_name,
                                                  alpha=0.7, s=60,
                                                  edgecolors='black', linewidths=0.8,
                                                  zorder=5)
            else:
                self.scatter_ax.scatter(db_data['vs'], db_data['vp'],
                                      alpha=0.6, s=50, edgecolors='black', linewidths=0.5,
                                      label='Database', zorder=5)
            
            # 绘制从模型添加的点（zorder=10，在最上层）
            if self.model_points:
                model_vp = [p['vp'] for p in self.model_points]
                model_vs = [p['vs'] for p in self.model_points]
                self.scatter_ax.scatter(model_vs, model_vp,
                                      c='red', marker='*', s=200,
                                      edgecolors='darkred', linewidths=1.5,
                                      label='Model Points', zorder=10)
                
                # 为每个点添加编号标签
                for i, point in enumerate(self.model_points):
                    if point.get('type') == 'point':  # 只对点选择的点添加编号
                        point_number = point.get('point_number', i + 1)
                        self.scatter_ax.annotate(f'{point_number}', 
                                               (point['vs'], point['vp']),
                                               xytext=(5, 5), textcoords='offset points',
                                               fontsize=8, color='white',
                                               bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                                               ha='left', va='bottom', zorder=11)
            
            # 设置坐标轴范围
            self.scatter_ax.set_xlim(vs_min, vs_max)
            self.scatter_ax.set_ylim(vp_min, vp_max)
            
            # 设置标签和图例
            self.scatter_ax.set_xlabel('S-wave Velocity (km/s)', fontsize=12)
            self.scatter_ax.set_ylabel('P-wave Velocity (km/s)', fontsize=12)
            self.scatter_ax.set_title('Vp-Vs Scatter Plot - Rock Database', fontsize=14, fontweight='bold')
            self.scatter_ax.grid(True, alpha=0.3)
            self.scatter_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            # 更新canvas
            self.scatter_canvas.draw()
            
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to update scatter plot: {e}")
    
    def _add_polygon_average_to_scatter(self, polygon):
        """Add polygon average point to scatter plot (internal method)"""
        if self.grid_data is None or self.property_calculator is None:
            return
        
        if len(polygon) < 3:
            return
        
        try:
            x_coord = self.profile_extractor.x_coord
            z_coord = self.profile_extractor.z_coord
            
            # 获取多边形边界
            poly_x = [p[0] for p in polygon]
            poly_z = [p[1] for p in polygon]
            x_min, x_max = min(poly_x), max(poly_x)
            z_min, z_max = min(poly_z), max(poly_z)
            
            # 在边界框内采样
            x_coords = self.grid_data[x_coord].values
            z_coords = self.grid_data[z_coord].values
            
            # 检查点是否在多边形内
            from matplotlib.path import Path as MplPath
            poly_path = MplPath(polygon)
            
            vp_values = []
            vs_values = []
            for x in x_coords:
                if x_min <= x <= x_max:
                    for z in z_coords:
                        if z_min <= z <= z_max:
                            if poly_path.contains_point((x, z)):
                                try:
                                    vp, vs = self._get_point_vp_vs(float(x), float(z))
                                    if self._is_valid_vp_vs_pair(vp, vs):
                                        vp_values.append(vp)
                                        vs_values.append(vs)
                                except:
                                    pass
            
            if not vp_values:
                return
            
            vp = float(np.mean(vp_values))
            vs = float(np.mean(vs_values))
            
            # 添加到模型点列表
            self.model_points.append({
                'vp': vp,
                'vs': vs,
                'x': np.mean(poly_x),
                'z': np.mean(poly_z),
                'type': 'polygon_average',
                'n_points': len(vp_values)
            })
            
            # 更新散点图
            self._update_scatter_plot()
            self.log_result(f"  [Auto-added polygon average (n={len(vp_values)}) to scatter plot]")
            
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to add polygon average: {e}")

    def _sample_primary_velocity_at_point(self, x: float, z: float) -> float:
        """Sample current primary velocity grid value at a point."""
        x_coord = self.profile_extractor.x_coord
        z_coord = self.profile_extractor.z_coord
        velocity_var = self.property_calculator.velocity_var
        return PropertyCalculator._sample_dataset_value(
            self.grid_data,
            velocity_var,
            x_coord,
            z_coord,
            float(x),
            float(z),
        )

    def _get_point_vp_vs(self, x: float, z: float) -> Tuple[float, float]:
        """
        Resolve Vp/Vs at point.
        Priority: dual-model mode (real Vs) > legacy single-model conversion.
        """
        velocity_method = self.velocity_method_var.get()
        has_vs_model = (
            self.property_calculator is not None
            and getattr(self.property_calculator, 'vs_grid_data', None) is not None
        )
        if has_vs_model:
            vp, vs, _ = self.property_calculator.get_vp_vs_at_point(
                x,
                z,
                velocity_method=velocity_method,
            )
            return float(vp), float(vs)

        model_type = self.model_type_var.get()
        original_velocity = self._sample_primary_velocity_at_point(x, z)
        if model_type == 'vp':
            vp = original_velocity
            vs = calculate_vs(vp, method=velocity_method)
            return float(vp), float(vs)
        vs = original_velocity
        vp = calculate_vp_from_vs_brocher(vs)
        return float(vp), float(vs)

    @staticmethod
    def _is_valid_vp_vs_pair(vp: float, vs: float) -> bool:
        try:
            vp_f = float(vp)
            vs_f = float(vs)
        except (TypeError, ValueError):
            return False
        if (not np.isfinite(vp_f)) or (not np.isfinite(vs_f)) or vs_f <= 0.0 or vp_f <= 0.0:
            return False
        ratio = vp_f / vs_f
        return np.isfinite(ratio) and (1.0 <= ratio <= 3.5)

    def _log_vp_vs_coverage_summary(self, max_samples: int = 3000) -> None:
        """Log sampled Vp/Vs pair coverage diagnostics for current model."""
        if self.grid_data is None or self.property_calculator is None or self.profile_extractor is None:
            return
        if max_samples <= 0:
            return

        x_coord = self.profile_extractor.x_coord
        z_coord = self.profile_extractor.z_coord
        x_vals = np.asarray(self.grid_data[x_coord].values, dtype=float)
        z_vals = np.asarray(self.grid_data[z_coord].values, dtype=float)
        if x_vals.size == 0 or z_vals.size == 0:
            return

        total_cells = int(x_vals.size * z_vals.size)
        target_nx = max(1, min(x_vals.size, int(np.sqrt(max_samples))))
        target_nz = max(1, min(z_vals.size, int(max_samples // target_nx)))
        x_idx = np.unique(np.linspace(0, x_vals.size - 1, num=target_nx, dtype=int))
        z_idx = np.unique(np.linspace(0, z_vals.size - 1, num=target_nz, dtype=int))

        sampled = 0
        valid = 0
        invalid = 0
        non_finite = 0
        source_model = 0
        source_empirical = 0
        has_vs_model = getattr(self.property_calculator, 'vs_grid_data', None) is not None
        velocity_method = self.velocity_method_var.get()

        for iz in z_idx:
            for ix in x_idx:
                x = float(x_vals[ix])
                z = float(z_vals[iz])
                sampled += 1
                if has_vs_model:
                    vp, vs, source = self.property_calculator.get_vp_vs_at_point(
                        x,
                        z,
                        velocity_method=velocity_method,
                    )
                    if source == 'model':
                        source_model += 1
                    else:
                        source_empirical += 1
                else:
                    vp, vs = self._get_point_vp_vs(x, z)
                    source_empirical += 1

                if (not np.isfinite(vp)) or (not np.isfinite(vs)):
                    non_finite += 1
                    continue
                if self._is_valid_vp_vs_pair(vp, vs):
                    valid += 1
                else:
                    invalid += 1

        if sampled <= 0:
            return
        valid_pct = 100.0 * valid / sampled
        invalid_pct = 100.0 * invalid / sampled
        non_finite_pct = 100.0 * non_finite / sampled
        self.log_result(
            f"Vp/Vs coverage diagnostics (sampled {sampled}/{total_cells} cells):"
        )
        self.log_result(
            f"  valid pairs: {valid} ({valid_pct:.1f}%), "
            f"invalid ratio/range: {invalid} ({invalid_pct:.1f}%), "
            f"non-finite: {non_finite} ({non_finite_pct:.1f}%)"
        )
        if has_vs_model:
            model_pct = 100.0 * source_model / sampled
            empirical_pct = 100.0 * source_empirical / sampled
            self.log_result(
                f"  Vs source mix: model={source_model} ({model_pct:.1f}%), "
                f"empirical={source_empirical} ({empirical_pct:.1f}%)"
            )
    
    def add_points_to_scatter(self):
        """Add selected points to scatter plot"""
        if self.grid_data is None or self.property_calculator is None:
            messagebox.showwarning('Warning', 'Please load model first')
            return
        
        # 从PointSelector获取点（优先使用PointSelector中的点，因为它是最新的）
        points_to_add = []
        if self.point_selector is not None:
            points_to_add = self.point_selector.get_points()
        
        # 如果PointSelector没有点，使用self.selected_points
        if not points_to_add:
            points_to_add = self.selected_points
        
        # 过滤掉None值（如果有）
        points_to_add = [p for p in points_to_add if p is not None]
        
        if not points_to_add:
            messagebox.showwarning('Warning', 'No points selected. Please select points first.')
            return
        
        # 如果散点图窗口未打开，先打开它
        if self.scatter_window is None or not self.scatter_window.winfo_exists():
            self.show_vp_vs_scatter()
        
        # 获取分类器用于校正
        classifier = None
        if self.property_calculator and self.property_calculator.rock_classifier:
            classifier = self.property_calculator.rock_classifier.classifier
        
        added_count = 0
        # 获取点的索引（从PointSelector获取，如果没有则使用顺序索引）
        point_indices = []
        if self.point_selector is not None:
            # PointSelector的点顺序与get_points()返回的顺序一致
            point_indices = list(range(len(points_to_add)))
        else:
            # 如果没有PointSelector，使用顺序索引
            point_indices = list(range(len(points_to_add)))
        
        for idx, (x, z) in enumerate(points_to_add):
            try:
                # 获取速度值（使用profile_extractor中的坐标名称）
                vp_original, vs_original = self._get_point_vp_vs(x, z)
                if not self._is_valid_vp_vs_pair(vp_original, vs_original):
                    self.log_result(f"  Warning: Skipped point ({x:.2f}, {z:.2f}) due to invalid Vp/Vs")
                    continue

                # 统一温压口径（总压力 + 海水温度）
                pressure, temperature = self._compute_pt_for_velocity_correction(
                    x=x,
                    z=z,
                    vp_for_zone=vp_original,
                )
                
                # 校正到标准条件（200MPa, 25°C，使用经验公式库）
                vp_corrected = correct_velocity(
                    vp_original, pressure=pressure, temperature=temperature,
                    target_pressure=200.0, target_temperature=25.0,
                    is_s_wave=False
                )
                vs_corrected = correct_velocity(
                    vs_original, pressure=pressure, temperature=temperature,
                    target_pressure=200.0, target_temperature=25.0,
                    is_s_wave=True
                )
                
                # 添加到模型点列表（使用校正后的值，保存点的编号）
                point_number = point_indices[idx] + 1 if idx < len(point_indices) else added_count + 1
                self.model_points.append({
                    'vp': vp_corrected,
                    'vs': vs_corrected,
                    'x': x,
                    'z': z,
                    'type': 'point',
                    'point_number': point_number,  # 保存点的编号
                    'vp_original': vp_original,
                    'vs_original': vs_original,
                    'pressure': pressure,
                    'temperature': temperature
                })
                added_count += 1
            except Exception as e:
                self.log_result(f"  Warning: Failed to add point ({x:.2f}, {z:.2f}): {e}")
        
        # 更新散点图
        self._update_scatter_plot()
        self.log_result(f"  Added {added_count} points to scatter plot")
        if added_count > 0:
            self.log_result(f"    All points corrected to 200 MPa, 25°C before adding to scatter plot")
    
    def add_polygon_to_scatter(self):
        """Add polygon average point to scatter plot"""
        if self.grid_data is None or self.property_calculator is None:
            messagebox.showwarning('Warning', 'Please load model first')
            return
        
        if not self.selected_polygons:
            messagebox.showwarning('Warning', 'No polygons selected. Please select polygon first.')
            return
        
        # 如果散点图窗口未打开，先打开它
        if self.scatter_window is None or not self.scatter_window.winfo_exists():
            self.show_vp_vs_scatter()
        
        # 处理最后一个多边形
        polygon = self.selected_polygons[-1]
        if len(polygon) < 3:
            messagebox.showwarning('Warning', 'Polygon must have at least 3 vertices.')
            return
        
        try:
            # 在多边形内采样点并计算平均值
            x_coord = self.profile_extractor.x_coord
            z_coord = self.profile_extractor.z_coord
            
            # 获取多边形边界
            poly_x = [p[0] for p in polygon]
            poly_z = [p[1] for p in polygon]
            x_min, x_max = min(poly_x), max(poly_x)
            z_min, z_max = min(poly_z), max(poly_z)
            
            # 在边界框内采样
            x_coords = self.grid_data[x_coord].values
            z_coords = self.grid_data[z_coord].values
            
            # 检查点是否在多边形内
            from matplotlib.path import Path as MplPath
            poly_path = MplPath(polygon)
            
            vp_values = []
            vs_values = []
            for x in x_coords:
                if x_min <= x <= x_max:
                    for z in z_coords:
                        if z_min <= z <= z_max:
                            if poly_path.contains_point((x, z)):
                                try:
                                    vp, vs = self._get_point_vp_vs(float(x), float(z))
                                    if self._is_valid_vp_vs_pair(vp, vs):
                                        vp_values.append(vp)
                                        vs_values.append(vs)
                                except:
                                    pass
            
            if not vp_values:
                messagebox.showwarning('Warning', 'No valid points found in polygon.')
                return
            
            vp_original = float(np.mean(vp_values))
            vs_original = float(np.mean(vs_values))
            
            # 计算平均位置的温压（统一口径）
            avg_x = float(np.mean(poly_x))
            avg_z = float(np.mean(poly_z))
            pressure, temperature = self._compute_pt_for_velocity_correction(
                x=avg_x,
                z=avg_z,
                vp_for_zone=vp_original,
            )
            
            # 校正到标准条件（200MPa, 25°C，使用经验公式库）
            vp_corrected = correct_velocity(
                vp_original, pressure=pressure, temperature=temperature,
                target_pressure=200.0, target_temperature=25.0,
                is_s_wave=False
            )
            vs_corrected = correct_velocity(
                vs_original, pressure=pressure, temperature=temperature,
                target_pressure=200.0, target_temperature=25.0,
                is_s_wave=True
            )
            
            # 添加到模型点列表（使用校正后的值）
            self.model_points.append({
                'vp': vp_corrected,
                'vs': vs_corrected,
                'x': avg_x,
                'z': avg_z,
                'type': 'polygon_average',
                'n_points': len(vp_values),
                'vp_original': vp_original,
                'vs_original': vs_original,
                'pressure': pressure,
                'temperature': temperature
            })
            
            # 更新散点图
            self._update_scatter_plot()
            self.log_result(f"  Added polygon average point (n={len(vp_values)}) to scatter plot")
            self.log_result(f"    Original: Vp={vp_original:.2f}, Vs={vs_original:.2f} km/s (P={pressure:.1f} MPa, T={temperature:.1f}°C)")
            self.log_result(f"    Corrected: Vp={vp_corrected:.2f}, Vs={vs_corrected:.2f} km/s (200 MPa, 25°C)")
            
        except Exception as e:
            messagebox.showerror('Error', f'Failed to add polygon average: {str(e)}')
            import traceback
            traceback.print_exc()
    
    def add_polygon_samples_to_scatter(self):
        """Add uniformly sampled points from polygon to scatter plot"""
        if self.grid_data is None or self.property_calculator is None:
            messagebox.showwarning('Warning', 'Please load model first')
            return
        
        if not self.selected_polygons:
            messagebox.showwarning('Warning', 'No polygons selected. Please select polygon first.')
            return
        
        # 如果散点图窗口未打开，先打开它
        if self.scatter_window is None or not self.scatter_window.winfo_exists():
            self.show_vp_vs_scatter()
        
        # 询问采样数量
        from tkinter.simpledialog import askinteger
        n_samples = askinteger('Sample Count', 'Enter number of samples:', 
                              initialvalue=10, minvalue=1, maxvalue=100)
        if n_samples is None:
            return
        
        # 处理最后一个多边形
        polygon = self.selected_polygons[-1]
        if len(polygon) < 3:
            messagebox.showwarning('Warning', 'Polygon must have at least 3 vertices.')
            return
        
        try:
            x_coord = self.profile_extractor.x_coord
            z_coord = self.profile_extractor.z_coord
            
            # 获取多边形边界
            poly_x = [p[0] for p in polygon]
            poly_z = [p[1] for p in polygon]
            x_min, x_max = min(poly_x), max(poly_x)
            z_min, z_max = min(poly_z), max(poly_z)
            
            # 在多边形内均匀采样
            # 使用改进的采样策略：先创建密集网格，然后使用空间均匀分布选择
            from matplotlib.path import Path as MplPath
            poly_path = MplPath(polygon)
            
            # 计算多边形面积（用于确定采样密度）
            def polygon_area(vertices):
                """使用Shoelace公式计算多边形面积"""
                n = len(vertices)
                area = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    area += vertices[i][0] * vertices[j][1]
                    area -= vertices[j][0] * vertices[i][1]
                return abs(area) / 2.0
            
            poly_area = polygon_area(polygon)
            samples = []
            
            # 方法1：如果多边形面积较大，使用网格采样
            # 创建足够密集的网格以确保有足够的候选点
            x_range = x_max - x_min
            z_range = z_max - z_min
            
            # 计算网格大小：确保有足够的候选点（至少是所需采样数的3-5倍）
            if poly_area > 0:
                # 估算需要的网格点数
                target_grid_points = n_samples * 5
                # 计算网格分辨率（假设网格是方形的）
                grid_size = np.sqrt(poly_area / target_grid_points)
                # 限制网格大小，避免过小
                grid_size = max(grid_size, min(x_range, z_range) / 50.0)
            else:
                grid_size = min(x_range, z_range) / 20.0
            
            n_x = max(int(x_range / grid_size) + 1, 10)
            n_z = max(int(z_range / grid_size) + 1, 10)
            
            # 生成网格点（使用稍微偏移的起始点，避免边界问题）
            x_grid = np.linspace(x_min, x_max, n_x)
            z_grid = np.linspace(z_min, z_max, n_z)
            
            # 在网格点中筛选多边形内的点
            grid_points = []
            for x in x_grid:
                for z in z_grid:
                    if poly_path.contains_point((x, z)):
                        grid_points.append((x, z))
            
            # 如果网格点足够，使用空间均匀分布选择
            if len(grid_points) >= n_samples:
                # 使用改进的选择策略：尽量选择空间上均匀分布的点
                selected_points = []
                remaining_points = grid_points.copy()
                
                # 第一点：随机选择
                if remaining_points:
                    idx = np.random.randint(0, len(remaining_points))
                    selected_points.append(remaining_points.pop(idx))
                
                # 后续点：选择与已选点距离最远的点（最大化最小距离）
                while len(selected_points) < n_samples and remaining_points:
                    max_min_dist = -1
                    best_point = None
                    best_idx = -1
                    
                    for idx, candidate in enumerate(remaining_points):
                        # 计算候选点到所有已选点的最小距离
                        min_dist = float('inf')
                        for selected in selected_points:
                            dist = np.sqrt((candidate[0] - selected[0])**2 + 
                                         (candidate[1] - selected[1])**2)
                            min_dist = min(min_dist, dist)
                        
                        # 选择最小距离最大的点（最远离已选点）
                        if min_dist > max_min_dist:
                            max_min_dist = min_dist
                            best_point = candidate
                            best_idx = idx
                    
                    if best_point is not None:
                        selected_points.append(best_point)
                        remaining_points.pop(best_idx)
                    else:
                        break
                
                # 如果还没选够，随机补充
                if len(selected_points) < n_samples and remaining_points:
                    n_needed = n_samples - len(selected_points)
                    indices = np.random.choice(len(remaining_points), 
                                              min(n_needed, len(remaining_points)), 
                                              replace=False)
                    for idx in indices:
                        selected_points.append(remaining_points[idx])
            else:
                # 如果网格点不够，使用随机采样
                selected_points = grid_points.copy()
                remaining = n_samples - len(selected_points)
                
                max_attempts = remaining * 200
                attempts = 0
                while len(selected_points) < n_samples and attempts < max_attempts:
                    attempts += 1
                    x = np.random.uniform(x_min, x_max)
                    z = np.random.uniform(z_min, z_max)
                    if poly_path.contains_point((x, z)):
                        # 检查是否与已有点太近（避免聚集）
                        too_close = False
                        for existing in selected_points:
                            dist = np.sqrt((x - existing[0])**2 + (z - existing[1])**2)
                            if dist < grid_size * 0.5:  # 如果距离太近，跳过
                                too_close = True
                                break
                        if not too_close:
                            selected_points.append((x, z))
            
            # 获取采样点坐标
            for x, z in selected_points[:n_samples]:
                try:
                    samples.append((float(x), float(z)))
                except:
                    pass
            
            if not samples:
                messagebox.showwarning('Warning', 'No valid samples found in polygon.')
                return
            
            # 转换为vp和vs并添加到模型点列表，同时在主图上显示
            added_count = 0
            
            # 在主图上绘制采样点
            sample_x = []
            sample_z = []
            
            invalid_count = 0
            for x, z in samples:
                vp_original, vs_original = self._get_point_vp_vs(x, z)
                if not self._is_valid_vp_vs_pair(vp_original, vs_original):
                    invalid_count += 1
                    continue

                pressure, temperature = self._compute_pt_for_velocity_correction(
                    x=x,
                    z=z,
                    vp_for_zone=vp_original,
                )
                
                # 校正到标准条件（200MPa, 25°C，使用经验公式库）
                vp_corrected = correct_velocity(
                    vp_original, pressure=pressure, temperature=temperature,
                    target_pressure=200.0, target_temperature=25.0,
                    is_s_wave=False
                )
                vs_corrected = correct_velocity(
                    vs_original, pressure=pressure, temperature=temperature,
                    target_pressure=200.0, target_temperature=25.0,
                    is_s_wave=True
                )
                
                self.model_points.append({
                    'vp': vp_corrected,
                    'vs': vs_corrected,
                    'x': x,
                    'z': z,
                    'type': 'polygon_sample',
                    'vp_original': vp_original,
                    'vs_original': vs_original,
                    'pressure': pressure,
                    'temperature': temperature
                })
                sample_x.append(x)
                sample_z.append(z)
                added_count += 1
            
            # 在主图上绘制采样点（使用不同的标记和颜色以区分）
            if sample_x:
                scatter_artist = self.ax_main.scatter(sample_x, sample_z, 
                                                     c='orange', marker='s', s=50, 
                                                     edgecolors='darkorange', linewidths=1,
                                                     alpha=0.7, zorder=5,
                                                     label='Polygon Samples')
                self.polygon_sample_artists.append(scatter_artist)
                self.canvas.draw()
            
            # 更新散点图
            self._update_scatter_plot()
            self.log_result(f"  Added {added_count} polygon samples to scatter plot")
            if invalid_count > 0:
                self.log_result(f"  Warning: skipped {invalid_count} invalid Vp/Vs samples")
            self.log_result(f"  Sample points displayed on main plot (orange squares)")
            if added_count > 0:
                self.log_result(f"    All samples corrected to 200 MPa, 25°C before adding to scatter plot")
            
        except Exception as e:
            messagebox.showerror('Error', f'Failed to add polygon samples: {str(e)}')
            import traceback
            traceback.print_exc()
    
    def clear_scatter_model_points(self):
        """Clear all model points from scatter plot"""
        self.model_points = []
        if self.scatter_window is not None and self.scatter_window.winfo_exists():
            self._update_scatter_plot()
        self.log_result("  Cleared all model points from scatter plot")
    
    def add_points_to_vp_vs_ratio(self):
        """Add selected points to Vp/Vs vs Vp plot"""
        if self.grid_data is None or self.property_calculator is None:
            messagebox.showwarning('Warning', 'Please load model first')
            return
        
        # 从PointSelector获取点
        points_to_add = []
        if self.point_selector is not None:
            points_to_add = self.point_selector.get_points()
        
        if not points_to_add:
            points_to_add = self.selected_points
        
        points_to_add = [p for p in points_to_add if p is not None]
        
        if not points_to_add:
            messagebox.showwarning('Warning', 'No points selected. Please select points first.')
            return
        
        # 如果窗口未打开，先打开它
        if self.vp_vs_ratio_window is None or not self.vp_vs_ratio_window.winfo_exists():
            self.show_vp_vs_ratio_plot()
        
        # 获取分类器用于校正
        classifier = None
        if self.property_calculator and self.property_calculator.rock_classifier:
            classifier = self.property_calculator.rock_classifier.classifier
        
        added_count = 0
        # 获取点的索引（从PointSelector获取，如果没有则使用顺序索引）
        point_indices = []
        if self.point_selector is not None:
            # PointSelector的点顺序与get_points()返回的顺序一致
            point_indices = list(range(len(points_to_add)))
        else:
            # 如果没有PointSelector，使用顺序索引
            point_indices = list(range(len(points_to_add)))
        
        for idx, (x, z) in enumerate(points_to_add):
            try:
                # 获取速度值（使用profile_extractor中的坐标名称）
                vp_original, vs_original = self._get_point_vp_vs(x, z)
                if not self._is_valid_vp_vs_pair(vp_original, vs_original):
                    self.log_result(f"  Warning: Skipped point ({x:.2f}, {z:.2f}) due to invalid Vp/Vs")
                    continue

                pressure, temperature = self._compute_pt_for_velocity_correction(
                    x=x,
                    z=z,
                    vp_for_zone=vp_original,
                )
                
                # 校正到标准条件（200MPa, 25°C，使用经验公式库）
                vp_corrected = correct_velocity(
                    vp_original, pressure=pressure, temperature=temperature,
                    target_pressure=200.0, target_temperature=25.0,
                    is_s_wave=False
                )
                vs_corrected = correct_velocity(
                    vs_original, pressure=pressure, temperature=temperature,
                    target_pressure=200.0, target_temperature=25.0,
                    is_s_wave=True
                )
                
                # 添加到模型点列表（使用校正后的值，保存点的编号）
                point_number = point_indices[idx] + 1 if idx < len(point_indices) else added_count + 1
                self.vp_vs_ratio_model_points.append({
                    'vp': vp_corrected,
                    'vs': vs_corrected,
                    'x': x,
                    'z': z,
                    'type': 'point',
                    'point_number': point_number,  # 保存点的编号
                    'vp_original': vp_original,
                    'vs_original': vs_original,
                    'pressure': pressure,
                    'temperature': temperature
                })
                added_count += 1
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to add point ({x}, {z}): {e}")
        
        # 更新图表
        self._update_vp_vs_ratio_plot()
        self.log_result(f"  Added {added_count} points to Vp/Vs vs Vp plot")
        if added_count > 0:
            self.log_result(f"    All points corrected to 200 MPa, 25°C before adding to plot")
    
    def add_polygon_to_vp_vs_ratio(self):
        """Add polygon average to Vp/Vs vs Vp plot"""
        if self.grid_data is None or self.property_calculator is None:
            messagebox.showwarning('Warning', 'Please load model first')
            return
        
        if not self.selected_polygons:
            messagebox.showwarning('Warning', 'No polygon selected. Please select a polygon first.')
            return
        
        # 如果窗口未打开，先打开它
        if self.vp_vs_ratio_window is None or not self.vp_vs_ratio_window.winfo_exists():
            self.show_vp_vs_ratio_plot()
        
        polygon = self.selected_polygons[-1]  # 使用最后一个多边形
        
        if len(polygon) < 3:
            return
        
        try:
            x_coord = self.profile_extractor.x_coord
            z_coord = self.profile_extractor.z_coord
            
            # 获取多边形边界
            poly_x = [p[0] for p in polygon]
            poly_z = [p[1] for p in polygon]
            x_min, x_max = min(poly_x), max(poly_x)
            z_min, z_max = min(poly_z), max(poly_z)
            
            # 在边界框内采样
            x_coords = self.grid_data[x_coord].values
            z_coords = self.grid_data[z_coord].values
            
            # 检查点是否在多边形内
            from matplotlib.path import Path as MplPath
            poly_path = MplPath(polygon)
            
            vp_values = []
            vs_values = []
            for x in x_coords:
                if x_min <= x <= x_max:
                    for z in z_coords:
                        if z_min <= z <= z_max:
                            if poly_path.contains_point((x, z)):
                                try:
                                    vp, vs = self._get_point_vp_vs(float(x), float(z))
                                    if self._is_valid_vp_vs_pair(vp, vs):
                                        vp_values.append(vp)
                                        vs_values.append(vs)
                                except:
                                    pass
            
            if not vp_values:
                messagebox.showwarning('Warning', 'No valid points found in polygon.')
                return
            
            vp_original = float(np.mean(vp_values))
            vs_original = float(np.mean(vs_values))

            avg_x = float(np.mean(poly_x))
            avg_z = float(np.mean(poly_z))
            pressure, temperature = self._compute_pt_for_velocity_correction(
                x=avg_x,
                z=avg_z,
                vp_for_zone=vp_original,
            )
            
            # 校正到标准条件（200MPa, 25°C，使用经验公式库）
            vp_corrected = correct_velocity(
                vp_original, pressure=pressure, temperature=temperature,
                target_pressure=200.0, target_temperature=25.0,
                is_s_wave=False
            )
            vs_corrected = correct_velocity(
                vs_original, pressure=pressure, temperature=temperature,
                target_pressure=200.0, target_temperature=25.0,
                is_s_wave=True
            )
            
            # 添加到模型点列表（使用校正后的值）
            self.vp_vs_ratio_model_points.append({
                'vp': vp_corrected,
                'vs': vs_corrected,
                'x': avg_x,
                'z': avg_z,
                'type': 'polygon_average',
                'n_points': len(vp_values),
                'vp_original': vp_original,
                'vs_original': vs_original,
                'pressure': pressure,
                'temperature': temperature
            })
            
            # 更新图表
            self._update_vp_vs_ratio_plot()
            self.log_result(f"  Added polygon average point (n={len(vp_values)}) to Vp/Vs vs Vp plot")
            self.log_result(f"    Original: Vp={vp_original:.2f}, Vs={vs_original:.2f} km/s (P={pressure:.1f} MPa, T={temperature:.1f}°C)")
            self.log_result(f"    Corrected: Vp={vp_corrected:.2f}, Vs={vs_corrected:.2f} km/s (200 MPa, 25°C)")
            
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to add polygon average: {e}")
    
    def add_polygon_samples_to_vp_vs_ratio(self):
        """Add polygon samples to Vp/Vs vs Vp plot"""
        if self.grid_data is None or self.property_calculator is None:
            messagebox.showwarning('Warning', 'Please load model first')
            return
        
        if not self.selected_polygons:
            messagebox.showwarning('Warning', 'No polygon selected. Please select a polygon first.')
            return
        
        # 如果窗口未打开，先打开它
        if self.vp_vs_ratio_window is None or not self.vp_vs_ratio_window.winfo_exists():
            self.show_vp_vs_ratio_plot()
        
        # 询问采样数量
        from tkinter.simpledialog import askinteger
        n_samples = askinteger('Sample Count', 'Enter number of samples:', 
                              initialvalue=10, minvalue=1, maxvalue=100)
        if n_samples is None:
            return
        
        polygon = self.selected_polygons[-1]  # 使用最后一个多边形
        
        if len(polygon) < 3:
            messagebox.showwarning('Warning', 'Polygon must have at least 3 vertices.')
            return
        
        try:
            x_coord = self.profile_extractor.x_coord
            z_coord = self.profile_extractor.z_coord
            
            # 获取多边形边界
            poly_x = [p[0] for p in polygon]
            poly_z = [p[1] for p in polygon]
            x_min, x_max = min(poly_x), max(poly_x)
            z_min, z_max = min(poly_z), max(poly_z)
            
            # 计算多边形面积
            def polygon_area(vertices):
                n = len(vertices)
                area = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    area += vertices[i][0] * vertices[j][1]
                    area -= vertices[j][0] * vertices[i][1]
                return abs(area) / 2.0
            
            poly_area = polygon_area(polygon)
            samples = []
            
            # 创建网格采样
            from matplotlib.path import Path as MplPath
            poly_path = MplPath(polygon)
            
            x_range = x_max - x_min
            z_range = z_max - z_min
            
            if poly_area > 0:
                target_grid_points = n_samples * 5
                grid_size = np.sqrt(poly_area / target_grid_points)
                grid_size = max(grid_size, min(x_range, z_range) / 50.0)
            else:
                grid_size = min(x_range, z_range) / 20.0
            
            n_x = max(int(x_range / grid_size) + 1, 10)
            n_z = max(int(z_range / grid_size) + 1, 10)
            
            x_grid = np.linspace(x_min, x_max, n_x)
            z_grid = np.linspace(z_min, z_max, n_z)
            
            grid_points = []
            for x in x_grid:
                for z in z_grid:
                    if poly_path.contains_point((x, z)):
                        grid_points.append((x, z))
            
            # 使用最大化最小距离算法选择均匀分布的点
            selected_points = []
            remaining_points = grid_points.copy()
            
            # 如果网格点足够，使用空间均匀分布选择
            if len(grid_points) >= n_samples:
                # 第一点：随机选择
                if remaining_points:
                    idx = np.random.randint(0, len(remaining_points))
                    selected_points.append(remaining_points.pop(idx))
                
                # 后续点：选择与已选点距离最远的点（最大化最小距离）
                while len(selected_points) < n_samples and remaining_points:
                    max_min_dist = -1
                    best_point = None
                    best_idx = -1
                    
                    for idx, candidate in enumerate(remaining_points):
                        # 计算候选点到所有已选点的最小距离
                        min_dist = float('inf')
                        for selected in selected_points:
                            dist = np.sqrt((candidate[0] - selected[0])**2 + 
                                           (candidate[1] - selected[1])**2)
                            min_dist = min(min_dist, dist)
                        
                        # 选择最小距离最大的点（最远离已选点）
                        if min_dist > max_min_dist:
                            max_min_dist = min_dist
                            best_point = candidate
                            best_idx = idx
                    
                    if best_point is not None:
                        selected_points.append(best_point)
                        remaining_points.pop(best_idx)
                    else:
                        break
            
                # 如果还没选够，随机补充
                if len(selected_points) < n_samples and remaining_points:
                    n_needed = n_samples - len(selected_points)
                    indices = np.random.choice(
                        len(remaining_points),
                        min(n_needed, len(remaining_points)),
                        replace=False,
                    )
                    for idx in indices:
                        selected_points.append(remaining_points[idx])
            else:
                # 如果网格点不够，使用随机采样
                selected_points = grid_points.copy()
                remaining = n_samples - len(selected_points)
                
                max_attempts = remaining * 200
                attempts = 0
                while len(selected_points) < n_samples and attempts < max_attempts:
                    attempts += 1
                    x = np.random.uniform(x_min, x_max)
                    z = np.random.uniform(z_min, z_max)
                    if poly_path.contains_point((x, z)):
                        # 检查是否与已有点太近（避免聚集）
                        too_close = False
                        for existing in selected_points:
                            dist = np.sqrt((x - existing[0])**2 + (z - existing[1])**2)
                            if dist < grid_size * 0.5:  # 如果距离太近，跳过
                                too_close = True
                                break
                        if not too_close:
                            selected_points.append((x, z))
            
            # 获取采样点坐标
            for x, z in selected_points[:n_samples]:
                try:
                    samples.append((float(x), float(z)))
                except:
                    pass
            
            if not samples:
                messagebox.showwarning('Warning', 'No valid samples found in polygon.')
                return
            
            # 转换为vp和vs并添加到模型点列表，同时在主图上显示
            added_count = 0
            invalid_count = 0
            
            # 在主图上绘制采样点
            sample_x = []
            sample_z = []
            
            for x, z in samples:
                vp_original, vs_original = self._get_point_vp_vs(x, z)
                if not self._is_valid_vp_vs_pair(vp_original, vs_original):
                    invalid_count += 1
                    continue

                pressure, temperature = self._compute_pt_for_velocity_correction(
                    x=x,
                    z=z,
                    vp_for_zone=vp_original,
                )
                
                # 校正到标准条件（200MPa, 25°C，使用经验公式库）
                vp_corrected = correct_velocity(
                    vp_original, pressure=pressure, temperature=temperature,
                    target_pressure=200.0, target_temperature=25.0,
                    is_s_wave=False
                )
                vs_corrected = correct_velocity(
                    vs_original, pressure=pressure, temperature=temperature,
                    target_pressure=200.0, target_temperature=25.0,
                    is_s_wave=True
                )
                
                self.vp_vs_ratio_model_points.append({
                    'vp': vp_corrected,
                    'vs': vs_corrected,
                    'x': x,
                    'z': z,
                    'type': 'polygon_sample',
                    'vp_original': vp_original,
                    'vs_original': vs_original,
                    'pressure': pressure,
                    'temperature': temperature
                })
                sample_x.append(x)
                sample_z.append(z)
                added_count += 1
            
            # 在主图上绘制采样点（使用不同的标记和颜色以区分）
            if sample_x:
                scatter_artist = self.ax_main.scatter(sample_x, sample_z, 
                                                     c='orange', marker='s', s=50, 
                                                     edgecolors='darkorange', linewidths=1,
                                                     alpha=0.7, zorder=5,
                                                     label='Polygon Samples')
                self.polygon_sample_artists.append(scatter_artist)
                self.canvas.draw()
            
            # 更新图表
            self._update_vp_vs_ratio_plot()
            self.log_result(f"  Added {added_count} polygon samples to Vp/Vs vs Vp plot")
            if invalid_count > 0:
                self.log_result(f"  Warning: skipped {invalid_count} invalid Vp/Vs samples")
            self.log_result(f"  Sample points displayed on main plot (orange squares)")
            if added_count > 0:
                self.log_result(f"    All samples corrected to 200 MPa, 25°C before adding to plot")
            
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to add polygon samples: {e}")
    
    def clear_vp_vs_ratio_model_points(self):
        """Clear all model points from Vp/Vs vs Vp plot"""
        self.vp_vs_ratio_model_points = []
        if self.vp_vs_ratio_window is not None and self.vp_vs_ratio_window.winfo_exists():
            self._update_vp_vs_ratio_plot()
        self.log_result("  Cleared all model points from Vp/Vs vs Vp plot")
    
    # ============================================================================
    # Gravity Simulation Functions
    # ============================================================================
    def save_scatter_figure(self):
        """Save scatter plot figure"""
        if self.scatter_fig is None:
            messagebox.showwarning('Warning', 'No scatter plot to save.')
            return
        
        filename = filedialog.asksaveasfilename(
            title='Save Scatter Plot',
            # 不设置defaultextension，让文件对话框根据用户选择的文件类型自动添加扩展名
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
            import os
            base_name, ext = os.path.splitext(filename)
            # 支持的文件扩展名列表
            supported_extensions = ['.png', '.pdf', '.ps', '.eps', '.jpg', '.jpeg', '.tif', '.tiff']
            # 如果文件名没有扩展名，默认使用PNG格式
            if not ext:
                filename = filename + '.png'
            # 如果扩展名不在支持列表中，替换为PNG
            elif ext.lower() not in supported_extensions:
                filename = base_name + '.png'
            
            self.scatter_fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.log_result(f"Scatter plot saved to: {filename}")
