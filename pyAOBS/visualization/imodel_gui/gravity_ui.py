"""重力图与整块模型重力计算。"""

from __future__ import annotations

from .deps import *  # noqa: F403


class GravityUIMixin:
    def _get_selected_full_model_gravity_method(self) -> str:
        method_var = getattr(self, 'gravity_method_var', None)
        method = str(method_var.get()).strip().lower() if method_var is not None else 'tomo2d_fft'
        if method not in {'tomo2d_fft', 'talwani_grid'}:
            method = 'tomo2d_fft'
            if method_var is not None:
                method_var.set(method)
        return method

    @staticmethod
    def _normalize_density_method_name(method: str) -> str:
        return str(method or 'gardner').strip().lower()

    def _uses_tomo2d_sediment_density(self, method: str) -> bool:
        try:
            return PropertyCalculator.uses_tomo2d_sediment_density(method)
        except Exception:
            return self._normalize_density_method_name(method) in {
                'tomo2d_sediment', 'sed_tomo2d', 'tomo2d-sediment'
            }

    def _density_method_for_library(self, method: str) -> str:
        m = self._normalize_density_method_name(method)
        if self._uses_tomo2d_sediment_density(m):
            return 'gardner'
        return m

    def _is_strict_jgrav_enabled(self) -> bool:
        return bool(getattr(self, 'strict_jgrav_var', None) and self.strict_jgrav_var.get())
    
    def show_gravity_plot(self):
        """Show gravity anomaly plot window"""
        if self.gravity_window is not None and self.gravity_window.winfo_exists():
            self.gravity_window.lift()
            return
        
        # 创建新窗口
        self.gravity_window = tk.Toplevel(self.root)
        self.gravity_window.title('Gravity Anomaly Plot')
        self.gravity_window.geometry('1000x700+200+100')
        
        # 创建matplotlib图形
        self.gravity_fig = plt.figure(figsize=(10, 7))
        self.gravity_ax = self.gravity_fig.add_subplot(111)
        
        # 嵌入到Tkinter窗口
        self.gravity_canvas = FigureCanvasTkAgg(self.gravity_fig, self.gravity_window)
        self.gravity_canvas.draw()
        self.gravity_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        toolbar = NavigationToolbar2Tk(self.gravity_canvas, self.gravity_window)
        toolbar.update()
        
        # 初始化图形
        self._update_gravity_plot()
        
        self.log_result("Gravity anomaly plot window opened")
    
    def _update_gravity_plot(self):
        """Update gravity anomaly plot"""
        if self.gravity_fig is None or self.gravity_ax is None:
            return
        
        self.gravity_ax.clear()
        
        if not self.gravity_bodies:
            self.gravity_ax.text(0.5, 0.5, 'No gravity bodies defined.\nAdd polygons as gravity bodies first.',
                                ha='center', va='center', transform=self.gravity_ax.transAxes,
                                fontsize=12, color='gray')
            self.gravity_ax.set_xlabel('Distance (km)', fontsize=12)
            self.gravity_ax.set_ylabel('Gravity Anomaly (mGal)', fontsize=12)
            self.gravity_ax.set_title('Gravity Anomaly Plot', fontsize=14, fontweight='bold')
            self.gravity_ax.grid(True, alpha=0.3)
            self.gravity_canvas.draw()
            return
        
        # 如果有重力体，显示提示信息
        self.gravity_ax.text(0.5, 0.5, f'{len(self.gravity_bodies)} gravity body/bodies defined.\nUse "Calculate Profile" to compute gravity anomaly.',
                            ha='center', va='center', transform=self.gravity_ax.transAxes,
                            fontsize=12, color='blue')
        self.gravity_ax.set_xlabel('Distance (km)', fontsize=12)
        self.gravity_ax.set_ylabel('Gravity Anomaly (mGal)', fontsize=12)
        self.gravity_ax.set_title('Gravity Anomaly Plot', fontsize=14, fontweight='bold')
        self.gravity_ax.grid(True, alpha=0.3)
        self.gravity_canvas.draw()
    
    def add_polygon_as_gravity_body(self):
        """Add selected polygon as gravity body"""
        if not TALWANI_AVAILABLE:
            messagebox.showerror('Error', 'Talwani gravity module not available.')
            return
        
        if not self.selected_polygons:
            messagebox.showwarning('Warning', 'No polygon selected. Please select a polygon first.')
            return
        
        # 使用最后一个多边形
        polygon = self.selected_polygons[-1]
        if len(polygon) < 3:
            messagebox.showwarning('Warning', 'Polygon must have at least 3 vertices.')
            return
        
        # 获取多边形内的平均密度
        try:
            x_coord = self.profile_extractor.x_coord
            z_coord = self.profile_extractor.z_coord
            velocity_var = self.property_calculator.velocity_var
            
            # 获取多边形边界
            poly_x = [p[0] for p in polygon]
            poly_z = [p[1] for p in polygon]
            x_min, x_max = min(poly_x), max(poly_x)
            z_min, z_max = min(poly_z), max(poly_z)
            
            # 在多边形内采样点
            from matplotlib.path import Path as MplPath
            poly_path = MplPath(polygon)
            
            x_coords = self.grid_data[x_coord].values
            z_coords = self.grid_data[z_coord].values
            
            velocities = []
            for x in x_coords:
                if x_min <= x <= x_max:
                    for z in z_coords:
                        if z_min <= z <= z_max:
                            if poly_path.contains_point((x, z)):
                                try:
                                    vel = float(self.grid_data[velocity_var].sel(
                                        {x_coord: x, z_coord: z}, method='nearest'
                                    ).values)
                                    velocities.append(vel)
                                except:
                                    pass
            
            if not velocities:
                messagebox.showwarning('Warning', 'Could not sample velocities in polygon.')
                return
            
            # 计算平均速度和密度
            avg_velocity = np.mean(velocities)
            model_type = self.model_type_var.get()
            density_method = self.density_method_var.get()
            
            if model_type == 'vp':
                vp = avg_velocity
                vs = calculate_vs(vp, method=velocity_method)
            else:
                vs = avg_velocity
                if velocity_method == 'brocher':
                    vp = calculate_vp_from_vs_brocher(vs)
                else:
                    vp = calculate_vp_from_vs_brocher(vs)
            
            if self._uses_tomo2d_sediment_density(density_method):
                avg_density = PropertyCalculator.tomo2d_sediment_density_from_vp(vp)
            else:
                avg_density = calculate_density(vp, method=self._density_method_for_library(density_method))  # g/cm³
            
            # 获取背景密度
            try:
                bg_density = float(self.background_density_var.get())  # g/cm³
            except:
                bg_density = 2.67  # 默认值
            
            # 计算密度对比度
            density_contrast = avg_density - bg_density  # g/cm³
            density_contrast_kg_m3 = convert_density_units(density_contrast, 'g/cm3', 'kg/m3')
            
            # 将多边形坐标转换为米（Talwani需要米为单位）
            poly_x_m = np.array(poly_x) * 1000.0  # km to m
            poly_z_m = np.array(poly_z) * 1000.0  # km to m
            
            # 添加到重力体列表
            self.gravity_bodies.append((poly_x_m, poly_z_m, density_contrast_kg_m3))
            
            self.log_result(f"\n{'='*50}")
            self.log_result(f"Added polygon as gravity body:")
            self.log_result(f"  Vertices: {len(polygon)}")
            self.log_result(f"  Average Vp: {vp:.2f} km/s")
            self.log_result(f"  Average Vs: {vs:.2f} km/s")
            self.log_result(f"  Average Density: {avg_density:.3f} g/cm³")
            self.log_result(f"  Background Density: {bg_density:.3f} g/cm³")
            self.log_result(f"  Density Contrast: {density_contrast:.3f} g/cm³ ({density_contrast_kg_m3:.1f} kg/m³)")
            self.log_result(f"  Total bodies: {len(self.gravity_bodies)}")
            
            # 更新图形
            if self.gravity_window is not None and self.gravity_window.winfo_exists():
                self._update_gravity_plot()
        
        except Exception as e:
            messagebox.showerror('Error', f'Failed to add polygon as gravity body: {str(e)}')
            import traceback
            traceback.print_exc()
    
    def calculate_gravity_profile(self):
        """Calculate gravity anomaly profile"""
        if not TALWANI_AVAILABLE:
            messagebox.showerror('Error', 'Talwani gravity module not available.')
            return
        
        if not self.gravity_bodies:
            messagebox.showwarning('Warning', 'No gravity bodies defined. Add polygons as gravity bodies first.')
            return
        
        # 获取观测点设置
        try:
            obs_level = float(self.obs_level_var.get())  # km
            obs_level_m = obs_level * 1000.0  # 转换为米
        except:
            obs_level = 0.0
            obs_level_m = 0.0
        
        # 从模型获取x坐标范围
        x_coord = self.profile_extractor.x_coord
        x_coords = self.grid_data[x_coord].values  # km
        x_min, x_max = x_coords.min(), x_coords.max()
        
        # 创建观测点数组（沿x方向，在地表）
        # 使用分箱中心，避免落在边界顶点上（Talwani端点奇点）
        n_obs_profile = 200
        x_obs_edges = np.linspace(x_min, x_max, n_obs_profile + 1)
        x_obs_km = 0.5 * (x_obs_edges[:-1] + x_obs_edges[1:])
        x_obs_m = x_obs_km * 1000.0  # 转换为米
        
        # 计算每个观测点的重力异常
        gravity_anomaly = np.zeros_like(x_obs_m)
        
        for i, x0_m in enumerate(x_obs_m):
            try:
                # 计算所有重力体的总贡献
                bodies_list = [(x, z, rho) for x, z, rho in self.gravity_bodies]
                gravity_anomaly[i] = talwani2d_gravity_multibody(
                    bodies_list, x0_m, obs_level_m, check_vertex=False
                )
            except Exception as e:
                gravity_anomaly[i] = np.nan
                if i == 0:  # 只警告一次
                    warnings.warn(f"Error calculating gravity at some points: {e}")
        
        # 更新图形
        if self.gravity_window is None or not self.gravity_window.winfo_exists():
            self.show_gravity_plot()
        
        self.gravity_ax.clear()
        self.gravity_ax.plot(x_obs_km, gravity_anomaly, 'b-', linewidth=2, label='Gravity Anomaly')
        self.gravity_ax.set_xlabel('Distance (km)', fontsize=12)
        self.gravity_ax.set_ylabel('Gravity Anomaly (mGal)', fontsize=12)
        self.gravity_ax.set_title(f'Gravity Anomaly Profile (Observation Level: {obs_level:.2f} km)', 
                                 fontsize=14, fontweight='bold')
        self.gravity_ax.grid(True, alpha=0.3)
        self.gravity_ax.legend(fontsize=10)
        
        # 添加零线
        self.gravity_ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        
        self.gravity_canvas.draw()
        
        self.log_result(f"\n{'='*50}")
        self.log_result(f"Gravity anomaly profile calculated:")
        self.log_result(f"  Observation level: {obs_level:.2f} km")
        self.log_result(f"  Number of bodies: {len(self.gravity_bodies)}")
        self.log_result(f"  Number of observation points: {len(x_obs_km)}")
        self.log_result(f"  Gravity range: {np.nanmin(gravity_anomaly):.2f} to {np.nanmax(gravity_anomaly):.2f} mGal")

    def _build_full_model_cache_key(self) -> str:
        """Build cache key for full-model gravity result."""
        file_key = str(getattr(self, 'current_model_file', '') or '').strip()
        if file_key:
            try:
                p = Path(file_key)
                st = p.stat()
                return f"file:{file_key}:{int(st.st_mtime_ns)}:{int(st.st_size)}"
            except Exception:
                return f"file:{file_key}"
        ds = self.grid_data
        if ds is None:
            return "none"
        dims_key = "|".join(f"{k}:{int(v)}" for k, v in sorted(ds.sizes.items()))
        var_key = ",".join(sorted(list(ds.data_vars)))
        return f"inmem:{id(ds)}:{dims_key}:{var_key}"

    @staticmethod
    def _build_full_model_param_signature(
        *,
        background_density: float,
        obs_level: float,
        max_grid_size: int,
        extension_dist: float,
        density_method: str,
        gravity_method: str,
        strict_jgrav: bool,
    ) -> str:
        """Build deterministic signature for full-model gravity parameters."""
        payload = {
            "background_density": float(background_density),
            "obs_level": float(obs_level),
            "max_grid_size": int(max_grid_size),
            "extension_dist": float(extension_dist),
            "density_method": str(density_method or ""),
            "gravity_method": str(gravity_method or ""),
            "strict_jgrav": bool(strict_jgrav),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    def _render_full_model_gravity_result(
        self,
        *,
        x_obs_km: np.ndarray,
        gravity_anomaly: np.ndarray,
        obs_level: float,
        gravity_method: str,
        model_shape: tuple[int, int] | None = None,
        from_cache: bool = False,
        anomaly_note: str = "",
    ) -> None:
        """Render full-model gravity curve and log summary."""
        if self.gravity_window is None or not self.gravity_window.winfo_exists():
            self.show_gravity_plot()

        self.gravity_ax.clear()
        line_label = f'Gravity Anomaly (Full Model, {gravity_method})'
        if anomaly_note:
            line_label = f'Gravity Anomaly (Full Model, {gravity_method}, {anomaly_note})'
        self.gravity_ax.plot(x_obs_km, gravity_anomaly, 'b-', linewidth=2, label=line_label)
        self.gravity_ax.set_xlabel('Distance (km)', fontsize=12)
        self.gravity_ax.set_ylabel('Gravity Anomaly (mGal)', fontsize=12)
        cache_suffix = " [cached]" if from_cache else ""
        self.gravity_ax.set_title(
            f'Gravity Anomaly from Full Model (Observation Level: {obs_level:.2f} km){cache_suffix}',
            fontsize=14,
            fontweight='bold',
        )
        self.gravity_ax.grid(True, alpha=0.3)
        self.gravity_ax.legend(fontsize=10)
        self.gravity_ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        self.gravity_canvas.draw()

        self.log_result(f"\n{'='*50}")
        if from_cache:
            self.log_result("Reusing cached gravity anomaly from full model (model unchanged).")
        else:
            self.log_result("Gravity anomaly from full model calculated:")
        if model_shape is not None:
            self.log_result(f"  Model grid: {model_shape[0]} x {model_shape[1]}")
        self.log_result(f"  Gravity method: {gravity_method}")
        self.log_result(f"  Number of observation points: {len(x_obs_km)}")
        self.log_result(f"  Gravity range: {np.nanmin(gravity_anomaly):.2f} to {np.nanmax(gravity_anomaly):.2f} mGal")
        self.log_result(f"  Mean gravity: {np.nanmean(gravity_anomaly):.2f} mGal")
        self.log_result(f"  Std gravity: {np.nanstd(gravity_anomaly):.2f} mGal")
        if anomaly_note:
            self.log_result(f"  Anomaly display mode: {anomaly_note}")

    def _render_full_model_gravity_compare_result(
        self,
        *,
        x_obs_km: np.ndarray,
        gravity_tomo2d: np.ndarray,
        gravity_talwani: np.ndarray,
        gravity_tomo2d_aligned: np.ndarray,
        gravity_talwani_aligned: np.ndarray,
        obs_level: float,
        strict_jgrav: bool,
        alignment_note: str = "",
    ) -> None:
        if self.gravity_window is None or not self.gravity_window.winfo_exists():
            self.show_gravity_plot()

        x_obs_km = np.asarray(x_obs_km, dtype=float)
        g_tomo = np.asarray(gravity_tomo2d, dtype=float)
        g_tal = np.asarray(gravity_talwani, dtype=float)
        g_tomo_a = np.asarray(gravity_tomo2d_aligned, dtype=float)
        g_tal_a = np.asarray(gravity_talwani_aligned, dtype=float)
        diff = g_tomo_a - g_tal_a

        self.gravity_ax.clear()
        self.gravity_ax.plot(x_obs_km, g_tomo_a, 'b-', linewidth=2, label='tomo2d_fft (aligned)')
        self.gravity_ax.plot(x_obs_km, g_tal_a, 'r-', linewidth=2, alpha=0.85, label='talwani_grid (aligned)')
        self.gravity_ax.plot(x_obs_km, diff, 'k--', linewidth=1.3, alpha=0.9, label='difference (aligned)')
        self.gravity_ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.8)
        self.gravity_ax.set_xlabel('Distance (km)', fontsize=12)
        self.gravity_ax.set_ylabel('Gravity Anomaly (mGal)', fontsize=12)
        title_suffix = f" | {alignment_note}" if alignment_note else ""
        self.gravity_ax.set_title(
            f'Full-Model Gravity Method Comparison (Observation Level: {obs_level:.2f} km){title_suffix}',
            fontsize=14,
            fontweight='bold',
        )
        self.gravity_ax.grid(True, alpha=0.3)
        self.gravity_ax.legend(fontsize=10)
        self.gravity_canvas.draw()

        valid = np.isfinite(g_tomo) & np.isfinite(g_tal)
        raw_diff = g_tomo - g_tal
        rmse_raw = float(np.sqrt(np.mean((raw_diff[valid]) ** 2))) if np.any(valid) else np.nan
        max_abs_raw = float(np.nanmax(np.abs(raw_diff[valid]))) if np.any(valid) else np.nan
        corr_raw = float(np.corrcoef(g_tomo[valid], g_tal[valid])[0, 1]) if np.sum(valid) > 2 else np.nan

        valid_a = np.isfinite(g_tomo_a) & np.isfinite(g_tal_a)
        rmse = float(np.sqrt(np.mean((diff[valid_a]) ** 2))) if np.any(valid_a) else np.nan
        max_abs = float(np.nanmax(np.abs(diff[valid_a]))) if np.any(valid_a) else np.nan
        corr = float(np.corrcoef(g_tomo_a[valid_a], g_tal_a[valid_a])[0, 1]) if np.sum(valid_a) > 2 else np.nan
        self.log_result(f"\n{'='*50}")
        self.log_result("Compared full-model gravity methods:")
        self.log_result(f"  Strict jgrav mode: {'ON' if strict_jgrav else 'OFF'}")
        self.log_result(f"  Observation points: {len(x_obs_km)}")
        self.log_result(f"  Raw tomo2d range: {np.nanmin(g_tomo):.2f} to {np.nanmax(g_tomo):.2f} mGal")
        self.log_result(f"  Raw talwani range: {np.nanmin(g_tal):.2f} to {np.nanmax(g_tal):.2f} mGal")
        self.log_result(f"  Raw difference RMSE: {rmse_raw:.2f} mGal")
        self.log_result(f"  Raw difference max|.|: {max_abs_raw:.2f} mGal")
        self.log_result(f"  Raw curve correlation: {corr_raw:.4f}")
        if alignment_note:
            self.log_result(f"  Alignment: {alignment_note}")
        self.log_result(f"  Aligned difference RMSE: {rmse:.2f} mGal")
        self.log_result(f"  Aligned difference max|.|: {max_abs:.2f} mGal")
        self.log_result(f"  Aligned curve correlation: {corr:.4f}")

    @staticmethod
    def _shared_reference_alignment(
        x_obs_km: np.ndarray,
        curve_a: np.ndarray,
        curve_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """
        在相同空间参考窗内对两条曲线做统一低频校正（线性边缘基线）。
        """
        x = np.asarray(x_obs_km, dtype=float)
        a = np.asarray(curve_a, dtype=float).copy()
        b = np.asarray(curve_b, dtype=float).copy()
        n = int(min(len(x), len(a), len(b)))
        if n < 6:
            return a, b, "no-alignment (insufficient points)"

        edge_n = max(3, min(12, n // 8))
        t = np.linspace(0.0, 1.0, num=n, dtype=float)

        def _align_one(y: np.ndarray) -> np.ndarray:
            y2 = np.asarray(y[:n], dtype=float).copy()
            left_mean = float(np.nanmean(y2[:edge_n]))
            right_mean = float(np.nanmean(y2[-edge_n:]))
            baseline = left_mean + (right_mean - left_mean) * t
            out = y2 - baseline
            return out

        a_out = _align_one(a)
        b_out = _align_one(b)
        note = f"shared-edge linear baseline removed (edge_n={edge_n})"
        return a_out, b_out, note

    def _run_synthetic_gravity_benchmark(self, strict_jgrav: bool) -> None:
        """
        合成矩形体 benchmark：定位两算法在统一目标上的基线差异。
        """
        try:
            x_coords = np.linspace(0.0, 200.0, 81)   # km
            z_coords = np.linspace(0.0, 60.0, 61)    # km
            dens = np.zeros((len(z_coords), len(x_coords)), dtype=float)

            rect_mask = (
                (x_coords[np.newaxis, :] >= 70.0)
                & (x_coords[np.newaxis, :] <= 130.0)
                & (z_coords[:, np.newaxis] >= 10.0)
                & (z_coords[:, np.newaxis] <= 35.0)
            )
            dens[rect_mask] = 320.0  # kg/m^3
            sig = np.abs(dens) > 1e-9

            x_obs = np.linspace(0.5, 199.5, 100)  # km

            g_tomo = self.gravity_calculator.calculate_tomo2d_fft_anomaly(
                x_coords_km=x_coords,
                z_coords_km=z_coords,
                density_contrast_kg_m3=dens,
                significant_mask=sig,
                x_obs_km=x_obs,
                obs_level_km=0.0,
                ref_x_range_km=(float(x_obs[0]), float(x_obs[-1])),
            )
            g_tal = self.gravity_calculator.calculate_talwani_grid_anomaly(
                x_coords_km=x_coords,
                z_coords_km=z_coords,
                density_contrast_kg_m3=dens,
                significant_mask=sig,
                x_obs_km=x_obs,
                obs_level_km=0.0,
            )

            valid = np.isfinite(g_tomo) & np.isfinite(g_tal)
            diff_raw = g_tomo - g_tal
            rmse_raw = float(np.sqrt(np.mean((diff_raw[valid]) ** 2))) if np.any(valid) else np.nan
            corr_raw = float(np.corrcoef(g_tomo[valid], g_tal[valid])[0, 1]) if np.sum(valid) > 2 else np.nan

            g_tomo_a, g_tal_a, note = self._shared_reference_alignment(x_obs, g_tomo, g_tal)
            valid_a = np.isfinite(g_tomo_a) & np.isfinite(g_tal_a)
            diff_a = g_tomo_a - g_tal_a
            rmse_a = float(np.sqrt(np.mean((diff_a[valid_a]) ** 2))) if np.any(valid_a) else np.nan
            corr_a = float(np.corrcoef(g_tomo_a[valid_a], g_tal_a[valid_a])[0, 1]) if np.sum(valid_a) > 2 else np.nan

            self.log_result("  Synthetic benchmark (single rectangle, +320 kg/m³):")
            self.log_result(f"    Strict jgrav mode: {'ON' if strict_jgrav else 'OFF'}")
            self.log_result(f"    Raw RMSE: {rmse_raw:.2f} mGal, Raw corr: {corr_raw:.4f}")
            self.log_result(f"    Aligned RMSE: {rmse_a:.2f} mGal, Aligned corr: {corr_a:.4f}")
            self.log_result(f"    Alignment used: {note}")
        except Exception as exc:
            self.log_result(f"  Synthetic benchmark failed: {exc}")

    def compare_full_model_gravity_methods(self):
        """Compute and compare tomo2d_fft vs talwani_grid on full model."""
        if self.grid_data is None:
            messagebox.showwarning('Warning', 'No model loaded. Please open a model first.')
            return
        if not TALWANI_AVAILABLE:
            messagebox.showerror('Error', 'Talwani gravity module not available.')
            return

        method_var = getattr(self, 'gravity_method_var', None)
        if method_var is None:
            messagebox.showerror('Error', 'Gravity method selector is not initialized.')
            return

        strict_jgrav = self._is_strict_jgrav_enabled()
        original_method = self._get_selected_full_model_gravity_method()
        cache_tomo = None
        cache_tal = None
        try:
            self.log_result(f"\n{'='*50}")
            self.log_result("Starting full-model method comparison...")
            self.log_result("  Step 1/2: tomo2d_fft")
            method_var.set('tomo2d_fft')
            self.calculate_gravity_from_full_model()
            if isinstance(self._full_model_gravity_cache, dict):
                cache_tomo = dict(self._full_model_gravity_cache)

            self.log_result("  Step 2/2: talwani_grid")
            method_var.set('talwani_grid')
            self.calculate_gravity_from_full_model()
            if isinstance(self._full_model_gravity_cache, dict):
                cache_tal = dict(self._full_model_gravity_cache)
        finally:
            method_var.set(original_method)

        if not isinstance(cache_tomo, dict) or not isinstance(cache_tal, dict):
            messagebox.showerror('Error', 'Failed to collect results for method comparison.')
            return

        x_tomo = np.asarray(cache_tomo.get('x_obs_km', []), dtype=float)
        g_tomo = np.asarray(cache_tomo.get('gravity_anomaly', []), dtype=float)
        x_tal = np.asarray(cache_tal.get('x_obs_km', []), dtype=float)
        g_tal = np.asarray(cache_tal.get('gravity_anomaly', []), dtype=float)
        obs_level = float(cache_tomo.get('obs_level', cache_tal.get('obs_level', 0.0)))

        if x_tomo.size < 2 or g_tomo.size != x_tomo.size or x_tal.size < 2 or g_tal.size != x_tal.size:
            messagebox.showerror('Error', 'Comparison data is invalid or incomplete.')
            return

        if x_tal.size != x_tomo.size or not np.allclose(x_tal, x_tomo, atol=1e-8, rtol=1e-6):
            g_tal = np.interp(x_tomo, x_tal, g_tal, left=np.nan, right=np.nan)

        g_tomo_aligned, g_tal_aligned, alignment_note = self._shared_reference_alignment(
            x_tomo, g_tomo, g_tal
        )

        self._render_full_model_gravity_compare_result(
            x_obs_km=x_tomo,
            gravity_tomo2d=g_tomo,
            gravity_talwani=g_tal,
            gravity_tomo2d_aligned=g_tomo_aligned,
            gravity_talwani_aligned=g_tal_aligned,
            obs_level=obs_level,
            strict_jgrav=strict_jgrav,
            alignment_note=alignment_note,
        )
        self._run_synthetic_gravity_benchmark(strict_jgrav=strict_jgrav)
    
    def calculate_gravity_from_full_model(self):
        """Calculate gravity anomaly from the entire velocity model"""
        if self.grid_data is None:
            messagebox.showwarning('Warning', 'No model loaded. Please open a model first.')
            return

        gravity_method = self._get_selected_full_model_gravity_method()
        strict_jgrav = self._is_strict_jgrav_enabled()
        if gravity_method == 'talwani_grid' and not TALWANI_AVAILABLE:
            messagebox.showerror('Error', 'Talwani gravity module not available.')
            return

        # 先解析关键参数；这些参数变化时必须强制重算而不是复用缓存
        try:
            bg_density = float(self.background_density_var.get())  # g/cm³
        except Exception:
            bg_density = 2.67

        try:
            obs_level = float(self.obs_level_var.get())  # km
        except Exception:
            obs_level = 0.0

        try:
            max_grid_size = int(self.max_grid_size_var.get())
            if max_grid_size < 10:
                max_grid_size = 10
                self.max_grid_size_var.set('10')
            elif max_grid_size > 500:
                max_grid_size = 500
                self.max_grid_size_var.set('500')
        except Exception:
            max_grid_size = 100
            self.max_grid_size_var.set('100')

        try:
            extension_dist = float(self.extension_dist_var.get())  # km
            if extension_dist < 0:
                extension_dist = 0.0
                self.extension_dist_var.set('0.0')
            elif extension_dist > 200:
                extension_dist = 200.0
                self.extension_dist_var.set('200.0')
        except Exception:
            extension_dist = 40.0
            self.extension_dist_var.set('40.0')

        density_method = self.density_method_var.get()
        param_signature = self._build_full_model_param_signature(
            background_density=bg_density,
            obs_level=obs_level,
            max_grid_size=max_grid_size,
            extension_dist=extension_dist,
            density_method=density_method,
            gravity_method=gravity_method,
            strict_jgrav=strict_jgrav,
        )

        model_key = self._build_full_model_cache_key()
        cache = self._full_model_gravity_cache if isinstance(self._full_model_gravity_cache, dict) else None
        cache_version = 6
        if (
            cache is not None
            and cache.get('model_key') == model_key
            and cache.get('param_signature') == param_signature
            and int(cache.get('cache_version', 0)) == cache_version
        ):
            x_obs_km_cached = np.asarray(cache.get('x_obs_km', []), dtype=float)
            gravity_cached = np.asarray(cache.get('gravity_anomaly', []), dtype=float)
            obs_level_cached = float(cache.get('obs_level', 0.0))
            if x_obs_km_cached.size > 1 and gravity_cached.size == x_obs_km_cached.size:
                self._render_full_model_gravity_result(
                    x_obs_km=x_obs_km_cached,
                    gravity_anomaly=gravity_cached,
                    obs_level=obs_level_cached,
                    gravity_method=str(cache.get('gravity_method', gravity_method) or gravity_method),
                    model_shape=cache.get('model_shape'),
                    from_cache=True,
                    anomaly_note=str(cache.get('anomaly_note', '') or ''),
                )
                return
        
        try:
            # 获取模型参数
            x_coord = self.profile_extractor.x_coord
            z_coord = self.profile_extractor.z_coord
            velocity_var = self.property_calculator.velocity_var
            
            # 获取坐标和速度数据
            x_coords = self.grid_data[x_coord].values  # km
            z_coords = self.grid_data[z_coord].values  # km
            velocity_data = self.grid_data[velocity_var].values  # 2D array
            
            # 获取计算方法
            density_method = self.density_method_var.get()
            obs_level_m = obs_level * 1000.0  # 转换为米
            obs_note = (
                f"Observation level convention: +down, sea level=0 "
                f"(z0={obs_level:.3f} km -> {obs_level_m:.1f} m)"
            )
            
            self.log_result(f"\n{'='*50}")
            self.log_result("Calculating gravity anomaly from full model...")
            self.log_result(f"  Original model size: {velocity_data.shape}")
            self.log_result(f"  Background density: {bg_density:.3f} g/cm³")
            self.log_result(f"  Observation level: {obs_level:.2f} km")
            self.log_result(f"  {obs_note}")
            self.log_result(f"  Density method: {density_method}")
            self.log_result(f"  Gravity method: {gravity_method}")
            self.log_result(f"  Strict jgrav mode: {'ON' if strict_jgrav else 'OFF'}")
            
            self.log_result(f"  Max grid size: {max_grid_size}")
            self.log_result(f"  Extension distance: {extension_dist:.1f} km")
            
            # 保存原始模型范围（用于观测点计算和延伸区域判断）
            original_x_min = x_coords[0]
            original_x_max = x_coords[-1]
            original_z_min = z_coords[0]
            original_z_max = z_coords[-1]  # 原始模型的底部
            nx_orig = len(x_coords)
            nz_orig = len(z_coords)
            
            # 模型延伸（避免边缘效应）
            if extension_dist > 0:
                self.log_result(f"  Extending model by {extension_dist:.1f} km on each side...")
                self.root.update()
                
                # 获取网格间距
                if len(x_coords) > 1:
                    dx_original = x_coords[1] - x_coords[0]  # km
                else:
                    dx_original = 1.0  # 默认1km
                
                if len(z_coords) > 1:
                    dz_original = z_coords[1] - z_coords[0]  # km
                else:
                    dz_original = 0.5  # 默认0.5km
                
                # 计算需要添加的网格点数
                n_extend_x = int(np.ceil(extension_dist / dx_original))
                n_extend_z = int(np.ceil(extension_dist / dz_original))
                
                # 创建延伸后的坐标数组
                x_min = x_coords[0]
                x_max = x_coords[-1]
                z_min = z_coords[0]
                z_max = z_coords[-1]
                
                # 左侧延伸
                x_left_extend = np.linspace(x_min - extension_dist, x_min - dx_original, n_extend_x)
                # 右侧延伸
                x_right_extend = np.linspace(x_max + dx_original, x_max + extension_dist, n_extend_x)
                # 底部延伸（如果需要）
                z_bottom_extend = np.linspace(z_max + dz_original, z_max + extension_dist, n_extend_z)
                
                # 合并坐标
                x_coords_extended = np.concatenate([x_left_extend, x_coords, x_right_extend])
                z_coords_extended = np.concatenate([z_coords, z_bottom_extend])
                
                # 创建延伸后的速度数组
                nz_extended = len(z_coords_extended)
                nx_extended = len(x_coords_extended)
                velocity_data_extended = np.zeros((nz_extended, nx_extended))
                
                # 填充原始数据区域（使用之前保存的nx_orig和nz_orig）
                velocity_data_extended[:nz_orig, n_extend_x:n_extend_x+nx_orig] = velocity_data
                
                # 填充延伸区域（使用边界值）
                # 左侧：使用最左侧列的值
                if nx_orig > 0:
                    left_column = velocity_data[:, 0]
                    for j in range(n_extend_x):
                        velocity_data_extended[:nz_orig, j] = left_column
                        # 底部延伸部分也使用边界值
                        if n_extend_z > 0:
                            velocity_data_extended[nz_orig:, j] = left_column[-1] if len(left_column) > 0 else velocity_data[-1, 0]
                
                # 右侧：使用最右侧列的值
                if nx_orig > 0:
                    right_column = velocity_data[:, -1]
                    right_start = n_extend_x + nx_orig
                    for j in range(n_extend_x):
                        velocity_data_extended[:nz_orig, right_start + j] = right_column
                        # 底部延伸部分也使用边界值
                        if n_extend_z > 0:
                            velocity_data_extended[nz_orig:, right_start + j] = right_column[-1] if len(right_column) > 0 else velocity_data[-1, -1]
                
                # 底部延伸：使用最底部行的值
                if n_extend_z > 0 and nx_orig > 0:
                    bottom_row = velocity_data[-1, :]
                    for i in range(n_extend_z):
                        velocity_data_extended[nz_orig + i, n_extend_x:n_extend_x+nx_orig] = bottom_row
                
                # 更新坐标和速度数据
                x_coords = x_coords_extended
                z_coords = z_coords_extended
                velocity_data = velocity_data_extended
                
                self.log_result(f"  Extended model size: {velocity_data.shape}")
                self.root.update()
            
            # 粗网格下采样（减少计算量）
            self.log_result("  Downsampling to coarse grid...")
            self.root.update()
            
            # 计算下采样因子（目标：每个方向最多max_grid_size个点）
            x_downsample = max(1, len(x_coords) // max_grid_size)
            z_downsample = max(1, len(z_coords) // max_grid_size)
            
            # 确保下采样因子至少为1
            x_downsample = max(1, x_downsample)
            z_downsample = max(1, z_downsample)
            
            # 如果进行了延伸，更新原始模型范围（用于观测点计算）
            if extension_dist > 0:
                # 原始模型范围在延伸后的坐标中的位置
                original_x_min = x_coords[n_extend_x]
                original_x_max = x_coords[n_extend_x + nx_orig - 1]
            
            # 保存下采样因子供后续使用
            original_x_coords = x_coords.copy()
            original_z_coords = z_coords.copy()
            
            # 下采样速度数据（使用平均值）
            if x_downsample > 1 or z_downsample > 1:
                # 计算粗网格大小
                nz_coarse = (len(z_coords) + z_downsample - 1) // z_downsample
                nx_coarse = (len(x_coords) + x_downsample - 1) // x_downsample
                
                # 初始化粗网格
                z_coarse = np.zeros(nz_coarse)
                x_coarse = np.zeros(nx_coarse)
                velocity_coarse = np.zeros((nz_coarse, nx_coarse))
                
                # 对z方向进行下采样
                for i in range(nz_coarse):
                    start_idx = i * z_downsample
                    end_idx = min(start_idx + z_downsample, len(z_coords))
                    z_coarse[i] = np.mean(z_coords[start_idx:end_idx])
                
                # 对x方向进行下采样
                for j in range(nx_coarse):
                    start_idx = j * x_downsample
                    end_idx = min(start_idx + x_downsample, len(x_coords))
                    x_coarse[j] = np.mean(x_coords[start_idx:end_idx])
                
                # 对速度数据进行下采样（使用平均值）
                for i in range(nz_coarse):
                    z_start = i * z_downsample
                    z_end = min(z_start + z_downsample, len(z_coords))
                    for j in range(nx_coarse):
                        x_start = j * x_downsample
                        x_end = min(x_start + x_downsample, len(x_coords))
                        # 计算该区域的平均速度
                        velocity_coarse[i, j] = np.nanmean(
                            velocity_data[z_start:z_end, x_start:x_end]
                        )
                
                # 更新坐标和速度数据
                x_coords = x_coarse
                z_coords = z_coarse
                velocity_data = velocity_coarse
                
                self.log_result(f"  Coarse grid size: {velocity_data.shape}")
                self.log_result(f"  Downsample factors: z={z_downsample}, x={x_downsample}")
            else:
                self.log_result("  Model already small enough, no downsampling needed")
            
            self.root.update()
            
            # 计算密度模型（向量化计算以提高效率）
            self.log_result("  Converting velocity to density...")
            self.root.update()  # 更新界面，避免卡死
            
            # 向量化处理速度数据
            valid_mask = ~(np.isnan(velocity_data) | (velocity_data <= 0))
            
            vp_data = np.where(valid_mask, velocity_data, np.nan)
            if str(self.model_type_var.get()).strip().lower() == 'vs':
                self.log_result(
                    "  Note: full-model gravity always treats primary grid as Vp; "
                    "Wave Type selector is ignored here."
                )
            
            density_method_library = self._density_method_for_library(density_method)
            # 向量化计算密度
            density_data = np.where(
                valid_mask,
                calculate_density(vp_data, method=density_method_library),
                bg_density,
            )

            # 与单点/属性剖面保持一致：海水层密度使用统一海水参数
            seafloor_depths = self._get_seafloor_depths(x_coords)
            seawater_density = (
                self.property_calculator.seawater_density_g_cm3
                if self.property_calculator is not None
                else 1.03
            )
            if self.property_calculator is not None:
                water_mask = np.zeros_like(valid_mask, dtype=bool)
                for i in range(len(z_coords)):
                    z_val = float(z_coords[i])
                    for j in range(len(x_coords)):
                        if not valid_mask[i, j]:
                            continue
                        seafloor_z = float(seafloor_depths[j]) if j < len(seafloor_depths) else 0.0
                        vp_val = float(vp_data[i, j])
                        if self.property_calculator.is_water_zone(vp=vp_val, z=z_val, seafloor_depth=seafloor_z):
                            water_mask[i, j] = True
                density_data[water_mask] = seawater_density
                n_water_cells = int(np.sum(water_mask))
                n_valid_cells = int(np.sum(valid_mask))
                water_ratio = (n_water_cells / n_valid_cells) if n_valid_cells > 0 else 0.0
                self.log_result(
                    f"  Water cells: {n_water_cells}/{n_valid_cells} ({water_ratio:.1%}), "
                    f"density fixed at {seawater_density:.3f} g/cm³"
                )
                if self._uses_tomo2d_sediment_density(density_method):
                    basement_depths = np.asarray(
                        [self._get_basement_depth(float(x_val)) for x_val in np.asarray(x_coords, dtype=float)],
                        dtype=float,
                    )
                    sed_mask = np.zeros_like(valid_mask, dtype=bool)
                    z_vals = np.asarray(z_coords, dtype=float)
                    for j in range(len(x_coords)):
                        seafloor_z = float(seafloor_depths[j]) if j < len(seafloor_depths) else 0.0
                        basement_z = float(basement_depths[j]) if j < len(basement_depths) else np.nan
                        if not np.isfinite(basement_z) or basement_z <= seafloor_z:
                            continue
                        sed_mask[:, j] = (z_vals >= seafloor_z) & (z_vals < basement_z)
                    sed_mask = sed_mask & valid_mask & (~water_mask)
                    if np.any(sed_mask):
                        vp_sed = np.asarray(vp_data, dtype=float)[sed_mask]
                        sed_density = 1.0 + 1.18 * np.power(np.maximum(vp_sed - 1.5, 0.0), 0.22)
                        sed_density = np.minimum(sed_density, 2.6)
                        density_data[sed_mask] = sed_density
                        self.log_result(
                            f"  Sediment cells (tomo2d mode): {int(np.sum(sed_mask))}, "
                            "rho = 1 + 1.18*(Vp-1.5)^0.22, capped at 2.6 g/cm³"
                        )
            
            # 计算密度对比度（相对于背景密度）
            effective_bg_density = float(bg_density)
            density_contrast = density_data - effective_bg_density  # g/cm³
            valid_density = density_data[valid_mask]
            if (not strict_jgrav) and valid_density.size > 0:
                pos_ratio0 = float(np.mean(density_contrast[valid_mask] > 0.0))
                neg_ratio0 = float(np.mean(density_contrast[valid_mask] < 0.0))
                if min(pos_ratio0, neg_ratio0) < 0.02:
                    auto_bg = float(np.nanmedian(valid_density))
                    if np.isfinite(auto_bg):
                        effective_bg_density = auto_bg
                        density_contrast = density_data - effective_bg_density
                        self.log_result(
                            "  Density contrast is strongly one-sided; "
                            f"using median density as effective background for anomaly centering: {effective_bg_density:.3f} g/cm³ "
                            f"(user background={bg_density:.3f} g/cm³)"
                        )
            
            # 转换为kg/m³
            density_contrast_kg_m3 = density_contrast * 1000.0
            
            # 创建有效单元掩码
            if extension_dist > 0:
                # 使用坐标范围判断延伸区域（下采样后仍然有效）
                # 原始模型范围
                x_orig_min = original_x_min
                x_orig_max = original_x_max
                z_orig_max = original_z_max  # 原始模型的底部
                
                # 创建延伸区域掩码（基于坐标范围）
                nx_coarse = len(x_coords)
                nz_coarse = len(z_coords)
                extension_mask = np.zeros((nz_coarse, nx_coarse), dtype=bool)
                extension_weight = np.ones((nz_coarse, nx_coarse), dtype=float)
                
                for i in range(nz_coarse):
                    for j in range(nx_coarse):
                        x_val = x_coords[j]
                        z_val = z_coords[i]
                        # 左侧延伸：x < 原始模型左边界
                        # 右侧延伸：x > 原始模型右边界
                        # 底部延伸：z > 原始模型底部
                        is_extension = x_val < x_orig_min or x_val > x_orig_max or z_val > z_orig_max
                        if is_extension:
                            extension_mask[i, j] = True
                            wx = 1.0
                            wz = 1.0
                            if x_val < x_orig_min:
                                wx = max(0.0, min(1.0, (x_val - (x_orig_min - extension_dist)) / max(extension_dist, 1e-6)))
                            elif x_val > x_orig_max:
                                wx = max(0.0, min(1.0, ((x_orig_max + extension_dist) - x_val) / max(extension_dist, 1e-6)))
                            if z_val > z_orig_max:
                                wz = max(0.0, min(1.0, ((z_orig_max + extension_dist) - z_val) / max(extension_dist, 1e-6)))
                            extension_weight[i, j] = max(0.0, min(wx, wz))

                # 默认模式对延伸区做平滑衰减，严格 jgrav 模式保持原对比度
                if not strict_jgrav:
                    density_contrast_kg_m3 = density_contrast_kg_m3 * extension_weight
                
                density_threshold = 10.0  # kg/m³，提高阈值以减少计算量
                significant_mask = (np.abs(density_contrast_kg_m3) > density_threshold) & valid_mask
                
                n_extension = np.sum(extension_mask)
                self.log_result(
                    f"  Extension cells: {n_extension} (density contrast tapered to suppress edge-induced trend)"
                )
            else:
                # 没有延伸，只使用密度阈值
                density_threshold = 10.0  # kg/m³
                significant_mask = (np.abs(density_contrast_kg_m3) > density_threshold) & valid_mask
            
            # 统计有效单元数量
            n_significant = np.sum(significant_mask)
            self.log_result(f"  Significant density cells: {n_significant} / {density_contrast_kg_m3.size}")
            if density_contrast_kg_m3.ndim == 2 and density_contrast_kg_m3.shape[1] >= 2:
                edge_n = max(1, min(5, density_contrast_kg_m3.shape[1] // 10))
                left_edge = density_contrast_kg_m3[:, :edge_n]
                right_edge = density_contrast_kg_m3[:, -edge_n:]
                self.log_result(
                    "  Edge density contrast mean (kg/m³): "
                    f"left={float(np.nanmean(left_edge)):.2f}, right={float(np.nanmean(right_edge)):.2f}"
                )
            if n_significant > 0:
                sig_vals = density_contrast_kg_m3[significant_mask]
                self.log_result(
                    "  Significant density contrast stats: "
                    f"mean={float(np.nanmean(sig_vals)):.2f} kg/m³, "
                    f"mean|.|={float(np.nanmean(np.abs(sig_vals))):.2f} kg/m³, "
                    f"min={float(np.nanmin(sig_vals)):.2f}, "
                    f"max={float(np.nanmax(sig_vals)):.2f}"
                )
                pos_ratio = float(np.mean(sig_vals > 0.0))
                neg_ratio = float(np.mean(sig_vals < 0.0))
                self.log_result(
                    f"  Significant contrast sign ratio: positive={pos_ratio:.1%}, negative={neg_ratio:.1%}"
                )
            
            if n_significant == 0:
                messagebox.showwarning('Warning', 'No significant density contrast found. Try adjusting background density.')
                return
            
            # 创建观测点数组（减少观测点数量以提高速度）
            # 使用原始模型范围，而不是延伸后的范围
            if extension_dist > 0:
                x_min, x_max = original_x_min, original_x_max
            else:
                x_min, x_max = x_coords.min(), x_coords.max()
            n_obs = min(50, len(x_coords))  # 最多50个观测点，或使用网格点数
            x_obs_edges = np.linspace(x_min, x_max, n_obs + 1)
            x_obs_km = 0.5 * (x_obs_edges[:-1] + x_obs_edges[1:])
            x_obs_m = x_obs_km * 1000.0  # 转换为米
            
            self.log_result(f"  Number of observation points: {n_obs}")
            
            valid_cells = int(np.sum(significant_mask))
            self.log_result(f"  Valid cells: {valid_cells}")
            if gravity_method == 'talwani_grid':
                self.log_result("  Calculating gravity by Talwani cell summation...")
                self.root.update()
                total_obs = int(len(x_obs_km))
                last_percent = {"value": -1}

                def _talwani_progress(done: int, total: int) -> None:
                    if total <= 0:
                        return
                    percent = int(done * 100 / total)
                    should_log = (
                        done == total
                        or done == 1
                        or (done % 5 == 0)
                        or percent >= last_percent["value"] + 10
                    )
                    if should_log:
                        last_percent["value"] = percent
                        self.log_result(f"    Progress: {percent:3d}% ({done}/{total})")
                        self.root.update()

                self.log_result(f"    Progress:   0% (0/{total_obs})")
                gravity_anomaly = self.gravity_calculator.calculate_talwani_grid_anomaly(
                    x_coords_km=np.asarray(x_coords, dtype=float),
                    z_coords_km=np.asarray(z_coords, dtype=float),
                    density_contrast_kg_m3=np.asarray(density_contrast_kg_m3, dtype=float),
                    significant_mask=np.asarray(significant_mask, dtype=bool),
                    x_obs_km=np.asarray(x_obs_km, dtype=float),
                    obs_level_km=float(obs_level),
                    progress_callback=_talwani_progress,
                )
            else:
                self.log_result("  Calculating gravity by tomo2d-style FFT integration...")
                self.root.update()
                gravity_anomaly = self.gravity_calculator.calculate_tomo2d_fft_anomaly(
                    x_coords_km=np.asarray(x_coords, dtype=float),
                    z_coords_km=np.asarray(z_coords, dtype=float),
                    density_contrast_kg_m3=np.asarray(density_contrast_kg_m3, dtype=float),
                    significant_mask=np.asarray(significant_mask, dtype=bool),
                    x_obs_km=np.asarray(x_obs_km, dtype=float),
                    obs_level_km=float(obs_level),
                    ref_x_range_km=(float(x_min), float(x_max)),
                )

            anomaly_note = ""
            gmin = float(np.nanmin(gravity_anomaly))
            gmax = float(np.nanmax(gravity_anomaly))
            if (not strict_jgrav) and np.isfinite(gmin) and np.isfinite(gmax) and (gmin >= 0.0 or gmax <= 0.0):
                # 绝对异常可能整体偏移（常见于背景密度或区域趋势项），
                # 对显示结果移除两端线性基线，便于查看正负相对异常。
                n_edge = max(2, min(8, len(gravity_anomaly) // 6))
                left_mean = float(np.nanmean(gravity_anomaly[:n_edge]))
                right_mean = float(np.nanmean(gravity_anomaly[-n_edge:]))
                if len(gravity_anomaly) > 1:
                    t = np.linspace(0.0, 1.0, num=len(gravity_anomaly))
                    baseline = left_mean + (right_mean - left_mean) * t
                    gravity_anomaly = gravity_anomaly - baseline
                    anomaly_note = "detrended"
                    self.log_result(
                        "  Raw anomaly is one-signed; removed linear edge baseline for display "
                        f"(left={left_mean:.2f}, right={right_mean:.2f} mGal)."
                    )
            
            self._full_model_gravity_cache = {
                'cache_version': cache_version,
                'model_key': model_key,
                'param_signature': param_signature,
                'x_obs_km': np.asarray(x_obs_km, dtype=float),
                'gravity_anomaly': np.asarray(gravity_anomaly, dtype=float),
                'obs_level': float(obs_level),
                'gravity_method': gravity_method,
                'strict_jgrav': bool(strict_jgrav),
                'model_shape': (int(velocity_data.shape[0]), int(velocity_data.shape[1])),
                'anomaly_note': anomaly_note,
            }
            self._render_full_model_gravity_result(
                x_obs_km=np.asarray(x_obs_km, dtype=float),
                gravity_anomaly=np.asarray(gravity_anomaly, dtype=float),
                obs_level=float(obs_level),
                gravity_method=gravity_method,
                model_shape=(int(velocity_data.shape[0]), int(velocity_data.shape[1])),
                from_cache=False,
                anomaly_note=anomaly_note,
            )
        
        except Exception as e:
            messagebox.showerror('Error', f'Failed to calculate gravity from full model: {str(e)}')
            import traceback
            traceback.print_exc()
            self.log_result(f"ERROR: {str(e)}")
    
    def clear_gravity_bodies(self):
        """Clear all gravity bodies"""
        self.gravity_bodies = []
        if self.gravity_window is not None and self.gravity_window.winfo_exists():
            self._update_gravity_plot()
        self.log_result("  Cleared all gravity bodies")
    
