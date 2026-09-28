"""点选与多边形交互。"""

from __future__ import annotations

try:
    from pyAOBS.visualization.imodel_qt.property_verbose import iter_rock_candidate_geology_lines
except ImportError:
    from ..imodel_qt.property_verbose import iter_rock_candidate_geology_lines

from .deps import *  # noqa: F403


class InteractionMixin:
    def enable_point_selection(self):
        """Enable point selection"""
        if self.grid_data is None:
            messagebox.showwarning('Warning', 'Please load model first')
            return
        
        def on_point_selected(x, z, index):
            """Point selection callback"""
            # 同步更新self.selected_points
            # 确保列表长度足够
            while len(self.selected_points) <= index:
                self.selected_points.append(None)
            self.selected_points[index] = (x, z)
            
            self.log_result(f"\n{'='*50}")
            self.log_result(f"Point {index+1}: x={x:.2f} km, z={z:.2f} km")
            # 自动计算该点的物性
            if self.property_calculator:
                density_method = self.density_method_var.get()  # 'gardner', 'brocher', 'nafe_drake'
                try:
                    seafloor_z = self._get_seafloor_depth(x)
                    basement_z = self._get_basement_depth(x)
                    props = self.property_calculator.calculate_all_properties(
                        x,
                        z,
                        density_method=density_method,
                        seafloor_depth=seafloor_z,
                        basement_depth=basement_z,
                    )
                except Exception as exc:
                    self.log_result(f"  Error calculating properties: {exc}")
                    return

                vp = float(props.get('vp', 0.0) or 0.0)
                vs = float(props.get('vs', 0.0) or 0.0)
                density = float(props.get('density', 0.0) or 0.0)
                pressure = float(props.get('pressure', 0.0) or 0.0)
                temperature = float(props.get('temperature', 0.0) or 0.0)
                zone = str(props.get('zone', 'deep') or 'deep').strip()
                vp_vs_ratio = vp / vs if vs > 0 else None
                rock_type = str(props.get('rock_type', 'UNKNOWN'))
                rock_candidates = props.get('rock_candidates', [])
                if not isinstance(rock_candidates, list):
                    rock_candidates = []

                # Display all information (single source of truth: PropertyCalculator)
                self.log_result(f"  P-wave velocity: {vp:.2f} km/s")
                self.log_result(f"  S-wave velocity: {vs:.2f} km/s")
                if vp_vs_ratio is None or zone == 'water':
                    self.log_result("  Vp/Vs ratio: N/A (fluid)")
                else:
                    self.log_result(f"  Vp/Vs ratio: {vp_vs_ratio:.2f}")
                self.log_result(f"  Density ({density_method}): {density:.2f} g/cm³")
                self.log_result(f"  Pressure: {pressure:.2f} MPa")
                self.log_result(f"  Temperature: {temperature:.2f} °C")
                self.log_result(f"  Zone: {self._format_zone_display(zone)}")

                if rock_candidates:
                    self.log_result(f"  Rock type candidates:")
                    for i, candidate in enumerate(rock_candidates, 1):
                        cand_type = candidate.get('rock_type', 'UNKNOWN')
                        cand_prob = candidate.get('probability', 0.0)
                        marker = " <-- Most likely" if i == 1 else ""
                        self.log_result(f"    {i}. {cand_type}: {cand_prob:.2%}{marker}")
                        for sub in iter_rock_candidate_geology_lines(candidate):
                            self.log_result(f"       {sub}")
                else:
                    self.log_result(
                        f"  Rock type: {rock_type} (probability: {float(props.get('rock_probability', 0.0) or 0.0):.2%})"
                    )
                foma = str(props.get('felsic_or_mafic', '') or '').strip()
                facies = str(props.get('rock_facies', '') or '').strip()
                sio2_raw = props.get('sio2_wt', None)
                if foma:
                    self.log_result(f"  Felsic/Mafic: {foma}")
                if facies:
                    self.log_result(f"  Rock facies: {facies}")
                try:
                    sio2_val = float(sio2_raw) if sio2_raw is not None and str(sio2_raw).strip() != '' else None
                except (TypeError, ValueError):
                    sio2_val = None
                if sio2_val is not None:
                    self.log_result(f"  SiO2: {sio2_val:.2f} wt.%")
                rock_classification = str(props.get('rock_classification', '') or '').strip()
                metamorphic_grade = str(props.get('metamorphic_grade', '') or '').strip()
                geological_meaning = str(props.get('geological_meaning', '') or '').strip()
                if rock_classification:
                    self.log_result(f"  Rock classification: {rock_classification}")
                if metamorphic_grade:
                    self.log_result(f"  Metamorphic grade: {metamorphic_grade}")
                if geological_meaning:
                    self.log_result(f"  Geological meaning: {geological_meaning}")
                if bool(props.get('zone_prior_applied', False)):
                    self.log_result("  Note: zone-first sample prior applied")
                if bool(props.get('rerank_applied', False)):
                    self.log_result("  Note: low-confidence rerank applied")
                if bool(props.get('zone_constraint_applied', False)):
                    self.log_result("  Note: zone semantic constraint applied")
                self.log_result(f"  Confidence: {self._format_confidence_text(props)}")
                self._update_fixed_geology_info(props)

                self.log_result(f"  Bulk modulus: {float(props.get('bulk_modulus', 0.0) or 0.0):.2f} GPa")
                self.log_result(f"  Shear modulus: {float(props.get('shear_modulus', 0.0) or 0.0):.2f} GPa")
                self.log_result(f"{'='*50}")
                self._record_property_result(x, z, props, source='point_selection')
        
        # 清除之前的点选择器（如果存在）
        if self.point_selector is not None:
            # 断开之前的事件连接
            if hasattr(self.point_selector, 'cid_click'):
                self.ax_main.figure.canvas.mpl_disconnect(self.point_selector.cid_click)
            if hasattr(self.point_selector, 'cid_key'):
                self.ax_main.figure.canvas.mpl_disconnect(self.point_selector.cid_key)
        
        # 清除之前的点列表
        self.selected_points = []
        
        self.point_selector = PointSelector(self.ax_main, callback=on_point_selected)
        self.log_result("Point selection enabled: Left-click to add, Right-click to remove, D to delete last, C to clear all")
    
    def enable_polygon_selection(self):
        """Enable polygon selection"""
        if self.grid_data is None:
            messagebox.showwarning('Warning', 'Please load model first')
            return
        
        def on_polygon_selected(vertices):
            """Polygon selection callback"""
            # PolygonSelector返回的vertices可能已经包含了闭合点
            # 确保多边形正确闭合：第一个点和最后一个点应该连接
            # 如果最后一个点等于第一个点，移除它（因为Polygon(closed=True)会自动添加）
            if len(vertices) > 0:
                # 检查最后一个点是否等于第一个点
                first_point = vertices[0]
                last_point = vertices[-1]
                if len(vertices) > 1 and abs(first_point[0] - last_point[0]) < 1e-6 and abs(first_point[1] - last_point[1]) < 1e-6:
                    # 如果最后一个点等于第一个点，移除它
                    vertices = vertices[:-1]
            
            # 保存处理后的顶点（确保第一个点和最后一个点会闭合）
            self.selected_polygons.append(vertices)
            
            # 绘制多边形（closed=True会自动将第一个点连接到最后一个点）
            poly = Polygon(vertices, closed=True, fill=False, 
                          edgecolor='red', linewidth=2, alpha=0.7)
            poly.set_label('_imodel_polygon')  # Add special label for identification
            self.ax_main.add_patch(poly)
            self.polygon_patches.append(poly)  # Save Polygon object reference
            self.canvas.draw()
            self.log_result(f"Polygon selected with {len(vertices)} vertices")
        
        self.polygon_selector = PolygonSelector(
            self.ax_main,
            on_polygon_selected,
            useblit=True,
            props=dict(color='red', linestyle='-', linewidth=2, alpha=0.5)
        )
        self.log_result("Polygon selection enabled: Left-click to add vertices, Right-click to complete")
    
    def clear_selections(self):
        """清除所有选择"""
        # 清除点选择
        if self.point_selector is not None:
            self.point_selector.clear_points()
        self.selected_points = []
        
        # 清除多边形采样点显示
        for artist in self.polygon_sample_artists:
            try:
                artist.remove()
            except:
                pass
        self.polygon_sample_artists = []
        
        # 清除多边形选择 - 采用更彻底的方法
        if self.polygon_selector:
            # 方法1：断开PolygonSelector的事件连接
            try:
                if hasattr(self.polygon_selector, 'disconnect_events'):
                    self.polygon_selector.disconnect_events()
            except:
                pass
            
            # 方法2：清除PolygonSelector的临时线条
            # PolygonSelector可能使用_line或其他属性存储线条
            for attr_name in ['_line', '_lines', 'line', 'lines']:
                if hasattr(self.polygon_selector, attr_name):
                    attr = getattr(self.polygon_selector, attr_name)
                    if attr is not None:
                        try:
                            if isinstance(attr, list):
                                for item in attr:
                                    if hasattr(item, 'remove'):
                                        item.remove()
                            elif hasattr(attr, 'remove'):
                                attr.remove()
                        except:
                            pass
            
            # 方法3：清除vertices
            try:
                self.polygon_selector.vertices = []
            except:
                pass
            
            # 方法4：尝试禁用selector
            try:
                if hasattr(self.polygon_selector, 'set_active'):
                    self.polygon_selector.set_active(False)
            except:
                pass
        
        # 清除所有保存的多边形patches
        for poly_patch in self.polygon_patches:
            try:
                poly_patch.remove()
            except:
                pass
        self.polygon_patches.clear()
        self.selected_polygons.clear()
        
        # 最彻底的方法：重新绘制整个模型（这会清除所有图形元素）
        # 但需要先保存当前的colormap设置
        if self.grid_data is not None:
            # 保存当前的colormap和范围设置（如果有的话）
            current_cmap = None
            current_vmin = None
            current_vmax = None
            
            # 尝试从当前图形中获取colormap设置
            try:
                collections = self.ax_main.collections
                if collections:
                    # 获取第一个colormap对象
                    coll = collections[0]
                    if hasattr(coll, 'get_cmap'):
                        current_cmap = coll.get_cmap().name
                    if hasattr(coll, 'get_clim'):
                        current_vmin, current_vmax = coll.get_clim()
            except:
                pass
            
            # 重新绘制模型（这会清除所有图形，包括PolygonSelector创建的临时线条）
            self.plot_model(cmap=current_cmap or 'viridis', 
                          vmin=current_vmin, 
                          vmax=current_vmax)
            
            # 重新绘制后，如果之前启用了多边形选择，需要重新创建PolygonSelector
            # 因为plot_model会调用ax_main.clear()，这会清除所有图形元素
            # 但PolygonSelector可能仍然存在，只是其内部状态可能不一致
            # 为了安全起见，如果之前有polygon_selector，我们将其设为None
            # 用户需要重新点击"启用多边形选择"按钮来重新启用
            if self.polygon_selector:
                # 完全断开并删除旧的selector
                try:
                    if hasattr(self.polygon_selector, 'disconnect_events'):
                        self.polygon_selector.disconnect_events()
                except:
                    pass
                self.polygon_selector = None
        else:
            # 如果没有模型，至少清除所有图形元素
            self.ax_main.clear()
            self.canvas.draw()
        
        self.log_result("All selections cleared")
    
