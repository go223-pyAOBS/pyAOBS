"""物性计算、日志、导出、快捷键与帮助。"""

from __future__ import annotations

try:
    from pyAOBS.visualization.imodel_qt.property_verbose import iter_rock_candidate_geology_lines
except ImportError:
    from ..imodel_qt.property_verbose import iter_rock_candidate_geology_lines

from .deps import *  # noqa: F403


class PropertiesUIMixin:
    def calculate_properties(self):
        """Calculate properties at specified point"""
        if self.grid_data is None or self.property_calculator is None:
            messagebox.showwarning('Warning', 'Please load model first')
            return
        
        try:
            x = float(self.prop_x_entry.get())
            z = float(self.prop_z_entry.get())
            
            # 获取该X位置的海底面深度
            seafloor_z = self._get_seafloor_depth(x)
            basement_z = self._get_basement_depth(x)
            props = self.property_calculator.calculate_all_properties(
                x,
                z,
                seafloor_depth=seafloor_z,
                basement_depth=basement_z,
            )
            self._record_property_result(x, z, props, source='manual_calculate')
            
            self.log_result(f"\nProperty Results (x={x:.2f}, z={z:.2f}):")
            self.log_result(f"  P-wave velocity: {props['vp']:.2f} km/s")
            self.log_result(f"  S-wave velocity: {props['vs']:.2f} km/s")
            vs_source = str(props.get('vs_source', '') or '').strip()
            if vs_source:
                self.log_result(f"  Vs source: {vs_source}")
            zone = str(props.get('zone', 'deep') or 'deep').strip()
            try:
                vp_vs_ratio = float(props['vp']) / float(props['vs']) if float(props['vs']) > 0 else None
            except Exception:
                vp_vs_ratio = None
            if vp_vs_ratio is None or zone == 'water':
                self.log_result("  Vp/Vs ratio: N/A (fluid)")
            else:
                self.log_result(f"  Vp/Vs ratio: {vp_vs_ratio:.2f}")
                if not (1.0 <= float(vp_vs_ratio) <= 3.5):
                    self.log_result("  Warning: Vp/Vs is outside expected range [1.0, 3.5]")
            self.log_result(f"  Density: {props['density']:.2f} g/cm³")
            self.log_result(f"  Pressure: {props['pressure']:.1f} MPa")
            self.log_result(f"  Temperature: {props['temperature']:.1f} °C")
            self.log_result(f"  Zone: {self._format_zone_display(zone)}")
            
            # Display rock type candidates
            rock_candidates = props.get('rock_candidates', [])
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
                self.log_result(f"  Rock type: {props['rock_type']} (probability: {props['rock_probability']:.2%})")
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
            
            self.log_result(f"  Bulk modulus: {props['bulk_modulus']:.2f} GPa")
            self.log_result(f"  Shear modulus: {props['shear_modulus']:.2f} GPa")
        except ValueError:
            messagebox.showerror('Error', 'Please enter valid coordinate values')
    
    def log_result(self, text: str):
        """Display text in result area"""
        self.result_text.insert(tk.END, text + '\n')
        self.result_text.see(tk.END)

    def _update_fixed_geology_info(self, props: dict | None) -> None:
        props = props or {}
        zone = str(props.get('zone', '') or '').strip() or '-'
        rock_classification = str(props.get('rock_classification', '') or '').strip() or '-'
        metamorphic_grade = str(props.get('metamorphic_grade', '') or '').strip() or '-'
        geological_meaning = str(props.get('geological_meaning', '') or '').strip() or '-'
        self.fixed_zone_var.set(self._format_zone_display(zone))
        self.fixed_rock_classification_var.set(rock_classification)
        self.fixed_metamorphic_grade_var.set(metamorphic_grade)
        self.fixed_geological_meaning_var.set(geological_meaning)

    def _format_zone_display(self, zone: str) -> str:
        key = str(zone or '').strip().lower()
        return self.zone_label_map.get(key, zone if zone else '-')

    def _load_zone_label_map(self) -> Dict[str, str]:
        labels = {
            'water': '海水层 (water)',
            'sediment': '沉积层 (sediment)',
            'deep': '深部地层 (deep)',
        }
        default_path = Path(__file__).resolve().parents[2] / 'utils' / 'rockdata' / 'imodel_priors.json'
        config_path = Path(os.environ.get('PYAOBS_IMODEL_PRIORS_FILE', str(default_path)))
        if not config_path.exists():
            return labels
        try:
            with config_path.open('r', encoding='utf-8') as f:
                config = json.load(f)
            raw_labels = config.get('zone_labels')
            if isinstance(raw_labels, dict):
                for k, v in raw_labels.items():
                    key = str(k or '').strip().lower()
                    text = str(v or '').strip()
                    if key and text:
                        labels[key] = text
        except Exception:
            pass
        return labels

    def _log_classifier_diagnostics(self) -> None:
        if self._classifier_diag_logged:
            return
        calc = getattr(self, 'property_calculator', None)
        diag = getattr(calc, 'classifier_diagnostics', None) if calc is not None else None
        if not isinstance(diag, dict):
            return

        if not diag.get('enabled'):
            self.log_result("Rock DB diagnostics: classifier unavailable")
            self._classifier_diag_logged = True
            return

        db_path = str(diag.get('database_path', '') or '').strip()
        rows = diag.get('row_count')
        rock_types = diag.get('rock_type_count')
        coverage = diag.get('coverage') if isinstance(diag.get('coverage'), dict) else {}
        error = str(diag.get('error', '') or '').strip()
        top_sources = diag.get('source_top5') if isinstance(diag.get('source_top5'), list) else []

        self.log_result("Rock DB diagnostics:")
        if db_path:
            self.log_result(f"  Database: {db_path}")
        if rows is not None:
            self.log_result(f"  Rows: {rows}")
        if rock_types is not None:
            self.log_result(f"  Rock types: {rock_types}")
        self.log_result(
            "  Coverage: "
            f"sio2={float(coverage.get('sio2_wt', 0.0)):.1%}, "
            f"facies={float(coverage.get('rock_facies', 0.0)):.1%}, "
            f"felsic_or_mafic={float(coverage.get('felsic_or_mafic', 0.0)):.1%}"
        )
        if top_sources:
            formatted = ', '.join(
                f"{str(item.get('source', 'unknown'))}({int(item.get('count', 0))})"
                for item in top_sources[:5]
                if isinstance(item, dict)
            )
            if formatted:
                self.log_result(f"  Top sources: {formatted}")
        if error:
            self.log_result(f"  Diagnostics note: {error}")
        self._classifier_diag_logged = True

    @staticmethod
    def _to_json_compatible(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): PropertiesUIMixin._to_json_compatible(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [PropertiesUIMixin._to_json_compatible(v) for v in value]
        return value

    def _record_property_result(self, x: float, z: float, props: dict, *, source: str) -> None:
        point_id = f'point_{x:.2f}_{z:.2f}'
        props_copy = dict(props or {})
        self.property_results[point_id] = props_copy
        rock_candidates = props_copy.get('rock_candidates', [])
        if not isinstance(rock_candidates, list):
            rock_candidates = []
        top_prob = 0.0
        if rock_candidates and isinstance(rock_candidates[0], dict):
            try:
                top_prob = float(rock_candidates[0].get('probability', 0.0) or 0.0)
            except (TypeError, ValueError):
                top_prob = 0.0
        row = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'point_id': point_id,
            'source': source,
            'x_km': float(x),
            'z_km': float(z),
            'vp_km_s': self._to_json_compatible(props_copy.get('vp')),
            'vs_km_s': self._to_json_compatible(props_copy.get('vs')),
            'vp_vs_ratio': self._to_json_compatible(props_copy.get('vp_vs_ratio')),
            'vs_source': str(props_copy.get('vs_source', '')),
            'density_g_cm3': self._to_json_compatible(props_copy.get('density')),
            'pressure_mpa': self._to_json_compatible(props_copy.get('pressure')),
            'temperature_c': self._to_json_compatible(props_copy.get('temperature')),
            'zone': str(props_copy.get('zone', 'deep') or 'deep'),
            'rock_type': str(props_copy.get('rock_type', 'UNKNOWN')),
            'felsic_or_mafic': self._to_json_compatible(props_copy.get('felsic_or_mafic')),
            'rock_facies': self._to_json_compatible(props_copy.get('rock_facies')),
            'sio2_wt': self._to_json_compatible(props_copy.get('sio2_wt')),
            'rock_classification': self._to_json_compatible(props_copy.get('rock_classification')),
            'metamorphic_grade': self._to_json_compatible(props_copy.get('metamorphic_grade')),
            'geological_meaning': self._to_json_compatible(props_copy.get('geological_meaning')),
            'rock_probability': self._to_json_compatible(props_copy.get('rock_probability', top_prob)),
            'confidence_level': str(props_copy.get('confidence_level', 'unknown')),
            'low_confidence': bool(props_copy.get('low_confidence', False)),
            'zone_prior_applied': bool(props_copy.get('zone_prior_applied', False)),
            'rerank_applied': bool(props_copy.get('rerank_applied', False)),
            'zone_constraint_applied': bool(props_copy.get('zone_constraint_applied', False)),
            'confidence_top1': self._to_json_compatible(props_copy.get('confidence_top1', top_prob)),
            'confidence_top2': self._to_json_compatible(props_copy.get('confidence_top2', 0.0)),
            'confidence_gap': self._to_json_compatible(props_copy.get('confidence_gap', 0.0)),
            'bulk_modulus_gpa': self._to_json_compatible(props_copy.get('bulk_modulus')),
            'shear_modulus_gpa': self._to_json_compatible(props_copy.get('shear_modulus')),
            'rock_candidates_json': json.dumps(
                self._to_json_compatible(rock_candidates),
                ensure_ascii=False,
            ),
        }
        self.property_result_rows.append(row)

    @staticmethod
    def _format_confidence_text(props: dict) -> str:
        try:
            top1 = float(props.get('confidence_top1', 0.0) or 0.0)
        except (TypeError, ValueError):
            top1 = 0.0
        try:
            top2 = float(props.get('confidence_top2', 0.0) or 0.0)
        except (TypeError, ValueError):
            top2 = 0.0
        try:
            gap = float(props.get('confidence_gap', max(0.0, top1 - top2)) or 0.0)
        except (TypeError, ValueError):
            gap = max(0.0, top1 - top2)
        is_low = bool(props.get('low_confidence', False))
        level = "低置信度" if is_low else "正常"
        return f"{level} (top1={top1:.2%}, top2={top2:.2%}, gap={gap:.2%})"

    def _compute_pt_for_velocity_correction(self, x: float, z: float, vp_for_zone: float) -> tuple[float, float]:
        """
        统一速度校正所用温压口径，与 PropertyCalculator 保持一致。
        返回: (pressure_mpa, temperature_c)
        """
        if self.property_calculator is None:
            pressure = max(0.0, float(z)) * 30.0
            temperature = 25.0 + max(0.0, float(z)) * 30.0
            return pressure, temperature

        seafloor_z = self._get_seafloor_depth(float(x))
        is_water = self.property_calculator.is_water_zone(
            vp=float(vp_for_zone),
            z=float(z),
            seafloor_depth=seafloor_z,
        )
        pressure = self.property_calculator.compute_total_pressure(
            z=float(z),
            seafloor_depth=seafloor_z,
            is_water=is_water,
            rock_pressure_gradient=30.0,
        )
        if is_water:
            temperature = float(self.property_calculator.seawater_temperature_c)
        else:
            temperature = float(
                calculate_temperature_from_depth(
                    float(z),
                    temperature_gradient=float(self.property_calculator.geothermal_gradient_c_per_km),
                    surface_temperature=float(self.property_calculator.seafloor_temperature_c),
                    seafloor_depth=seafloor_z,
                )
            )
        return pressure, temperature
    
    def save_figure(self):
        """Save figure"""
        filename = filedialog.asksaveasfilename(
            title='Save Figure',
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
            
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.log_result(f"Figure saved to: {filename}")
    
    def export_results(self):
        """Export all results"""
        output_dir = filedialog.askdirectory(title='Select Export Directory')
        if not output_dir:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export profiles
        for name, profile in self.profiles.items():
            profile.to_csv(output_path / f'{name}.csv', index=False)
        
        # Export property results
        if self.property_result_rows:
            props_df = pd.DataFrame(self.property_result_rows)
            props_df.to_csv(output_path / 'property_point_results.csv', index=False)
            with open(output_path / 'property_point_results.json', 'w', encoding='utf-8') as f:
                json.dump(self._to_json_compatible(self.property_result_rows), f, ensure_ascii=False, indent=2)
        elif self.property_results:
            # 兼容老数据结构：至少保证旧结果仍可导出
            props_df = pd.DataFrame(list(self.property_results.values()))
            props_df.to_csv(output_path / 'property_results.csv', index=False)
        
        self.log_result(f"\nResults exported to: {output_dir}")
        messagebox.showinfo('Success', f'Results exported to:\n{output_dir}')

    def export_point_results(self):
        """Export point-wise property calculation results only."""
        rows = list(self.property_result_rows)
        if not rows and self.property_results:
            for point_id, props in self.property_results.items():
                item = {'point_id': point_id}
                if isinstance(props, dict):
                    item.update(self._to_json_compatible(props))
                rows.append(item)
        if not rows:
            messagebox.showinfo('No Data', 'No point calculation results to export yet.')
            return

        filename = filedialog.asksaveasfilename(
            title='Export Point Results',
            defaultextension='.csv',
            filetypes=[
                ('CSV files', '*.csv'),
                ('JSON files', '*.json'),
                ('All files', '*.*'),
            ],
        )
        if not filename:
            return

        try:
            target = Path(filename)
            ext = target.suffix.lower()
            if ext == '.json':
                with target.open('w', encoding='utf-8') as f:
                    json.dump(self._to_json_compatible(rows), f, ensure_ascii=False, indent=2)
            else:
                pd.DataFrame(rows).to_csv(target, index=False)
            self.log_result(f"Point results exported: {target}")
            messagebox.showinfo('Success', f'Point results exported to:\n{target}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to export point results: {str(e)}')
    
    def on_key_press(self, event):
        """处理键盘事件"""
        # 可以添加快捷键
        pass
    
    def show_help(self):
        """Show help information"""
        help_text = """
Interactive Velocity Model Viewer - User Guide

1. File Operations:
   - Load Vp Model: Load primary grid/v.in velocity model file
   - Load Vs Model: Load optional Vs grid model file
   - Save Figure: Save current displayed figure
   - Export Results: Export all calculation results

2. Interactive Tools:
   - Point Selection: Left-click to add, Right-click to remove, D to delete last, C to clear all
   - Polygon Selection: Left-click to add vertices, Right-click to complete polygon

3. Profile Extraction:
   - Enter X coordinate to extract vertical profile
   - Enter X range and sampling interval to extract 1D vertical profile

4. Property Calculation:
   - Enter coordinates to calculate all property parameters at that point

5. Shortcuts:
   - Ctrl+O: Open model
        """
        messagebox.showinfo('User Guide', help_text)
