#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
站位回收策略规划工具 - GUI版本

功能：
1. 加载站位文件
2. 设置回收参数
3. 规划回收策略
4. 可视化显示回收路径、释放位置、时间线等

Author: Haibo Huang
Date: 2025
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys

# 导入回收策略模块
try:
    from .recovery_strategy import (
        RecoveryStationPlanner, 
        load_stations_file,
        haversine_distance,
        print_recovery_plan
    )
except ImportError:
    # 如果作为独立脚本运行
    from recovery_strategy import (
        RecoveryStationPlanner, 
        load_stations_file,
        haversine_distance,
        print_recovery_plan
    )

# 如果haversine_distance未导入，定义它
try:
    haversine_distance
except NameError:
    import math
    def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """Calculate Haversine distance between two points"""
        R = 6371.0  # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c


class RecoveryStrategyGUI:
    """Station Recovery Strategy Planning GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Station Recovery Strategy Planner")
        self.root.geometry("1400x900")
        
        # Data
        self.stations = []
        self.station_names = []
        self.station_depths = []
        self.station_ascent_speeds = []
        self.station_release_distances = []
        self.recovery_path = []
        self.recovery_result = None
        
        # Parameters
        self.ship_speed = 10.0  # knots
        self.num_stations = 2
        self.pickup_time = 30.0  # minutes
        self.release_time = 10.0  # minutes
        self.ascent_speed = 0.65  # m/s
        
        # Create interface
        self.create_widgets()
        
    def create_widgets(self):
        """Create GUI components"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Left: Control panel
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        main_frame.columnconfigure(0, weight=0)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Right: Visualization area
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # === Left Control Panel ===
        # File Loading
        file_frame = ttk.LabelFrame(left_frame, text="File Loading", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load Station File", 
                   command=self.load_stations_file).pack(fill=tk.X, pady=5)
        
        self.file_label = ttk.Label(file_frame, text="No file loaded", foreground="gray")
        self.file_label.pack(fill=tk.X)
        
        # Parameters
        param_frame = ttk.LabelFrame(left_frame, text="Parameters", padding="10")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Ship Speed
        ttk.Label(param_frame, text="Ship Speed (knots):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.ship_speed_var = tk.StringVar(value="10.0")
        ttk.Entry(param_frame, textvariable=self.ship_speed_var, width=15).grid(row=0, column=1, pady=5)
        
        # Number of Stations
        ttk.Label(param_frame, text="Concurrent Stations:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.num_stations_var = tk.StringVar(value="2")
        ttk.Entry(param_frame, textvariable=self.num_stations_var, width=15).grid(row=1, column=1, pady=5)
        
        # Pickup Time
        ttk.Label(param_frame, text="Pickup Time (min):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.pickup_time_var = tk.StringVar(value="30.0")
        ttk.Entry(param_frame, textvariable=self.pickup_time_var, width=15).grid(row=2, column=1, pady=5)
        
        # Release Time
        ttk.Label(param_frame, text="Release Time (min):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.release_time_var = tk.StringVar(value="10.0")
        ttk.Entry(param_frame, textvariable=self.release_time_var, width=15).grid(row=3, column=1, pady=5)
        
        # Ascent Speed (default)
        ttk.Label(param_frame, text="Default Ascent Speed (m/s):").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.ascent_speed_var = tk.StringVar(value="0.65")
        ttk.Entry(param_frame, textvariable=self.ascent_speed_var, width=15).grid(row=4, column=1, pady=5)
        
        # Action Buttons
        action_frame = ttk.LabelFrame(left_frame, text="Actions", padding="10")
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(action_frame, text="Plan Recovery Strategy", 
                   command=self.plan_recovery).pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="Optimize Recovery Order", 
                   command=self.optimize_recovery).pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="Export Plan", 
                   command=self.export_plan).pack(fill=tk.X, pady=5)
        
        # Results Display
        result_frame = ttk.LabelFrame(left_frame, text="Results", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=15, width=35)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # === Right Visualization Area ===
        # Create matplotlib figure
        self.fig = plt.Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, right_frame)
        toolbar.update()
        
        # Initialize plot
        self.update_plot()
        
    def load_stations_file(self):
        """Load station file"""
        filename = filedialog.askopenfilename(
            title="Select Station File",
            filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            stations, station_names, station_ascent_speeds, station_release_distances, station_depths = \
                load_stations_file(filename)
            
            self.stations = stations
            self.station_names = station_names
            self.station_ascent_speeds = station_ascent_speeds
            self.station_release_distances = station_release_distances
            self.station_depths = station_depths
            
            # Use file order as recovery path
            self.recovery_path = list(range(len(stations)))
            
            self.file_label.config(text=os.path.basename(filename), foreground="black")
            self.update_plot()
            
            messagebox.showinfo("Success", f"Loaded {len(stations)} stations")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def get_parameters(self):
        """Get parameters"""
        try:
            self.ship_speed = float(self.ship_speed_var.get())
            self.num_stations = int(self.num_stations_var.get())
            self.pickup_time = float(self.pickup_time_var.get())
            self.release_time = float(self.release_time_var.get())
            self.ascent_speed = float(self.ascent_speed_var.get())
            
            if self.num_stations < 2:
                raise ValueError("Number of concurrent stations must be at least 2")
            
            return True
        except ValueError as e:
            messagebox.showerror("Parameter Error", f"Parameter setting error: {str(e)}")
            return False
    
    def plan_recovery(self):
        """Plan recovery strategy"""
        if not self.stations:
            messagebox.showwarning("Warning", "Please load station file first")
            return
        
        if not self.get_parameters():
            return
        
        try:
            # Create planner
            planner = RecoveryStationPlanner(
                recovery_path=self.recovery_path,
                stations=self.stations,
                station_depths=self.station_depths,
                station_ascent_speeds=self.station_ascent_speeds,
                station_release_distances=self.station_release_distances,
                ascent_speed=self.ascent_speed,
                ship_speed=self.ship_speed,
                pickup_time=self.pickup_time,
                release_time=self.release_time
            )
            
            # Plan recovery
            self.recovery_result = planner.plan_rolling_recovery(self.num_stations)
            
            # Verify all stations are recovered (critical rule)
            self.verify_all_stations_recovered()
            
            # Update display
            self.update_result_text()
            self.update_plot()
            
            messagebox.showinfo("Success", "Recovery strategy planning completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Planning failed: {str(e)}")
    
    def optimize_recovery(self):
        """Optimize recovery order"""
        if not self.stations:
            messagebox.showwarning("Warning", "Please load station file first")
            return
        
        if not self.get_parameters():
            return
        
        try:
            # Create planner
            planner = RecoveryStationPlanner(
                recovery_path=self.recovery_path,
                stations=self.stations,
                station_depths=self.station_depths,
                station_ascent_speeds=self.station_ascent_speeds,
                station_release_distances=self.station_release_distances,
                ascent_speed=self.ascent_speed,
                ship_speed=self.ship_speed,
                pickup_time=self.pickup_time,
                release_time=self.release_time
            )
            
            # Optimize recovery order
            self.recovery_result = planner.optimize_recovery_order(self.num_stations)
            
            # Verify all stations are recovered (critical rule)
            self.verify_all_stations_recovered()
            
            # Update display
            self.update_result_text()
            self.update_plot()
            
            messagebox.showinfo("Success", "Recovery order optimization completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Optimization failed: {str(e)}")
    
    def verify_all_stations_recovered(self):
        """Verify that all stations are recovered (critical rule)"""
        if not self.recovery_result or not self.stations:
            return
        
        recovered_stations = set(self.recovery_result['recovery_order'])
        all_stations = set(self.recovery_path)
        missing_stations = all_stations - recovered_stations
        
        if missing_stations:
            # This should not happen if the algorithm is correct, but warn the user
            missing_list = sorted(missing_stations)
            missing_names = [self.station_names[idx] if idx < len(self.station_names) 
                           else f'Station {idx}' for idx in missing_list]
            messagebox.showwarning(
                "Warning", 
                f"Critical Rule Violation: {len(missing_stations)} station(s) were not recovered:\n" +
                ", ".join(missing_names) + 
                "\n\nThis should not happen. Please check the algorithm."
            )
    
    def update_result_text(self):
        """Update result display text"""
        if not self.recovery_result:
            return
        
        result = self.recovery_result
        text = ""
        
        # Verify all stations are recovered (critical rule)
        recovered_stations = set(result['recovery_order'])
        all_stations = set(self.recovery_path)
        missing_stations = all_stations - recovered_stations
        
        text += f"Total Recovery Time: {result['total_time']/3600:.2f} hours ({result['total_time']:.0f} seconds)\n"
        text += f"Total Stations: {len(self.recovery_path)}\n"
        text += f"Recovered Stations: {len(result['recovery_order'])}\n"
        
        if missing_stations:
            text += f"⚠️ MISSING STATIONS: {len(missing_stations)} station(s) not recovered!\n"
            text += f"   Missing: {sorted(missing_stations)}\n"
        else:
            text += f"✓ All stations recovered (Critical rule satisfied)\n"
            text += f"  - Recovery route includes both rolling and single-station recovery\n"
            text += f"  - All stations are in the continuous recovery path\n"
        
        text += "\n"
        
        feasible_count = len(result['recovery_order']) - len(result['violations'])
        text += f"Feasible Stations: {feasible_count} / {len(result['recovery_order'])}\n"
        text += f"All Constraints Met: {'Yes' if result['is_feasible'] else 'No'}\n\n"
        
        if result['violations']:
            text += f"⚠️ Violations ({len(result['violations'])} stations):\n"
            for v in result['violations']:
                idx = v['station_idx']
                name = self.station_names[idx] if idx < len(self.station_names) else f'Station {idx}'
                delay = v.get('delay', 0) / 60
                text += f"  - {name}: delay {delay:.2f} minutes\n"
            text += "\n"
        
        text += "Recovery Order:\n"
        for i, idx in enumerate(result['recovery_order'][:10]):  # Show first 10
            name = self.station_names[idx] if idx < len(self.station_names) else f'Station {idx}'
            text += f"  {i+1}. {name}\n"
        if len(result['recovery_order']) > 10:
            text += f"  ... (total {len(result['recovery_order'])} stations)\n"
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, text)
    
    def update_plot(self):
        """Update plot display"""
        self.ax.clear()
        
        if not self.stations:
            self.ax.text(0.5, 0.5, "Please load station file", 
                        ha='center', va='center', fontsize=16, color='gray')
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.canvas.draw()
            return
        
        # Extract coordinates
        lons = [s[0] for s in self.stations]
        lats = [s[1] for s in self.stations]
        
        # Plot stations
        self.ax.scatter(lons, lats, c='blue', s=100, marker='o', 
                       label='Stations', zorder=5, edgecolors='black', linewidths=1)
        
        # Annotate station names
        for i, (lon, lat) in enumerate(self.stations):
            name = self.station_names[i] if i < len(self.station_names) else f'S{i}'
            self.ax.annotate(name, (lon, lat), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        # If recovery result exists, plot recovery path and release positions
        if self.recovery_result:
            result = self.recovery_result
            
            # Plot recovery path (by recovery order)
            if result['recovery_order']:
                recovery_lons = [self.stations[idx][0] for idx in result['recovery_order']]
                recovery_lats = [self.stations[idx][1] for idx in result['recovery_order']]
                
                # Plot path line
                self.ax.plot(recovery_lons, recovery_lats, 'r--', 
                           linewidth=2, alpha=0.5, label='Recovery Path', zorder=1)
                
                # Plot arrows indicating direction
                for i in range(len(recovery_lons) - 1):
                    dx = recovery_lons[i+1] - recovery_lons[i]
                    dy = recovery_lats[i+1] - recovery_lats[i]
                    self.ax.arrow(recovery_lons[i], recovery_lats[i], dx*0.7, dy*0.7,
                                head_width=0.01, head_length=0.01, fc='red', ec='red', zorder=2)
            
            # Plot release positions
            if result.get('release_positions'):
                release_lons = [pos[0] for pos in result['release_positions']]
                release_lats = [pos[1] for pos in result['release_positions']]
                self.ax.scatter(release_lons, release_lats, c='green', s=80, 
                              marker='s', label='Release Positions', zorder=4, edgecolors='black', linewidths=1)
            
            # Plot release distance ranges
            if result.get('release_positions') and self.station_release_distances:
                for i, (release_lon, release_lat) in enumerate(result['release_positions'][:10]):  # Show first 10
                    if i < len(result['recovery_order']):
                        station_idx = result['recovery_order'][i]
                        if station_idx < len(self.station_release_distances):
                            release_distance = self.station_release_distances[station_idx]
                            # Plot release distance circle (convert km to degrees, approx: 1 degree ≈ 111 km)
                            circle = Circle((release_lon, release_lat), release_distance/111.0, 
                                          fill=False, linestyle='--', color='green', alpha=0.2, linewidth=1)
                            self.ax.add_patch(circle)
            
            # Mark start and end points
            if result['recovery_order']:
                start_idx = result['recovery_order'][0]
                end_idx = result['recovery_order'][-1]
                start_lon, start_lat = self.stations[start_idx]
                end_lon, end_lat = self.stations[end_idx]
                
                self.ax.scatter([start_lon], [start_lat], c='green', s=200, 
                              marker='*', label='Start', zorder=6, edgecolors='black', linewidths=2)
                self.ax.scatter([end_lon], [end_lat], c='red', s=200, 
                              marker='*', label='End', zorder=6, edgecolors='black', linewidths=2)
            
            # Mark constraint violations
            if result['violations']:
                violation_indices = [v['station_idx'] for v in result['violations']]
                violation_lons = [self.stations[idx][0] for idx in violation_indices]
                violation_lats = [self.stations[idx][1] for idx in violation_indices]
                self.ax.scatter(violation_lons, violation_lats, c='red', s=150, 
                              marker='X', label='Violations', zorder=7, edgecolors='black', linewidths=2)
            
            # Mark missing stations (if any) - Critical rule check
            # This should not happen if the algorithm is correct
            recovered_stations = set(result['recovery_order'])
            all_stations = set(self.recovery_path)
            missing_stations = all_stations - recovered_stations
            if missing_stations:
                missing_lons = [self.stations[idx][0] for idx in missing_stations]
                missing_lats = [self.stations[idx][1] for idx in missing_stations]
                self.ax.scatter(missing_lons, missing_lats, c='orange', s=200, 
                              marker='D', label='Missing (Not Recovered!)', zorder=8, 
                              edgecolors='red', linewidths=3)
        
        # Set plot properties
        self.ax.set_xlabel('Longitude (°)', fontsize=12)
        self.ax.set_ylabel('Latitude (°)', fontsize=12)
        self.ax.set_title('Station Recovery Strategy Visualization', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', fontsize=9)
        
        # 调整坐标轴范围
        if lons:
            lon_margin = (max(lons) - min(lons)) * 0.1 if max(lons) != min(lons) else 0.1
            lat_margin = (max(lats) - min(lats)) * 0.1 if max(lats) != min(lats) else 0.1
            self.ax.set_xlim(min(lons) - lon_margin, max(lons) + lon_margin)
            self.ax.set_ylim(min(lats) - lat_margin, max(lats) + lat_margin)
        
        self.canvas.draw()
    
    def export_plan(self):
        """Export recovery plan"""
        if not self.recovery_result:
            messagebox.showwarning("Warning", "Please plan recovery strategy first")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Recovery Plan",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                result = self.recovery_result
                f.write("Station Recovery Strategy Plan\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Recovery Time: {result['total_time']/3600:.2f} hours\n")
                f.write(f"Total Stations: {len(self.recovery_path)}\n")
                f.write(f"Recovered Stations: {len(result['recovery_order'])}\n")
                
                # Verify all stations are recovered (critical rule)
                recovered_stations = set(result['recovery_order'])
                all_stations = set(self.recovery_path)
                missing_stations = all_stations - recovered_stations
                
                if missing_stations:
                    f.write(f"⚠️ WARNING: {len(missing_stations)} station(s) were NOT recovered!\n")
                    f.write(f"   Missing stations: {sorted(missing_stations)}\n")
                else:
                    f.write("✓ All stations recovered (Critical rule satisfied)\n")
                
                f.write("\n")
                
                feasible_count = len(result['recovery_order']) - len(result['violations'])
                f.write(f"Feasible Stations: {feasible_count} / {len(result['recovery_order'])}\n\n")
                
                f.write("\nDetailed Recovery Plan:\n")
                f.write("=" * 80 + "\n")
                
                # Track ship position and time for calculating travel times
                current_lon, current_lat = self.stations[result['recovery_order'][0]] if result['recovery_order'] else (0.0, 0.0)
                current_time = 0.0
                
                for i, station_idx in enumerate(result['recovery_order']):
                    name = self.station_names[station_idx] if station_idx < len(self.station_names) else f'Station {station_idx}'
                    lon, lat = self.stations[station_idx]
                    depth = self.station_depths[station_idx] if station_idx < len(self.station_depths) and self.station_depths[station_idx] is not None else 0.0
                    
                    # Get release information
                    if i < len(result.get('release_times', [])):
                        release_time = result['release_times'][i]
                        release_pos = result['release_positions'][i] if i < len(result.get('release_positions', [])) else (lon, lat)
                        release_lon, release_lat = release_pos
                    else:
                        release_lon, release_lat = lon, lat
                        release_time = result['recovery_arrival_times'][i] - 3600
                    
                    recovery_arrival_time = result['recovery_arrival_times'][i]
                    recovery_time = result['recovery_times'][i]
                    recovery_end_time = recovery_arrival_time + recovery_time
                    ascent_start_time = result['ascent_start_times'][i]
                    ascent_end_time = result['ascent_end_times'][i]
                    ascent_time = ascent_end_time - ascent_start_time
                    
                    release_distance = self.station_release_distances[station_idx] if station_idx < len(self.station_release_distances) else 5.0
                    distance_to_station = haversine_distance(release_lon, release_lat, lon, lat)
                    
                    # Calculate travel times
                    travel_to_release_distance = haversine_distance(current_lon, current_lat, release_lon, release_lat)
                    travel_to_release_time = travel_to_release_distance / (self.ship_speed * 1.852) * 3600
                    travel_to_station_time = distance_to_station / (self.ship_speed * 1.852) * 3600
                    time_since_release = recovery_arrival_time - release_time
                    
                    f.write(f"\n【Operation {i+1}】Station {i+1}: {name} (Index {station_idx})\n")
                    f.write("-" * 80 + "\n")
                    
                    # Step 1: Travel to release position
                    f.write(f"\n  Step 1: Travel to Release Position\n")
                    f.write(f"    Current Position: ({current_lon:.4f}°, {current_lat:.4f}°)\n")
                    f.write(f"    Release Position: ({release_lon:.4f}°, {release_lat:.4f}°)\n")
                    f.write(f"    Travel Distance: {travel_to_release_distance:.2f} km\n")
                    f.write(f"    Travel Time: {travel_to_release_time/60:.2f} minutes ({travel_to_release_time:.0f} seconds)\n")
                    f.write(f"    Departure Time: {current_time/60:.2f} minutes ({current_time:.0f} seconds)\n")
                    f.write(f"    Arrival at Release Position: {release_time/60:.2f} minutes ({release_time:.0f} seconds)\n")
                    
                    # Step 2: Release operation
                    f.write(f"\n  Step 2: Release Instrument\n")
                    f.write(f"    Release Position: ({release_lon:.4f}°, {release_lat:.4f}°)\n")
                    f.write(f"    Release Distance Range: {release_distance:.2f} km\n")
                    f.write(f"    Release Time: {release_time/60:.2f} minutes ({release_time:.0f} seconds)\n")
                    f.write(f"    Release Operation Time: {self.release_time:.1f} minutes\n")
                    f.write(f"    Release Completed: {(release_time + self.release_time * 60)/60:.2f} minutes\n")
                    
                    # Step 3: Instrument ascent
                    f.write(f"\n  Step 3: Instrument Ascent\n")
                    f.write(f"    Station Coordinates: ({lon:.4f}°, {lat:.4f}°)\n")
                    f.write(f"    Depth: {depth:.1f} m\n")
                    speed_m_per_min = self.station_ascent_speeds[station_idx] if station_idx < len(self.station_ascent_speeds) else 39.0
                    f.write(f"    Ascent Speed: {speed_m_per_min:.2f} m/min ({speed_m_per_min/60:.4f} m/s)\n")
                    f.write(f"    Ascent Start Time: {ascent_start_time/60:.2f} minutes ({ascent_start_time:.0f} seconds)\n")
                    f.write(f"    Ascent End Time: {ascent_end_time/60:.2f} minutes ({ascent_end_time:.0f} seconds)\n")
                    f.write(f"    Ascent Duration: {ascent_time/60:.2f} minutes ({ascent_time:.0f} seconds)\n")
                    
                    # Step 4: Travel to station position
                    f.write(f"\n  Step 4: Travel to Station Position for Recovery\n")
                    f.write(f"    Release Position: ({release_lon:.4f}°, {release_lat:.4f}°)\n")
                    f.write(f"    Station Position: ({lon:.4f}°, {lat:.4f}°)\n")
                    f.write(f"    Travel Distance: {distance_to_station:.2f} km\n")
                    f.write(f"    Travel Time: {travel_to_station_time/60:.2f} minutes ({travel_to_station_time:.0f} seconds)\n")
                    f.write(f"    Departure Time: {(release_time + self.release_time * 60)/60:.2f} minutes\n")
                    f.write(f"    Arrival at Station: {recovery_arrival_time/60:.2f} minutes ({recovery_arrival_time:.0f} seconds)\n")
                    
                    # Step 5: Recovery operation
                    f.write(f"\n  Step 5: Recovery Operation\n")
                    f.write(f"    Station Position: ({lon:.4f}°, {lat:.4f}°)\n")
                    f.write(f"    Arrival Time: {recovery_arrival_time/60:.2f} minutes ({recovery_arrival_time:.0f} seconds)\n")
                    f.write(f"    Pickup Time: {self.pickup_time:.1f} minutes\n")
                    f.write(f"    Recovery Completed: {recovery_end_time/60:.2f} minutes ({recovery_end_time:.0f} seconds)\n")
                    
                    # Time summary
                    f.write(f"\n  Time Summary:\n")
                    f.write(f"    Total Time from Release to Recovery: {time_since_release/60:.2f} minutes ({time_since_release:.0f} seconds)\n")
                    f.write(f"    Instrument Ascent Time: {ascent_time/60:.2f} minutes ({ascent_time:.0f} seconds)\n")
                    
                    if time_since_release >= ascent_time:
                        delay = time_since_release - ascent_time
                        f.write(f"    ⚠️  Constraint Violation: Time from release to recovery exceeds ascent time by {delay/60:.2f} minutes\n")
                    else:
                        margin = ascent_time - time_since_release
                        f.write(f"    ✓ Time Margin: {margin/60:.2f} minutes (Time from release to recovery is less than ascent time)\n")
                    
                    # Update current position and time
                    current_lon, current_lat = lon, lat
                    current_time = recovery_end_time
            
            messagebox.showinfo("Success", f"Plan exported to: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {str(e)}")


def main():
    """主函数"""
    root = tk.Tk()
    app = RecoveryStrategyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
