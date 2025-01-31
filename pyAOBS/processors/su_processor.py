#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pygmt
import os
from obspy import read
import xarray as xr
from scipy.interpolate import griddata, interp1d
from typing import Dict, List, Tuple, Optional, Union
import segyio

class SUProcessor:
    """
    A class for processing and visualizing Seismic Unix (SU) files.
    
    Attributes:
        su_file (str): Path to the SU file
        su_data: ObsPy Stream object containing the SU data
        header_info (dict): Dictionary containing header information
        header_values (dict): Dictionary containing all header values for all traces
        stats_values (dict): Dictionary containing all stats values for all traces
    """
    
    # Map between standard names and internal SU header names
    su_header_map = {
        # Trace sequence and identification
        'tracl': 'trace_sequence_number_within_line',
        'tracr': 'trace_sequence_number_within_segy_file',
        'fldr': 'original_field_record_number',
        'tracf': 'trace_number_within_the_original_field_record',
        'ep': 'energy_source_point_number',
        'cdp': 'ensemble_number',
        'trid': 'trace_identification_code',
        'nvs': 'number_of_vertically_summed_traces_yielding_this_trace',
        'nhs': 'number_of_horizontally_stacked_traces_yielding_this_trace',
        
        # Trace header position information
        'offset': 'distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group',
        'gelev': 'receiver_group_elevation',
        'selev': 'surface_elevation_at_source',
        'sdepth': 'source_depth_below_surface',
        'gdel': 'datum_elevation_at_receiver_group',
        'sdel': 'datum_elevation_at_source',
        'swdep': 'water_depth_at_source',
        'gwdep': 'water_depth_at_group',
        
        # Source-receiver coordinates
        'sx': 'source_coordinate_x',
        'sy': 'source_coordinate_y',
        'gx': 'group_coordinate_x',
        'gy': 'group_coordinate_y',
        'counit': 'coordinate_units',
        
        # Timing and sampling
        'delrt': 'delay_recording_time',
        'muts': 'mute_time_start_time_in_ms',
        'mute': 'mute_time_end_time_in_ms',
        'dt': 'sample_interval_in_ms_for_this_trace',
        'gain': 'gain_type_of_field_instruments',
        'igc': 'instrument_gain_constant',
        'igi': 'instrument_early_or_initial_gain',
        'corr': 'correlated',
        'ns': 'number_of_samples_in_this_trace',
        
        # Filter parameters
        'sfs': 'sweep_frequency_at_start',
        'sfe': 'sweep_frequency_at_end',
        'slen': 'sweep_length_in_ms',
        'styp': 'sweep_type',
        'stat': 'sweep_trace_taper_length_at_start_in_ms',
        'stae': 'sweep_trace_taper_length_at_end_in_ms',
        
        # Additional parameters
        'year': 'year_data_recorded',
        'day': 'day_of_year',
        'hour': 'hour_of_day',
        'minute': 'minute_of_hour',
        'sec': 'second_of_minute',
        'scalel': 'scalar_to_be_applied_to_all_elevations_and_depths',
        'scalco': 'scalar_to_be_applied_to_all_coordinates',
        # weathering velocity and static corrections
        'swevel': 'subweathering_velocity',
        'wevel': 'weathering_velocity',
        'sut': 'uphole_time_at_source_in_ms',
        'gut': 'uphole_time_at_group_in_ms',
        'sstat': 'source_static_correction_in_ms',
        'gstat': 'group_static_correction_in_ms',
        'tstat': 'total_static_applied_in_ms',
    }

    # Map between standard names and ObsPy stats fields
    stats_header_map = {
        'network': 'network',
        'station': 'station',
        'location': 'location',
        'channel': 'channel',
        'starttime': 'starttime',
        'endtime': 'endtime',
        'sampling_rate': 'sampling_rate',
        'delta': 'delta',
        'npts': 'npts',
        'calib': 'calib'
    }

    # Description for each field
    field_descriptions = {
        # Trace sequence and identification
        'tracl': 'Trace sequence number within line',
        'tracr': 'Trace sequence number within SEG-Y file',
        'fldr': 'Original field record number',
        'tracf': 'Trace number within the original field record',
        'ep': 'Energy source point number',
        'cdp': 'CDP ensemble number',
        'trid': 'Trace identification code',
        'nvs': 'Number of vertically summed traces yielding this trace',
        'nhs': 'Number of horizontally stacked traces yielding this trace',
        
        # Trace header position information
        'offset': 'Distance from source to receiver group',
        'gelev': 'Receiver group elevation',
        'selev': 'Surface elevation at source',
        'sdepth': 'Source depth below surface',
        'gdel': 'Datum elevation at receiver group',
        'sdel': 'Datum elevation at source',
        'swdep': 'Water depth at source',
        'gwdep': 'Water depth at group',
        
        # Source-receiver coordinates
        'sx': 'Source X coordinate',
        'sy': 'Source Y coordinate',
        'gx': 'Group X coordinate',
        'gy': 'Group Y coordinate',
        'counit': 'Coordinate units',
        
        # Timing and sampling
        'delrt': 'Delay recording time',
        'muts': 'Mute time start time in ms',
        'mute': 'Mute time end time in ms',
        'dt': 'Sample interval in ms for this trace',
        'gain': 'Gain type of field instruments',
        'igc': 'Instrument gain constant',
        'igi': 'Instrument early or initial gain',
        'corr': 'Correlated data flag',
        'ns': 'Number of samples in this trace',
        
        # Filter parameters
        'sfs': 'Sweep frequency at start',
        'sfe': 'Sweep frequency at end',
        'slen': 'Sweep length in ms',
        'styp': 'Sweep type',
        'stat': 'Sweep trace taper length at start in ms',
        'stae': 'Sweep trace taper length at end in ms',
        
        # Additional parameters
        'year': 'Year data recorded',
        'day': 'Day of year',
        'hour': 'Hour of day',
        'minute': 'Minute of hour',
        'sec': 'Second of minute',
        'scalel': 'Scalar to be applied to all elevations and depths',
        'scalco': 'Scalar to be applied to all coordinates',
        
        # Weathering velocity and static corrections
        'swevel': 'Subweathering velocity',
        'wevel': 'Weathering velocity',
        'sut': 'Uphole time at source in ms',
        'gut': 'Uphole time at group in ms',
        'sstat': 'Source static correction in ms',
        'gstat': 'Group static correction in ms',
        'tstat': 'Total static applied in ms'
    }

    # Description for ObsPy stats fields
    stats_descriptions = {
        'network': 'Network code (e.g., station network/array)',
        'station': 'Station code (e.g., seismic station identifier)',
        'location': 'Location code (e.g., survey area identifier)',
        'channel': 'Channel code (e.g., component identifier)',
        'starttime': 'Start time of the trace',
        'endtime': 'End time of the trace',
        'sampling_rate': 'Sampling rate in samples per second',
        'delta': 'Sampling interval in seconds',
        'npts': 'Number of sample points in the trace',
        'calib': 'Calibration factor (conversion to ground motion units)'
    }
    
    # Define field types for different categories
    field_types = {
        'string_fields': ['network', 'station', 'location', 'channel'],
        'integer_fields': ['tracl', 'tracr', 'fldr', 'tracf', 'ep', 'cdp', 'trid', 'nvs', 'nhs', 'ns',
                          'year', 'day', 'hour', 'minute', 'sec', 'counit', 'corr', 'styp',
                          'offset', 'gelev', 'selev', 'sdepth', 'gdel', 'sdel', 'swdep', 'gwdep',
                          'sx', 'sy', 'gx', 'gy', 'dt', 'delrt', 'muts', 'mute',
                          'gain', 'igc', 'igi', 'sfs', 'sfe', 'slen', 'stat', 'stae',
                          'scalel', 'scalco', 'swevel', 'wevel', 'sut', 'gut', 'sstat', 'gstat', 'tstat'],
        'float_fields': ['sampling_rate', 'delta', 'calib'],
        'datetime_fields': ['starttime', 'endtime']
    }

    def __init__(self, su_file: str, headonly: bool = True):
        """
        Initialize SUProcessor with a SU file.
        
        Args:
            su_file (str): Path to the SU file
            headonly (bool, optional): If True, only load header information.
                                     If False, load complete data including traces.
                                     Defaults to True for memory efficiency.
        """
        self.su_file = su_file
        self.su_data = None
        self.header_info = None
        self.header_values = {}  # Store SU header values
        self.stats_values = {}   # Store stats values
        
        if headonly:
            self._load_header()
        else:
            self._load_data()
    
    def _load_header_values(self) -> None:
        """Load and store all header values from traces."""
        if not self.su_data or not self.su_data.traces:
            return
            
        # Initialize arrays for each available field
        n_traces = len(self.su_data.traces)
        
        # Store SU header values
        for field in self.su_header_map:
            if self.header_info.get(field, False):
                internal_name = self.su_header_map[field]
                
                # Determine field type and create appropriate array
                if field in self.field_types['string_fields']:
                    values = np.array([''] * n_traces, dtype=object)
                elif field in self.field_types['integer_fields']:
                    values = np.zeros(n_traces, dtype=np.int32)
                else:  # float fields
                    values = np.zeros(n_traces, dtype=np.float64)
                
                # Fill values
                for i, trace in enumerate(self.su_data.traces):
                    if hasattr(trace.stats.su.trace_header, internal_name):
                        value = getattr(trace.stats.su.trace_header, internal_name)
                        if value == '':
                            if field in self.field_types['string_fields']:
                                values[i] = ''
                            else:
                                values[i] = np.nan
                        else:
                            try:
                                if field in self.field_types['string_fields']:
                                    values[i] = str(value)
                                elif field in self.field_types['integer_fields']:
                                    values[i] = int(value)
                                else:
                                    values[i] = float(value)
                            except (ValueError, TypeError):
                                if field in self.field_types['string_fields']:
                                    values[i] = ''
                                else:
                                    values[i] = np.nan
                    else:
                        if field in self.field_types['string_fields']:
                            values[i] = ''
                        else:
                            values[i] = np.nan
                            
                self.header_values[field] = values
        
        # Store stats values
        for field in self.stats_header_map:
            if self.header_info.get(f"stats_{field}", False):
                # Determine field type and create appropriate array
                if field in self.field_types['string_fields']:
                    values = np.array([''] * n_traces, dtype=object)
                elif field in self.field_types['integer_fields']:
                    values = np.zeros(n_traces, dtype=np.int32)
                elif field in self.field_types['datetime_fields']:
                    values = np.zeros(n_traces, dtype=np.float64)  # store as timestamps
                else:  # float fields
                    values = np.zeros(n_traces, dtype=np.float64)
                
                # Fill values
                for i, trace in enumerate(self.su_data.traces):
                    if hasattr(trace.stats, field):
                        value = getattr(trace.stats, field)
                        try:
                            if field in self.field_types['string_fields']:
                                values[i] = str(value) if value != '' else ''
                            elif field in self.field_types['datetime_fields']:
                                values[i] = value.timestamp
                            elif field in self.field_types['integer_fields']:
                                values[i] = int(value)
                            else:
                                values[i] = float(value)
                        except (ValueError, TypeError):
                            if field in self.field_types['string_fields']:
                                values[i] = ''
                            else:
                                values[i] = np.nan
                    else:
                        if field in self.field_types['string_fields']:
                            values[i] = ''
                        else:
                            values[i] = np.nan
                            
                self.stats_values[field] = values
    
    def _load_header(self) -> None:
        """Load only headers from SU file."""
        if not os.path.exists(self.su_file):
            raise FileNotFoundError(f"SU file not found: {self.su_file}")
        
        # Read headers only
        self.su_data = read(self.su_file, headonly=True)
        self.header_info = self._check_headers()
        self._load_header_values()
    
    def _load_data(self) -> None:
        """Load complete SU data including traces."""
        if not os.path.exists(self.su_file):
            raise FileNotFoundError(f"SU file not found: {self.su_file}")
        
        print("Loading complete SU data...")
        self.su_data = read(self.su_file)
        print(f"Loaded {len(self.su_data.traces)} traces with {len(self.su_data.traces[0].data)} samples each")
        self.header_info = self._check_headers()
        self._load_header_values()
        
    def _check_headers(self) -> Dict:
        """
        Check SU file trace headers and return available information.
        
        Returns:
            dict: Dictionary containing header information about coordinate fields
        """
        if not self.su_data.traces:
            return None
        
        first_trace = self.su_data.traces[0]
        header_info = {}

        # Check SU trace headers
        for standard_name, internal_name in self.su_header_map.items():
            header_info[standard_name] = hasattr(first_trace.stats.su.trace_header, internal_name)
        
        # Check ObsPy stats headers
        for standard_name in self.stats_header_map:
            header_info[f"stats_{standard_name}"] = hasattr(first_trace.stats, standard_name)
        
        return header_info
    
    def print_header_info(self) -> None:
        """Print all available header information."""
        if not self.su_data.traces:
            print("No traces found in SU file")
            return
            
        first_trace = self.su_data.traces[0]
        
        print("\nAll available header information:")
        print("-" * 30)
        for key, value in first_trace.stats.items():
            print(f"{key}: {value}")
    
    def apply_bandpass_filter(self, 
                            freqmin: float, 
                            freqmax: float, 
                            corners: int = 4) -> None:
        """
        Apply bandpass filter to seismic data.
        
        Args:
            freqmin: Lower frequency bound
            freqmax: Upper frequency bound
            corners: Number of filter corners
        """
        # Ensure complete data is loaded before filtering
        if not self.su_data or self.su_data.traces[0].data is None:
            self._load_data()
            
        for trace in self.su_data.traces:
            trace.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=corners)
    
    def apply_agc(self, window_length: float) -> None:
        """
        Apply Automatic Gain Control to normalize amplitudes within a sliding window.
        
        Args:
            window_length (float): Length of AGC window in seconds
        """
        # Ensure complete data is loaded before applying AGC
        if not self.su_data or self.su_data.traces[0].data is None:
            self._load_data()
            
        # Get sampling rate from first trace
        dt = self.su_data.traces[0].stats.delta
        window_samples = int(window_length / dt)
        
        # Process each trace
        for trace in self.su_data.traces:
            data = trace.data
            n_samples = len(data)
            
            # Create output array
            output = np.zeros_like(data)
            
            # Calculate AGC weights
            for i in range(n_samples):
                # Define window boundaries
                half_window = window_samples // 2
                start = max(0, i - half_window)
                end = min(n_samples, i + half_window)
                
                # Calculate RMS in window
                window_data = data[start:end]
                rms = np.sqrt(np.mean(window_data**2))
                
                # Apply AGC weight
                if rms > 0:
                    output[i] = data[i] / rms
                    
            # Update trace data
            trace.data = output
    
    def compute_spectrum(self, trace_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute frequency spectrum for a trace.
        
        Args:
            trace_index: Index of trace to analyze
            
        Returns:
            Tuple of frequencies and amplitudes
        """
        # Ensure complete data is loaded before computing spectrum
        if not self.su_data or self.su_data.traces[0].data is None:
            self._load_data()
            
        # Implementation of spectrum computation
        pass
    
    def get_amplitude_statistics(self) -> Dict[str, float]:
        """
        Calculate basic statistics of trace amplitudes.
        
        Returns:
            Dictionary containing min, max, mean, std of amplitudes
        """
        # Ensure complete data is loaded before calculating statistics
        if not self.su_data or self.su_data.traces[0].data is None:
            self._load_data()
            
        # Implementation of statistics calculation
        pass
    
    def to_grid(self, 
                output_grid: str,
                dy: float,
                y_min: float,
                x_field: Optional[str] = None,
                dx: Optional[float] = None,
                x_min: Optional[float] = None,
                method: str = 'linear',
                max_distance: Optional[float] = None,
                spacing_threshold: Optional[float] = None
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert SU data to regular grid format.
        
        Args:
            output_grid (str): Output grid file path
            dy (float): Grid spacing in y direction
            y_min (float): Minimum coordinate for y
            x_field (str, optional): Field to use for x-coordinates. If not provided,
                                   will try to use fields in default order: ['offset', 'cdp', 'fldr', 'sx', 'gx']
            dx (float, optional): Grid spacing in x direction
            x_min (float, optional): Minimum coordinate for x
            method (str, optional): Interpolation method ('linear', 'cubic', 'nearest')
            max_distance (float, optional): Maximum allowed distance for interpolation
            spacing_threshold (float, optional): Threshold for identifying gaps
            
        Returns:
            tuple: (data, x, y) arrays of the gridded data
        """
        # Ensure complete data is loaded before gridding
        try:
            first_trace_data = self.su_data.traces[0].data
            if first_trace_data is None or len(first_trace_data) == 0:
                self._load_data()
        except (AttributeError, IndexError):
            self._load_data()
            
        # Get number of samples from first trace's data
        n_samples = len(self.su_data.traces[0].data)
        print(f"Number of samples from trace data: {n_samples}")
        
        # Create y-coordinates first
        y = np.arange(y_min, y_min + n_samples * dy, dy)
        if len(y) == 0:
            raise ValueError(f"Invalid y-coordinates: min={y_min}, dy={dy}, n_samples={n_samples}, y_length={len(y)}")
        
        # Default order for x-coordinate fields
        default_x_fields = ['offset', 'cdp', 'fldr', 'sx', 'gx']
        
        # If x_field is specified, try to use it
        if x_field is not None:
            if x_field not in self.header_info:
                available_fields = [f for f in default_x_fields if self.header_info.get(f, False)]
                raise ValueError(f"Specified field '{x_field}' not found in headers. "
                               f"Available coordinate fields: {available_fields}")
            
            x_coords = self.get_header_values(x_field)
            # Convert offset from meters to kilometers if x_field is 'offset'
            if x_field == 'offset':
                x_coords = x_coords / 1000.0  # Convert to km
                
            if np.all(np.isnan(x_coords)):
                available_fields = [f for f in default_x_fields if not np.all(np.isnan(self.get_header_values(f)))]
                raise ValueError(f"Specified field '{x_field}' contains no valid values. "
                               f"Available coordinate fields with valid values: {available_fields}")
            
            print(f"\nUsing specified field {x_field} ({self.field_descriptions[x_field]}) for x-coordinates")
        
        # If x_field is not specified, try default fields in order
        else:
            x_coords = None
            used_field = None
            
            for field in default_x_fields:
                if self.header_info.get(field, False):
                    x_coords = self.get_header_values(field)
                    # Convert offset from meters to kilometers if the field is 'offset'
                    if field == 'offset':
                        x_coords = x_coords / 1000.0  # Convert to km
                        
                    if not np.all(np.isnan(x_coords)):
                        used_field = field
                        print(f"\nUsing {field} ({self.field_descriptions[field]}) for x-coordinates")
                        break
            
            if x_coords is None or used_field is None:
                available_fields = [f for f in default_x_fields if self.header_info.get(f, False)]
                raise ValueError(f"No valid x-coordinate field found. Available fields: {available_fields}")
        
        # Remove any NaN values
        valid_mask = ~np.isnan(x_coords)
        x_coords = x_coords[valid_mask]

        
        # Get traces data for valid coordinates only
        valid_traces = []
        for i, mask in enumerate(valid_mask):
            if mask:
                valid_traces.append(self.su_data.traces[i].data)
        
        if not valid_traces:
            raise ValueError("No valid traces found after filtering")
           
        data = np.array(valid_traces).T  # Transpose to get (time, trace) shape
        print("data shape",data.shape) 
        # Sort coordinates and data
        sort_idx = np.argsort(x_coords)
        x_orig = x_coords[sort_idx]
        data = data[:, sort_idx]
        
        # Calculate trace spacing
        trace_spacing = np.diff(x_orig)
        if len(trace_spacing) == 0:
            raise ValueError("Not enough valid traces for gridding")
            
        # Identify and analyze gaps in trace spacing
        if spacing_threshold is None:
            spacing_threshold = 3.0
            
        median_spacing = np.median(trace_spacing)
        gap_mask = trace_spacing > (median_spacing * spacing_threshold)
        regular_spacing = trace_spacing[~gap_mask]
        
        if len(regular_spacing) == 0:
            raise ValueError("No regular trace spacing found after gap analysis")
            
        # Calculate statistics using only regular spacings
        mean_spacing = np.mean(regular_spacing)
        std_spacing = np.std(regular_spacing)
        
        # Print spacing statistics
        print(f"\nTrace spacing statistics:")
        print(f"Median spacing: {median_spacing:.2f}")
        print(f"Mean regular spacing: {mean_spacing:.2f}")
        print(f"Std regular spacing: {std_spacing:.2f}")
        
        if np.any(gap_mask):
            gap_spacings = trace_spacing[gap_mask]
            print(f"\nFound {len(gap_spacings)} gaps in trace spacing:")
            for i, gap in enumerate(gap_spacings):
                gap_start = x_orig[np.where(gap_mask)[0][i]]
                print(f"Gap {i+1}: {gap:.2f} units at position {gap_start:.2f}")
        
        # Set maximum interpolation distance if not provided
        if max_distance is None:
            max_distance = 1.5 * mean_spacing
        print(f"\nMaximum interpolation distance: {max_distance:.2f}")
        
        # Create regular grid for interpolation
        if dx is None:
            # If dx not provided, use mean regular spacing
            dx = mean_spacing
        
        if x_min is None:
            x_min = x_orig.min()
        x_reg = np.arange(x_min, x_orig.max() + dx, dx)
        
        if len(x_reg) == 0:
            raise ValueError(f"Invalid x-coordinates: min={x_min}, dx={dx}, max={x_orig.max()}")
            
        # Interpolate data to regular grid
        print(f"\nInterpolating to regular grid using {method} method...")
        data_reg = np.zeros((len(y), len(x_reg)))
        
        # Create interpolation functions for each time sample
        if method == 'linear':
            for i in range(len(y)):
                # Find points within maximum distance
                valid_points = []
                valid_values = []
                for j, x_point in enumerate(x_reg):
                    distances = np.abs(x_orig - x_point)
                    min_dist = np.min(distances)
                    if min_dist <= max_distance:
                        valid_points.append(x_point)
                        nearest_idx = np.argmin(distances)
                        valid_values.append(data[i, nearest_idx])
                
                if valid_points:
                    data_reg[i, :] = np.interp(x_reg, np.array(valid_points), np.array(valid_values))
                
        elif method == 'cubic':
            from scipy.interpolate import CubicSpline
            for i in range(len(y)):
                # Sort x_orig and corresponding data
                sort_idx = np.argsort(x_orig)
                x_sorted = x_orig[sort_idx]
                data_sorted = data[i, sort_idx]
                
                # Create cubic spline interpolator
                cs = CubicSpline(x_sorted, data_sorted, extrapolate=False)
                
                # Interpolate only within the valid range
                mask = (x_reg >= x_sorted[0]) & (x_reg <= x_sorted[-1])
                data_reg[i, mask] = cs(x_reg[mask])
                
        else:  # nearest
            for i in range(len(y)):
                for j, x_point in enumerate(x_reg):
                    distances = np.abs(x_orig - x_point)
                    min_dist = np.min(distances)
                    if min_dist <= max_distance:
                        nearest_idx = np.argmin(distances)
                        data_reg[i, j] = data[i, nearest_idx]
        
        # Print final grid information
        print(f"\nFinal grid dimensions: {len(y)} x {len(x_reg)}")
        print(f"X range: {x_reg.min():.2f} to {x_reg.max():.2f}")
        print(f"Y range: {y.min():.2f} to {y.max():.2f}")
        
        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars=dict(z=(["y", "x"], data_reg)),
            coords=dict(x=x_reg, y=y)
        )
        
        # Save as netCDF grid file
        if output_grid:
            ds.to_netcdf(output_grid)
        return ds


    def export_segy(self, output_file: str) -> None:
        """
        Export data to SEG-Y format.
        
        Args:
            output_file: Path to output SEG-Y file
        """
        # Implementation of SEG-Y export
        pass
    
    def plot_trace(self, 
                  trace_index: int = 0, 
                  output_file: Optional[str] = None) -> None:
        """
        Plot a single trace.
        
        Args:
            trace_index: Index of trace to plot
            output_file: Optional path to save plot
        """
        # Ensure complete data is loaded before plotting
        if not self.su_data or self.su_data.traces[0].data is None:
            self._load_data()
            
        # Implementation of trace plotting
        pass

    @staticmethod
    def plot_grids(grid_files: List[str],
                   output_fig: str,
                   region: List[float],
                   projection: str = "X15c/-8c",  # Negative y scale to reverse y-axis
                   transparency_values: Optional[List[float]] = None,
                   colormaps: Optional[List[str]] = None,
                   use_shading: bool = False,
                   azimuth: float = 45,
                   norm_method: Union[bool, str] = "e",
                   norm_amp: Optional[float] = None,
                   norm_sigma: Optional[float] = None,
                   norm_offset: Optional[float] = None,
                   ambient: Optional[float] = None,
                   cpt_reverse: bool = False,  # Whether to reverse the colormap
                   cpt_continuous: bool = True,  # Whether to use continuous colormap
                   cpt_series: Optional[Union[str, List[str]]] = None  # Series parameter for CPT creation
                   ) -> None:
        """
        Plot multiple grid files with transparency in a vertical layout.
        
        Args:
            grid_files (list): List of grid files to plot
            output_fig (str): Output figure path
            region (list): Plot region [xmin, xmax, ymin, ymax]
            projection (str): GMT projection (default: X15c/-8c for Cartesian with reversed y-axis)
            transparency_values (list, optional): Transparency values for each grid.
                                               If list is shorter than grid_files, the last value
                                               will be used for remaining grids.
            colormaps (list, optional): Colormaps for each grid. Can be:
                                      1. Path to a .cpt file
                                      2. Built-in GMT colormap name (e.g., 'polar', 'seis')
                                      If list is shorter than grid_files, the last value
                                      will be used for remaining grids.
            use_shading (bool, optional): Whether to apply shading effect. Default is False.
            azimuth (float, optional): Azimuth angle for shading in degrees. Default is 45.
            norm_method (str or bool, optional): Normalization method for shading.
            norm_amp (float, optional): Maximum output magnitude after normalization.
            norm_sigma (float, optional): Sigma parameter for normalization.
            norm_offset (float, optional): Offset parameter for normalization.
            ambient (float, optional): Ambient light to add after normalization.
            cpt_reverse (bool, optional): Whether to reverse the colormap. Default is False.
            cpt_continuous (bool, optional): Whether to use continuous colormap. Default is True.
            cpt_series (str or list, optional): Series parameter for CPT creation. Can be:
                                              1. A string in format "min/max/inc[+b|l|n]"
                                              2. A list of strings, one for each grid
                                              If not provided, pygmt will make default color boundaries.
             """
        n_grids = len(grid_files)
        
        # Handle transparency values
        if transparency_values is None:
            transparency_values = [0] * n_grids
        elif len(transparency_values) < n_grids:
            last_value = transparency_values[-1]
            transparency_values.extend([last_value] * (n_grids - len(transparency_values)))
        
        # Handle colormaps
        if colormaps is None:
            colormaps = ["polar"] * n_grids
        elif len(colormaps) < n_grids:
            last_cmap = colormaps[-1]
            colormaps.extend([last_cmap] * (n_grids - len(colormaps)))
            
        # Handle series parameter
        if isinstance(cpt_series, str):
            cpt_series = [cpt_series] * n_grids
        elif isinstance(cpt_series, list) and len(cpt_series) < n_grids:
            last_series = cpt_series[-1]
            cpt_series = cpt_series + [last_series] * (n_grids - len(cpt_series))
            
        fig = pygmt.Figure()
        
        if n_grids > 1:
            # Set up vertical subplot layout
            fig.subplot(
                nrows=n_grids,
                ncols=1,
                figsize=("15c", f"{8*n_grids}c"),
                frame=["WSen", "xa", "ya"],
                margins=["1c", "1c"]
            )
        
        for i, (grid_file, transparency, cmap) in enumerate(zip(grid_files, transparency_values, colormaps)):
            # Check if cmap is a path to a .cpt file
            if isinstance(cmap, str) and cmap.lower().endswith('.cpt') and os.path.exists(cmap):
                cpt_file = cmap
            else:
                # Create custom CPT for this grid using built-in colormap
                cpt_file = f"grid_{i}.cpt"
                
                # Determine series parameter
                if cpt_series is not None:
                    series = cpt_series[i]
                else:
                    series = None  # Let PyGMT handle the data range automatically
                
                pygmt.grd2cpt(
                    grid=grid_file,
                    cmap=cmap,
                    continuous=cpt_continuous,
                    reverse=cpt_reverse,
                    series=series,  # Will be None if cpt_series not provided
                    output=cpt_file  # Output CPT file
                )
            
            if n_grids > 1:
                with fig.set_panel(panel=i):
                    fig.basemap(
                        region=region,
                        projection=projection,
                        frame=["xa", "ya", "WSen"]
                    )
                    
                    if use_shading:
                        # Build normalization string
                        norm_str = ""
                        if isinstance(norm_method, str):
                            norm_str = norm_method
                        if norm_amp is not None:
                            norm_str += str(norm_amp)
                        if norm_sigma is not None:
                            norm_str += f"+s{norm_sigma}"
                        if norm_offset is not None:
                            norm_str += f"+o{norm_offset}"
                        if ambient is not None:
                            norm_str += f"+a{ambient}"
                        
                        # Create shading effect
                        shade = pygmt.grdgradient(
                            grid=grid_file,
                            azimuth=azimuth,
                            normalize=norm_str if norm_str else norm_method
                        )
                        # Plot grid image with shading
                        fig.grdimage(
                            grid=grid_file,
                            shading=shade,
                            cmap=cpt_file,  # Use the CPT file
                            transparency=transparency * 100,
                            nan_transparent=True  # Make NaN values transparent
                        )
                    else:
                        # Plot grid image without shading
                        fig.grdimage(
                            grid=grid_file,
                            cmap=cpt_file,  # Use the CPT file
                            transparency=transparency * 100,
                            nan_transparent=True  # Make NaN values transparent
                        )
                    
                    # Add colorbar for each subplot
                    fig.colorbar(
                        cmap=cpt_file,  # Use the same CPT file
                        frame=["af", "+lAmplitude"],
                        position="JMR+o0.5c/0c+w8c"  # Place colorbar on the right side
                    )
            else:
                # Single plot without subplot
                fig.basemap(
                    region=region,
                    projection=projection,
                    frame=["xa", "ya", "WSen"]
                )
                
                if use_shading:
                    # Build normalization string
                    norm_str = ""
                    if isinstance(norm_method, str):
                        norm_str = norm_method
                    if norm_amp is not None:
                        norm_str += str(norm_amp)
                    if norm_sigma is not None:
                        norm_str += f"+s{norm_sigma}"
                    if norm_offset is not None:
                        norm_str += f"+o{norm_offset}"
                    if ambient is not None:
                        norm_str += f"+a{ambient}"
                    
                    # Create shading effect
                    shade = pygmt.grdgradient(
                        grid=grid_file,
                        azimuth=azimuth,
                        normalize=norm_str if norm_str else norm_method
                    )
                    # Plot grid image with shading
                    fig.grdimage(
                        grid=grid_file,
                        shading=shade,
                        cmap=cpt_file,  # Use the CPT file
                        transparency=transparency * 100,
                        nan_transparent=True  # Make NaN values transparent
                    )
                else:
                    # Plot grid image without shading
                    fig.grdimage(
                        grid=grid_file,
                        cmap=cpt_file,  # Use the CPT file
                        transparency=transparency * 100,
                        nan_transparent=True  # Make NaN values transparent
                    )
                
                # Add colorbar
                fig.colorbar(
                    cmap=cpt_file,  # Use the same CPT file
                    frame=["af", "+lAmplitude"],
                    position="JMR+o0.5c/0c+w8c"  # Place colorbar on the right side
                )
            
            # Clean up temporary CPT file only if we created it
            if not cmap.lower().endswith('.cpt') and os.path.exists(cpt_file):
                os.remove(cpt_file)
        
        if n_grids > 1:
            fig.subplot(endplot=True)  # End the subplot
            
        fig.savefig(output_fig)

    def get_header_value(self, trace_index: int, field: str) -> Optional[float]:
        """
        Get the value of a header field for a specific trace using standard field name.
        
        Args:
            trace_index: Index of the trace
            field: Standard field name (e.g., 'offset', 'sx', 'cdp', or 'stats_sampling_rate')
            
        Returns:
            Value of the header field or None if field is not available
        """
        if not self.su_data or trace_index >= len(self.su_data.traces):
            return None
        
        # Check if it's a stats field
        if field.startswith('stats_'):
            stats_field = field[6:]  # Remove 'stats_' prefix
            if stats_field not in self.stats_header_map:
                raise ValueError(f"Unknown stats field name: {stats_field}")
            if stats_field in self.stats_values:
                return self.stats_values[stats_field][trace_index]
            return None
        
        # Handle regular SU header fields
        if field not in self.su_header_map:
            raise ValueError(f"Unknown field name: {field}")
        
        if field in self.header_values:
            return self.header_values[field][trace_index]
        return None
    
    def get_header_values(self, field: str) -> np.ndarray:
        """
        Get the values of a header field for all traces using standard field name.
        
        Args:
            field: Standard field name (e.g., 'offset', 'sx', 'cdp', or 'stats_sampling_rate')
            
        Returns:
            Array of header field values
        """
        if field.startswith('stats_'):
            stats_field = field[6:]
            if stats_field not in self.stats_header_map:
                raise ValueError(f"Unknown stats field name: {stats_field}")
            return self.stats_values.get(stats_field, np.array([]))
            
        if field not in self.su_header_map:
            raise ValueError(f"Unknown field name: {field}")
        return self.header_values.get(field, np.array([]))
    
    def get_available_fields(self) -> Dict[str, str]:
        """
        Get a dictionary of available header fields and their descriptions.
        
        Returns:
            Dictionary mapping available standard field names to their descriptions
        """
        if not self.header_info:
            return {}
            
        available_fields = {}
        
        # Add available SU header fields
        for field in self.header_info:
            if not field.startswith('stats_') and self.header_info[field]:
                available_fields[field] = self.field_descriptions[field]
        
        # Add available stats fields
        for field in self.header_info:
            if field.startswith('stats_') and self.header_info[field]:
                stats_field = field[6:]  # Remove 'stats_' prefix
                available_fields[field] = self.stats_descriptions[stats_field]
        
        return available_fields
    
    def print_available_fields(self) -> None:
        """Print all available header fields and their descriptions."""
        available_fields = self.get_available_fields()
        
        if not available_fields:
            print("No header fields available")
            return
            
        print("\nAvailable header fields:")
        print("-" * 30)
        for field, description in available_fields.items():
            print(f"{field}: {description}")

    def get_header_stats(self, field: str) -> Dict[str, float]:
        """
        Get basic statistics for a header field across all traces.
        
        Args:
            field: Standard field name (e.g., 'offset', 'sx', 'cdp')
            
        Returns:
            Dictionary containing min, max, mean, std of the field values
        """
        values = self.get_header_values(field)
        if len(values) == 0:
            return {}
            
        return {
            'min': np.nanmin(values),
            'max': np.nanmax(values),
            'mean': np.nanmean(values),
            'std': np.nanstd(values),
            'count': np.sum(~np.isnan(values))
        }

    def process_su_file(self, input_file: str, output_file: str, 
                       dx: Optional[float] = None, dy: float = 0.01,
                       method: str = 'linear',
                       max_dist: Optional[float] = None,
                       field: str = 'offset') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """处理SU文件并输出为网格文件。
        
        Args:
            input_file: 输入的SU文件路径
            output_file: 输出的网格文件路径
            dx: x方向的网格间距，默认为None时使用平均道间距
            dy: y方向的网格间距
            method: 插值方法，可选 'linear', 'nearest', 'cubic'
            max_dist: 最大插值距离，默认为dx的2倍
            field: 用于x坐标的字段，可选 'offset' 或 'cdp'
        
        Returns:
            tuple: (data, x, y) arrays of the gridded data
        """
        # 读取SU文件
        stream = read(input_file)
        
        # 获取道集数据和头文件信息
        traces = []
        x_coords = []
        
        for trace in stream:
            traces.append(trace.data)
            if field == 'offset':
                x_coord = trace.stats.su.trace_header.distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group / 1000.0  # 将米转换为公里
            elif field == 'cdp':
                x_coord = trace.stats.su.trace_header.ensemble_number
            else:
                raise ValueError(f"不支持的字段类型: {field}")
            x_coords.append(x_coord)
            
        # 转换为numpy数组
        traces = np.array(traces).T  # 转置以获得 (time, trace) 形状
        x = np.array(x_coords)
        
        # 获取时间采样
        dt = stream[0].stats.delta
        nt = stream[0].stats.npts
        t = np.arange(nt) * dt
        
        # 创建输出网格的坐标
        x_min, x_max = np.min(x), np.max(x)
        t_min, t_max = np.min(t), np.max(t)
        
        # 计算平均道间距
        mean_spacing = np.mean(np.diff(np.sort(x)))
        if dx is None:
            dx = mean_spacing
            print(f"Using mean trace spacing as dx: {dx:.4f}")
        
        x_grid = np.arange(x_min, x_max + dx, dx)
        t_grid = np.arange(t_min, t_max + dy, dy)
        
        print(f"\nTrace spacing statistics:")
        print(f"Median spacing: {np.median(np.diff(np.sort(x))):.4f}")
        print(f"Mean regular spacing: {mean_spacing:.4f}")
        print(f"Std regular spacing: {np.std(np.diff(np.sort(x))):.4f}")
        
        if max_dist is None:
            max_dist = 2 * dx
        print(f"\nMaximum interpolation distance: {max_dist:.2f}\n")
        
        # 创建输出网格
        X, T = np.meshgrid(x_grid, t_grid)
        grid = np.zeros_like(X)
        
        print("Interpolating to regular grid using linear method...")
        
        # 对每个时间采样进行插值
        for i in range(nt):
            data = traces[i, :]
            
            # 使用指定方法进行插值
            if method == 'linear':
                interp = interp1d(x, data, 
                                kind='linear', 
                                bounds_error=False, 
                                fill_value=np.nan)
            elif method == 'nearest':
                interp = interp1d(x, data, 
                                kind='nearest', 
                                bounds_error=False, 
                                fill_value=np.nan)
            elif method == 'cubic':
                interp = interp1d(x, data, 
                                kind='cubic', 
                                bounds_error=False, 
                                fill_value=np.nan)
            else:
                raise ValueError(f"不支持的插值方法: {method}")
                
            grid[i, :] = interp(x_grid)
        
        print(f"\nFinal grid dimensions: {grid.shape[0]} x {grid.shape[1]}")
        print(f"X range: {x_min:.2f} to {x_max:.2f}")
        print(f"Y range: {t_min:.2f} to {t_max:.2f}")
        
        # 创建xarray数据集
        ds = xr.Dataset(
            data_vars={
                'amplitude': (('t', 'x'), grid)
            },
            coords={
                'x': x_grid,
                't': t_grid
            }
        )
        
        # 保存为netCDF格式
        ds.to_netcdf(output_file)
        return ds

def main():
    """Test SU data processing and visualization functionality."""
    su_file = "207b_TWIOBS02_offset.su"  # Replace with your SU file
    
    if os.path.exists(su_file):
        try:
            # Create processor instance
            processor = SUProcessor(su_file)
            
            # Test to_grid with different parameters
            print("\n1. Testing to_grid method:")
            print("-" * 50)
            
            # Get sampling parameters from the data
            dt = processor.get_header_value(0, 'dt')  # sampling interval in microseconds
            if dt is None or np.isnan(dt):
                dt = 4000  # default to 4ms
            dy = dt / 1_000_000.0  # convert microseconds to seconds
            
            n_samples = processor.get_header_value(0, 'ns')  # number of samples
            if n_samples is None or np.isnan(n_samples):
                n_samples = 3001  # default value
            
            y_min = 0.0  # start from zero time
            
            print(f"\nSampling parameters:")
            print(f"dt = {dt} microseconds ({dy:.6f} seconds)")
            print(f"n_samples = {n_samples}")
            print(f"y_min = {y_min}")
            
            # Create output directory if it doesn't exist
            os.makedirs('grid_outputs', exist_ok=True)
            
            # Test linear interpolation methods
            grid_files = []
            
            # Test with specific x-coordinate field
            try:
                print("\nTesting with specified x-coordinate field (offset)...")
                output_grid = 'su_outputs/grid_offset_north.grd'
                if False:
                        ds = processor.to_grid(
                        output_grid=output_grid,
                        dy=dy,
                        y_min=y_min,
                        x_field='offset',
                        method='linear',
                        spacing_threshold=2.0
                    )
                ds=processor.process_su_file(su_file, output_grid, dy=dy, method='linear', field='offset')
                grid_files.append(output_grid)
                print(f"Grid created successfully using 'offset' field")
                
            except ValueError as e:
                print(f"Error using 'offset' field: {e}")
            
            
            # Test plot_grids if we have created any grids
            if grid_files:
                print("\n2. Testing plot_grids method:")
                print("-" * 50)
                
                # Get data extent for plotting using the first two default fields
                x_min = processor.get_header_values('offset').min() * 0.001
                x_max = processor.get_header_values('offset').max() * 0.001
                y_max = n_samples * dy
                
                # Create comparison plot
                output_fig = 'su_outputs/comparison_offset.png'
                print(f"\nCreating comparison plot: {output_fig}")
                
                try:
                    SUProcessor.plot_grids(
                        grid_files=grid_files,
                        output_fig=output_fig,
                        region=[x_min, x_max, y_min, y_max],
                        transparency_values=[0],  # no transparency
                        colormaps=['polar'],  # use seis colormap
                        use_shading=True,  # enable shading effect
                        azimuth=135,  # light source direction
                        norm_method='e',  # use Laplace distribution for normalization
                        norm_amp=1.0,  # maximum amplitude
                        cpt_reverse=False,  # don't reverse colormap
                        cpt_continuous=True,  # use continuous colormap
                        cpt_series=None  # no series parameter
                    )
                    print("Plot created successfully")
                    
                except Exception as e:
                    print(f"Error creating comparison plot: {e}")
            
        except Exception as e:
            print(f"Error processing SU file: {e}")
            raise  # Re-raise the exception for debugging
    else:
        print(f"Input file {su_file} not found!")

if __name__ == "__main__":
    main() 