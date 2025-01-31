"""
Velocity Model Reader Module

This module provides functions for reading and processing various formats
of velocity model files (v.in, zeltform, etc.).

Functions:
    read_vin_model: Read a ZELT format velocity model file
    process_velocity_model: Process velocity model data (e.g., compute averages)
    write_vin_model: Write processed model data to file
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..utils import get_logger
from .models import Point2d, ZNode2d

logger = get_logger(__name__)

def read_three_lines(file) -> Optional[Tuple[List[float], List[float], List[int], bool]]:
    """Read three lines of ZELT format data.
    
    Args:
        file: File object to read from
        
    Returns:
        Optional[Tuple[List[float], List[float], List[int], bool]]: 
            Tuple of (x_coordinates, values, flags, is_bottom_boundary) if successful, None otherwise
            For bottom boundary points (less than 10 points), returns coordinates and True flag
    """
    nrz = 1
    j1 = 0
    j2 = 9
    line1_row, line2_row, line3_row = [], [], []
    
    while True:
        try:
            # Read first line (x coordinates)
            line1 = file.readline().strip()
            if not line1:  # End of file
                break
            parts1 = line1.split()
            ilyr = int(parts1[0])
            line1_row.extend([float(x) for x in parts1[1:]])
            
            # Read second line (values)
            line2 = file.readline().strip()
            if not line2:  # End of file
                break
            parts2 = line2.split()
            icnt = int(parts2[0])
            line2_row.extend([float(x) for x in parts2[1:]])
            
            # Read third line (flags) if not bottom boundary
            line3 = file.readline().strip()
            if line3:  # If third line exists
                parts3 = line3.split()
                line3_row.extend([int(x) for x in parts3[0:]])
            elif icnt == 0:  # Bottom boundary case
                line3_row.extend([0] * len(line1_row))  # Fill with zeros
                return line1_row, line2_row, line3_row, True
            
            nrz += 1
            if icnt != 1:
                return line1_row, line2_row, line3_row, False
                
            j1 += 10
            j2 += 10
            logger.debug(f"Read data for layer {ilyr}")
            
        except Exception as e:
            logger.error(f"Error reading layer data: {e}")
            return None
            
    return None

def check_lengths(velocity_model: Dict) -> bool:
    """Validate the consistency of array lengths in the velocity model.
    
    Args:
        velocity_model (Dict): Dictionary containing velocity model data
        
    Returns:
        bool: True if all array lengths are consistent, False otherwise
    """
    layer_boundary_x = velocity_model["layer_boundary_x"]
    layer_boundary_z = velocity_model["layer_boundary_z"]
    upper_velocities = velocity_model["upper_velocities"]
    upper_x_velocities = velocity_model["upper_x_velocities"]
    lower_velocities = velocity_model["lower_velocities"]
    lower_x_velocities = velocity_model["lower_x_velocities"]
    
    for i in range(len(layer_boundary_x)):
        # Check layer_boundary_x and layer_boundary_z lengths
        if len(layer_boundary_x[i]) != len(layer_boundary_z[i]):
            logger.error(f"Layer boundary x and z lengths do not match at layer {i}")
            return False
            
        # Check upper velocity data lengths
        if len(upper_velocities[i]) != len(upper_x_velocities[i]):
            logger.error(f"Upper velocity data lengths do not match at layer {i}")
            return False
            
        # Check lower velocity data lengths
        if len(lower_velocities[i]) != len(lower_x_velocities[i]):
            logger.error(f"Lower velocity data lengths do not match at layer {i}")
            return False
            
    return True

def read_vin_model(file_path: str) -> Dict:
    """Read a ZELT format velocity model file.
    
    The ZELT format consists of layer boundaries and velocity nodes. Each layer has:
    - A boundary definition (x-coordinates, z-coordinates, and flags)
    - Upper velocity nodes (x-coordinates, velocities, and flags)
    - Lower velocity nodes (x-coordinates, velocities, and flags)
    - Optional bottom boundary for the last layer
    
    Args:
        file_path (str): Path to the velocity model file
        
    Returns:
        Dict: Dictionary containing model data with keys:
            - layer_boundary_x: List of x-coordinates for each layer boundary
            - layer_boundary_z: List of z-coordinates for each layer boundary
            - layer_boundary_flags: List of flags for each boundary point
            - upper_x_velocities: List of x-coordinates for upper velocities
            - upper_velocities: List of upper velocity values
            - upper_velocity_flags: List of flags for upper velocities
            - lower_x_velocities: List of x-coordinates for lower velocities
            - lower_velocities: List of lower velocity values
            - lower_velocity_flags: List of flags for lower velocities
            - bottom_boundary: Optional tuple (x1, x2, z) for bottom boundary
            
    Raises:
        ValueError: If the model data is invalid or inconsistent
    """
    layers = {
        "layer_boundary_x": [],
        "layer_boundary_z": [],
        "layer_boundary_flags": [],
        "upper_x_velocities": [],
        "upper_velocities": [],
        "upper_velocity_flags": [],
        "lower_x_velocities": [],
        "lower_velocities": [],
        "lower_velocity_flags": [],
        "bottom_boundary_x": [],
        "bottom_boundary_z": [],
        "bottom_boundary_flags": []
    }
    
    logger.info(f"Reading velocity model from {file_path}")
    with open(file_path, 'r') as file:
        nrzmax = 300  # Assuming ppcntr = 3000, nrzmax = ppcntr / 10
        nrvmax = 30   # Assuming ppvel = 300, nrvmax = ppvel / 10
        
        while True:
            # Read layer boundary coordinates
            result = read_three_lines(file)
            if not result:
                break
                
            if len(result) == 4:  # New format with bottom boundary flag
                xm_row, zm_row, ivarz_row, is_bottom = result
                if is_bottom:
                    # Store bottom boundary and exit
                    if len(xm_row) >= 2:
                        layers["bottom_boundary_x"] = xm_row
                        layers["bottom_boundary_z"] = zm_row
                        layers["bottom_boundary_flags"] = ivarz_row
                    break
            else:
                xm_row, zm_row, ivarz_row = result
                
            layers["layer_boundary_x"].append(xm_row)
            layers["layer_boundary_z"].append(zm_row)
            layers["layer_boundary_flags"].append(ivarz_row)
            
            # Read upper and lower velocity data
            for key in ["upper", "lower"]:
                vel_data = read_three_lines(file)
                if vel_data:
                    xvel_row, vel_row, flag_row = vel_data[:3]  # Ignore potential bottom flag
                else:
                    xvel_row = vel_row = flag_row = [np.nan] * 10
                    
                layers[f"{key}_x_velocities"].append(xvel_row)
                layers[f"{key}_velocities"].append(vel_row)
                layers[f"{key}_velocity_flags"].append(flag_row)
    
    if not check_lengths(layers):
        raise ValueError("Invalid model data: inconsistent array lengths")
        
    logger.info(f"Successfully read model with {len(layers['layer_boundary_x'])} layers")
    return layers

def process_velocity_model(model_data: Dict) -> Dict:
    """Process velocity model data, computing averages for each layer.
    
    Args:
        model_data (Dict): Original model data from read_vin_model
        
    Returns:
        Dict: Processed model data with averaged velocities
    """
    processed = model_data.copy()
    
    for i in range(len(model_data["layer_boundary_x"])):
        upper_vel = np.array(model_data["upper_velocities"][i])
        lower_vel = np.array(model_data["lower_velocities"][i])
        
        # Compute averages
        upper_avg = float(np.mean(upper_vel[~np.isnan(upper_vel)]))
        lower_avg = float(np.mean(lower_vel[~np.isnan(lower_vel)]))
        
        # Replace with averaged values
        processed["upper_velocities"][i] = [upper_avg] * len(upper_vel)
        processed["lower_velocities"][i] = [lower_avg] * len(lower_vel)
    
    logger.info("Processed model data: computed layer averages")
    return processed

def write_three_lines(file, data: List[Union[List[float], List[int]]], row_prefix: Optional[int], is_bottom: bool = False) -> None:
    """Write three lines of ZELT format data to a file.
    
    Args:
        file: File object to write to
        data (List[Union[List[float], List[int]]]): Three rows of data to write
        row_prefix (Optional[int]): Prefix number for the first row (layer number)
        is_bottom (bool): Whether this is a bottom boundary (default: False)
        
    Raises:
        ValueError: If data does not contain exactly three rows
    """
    if len(data) != 3:
        raise ValueError("Data must contain exactly three rows.")
        
    def format_row(row: Union[List[float], List[int]], prefix: Optional[int], dtype: str) -> str:
        """Format a single row of data with appropriate prefix and number format."""
        prefix_str = f"{prefix:2d} " if prefix is not None else "   "
        fmt = "{:7.2f}" if dtype == "float" else "{:7d}"
        return prefix_str + "".join(fmt.format(x) for x in row) + "\n"
    
    chunk_size = 10
    for i in range(0, len(data[0]), chunk_size):
        # Split data into chunks
        sub_rows = [row[i:i + chunk_size] for row in data]
        
        # For bottom boundary, always use 0 as continue flag
        icontinue = 0 if is_bottom else (1 if len(sub_rows[0]) == chunk_size else 0)
        
        # Write data chunks if they contain valid values
        if not np.any(np.isnan(sub_rows[0])):
            file.write(format_row(sub_rows[0], row_prefix, "float"))
        if not np.any(np.isnan(sub_rows[1])):
            file.write(format_row(sub_rows[1], icontinue, "float"))
        if not is_bottom and not np.any(np.isnan(sub_rows[2])):  # Skip flags for bottom boundary
            file.write(format_row(sub_rows[2], None, "int"))

def write_vin_model(file_path: str, processed_model: Dict) -> None:
    """Write a processed velocity model to a ZELT format file.
    
    The ZELT format consists of layer boundaries and velocity nodes. Each layer has:
    - A boundary definition (x-coordinates, z-coordinates, and flags)
    - Upper velocity nodes (x-coordinates, velocities, and flags)
    - Lower velocity nodes (x-coordinates, velocities, and flags)
    - Bottom boundary for the last layer
    
    Args:
        file_path (str): Path to the output file
        processed_model (Dict): Dictionary containing processed model data with keys:
            - layer_boundary_x: List of x-coordinates for each layer boundary
            - layer_boundary_z: List of z-coordinates for each layer boundary
            - layer_boundary_flags: List of flags for each boundary point
            - upper_x_velocities: List of x-coordinates for upper velocities
            - upper_velocities: List of upper velocity values
            - upper_velocity_flags: List of flags for upper velocities
            - lower_x_velocities: List of x-coordinates for lower velocities
            - lower_velocities: List of lower velocity values
            - lower_velocity_flags: List of flags for lower velocities
            - bottom_boundary_x: List of x-coordinates for boundary bottom boundary
            - bottom_boundary_z: List of z-coordinates for boundary bottom boundary
            - bottom_boundary_flags: List of flags for boundary bottom boundary
    """
    logger.info(f"Writing processed model to {file_path}")
    with open(file_path, 'w') as file:
        for i in range(len(processed_model["layer_boundary_x"])):
            # Write layer boundary coordinates
            layer_number = i + 1
            boundary_data = processed_model["layer_boundary_x"][i]
            z_data = processed_model["layer_boundary_z"][i]
            flag_data = processed_model["layer_boundary_flags"][i]
            write_three_lines(file, [boundary_data, z_data, flag_data], layer_number)
            
            # Write upper velocity data
            upper_vel_data = processed_model["upper_x_velocities"][i]
            upper_vel = processed_model["upper_velocities"][i]
            upper_flag = processed_model["upper_velocity_flags"][i]
            write_three_lines(file, [upper_vel_data, upper_vel, upper_flag], layer_number)
            
            # Write lower velocity data
            lower_vel_data = processed_model["lower_x_velocities"][i]
            lower_vel = processed_model["lower_velocities"][i]
            lower_flag = processed_model["lower_velocity_flags"][i]
            write_three_lines(file, [lower_vel_data, lower_vel, lower_flag], layer_number)
        
        # Write bottom boundary if present
        if processed_model.get("bottom_boundary_x"):
            layer_number = len(processed_model["layer_boundary_x"]) + 1
            write_three_lines(
                file, 
                [
                    processed_model["bottom_boundary_x"],
                    processed_model["bottom_boundary_z"],
                    processed_model["bottom_boundary_flags"]
                ],
                layer_number,
                is_bottom=True
            )
    
    logger.info(f"Successfully wrote model with {len(processed_model['layer_boundary_x'])} layers")

# Add standalone functions to __all__
__all__ = [
    'read_vin_model',
    'process_velocity_model',
    'write_vin_model',
    'read_three_lines',
    'check_lengths'
]

# Example usage
if __name__ == "__main__":
    # Example 1: Using standalone functions
    file_path = "v_all.in"
    output_path = "v_averaged_all.in"
    
    # Read and process model
    model_data = read_vin_model(file_path)
    processed_data = process_velocity_model(model_data)
    write_vin_model(output_path, processed_data)
    
    print("\nModel Statistics:")
    print(f"Number of layers: {len(model_data['layer_boundary_x'])}")
    