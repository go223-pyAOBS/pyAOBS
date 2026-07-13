#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python implementation of fntime_v1.1.c

Parse OBS file timestamps from hexadecimal filenames.

Original C program:
    program name: fntime.c
    written by: XL Qiu
    first attempt: 2009/4/26
    modified by: JZ Zhang, HY Zhang
    modified: 2016/08/31 by HY Zhang, add TC, sps_pre as new input parameters

Usage:
    python fntime_v1_1.py fileNameList TC sps_pre

    fileNameList: Input file containing list of filenames (one per line)
    TC: Clock parameter read from A*.LOG file
    sps_pre: Preset sampling rate of OBS
"""

import os
import sys
import math
from typing import Tuple, Dict, Optional


def hexto_int(hex_char: str) -> int:
    """Convert a single hexadecimal character to integer.
    
    Corresponds to C function: HextoInt(unsigned char datebyte)
    
    Args:
        hex_char: Single hexadecimal character (0-9, A-F, a-f)
        
    Returns:
        Integer value (0-15)
        
    Raises:
        ValueError: If input is not a valid hexadecimal character
    """
    if len(hex_char) != 1:
        raise ValueError(f"Expected single character, got: {hex_char}")
    
    hex_char = hex_char.upper()
    if not (('0' <= hex_char <= '9') or ('A' <= hex_char <= 'F')):
        raise ValueError(f"Invalid hex character: {hex_char}")
    
    if ord(hex_char) < 58:  # '0'-'9'
        return ord(hex_char) - 48
    else:  # 'A'-'F'
        return ord(hex_char) - 55


def hexto_dec(hex_str: str) -> int:
    """Convert hexadecimal string to decimal integer.
    
    Corresponds to C function: HextoDec(unsigned int a0, unsigned char *p)
    Uses weighted conversion: sum(pow(16, a0-i) * j)
    
    Args:
        hex_str: Hexadecimal string (e.g., "5DD", "FFF")
        
    Returns:
        Decimal integer value
    """
    if not hex_str:
        return 0
    
    result = 0
    a0 = len(hex_str) - 1  # Last index (equivalent to --a1 in C code)
    
    for i, char in enumerate(hex_str):
        char_lower = char.lower()
        
        # Convert character to integer value
        if 'a' <= char_lower <= 'f':
            j = ord(char_lower) - 87
        elif 'A' <= char <= 'F':
            j = ord(char) - 55
        else:
            j = ord(char) - 48
        
        # Weighted sum: pow(16, a0 - i) * j
        weight = pow(16, a0 - i)
        result += weight * j
    
    return result


def julian_day(year: int, mon: int, day: int) -> int:
    """Calculate Julian day number.
    
    Corresponds to C function: julian_day(long year, long mon, long day)
    Formula: day-32075+1461*(year+4800-(14-mon)/12)/4+367*(mon-2+(14-mon)/12*12)/12-3*((year+4900-(14-mon)/12)/100)/4
    
    Note: In C, integer division truncates (rounds toward zero). We must ensure
    each division step uses integer division to match C behavior exactly.
    
    Args:
        year: Year (e.g., 2020)
        mon: Month (1-12)
        day: Day (1-31)
        
    Returns:
        Julian day number
    """
    # Use integer division (//) to match C behavior exactly
    # C code: (14-mon)/12 is integer division
    temp = (14 - mon) // 12
    return int(day - 32075 + 
               1461 * (year + 4800 - temp) // 4 +
               367 * (mon - 2 + temp * 12) // 12 -
               3 * ((year + 4900 - temp) // 100) // 4)


def parse_datetime_from_filename(filename: str) -> Dict[str, int]:
    """Parse date and time from first 8 hexadecimal characters of filename.
    
    Args:
        filename: must match format AABBCCDD.YYY
        
    Returns:
        Dictionary with keys: year, mon, day, hour, min, sec
    """
    if len(filename) < 8:
        raise ValueError(f"Filename too short: {filename}")
    
    # Convert first 8 characters to integer list
    date = [hexto_int(filename[i]) for i in range(8)]
    
    # Bitwise extraction (matching C code exactly)
    year = (date[0] << 2 | date[1] >> 2) & 0x0f
    mon = (date[1] << 2 | date[2] >> 2) & 0x0f
    day = (date[2] << 3 | date[3] >> 1) & 0x1f
    hour = (date[3] << 4 | date[4]) & 0x1f
    min = (date[5] << 2 | date[6] >> 2) & 0x3f
    sec = (date[6] << 4 | date[7]) & 0x3f
    
    # Year adjustment (2009-2024 range)
    year += 2000
    if year < 2009:
        year += 16
    
    return {
        'year': year,
        'mon': mon,
        'day': day,
        'hour': hour,
        'min': min,
        'sec': sec,
        'date0': date[0]  # Store for file type detection
    }


def parse_milliseconds(filename: str, pcik: float) -> float:
    """Parse milliseconds from filename.
    
    Args:
        filename: Filename (must match format AABBCCDD.YYY)
        pcik: Internal clock frequency (TC / 256.0)
        
    Returns:
        Actual milliseconds (in seconds): nzmsec * 4096 / PCIk
        
    Raises:
        ValueError: If filename doesn't match the required format
    """
    # Strict format validation: must be "XXXXXXXX.YYY" format
    # where XXXXXXXX is exactly 8 hex characters and YYY is exactly 3 hex characters
    if '.' not in filename:
        raise ValueError(f"Filename must contain a dot separator: {filename}")
    
    parts = filename.split('.', 1)
    if len(parts) != 2:
        raise ValueError(f"Filename must have exactly one dot: {filename}")
    
    date_part = parts[0]
    msec_part = parts[1]
    
    # Validate date part: must be exactly 8 hexadecimal characters
    if len(date_part) != 8:
        raise ValueError(f"Date part must be exactly 8 hexadecimal characters, got {len(date_part)}: {date_part} in {filename}")
    
    if not all(c in '0123456789ABCDEFabcdef' for c in date_part):
        raise ValueError(f"Date part must contain only hexadecimal characters: {date_part} in {filename}")
    
    # Validate msec part: must be exactly 3 hexadecimal characters
    if len(msec_part) != 3:
        raise ValueError(f"Millisecond part must be exactly 3 hexadecimal characters, got {len(msec_part)}: {msec_part} in {filename}")
    
    if not all(c in '0123456789ABCDEFabcdef' for c in msec_part):
        raise ValueError(f"Millisecond part must contain only hexadecimal characters: {msec_part} in {filename}")  
    # Extract msec hex part (exactly 3 characters)
    msec_hex = msec_part  
    # Convert to decimal (a0 = 2 for 3 characters)
    nzmsec = hexto_dec(msec_hex)  
    # Convert to actual milliseconds (in seconds as decimal part)
    actual_msec = nzmsec * 4096.0 / pcik   
    return actual_msec


def get_file_type(date0: int) -> Tuple[int, bool]:
    """Determine file type based on first hexadecimal character.
    
    Args:
        date0: Integer value of first hex character
        
    Returns:
        Tuple of (scandata, is_4channel)
        scandata: Bytes per data point (12 for 4-channel, 9 for 3-channel)
        is_4channel: True for 4-channel data, False for 3-channel
    """
    if (date0 >> 3) == 0:
        # 4 channel data (3 broadband + 1 hydrophone)
        return 12, True
    elif (date0 >> 3) == 1:
        # 3 channels of short period
        return 9, False
    else:
        raise ValueError(f"Unknown file type for date0={date0}")


def npts_from_file(filename: str, scandata: int) -> int:
    """Analyze file to get number of data points.
    
    Args:
        filename: Path to data file
        scandata: Bytes per data point
        
    Returns:
        Number of data points (npts)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot open the data file: {filename}")
    
    file_size = os.path.getsize(filename)
    npts = file_size // scandata
    
    return npts


def calculate_absolute_time(year: int, mon: int, day: int, hour: int, 
                           min: int, sec: int, msec: float) -> float:
    """Calculate absolute time in seconds.
    
    Args:
        year, mon, day, hour, min, sec: Date and time components
        msec: Milliseconds
        
    Returns:
        Absolute time in seconds: ((jday*24+hour)*60+min)*60+sec+msec
    """
    jday = julian_day(year, mon, day)
    time_seconds = ((jday * 24 + hour) * 60 + min) * 60 + sec + msec
    return time_seconds


def process_fntime(filelist_path: str, tc: float, sps_pre: float, 
                  output_path: str = "fntime.txt", verbose: bool = False) -> None:
    """Main function to process file list and generate time information.
    
    Args:
        filelist_path: Path to input file containing list of filenames
        tc: Clock parameter from A*.LOG file
        sps_pre: Preset sampling rate
        output_path: Path to output file (default: "fntime.txt")
        verbose: Whether to print debug information
    """
    # Validate inputs
    if not os.path.exists(filelist_path):
        raise FileNotFoundError(f"Cannot open infile: {filelist_path}")
    
    if tc <= 0:
        raise ValueError(f"TC must be positive, got: {tc}")
    if sps_pre <= 0:
        raise ValueError(f"sps_pre must be positive, got: {sps_pre}")
    
    # Calculate internal clock frequency
    pcik = tc / 256.0
    
    if verbose:
        print(f"TC: {tc}")
        print(f"PCIk: {pcik}")
    
    # Read file list
    # C code: fgets(fnamein,13,fpin) reads filename (max 12 chars)
    #         fgets(linefd,5,fpin) reads next line (skip newline/empty line)
    # However, if input file has filenames on consecutive lines (no empty lines),
    # we should handle both formats:
    # 1. Format with skip lines: filename + empty line + filename + empty line ...
    # 2. Format without skip lines: filename + filename + filename ...
    filenames = []
    
    def is_valid_filename(line):
        """Check if line matches the required format: AABBCCDD.YYY
        
        Args:
            line: Line to check
            
        Returns:
            True if format is valid (8 hex chars + dot + 3 hex chars), False otherwise
        """
        line = line.strip()
        if not line:
            return False
        
        # Must contain exactly one dot
        if '.' not in line:
            return False
        
        parts = line.split('.', 1)
        if len(parts) != 2:
            return False
        
        date_part = parts[0]
        msec_part = parts[1]
        
        # Date part must be exactly 8 hexadecimal characters
        if len(date_part) != 8:
            return False
        if not all(c in '0123456789ABCDEFabcdef' for c in date_part):
            return False
        
        # Msec part must be exactly 3 hexadecimal characters
        if len(msec_part) != 3:
            return False
        if not all(c in '0123456789ABCDEFabcdef' for c in msec_part):
            return False
        
        return True
    
    with open(filelist_path, 'r') as f:
        lines = f.readlines()
    
    # Process all lines
    i = 0
    while i < len(lines):
        filename = lines[i].strip()
        i += 1
        
        # Skip empty lines
        if not filename:
            continue
        
        # Strict format validation: must be AABBCCDD.YYY format
        if not is_valid_filename(filename):
            raise ValueError(f"Invalid filename format (must be AABBCCDD.YYY): {filename}")
        
        # Save full filename for file operations
        full_filename = filename
        
        # Truncate to 12 characters for parsing (matching fgets limit)
        # C code reads max 12 chars: fgets(fnamein,13,fpin)
        # For "51A6D159.5DD" (13 chars), it reads "51A6D159.5D" (12 chars)
        parse_filename = filename[:12] if len(filename) > 12 else filename
        
        # Store both full filename (for file operations) and parse filename (for parsing)
        filenames.append({
            'full': full_filename,      # Full filename for file operations
            'parse': parse_filename     # First 12 chars for hex parsing
        })
        
        # C code: fgets(linefd,5,fpin) - reads next line (skip line)
        # If next line is a filename, we still process it in next iteration
        # So we don't need to skip anything here if it's a filename
    
    if not filenames:
        raise ValueError("No filenames found in input file")
    
    if verbose:
        print(f"Found {len(filenames)} filenames:")
        for i, fn in enumerate(filenames, 1):
            print(f"  {i}. {fn}")
    
    # Initialize counters
    nfa = 0  # Count of 4-channel files
    nf2 = 0  # Count of 3-channel files
    time1 = 0.0
    
    # Previous file info (for output)
    prev_info = None
    
    # Open output file
    with open(output_path, 'w', encoding='utf-8') as fpout:
        # Process each file
        for file_idx, file_info in enumerate(filenames, 1):
            full_filename = file_info['full']      # Full filename for file operations
            parse_filename = file_info['parse']    # First 12 chars for hex parsing
            
            if verbose:
                print(f"\nProcessing file {file_idx}/{len(filenames)}: {full_filename}")
                print(f"  Parse filename (first 12 chars): {parse_filename}")
            
            try:
                # Filename format is strictly validated: AABBCCDD.YYY
                # Extract date part (first 8 hex characters before dot)
                # C code: for i=0;i<8;i++) date[i] = HextoInt(fnamein[i])
                parts = full_filename.split('.', 1)
                date_hex = parts[0]  # First 8 hex characters (already validated)
                msec_hex = parts[1]  # Last 3 hex characters (already validated)
                
                # Parse date/time from first 8 hex characters
                dt = parse_datetime_from_filename(date_hex)
                year, mon, day = dt['year'], dt['mon'], dt['day']
                hour, min, sec = dt['hour'], dt['min'], dt['sec']
                date0 = dt['date0']
                
                # Parse milliseconds (validates format and extracts msec)
                actual_msec = parse_milliseconds(full_filename, pcik)
                
                # Convert msec hex part to decimal for display
                nzmsec = hexto_dec(msec_hex)
                
                if verbose:
                    print("**************************************************")
                    print(f"fileName: {full_filename}")
                    print(f"extend fileName (msec hex): {msec_hex}")
                    print(f"year: {year}, mon: {mon}, day: {day}")
                    print(f"hour: {hour}, min: {min}, sec: {sec}")
                    print(f"PCIk: {pcik}")
                    print(f"nzmsec (hex->dec): {nzmsec} (from hex '{msec_hex}')")
                    print(f"nzmsec*4096/PCIk: {actual_msec}")
                
                # Calculate absolute time
                time = calculate_absolute_time(year, mon, day, hour, min, sec, actual_msec)
                
                if verbose:
                    jday = julian_day(year, mon, day)
                    print(f"julian_day: {jday}")
                    print(f"absolute time: {time}")
                
                # Get file type and analyze file
                scandata, is_4channel = get_file_type(date0)
                if is_4channel:
                    nfa += 1
                else:
                    nf2 += 1
                
                # Analyze file (use full filename for file operations)
                npts = npts_from_file(full_filename, scandata)
                
                if verbose:
                    print(f"scandata: {scandata}, npts: {npts}")
                    print(f"time1: {time1}")
                    print(f"time: {time}")
                
                # Output logic (matches C code behavior)
                # C code: if(nfa == 1 || nf2 == 1) ; // do nothing for first file
                #         else { // output previous file info for subsequent files }
                # 
                # Logic: First file (nfa==1 or nf2==1) does not output anything
                #        Second and subsequent files output previous file info
                #        This means: when processing file N (N >= 2), output file (N-1) info
                is_first_file = (nfa == 1 or nf2 == 1)
                
                if verbose:
                    print(f"  is_first_file: {is_first_file}, nfa: {nfa}, nf2: {nf2}")
                    print(f"  prev_info exists: {prev_info is not None}")
                
                # Output previous file info if we have one (this happens starting from file 2)
                # The first file (nfa==1 or nf2==1) itself doesn't output, but its info is stored
                # and will be output when processing the second file
                if prev_info is not None:
                    # Calculate delta and gap (using previous file's npts)
                    # delta: sampling interval in seconds for previous file
                    delta = abs(time - time1) / prev_info['npts']
                    # Gap: difference between preset and actual sampling rate (in milliseconds)
                    gap = (1.0 / sps_pre - delta) * prev_info['npts'] * 1000.0
                    
                    if verbose:
                        print(f"  -> Writing output for previous file: {prev_info['filename']}")
                    
                    # Format output (matching C code format exactly)
                    # C code format: "%s  %4ld-%02d-%02d %02ld:%02ld:%06.3f UTC time_length=%10.6lf second, delta=%10.6lf msec, Gap=%10.6lf msec\n"
                    # Note: delta format is %10.6lf which gives space before number if positive
                    delta_ms = delta * 1000.0
                    gap_ms = gap
                    
                    output_line = (
                        f"{prev_info['filename']}  "
                        f"{prev_info['year']:4d}-{prev_info['mon']:02d}-{prev_info['day']:02d} "
                        f"{prev_info['hour']:02d}:{prev_info['min']:02d}:"
                        f"{prev_info['sec']+prev_info['msec']:06.3f} UTC "
                        f"time_length={abs(time - time1):10.6f} second, "
                        f"delta={delta_ms:10.6f} msec, "
                        f"Gap={gap_ms:10.6f} msec\n"
                    )
                    fpout.write(output_line)
                
                # Store current file info for next iteration
                prev_info = {
                    'filename': full_filename,  # Store full filename for output
                    'year': year,
                    'mon': mon,
                    'day': day,
                    'hour': hour,
                    'min': min,
                    'sec': sec,
                    'msec': actual_msec,
                    'nzmsec': nzmsec,
                    'npts': npts,
                    'time': time
                }
                
                time1 = time
                
            except Exception as e:
                print(f"Error processing {full_filename}: {e}", file=sys.stderr)
                if verbose:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Output last file info (after loop, matching C code)
        if prev_info is not None:
            if verbose:
                print(f"\n-> Writing output for last file: {prev_info['filename']}")
            output_line = (
                f"{prev_info['filename']}  "
                f"{prev_info['year']:4d}-{prev_info['mon']:02d}-{prev_info['day']:02d} "
                f"{prev_info['hour']:02d}:{prev_info['min']:02d}:"
                f"{prev_info['sec']+prev_info['msec']:06.3f} UTC\n"
            )
            fpout.write(output_line)
    
    if verbose:
        print(f"\nOutput written to: {output_path}")


def main(verbose: bool = False):
    """Command-line interface."""
    if len(sys.argv) != 4:
        print("Usage: python fntime_v1_1.py fileNameList TC sps_pre")
        print("  fileNameList: Input file containing list of filenames")
        print("  TC: Clock parameter from A*.LOG file")
        print("  sps_pre: Preset sampling rate")
        sys.exit(1)
    
    filelist_path = sys.argv[1]
    try:
        tc = float(sys.argv[2])
        sps_pre = float(sys.argv[3])
    except ValueError as e:
        print(f"Error: Invalid numeric argument: {e}")
        sys.exit(1)
    
    try:
        process_fntime(filelist_path, tc, sps_pre, verbose=verbose)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(verbose=True)

