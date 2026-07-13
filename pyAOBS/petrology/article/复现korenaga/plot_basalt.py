#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse BASALT_MODERN output and generate visualization plots.

Usage:
    python plot_basalt.py <output_file.txt> [png_prefix]
"""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt


def parse_output(path):
    """Parse basalt_modern full-output text into structured records."""
    # Try UTF-16 first (PowerShell redirection default), then UTF-8
    try:
        with open(path, 'r', encoding='utf-16') as f:
            lines = f.readlines()
    except (UnicodeError, UnicodeDecodeError):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    records = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for a phase table header: first token is a label and second token is 'LIQ'
        parts = line.split()
        if len(parts) < 3 or parts[1] != 'LIQ':
            i += 1
            continue

        # Header line: [FRAC|SYS] LIQ PLAG OL CPX
        phases = parts[2:]  # ['PLAG', 'OL', 'CPX']

        # Next non-empty line is FRAC row
        i += 1
        while i < len(lines) and lines[i].strip() == '':
            i += 1
        if i >= len(lines):
            break
        frac_parts = lines[i].strip().split()
        if len(frac_parts) < 2 or frac_parts[0] != 'FRAC':
            i += 1
            continue
        fracs = [float(x) for x in frac_parts[1:]]
        liquid_frac = fracs[0]
        phase_fracs = fracs[1:]

        # Read component rows until TEMP line
        comp_data = {}
        i += 1
        while i < len(lines):
            cline = lines[i].strip()
            if cline.startswith('TEMP='):
                break
            if not cline:
                i += 1
                continue
            cparts = cline.split()
            if len(cparts) < 2 + len(phases):
                i += 1
                continue
            comp_name = cparts[0]
            values = [float(x) for x in cparts[1:]]
            comp_data[comp_name] = values
            i += 1

        # Parse TEMP line
        if i >= len(lines):
            break
        temp_line = lines[i].strip()
        m = re.search(r'TEMP=\s*([\d.E+-]+)', temp_line)
        temp = float(m.group(1)) if m else np.nan
        m_p = re.search(r'P\(kbar\)=\s*([\d.E+-]+)', temp_line)
        p = float(m_p.group(1)) if m_p else 0.0
        m_flr = re.search(r'FLR\s*=\s*([\d.E+-]+)', temp_line)
        flr = float(m_flr.group(1)) if m_flr else np.nan

        records.append({
            'temp': temp,
            'p': p,
            'flr': flr,
            'liquid_frac': liquid_frac,
            'phase_fracs': phase_fracs,
            'phases': phases,
            'comp': comp_data,
        })
        i += 1

    return records


def records_to_arrays(records):
    """Convert list of records to numpy arrays for plotting."""
    temps = np.array([r['temp'] for r in records])
    pressures = np.array([r['p'] for r in records])
    flrs = np.array([r['flr'] for r in records])
    liq_frac = np.array([r['liquid_frac'] for r in records])

    phases = records[0]['phases'] if records else []
    nphas = len(phases)
    phase_fracs = np.zeros((len(records), nphas))
    for i, r in enumerate(records):
        phase_fracs[i, :] = r['phase_fracs'][:nphas]

    # Components
    comp_names = list(records[0]['comp'].keys()) if records else []
    comp_liq = {c: np.zeros(len(records)) for c in comp_names}
    comp_phas = {c: np.zeros((len(records), nphas)) for c in comp_names}
    for i, r in enumerate(records):
        for c in comp_names:
            vals = r['comp'].get(c, [])
            if len(vals) >= 1 + nphas:
                comp_liq[c][i] = vals[0]
                comp_phas[c][i, :] = vals[1:1+nphas]

    return {
        'temps': temps,
        'p': pressures,
        'flr': flrs,
        'liq_frac': liq_frac,
        'phases': phases,
        'phase_fracs': phase_fracs,
        'comp_names': comp_names,
        'comp_liq': comp_liq,
        'comp_phas': comp_phas,
    }


def plot_phase_fractions(data, prefix='basalt'):
    """Plot liquid + phase fractions vs temperature."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data['temps'] - 273.16, data['liq_frac'], 'k-', linewidth=2, label='LIQ')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, ph in enumerate(data['phases']):
        ax.plot(data['temps'] - 273.16, data['phase_fracs'][:, idx],
                color=colors[idx % len(colors)], linewidth=2, label=ph)
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Phase fraction', fontsize=12)
    ax.set_title('Phase fractions vs temperature', fontsize=13)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{prefix}_phase_fractions.png', dpi=200)
    plt.close(fig)


def plot_liquid_composition(data, prefix='basalt'):
    """Plot liquid composition (wt fraction) vs temperature."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in data['comp_names']:
        ax.plot(data['temps'] - 273.16, data['comp_liq'][c], linewidth=2, label=c)
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Liquid composition', fontsize=12)
    ax.set_title('Liquid composition vs temperature', fontsize=13)
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{prefix}_liquid_composition.png', dpi=200)
    plt.close(fig)


def plot_liquidus_summary(data, prefix='basalt'):
    """Plot pressure-temperature path and first appearance of phases."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(data['temps'] - 273.16, data['p'], c=data['liq_frac'],
                    cmap='viridis', s=40, edgecolors='none')
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Pressure (kbar)', fontsize=12)
    ax.set_title('P-T path colored by liquid fraction', fontsize=13)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Liquid fraction')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{prefix}_PT_path.png', dpi=200)
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    out_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else out_path.replace('.txt', '')

    records = parse_output(out_path)
    if not records:
        print('No valid records found in', out_path)
        sys.exit(1)

    print(f'Parsed {len(records)} temperature steps from {out_path}')
    data = records_to_arrays(records)

    plot_phase_fractions(data, prefix)
    plot_liquid_composition(data, prefix)
    plot_liquidus_summary(data, prefix)

    print(f'Saved plots:')
    print(f'  {prefix}_phase_fractions.png')
    print(f'  {prefix}_liquid_composition.png')
    print(f'  {prefix}_PT_path.png')


if __name__ == '__main__':
    main()
