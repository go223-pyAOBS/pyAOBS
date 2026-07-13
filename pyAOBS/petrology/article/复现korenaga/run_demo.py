#!/usr/bin/env python3
"""Run the basalt model for the demo composition and capture output to a UTF-8 file."""

import subprocess
import sys

cmd = ['.\\basalt_modern.exe', 'input_demo_1atm.txt']
out_path = 'output_demo_1atm.txt'

result = subprocess.run(cmd, cwd='.', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if result.returncode != 0:
    print('Model failed:', result.returncode, file=sys.stderr)
    print(result.stderr.decode('utf-8', errors='replace'), file=sys.stderr)
    sys.exit(1)

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(result.stdout.decode('utf-8', errors='replace'))

print(f'Wrote {out_path} ({len(result.stdout)} bytes)')
