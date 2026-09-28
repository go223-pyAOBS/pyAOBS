from pathlib import Path
import sys
p = Path(sys.argv[1])
p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
print("lf", p)
