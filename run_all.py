"""One-command reproduction of the fast parts (statistic + figures)."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for cmd in ([sys.executable, "src/run_slr.py"],
            [sys.executable, "src/fig1.py"],
            [sys.executable, "src/fig2.py"],
            [sys.executable, "src/fig3.py"]):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=HERE, check=False)
