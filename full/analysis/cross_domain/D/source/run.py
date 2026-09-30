from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
subprocess.run([sys.executable,str(ROOT/'run.py'),'--panels','D',*sys.argv[1:]],check=True)
