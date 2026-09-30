
from pathlib import Path
import subprocess,sys
p=Path(__file__).resolve().with_name('run.py')
subprocess.run([sys.executable,str(p),*sys.argv[1:]],check=True)
