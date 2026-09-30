import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description='Generate available Figure 2 panels.')
    parser.add_argument('--panels',nargs='+',choices=list('BCDFGH'),default=list('BCDFGH'))
    parser.add_argument('--device',default=None)
    parser.add_argument('--render-python',default=sys.executable,help='Python with pycortex for B/D rendering.')
    parser.add_argument('--check',action='store_true',help='Check required files against the manifest.')
    parser.add_argument('--verify-hashes',action='store_true',help='Verify SHA-256 checksums with --check.')
    args=parser.parse_args()
    if args.device:os.environ['FIG2_DEVICE']=args.device
    os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4';os.environ['MPLBACKEND']='Agg'
    if args.check:
        import json
        manifest=json.loads((ROOT/'manifest.json').read_text())
        for item in manifest:
            path=ROOT/item['path']
            if not path.is_file() or path.stat().st_size!=item['bytes']:raise RuntimeError(f'Missing or incomplete file: {path}')
            if args.verify_hashes:
                import hashlib
                digest=hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
                if digest.hexdigest()!=item['sha256']:raise RuntimeError(f'Checksum mismatch: {path}')
        print('File check passed.');return
    panels=list(dict.fromkeys(args.panels))
    for panel in panels:
        print(f'Figure 2{panel}',flush=True)
        subprocess.run([sys.executable,str(ROOT/panel/'run.py')],cwd=ROOT,check=True)
        if panel in ['B','D']:
            subprocess.run([args.render_python,str(ROOT/'common/cortical.py'),panel],cwd=ROOT,check=True)
    print('Done. Results are in the panel outputs/ directories.')

if __name__=='__main__':main()
