
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panels',nargs='+',choices=list('ABCDE'));p.add_argument('--check',action='store_true');a,extra=p.parse_known_args()
    if a.check:
        for name in ['geometry.py','infer_classification.py','resources.json','inputs/geometry','experiment']:
            if not (ROOT/name).exists():raise FileNotFoundError(ROOT/name)
        print('Entry and resource presence checked; no inference performed.');return
    if not a.panels:p.error('Choose --panels A B C D E')
    geometry_done=False
    for panel in a.panels:
        if panel in 'AB':
            if geometry_done:continue
            cmd=[str(ROOT/'geometry.py'),'--output',str(ROOT/'AB_outputs')];geometry_done=True
        else:cmd=[str(ROOT/'infer_classification.py'),'--panel',panel]
        subprocess.run([sys.executable,*cmd,*extra],check=True,cwd=ROOT)
if __name__=='__main__':main()
