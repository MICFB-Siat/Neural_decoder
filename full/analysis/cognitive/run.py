
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panels',nargs='+',choices=list('BCEFGIJKLMN'),required=True);p.add_argument('--output',type=Path,default=ROOT/'outputs');a,extra=p.parse_known_args()
geometry_done=False
for panel in a.panels:
    out=a.output/panel
    if panel in 'BN':cmd=['rsa.py','--panel',panel]
    elif panel=='C':cmd=['meg_classification.py']
    elif panel in 'EF':cmd=['state_classification.py','--panel',panel]
    elif panel=='G':cmd=['G/code/inference_g.py']
    elif panel in 'IJ':cmd=['regression.py','--panel',panel]
    else:
        if geometry_done:continue
        cmd=['affect_geometry.py','--panels',*[s for s in a.panels if s in 'KLM']];out=a.output/'KLM';geometry_done=True
    subprocess.run([sys.executable,str(ROOT/cmd[0]),*cmd[1:],'--output',str(out),*extra],check=True,cwd=ROOT)
