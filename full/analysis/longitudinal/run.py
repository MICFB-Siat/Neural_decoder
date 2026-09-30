
from pathlib import Path
import argparse,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panels',nargs='+',choices=list('ABCDEFGHJKLM'));p.add_argument('--output',type=Path,default=ROOT/'outputs');p.add_argument('--paths',type=Path);a,extra=p.parse_known_args()
manifest=json.loads((ROOT/'manifest.json').read_text())
if not a.panels:p.error('Specify --panels')
env=os.environ.copy()
config=a.paths or env.get('FIGURE_PATHS')
if config:
    env['FIGURE_PATHS']=str(Path(os.path.expandvars(str(config))).expanduser().resolve())
completed=set()
for panel in a.panels:
    if panel in 'ACE':cmd=['signals.py','--panel',panel]
    elif panel=='L':cmd=['L/infer.py']
    else:
        key='J' if panel in 'JK' else panel
        if key in completed:continue
        completed.add(key);cmd=[key+'/source/'+manifest['panels'][key]['script']]
    subprocess.run([sys.executable,str(ROOT/cmd[0]),*cmd[1:],'--output',str(a.output/panel),*extra],check=True,cwd=ROOT,env=env)
