from pathlib import Path
import argparse,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description='Run a bundled training/evaluation script. No historical scores are used by this launcher.')
p.add_argument('--script',type=Path,required=True);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
script=(ROOT/a.script).resolve()
if not script.is_relative_to(ROOT) or not script.is_file():raise ValueError('Choose a script contained in this figure folder')
for rel,item in json.loads((ROOT/'external_datasets.json').read_text()).items():
    dst=ROOT/'_source'/rel;target=Path(item['path']).expanduser()
    if dst.is_symlink() and dst.readlink()!=target:dst.unlink()
    if not dst.exists() and not dst.is_symlink():dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(target,target_is_directory=True)
args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'_source')+os.pathsep+env.get('PYTHONPATH','')
subprocess.run([sys.executable,str(script),*args],cwd=ROOT,env=env,check=True)
