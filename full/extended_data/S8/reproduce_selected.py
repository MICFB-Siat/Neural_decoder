
from pathlib import Path
import argparse,subprocess,sys,json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panels',nargs='+',choices=list('ABC'),default=list('ABC'));p.add_argument('--device',default='cpu');p.add_argument('--limit-folds',type=int,default=1);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Use a new output directory')
 a.output.mkdir(parents=True);rows=[];commands=[]
 for panel in a.panels:
  for config in sorted((ROOT/'weights'/panel/'SEEDV').rglob('args.json')):
   relative=config.parent.relative_to(ROOT);condition=relative.parts[3];out=a.output/panel/condition
   cmd=[sys.executable,str(ROOT/'infer_saved.py'),'--run',str(relative),'--device',a.device,'--limit-folds',str(a.limit_folds),'--output',str(out)]
   subprocess.run(cmd,cwd=ROOT,check=True);commands.append(cmd);f=pd.read_csv(out/'fresh_metrics.csv');f['panel']=panel;f['condition']=condition;rows.append(f)
 if not rows:raise ValueError('No retained conditions')
 frame=pd.concat(rows,ignore_index=True);frame.to_csv(a.output/'fresh_conditions.csv',index=False)
 for panel in a.panels:
  part=frame[frame.panel==panel];values=part.groupby('condition').accuracy.mean();fig,ax=plt.subplots(figsize=(6,4));ax.bar(values.index,values.values);ax.tick_params(axis='x',rotation=25);ax.set(title=f'S8{panel}: SEED-V retained cross5 fold',ylabel='Accuracy',ylim=(0,1));fig.tight_layout();fig.savefig(a.output/f'S8{panel}.pdf');fig.savefig(a.output/f'S8{panel}.png',dpi=200);plt.close(fig)
 (a.output/'run_manifest.json').write_text(json.dumps(dict(commands=commands,limit_folds=a.limit_folds,protocol='retained SEED-V cross-five-fold candidate; not LOSO or full paper cohort'),indent=2))
if __name__=='__main__':main()
