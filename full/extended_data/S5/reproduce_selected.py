
from pathlib import Path
import argparse,subprocess,sys,json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panels',nargs='+',choices=list('ABC'),default=list('ABC'));p.add_argument('--datasets',nargs='+',choices=['BCIC','FACED','MOTOR','SEEDV'],default=['BCIC','FACED','MOTOR','SEEDV']);p.add_argument('--device',default='cpu');p.add_argument('--limit-folds',type=int,default=1);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Use a new output directory to keep this run distinct')
 a.output.mkdir(parents=True);rows=[];commands=[]
 for panel in a.panels:
  for dataset in a.datasets:
   configs=sorted((ROOT/'weights'/panel/dataset).rglob('args.json'))
   if not configs:raise FileNotFoundError(f'No retained conditions: {panel}/{dataset}')
   for config in configs:
    relative=config.parent.relative_to(ROOT);condition=relative.parts[3];out=a.output/panel/dataset/condition
    cmd=[sys.executable,str(ROOT/('infer_mode_count.py' if panel=='C' else 'infer_saved.py')),'--run',str(relative),'--device',a.device,'--limit-folds',str(a.limit_folds),'--output',str(out)]
    subprocess.run(cmd,cwd=ROOT,check=True);commands.append(cmd)
    frame=pd.read_csv(out/'fresh_metrics.csv');frame['panel']=panel;frame['dataset']=dataset;frame['condition']=condition;rows.append(frame)
 frame=pd.concat(rows,ignore_index=True);frame.to_csv(a.output/'fresh_conditions.csv',index=False)
 for panel in a.panels:
  fig,axes=plt.subplots(1,len(a.datasets),figsize=(4*len(a.datasets),4),squeeze=False)
  for ax,dataset in zip(axes[0],a.datasets):
   part=frame[(frame.panel==panel)&(frame.dataset==dataset)];groups=part.groupby('condition').accuracy.mean()
   if panel=='C':
    groups=groups.reindex(sorted(groups.index,key=lambda x:int(x[1:])));ax.plot([int(x[1:]) for x in groups.index],groups.values,'o-');ax.set_xlabel('Modes K')
   else:ax.bar(groups.index,groups.values);ax.tick_params(axis='x',rotation=30)
   ax.set(title=dataset,ylabel='Accuracy',ylim=(0,1))
  fig.suptitle(f'S5{panel}: retained-fold evaluation');fig.tight_layout();fig.savefig(a.output/f'S5{panel}.pdf');fig.savefig(a.output/f'S5{panel}.png',dpi=200);plt.close(fig)
 (a.output/'run_manifest.json').write_text(json.dumps(dict(commands=commands,limit_folds=a.limit_folds),indent=2))
if __name__=='__main__':main()
