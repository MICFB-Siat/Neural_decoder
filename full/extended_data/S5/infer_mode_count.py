
from pathlib import Path
import argparse,json,sys,csv
import numpy as np,torch
import infer_saved as replay
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'_source'
PREP=SOURCE/'data_check_20260507/Exp_tyf_Results/Exp_ablation_bcic_faced_motor/code'
sys.path.insert(0,str(PREP))
from build_k_datasets import build_subject

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--limit-folds',type=int);p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=32);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=(ROOT/a.run).resolve();rel=root.relative_to(ROOT).parts
 if rel[:2]!=('weights','C'):raise ValueError('Select a retained S5C run')
 dataset,kname=rel[2:4];k=int(kname[1:]);cfg=replay.read(root/'args.json');a.output.mkdir(parents=True,exist_ok=True)
 templates={'BCIC':'BCIC/d_template_eeg_17_5e253942b967.npy','FACED':'FACED/d_template_eeg_25.npy','MOTOR':'MOTOR_eeg-fmri/D_eeg.npy','SEEDV':'SEED-V/sensor_geom_eigenmode_template_60x48.npy'}
 geom=SOURCE/'guoyi_exp/data/eigenmode_test';template=geom/templates[dataset];raw=SOURCE/'data_check_20260507/full_run_5fold/stage2_new'/{'FACED':'FACED_new','SEEDV':'SEED-V'}.get(dataset,dataset)
 d=np.load(template).astype(np.float64);u,s,vt=np.linalg.svd(d,full_matrices=False)
 if k<1 or k>int((s/s[0]>=1e-12).sum()):raise ValueError('K exceeds original template rank')
 prep=a.output/'prepared_inputs';prep.mkdir(exist_ok=True);dpath=prep/f'D_rank{k}.npy';np.save(dpath,((u[:,:k]*s[:k])@vt[:k]).astype(np.float64))
 if dataset=='SEEDV':lam=np.load(ROOT/'templates/cortex_eigval_48.npy')
 else:
  left=np.load(geom/'fs32k/fsLR_32k_lh_eval_1024.npy');right=np.load(geom/'fs32k/fsLR_32k_rh_eval_1024.npy');lam=np.concatenate([left[:(d.shape[1]+1)//2],right[:d.shape[1]//2]]).astype(np.float32)
 lpath=prep/'cortical_eigenvalues.npy';np.save(lpath,lam)
 folds=sorted(p.parent for p in root.glob('sub-*/fold_*/split.json'))
 if a.limit_folds:folds=folds[:a.limit_folds]
 if not folds:raise ValueError('No retained split')
 for subject in sorted(set(f.parent.name for f in folds)):
  build_subject(raw/(subject+'.h5'),prep/(subject+'.h5'),u[:,:k].astype(np.float32),k,template,cfg['label_key'],8)

 prepared_values={str(p.resolve()) for p in [prep,dpath,lpath]}
 original_resolve=replay.resolve
 replay.resolve=lambda value: Path(value) if str(value) in prepared_values else original_resolve(value)
 cls=replay.load_module('classification_runtime',replay.EXP/'Exp_Classification/code/classify_baseline_v2.py')
 rows=[]
 for fold in folds:
  result=replay.infer_fold(cls,root,fold,a.output/fold.relative_to(root),torch.device(a.device),a.batch_size,config_override={'h5_dir':str(prep.resolve()),'d_path':str(dpath.resolve()),'lam_cortex_path':str(lpath.resolve())})
  rows.append(dict(dataset=dataset,K=k,fold=str(fold.relative_to(root)),accuracy=result['accuracy'],n_trials=result['n_trials']))
 with (a.output/'fresh_metrics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(4,4));ax.scatter([k]*len(rows),[r['accuracy'] for r in rows]);ax.set(xlabel='Retained modes K',ylabel='Accuracy',ylim=(0,1));fig.tight_layout();fig.savefig(a.output/'accuracy.pdf');fig.savefig(a.output/'accuracy.png',dpi=200);plt.close(fig)
 (a.output/'preparation.json').write_text(json.dumps(dict(dataset=dataset,K=k,source=str(raw),template=str(template),algorithm='Nested leading left singular vectors; rank-K D reconstruction; original cortical eigenvalue convention',run=str(root)),indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
