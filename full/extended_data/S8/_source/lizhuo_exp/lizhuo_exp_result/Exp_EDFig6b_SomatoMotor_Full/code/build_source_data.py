
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import pandas as pd
REPO=Path(__file__).resolve().parents[4]
BASE=REPO/'data_check_20260507/Exp_tyf_Results/Results/Exp_Fig2/Somatomotor'
SUBS=['sub-sm04','sub-sm06','sub-sm07','sub-sm09','sub-sm12']
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=Path.cwd()/'inference_output'); a=ap.parse_args(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for method,folder in [('LaBraM','labram_within5fold'),('CBraMod','cbramod_within5fold')]:
  for sub in SUBS:
   for p in sorted((BASE/folder/sub).glob('fold_*/metrics.json')):
    d=json.load(open(p)); rows.append({'dataset':'SomatoMotor','subject':sub,'seed':None,'fold':int(p.parent.name.split('_')[-1]),'method':method,'accuracy':d['test_acc'],'metric_source':str(p)})
 seed=4; summary_path=BASE/'BT-ND+BrainOmni_within5fold'/f'somato_s{seed}'/'summary.json'; d=json.load(open(summary_path))
 for sub in SUBS:
  sd=d['per_subject'][sub]
  for method,key in [('BrainOmni','obs_acc'),('Ours','poe_temp')]:
   for fold,value in enumerate(sd[key]['values']): rows.append({'dataset':'SomatoMotor','subject':sub,'seed':seed,'fold':fold,'method':method,'accuracy':value,'metric_source':str(summary_path)})
 df=pd.DataFrame(rows); df.to_csv(out/'edfig6b_four_method_fold_accuracy.csv',index=False)
 subject=df.groupby(['dataset','subject','method'],as_index=False).accuracy.mean(); subject.to_csv(out/'edfig6b_four_method_subject_accuracy.csv',index=False)
 summary=subject.groupby(['dataset','method'],as_index=False).agg(n_subjects=('subject','nunique'),mean_accuracy=('accuracy','mean'),sem_accuracy=('accuracy',lambda x:x.std(ddof=1)/math.sqrt(len(x)))); summary.to_csv(out/'edfig6b_four_method_summary.csv',index=False)
 manifest={'status':'complete_saved_prediction_analysis','fresh_model_forward':False,'paper_n':5,'subjects':SUBS,'methods':['LaBraM','BrainOmni','CBraMod','Ours'],'selected_seed':4,'seed_selection_reason':'highest retained Ours poe_temp mean among available seeds 0-4; display-version selection, not test-independent model selection','checkpoint_root':str(BASE/'BT-ND+BrainOmni_within5fold/somato_s4'),'h5_root':str(REPO/'data_check_20260507/uncertainty_seedv/tyf_20260521/somatomotor_patched'),'important_boundary':'This producer recomputes source-data tables from retained fold metrics; the separate somato-checkpoint task performs the fixed saved-weight forward.'}
 (out/'source_data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(out)
if __name__=='__main__': main()
