from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'I/code'))
import decode_cross_subject_residual_5fold as dec

def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',choices=['I','J'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--subjects',nargs='+');p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(2);device=torch.device(a.device);a.output.mkdir(parents=True,exist_ok=True)
    selected=pd.DataFrame(json.loads((ROOT/a.panel/'fixed_checkpoints.json').read_text())['rows'])
    if a.subjects:selected=selected[selected.subject.isin(a.subjects)]
    registry=json.loads((ROOT/'regression_inputs.json').read_text());predictions={};target_names={}
    for checkpoint in sorted({p for group in selected.checkpoints for p in group}):
        path=ROOT/checkpoint
        ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        method='Ours' if ck['method']=='poe' else 'NeuroStorm'

        subjects=[]
        for subject in ck['outer_test_subjects']:
            rows=selected[(selected.subject==subject)&(selected.method.str.lower()==method.lower())]
            if len(rows) and any(checkpoint in paths for paths in rows.checkpoints):
                subjects.append(subject)
        if not subjects:
            continue
        folder=registry[str(path.relative_to(ROOT))]
        model=dec.build_model_from_checkpoint(ck,device)
        for subject in subjects:
            with np.load(ROOT/'inputs/regression'/folder/(subject+'.npz')) as z:
                u=dec.SubjectData(subject,**{k:z[k].copy() for k in ['obs','indiv','neurostorm','target_values','stim_file','trial_index']})
            result=dec.predict_subject(model,u,ck['method'],[dec.ArrayScaler.from_state(v) for v in ck['feature_scalers']],dec.ArrayScaler.from_state(ck['target_scaler']),dec.ResidualReference.from_state(ck['outer_training_reference']),ck['min_reference_subjects'],device,ck['target_mode'])
            predictions[(method.lower(),subject,checkpoint)]=result
            target_names[(method.lower(),subject)]=ck['target_names']
            np.savez_compressed(a.output/f'{method}_{subject}_model_{len(predictions):04d}.npz',**result)
    values=[]
    undefined=[]
    for row in selected.itertuples():
        method=row.method.lower()
        if str(getattr(row,'selection_status','')).startswith('undefined_'):

            available=next((v for (m,sub,_),v in predictions.items() if m==method and sub==row.subject),None)
            if available is None:raise RuntimeError('No predictions to verify undefined target: '+row.subject)
            idx=target_names[(method,row.subject)].index(row.target)
            target=available['target_values_raw'][:,idx]
            finite=target[np.isfinite(target)]
            if len(finite)>1 and np.ptp(finite)>1e-8:
                raise RuntimeError('Historical undefined status conflicts with a varying target')
            undefined.append(dict(subject=row.subject,target=row.target,method=row.method,reason=row.selection_status,n_finite=int(len(finite)),target_range=float(np.ptp(finite)) if len(finite) else None))
            values.append(dict(subject=row.subject,target=row.target,method=row.method,value=float('nan')))
            continue
        selected_predictions=[predictions[(method,row.subject,p)] for p in row.checkpoints if (method,row.subject,p) in predictions]
        if len(selected_predictions)!=row.n_models:raise RuntimeError('Fixed checkpoint membership mismatch: '+row.subject)
        idx=target_names[(method,row.subject)].index(row.target)
        pk,tk=('prediction_raw_direct','target_values_raw') if a.panel=='I' else ('prediction_residual','target_values_residual')
        pred=np.mean(np.stack([v[pk] for v in selected_predictions]),axis=0).astype(np.float32)
        metric=dec.regression_metrics(selected_predictions[0][tk][:,idx],pred[:,idx])
        values.append(dict(subject=row.subject,target=row.target,method=row.method,value=metric['pearson_r']))
    (a.output/'undefined_metrics.json').write_text(json.dumps(undefined,indent=2))
    frame=pd.DataFrame(values)
    if frame.empty:raise ValueError('No selected predictions')
    frame.to_csv(a.output/'metrics.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    targets=sorted(frame.target.unique());fig,axes=plt.subplots(1,len(targets),figsize=(3*len(targets),4),squeeze=False)
    for ax,target in zip(axes[0],targets):
        d=frame[frame.target==target];methods=sorted(d.method.unique());ax.boxplot([d[d.method==m].value.dropna() for m in methods],labels=methods);ax.set_title(target);ax.set_ylabel('Pearson r')
    fig.tight_layout();fig.savefig(a.output/(a.panel+'.pdf'));fig.savefig(a.output/(a.panel+'.png'),dpi=200);plt.close(fig)
    print(frame.groupby(['target','method']).value.median())
if __name__=='__main__':main()
