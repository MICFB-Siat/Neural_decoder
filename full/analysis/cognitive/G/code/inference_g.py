from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from weight_bundle import DEFAULT_BUNDLE, load_bundle, load_checkpoint

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
OUT = HERE / 'outputs'
FEATURES = {'ds000210':'full55_joint_mae_all_re_v1', 'ds002835':'ds002835_26_continual_mae_re_v1'}

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--subjects',nargs='+');parser.add_argument('--bundle',type=Path,default=DEFAULT_BUNDLE);args=parser.parse_args();OUT=args.output
    OUT.mkdir(parents=True,exist_ok=True)
    bundle=load_bundle(args.bundle)
    table=pd.DataFrame(bundle['rows'])
    if args.subjects:table=table[table.subject.isin(args.subjects)]
    if table.empty:raise ValueError('No matching subjects in the Fig. 6G bundle')
    jobs=set()
    for r in table.itertuples():
        method='poe' if r.method=='Ours' else 'neurostorm'
        for checkpoint in r.checkpoints:jobs.add((r.dataset,method,r.subject,checkpoint))
    sys.path.insert(0,str(HERE/'code'))
    import decode_loso_full55_v2 as dec
    config=json.loads((HERE/'datasets.json').read_text())
    dec.experiment_paths=lambda _: {'features':ROOT/'inputs/state_features'}
    torch.set_num_threads(1)
    device=torch.device('cpu')
    metrics=[]; predictions=[]
    for ds in FEATURES:
        for method in ['poe','neurostorm']:
            subjects,_=dec.read_subjects(config,ds,method,FEATURES[ds],31 if ds=='ds000210' else 26)
            for _,_,subject,checkpoint in sorted(j for j in jobs if j[:2]==(ds,method)):
                payload=load_checkpoint(bundle,checkpoint)
                prov=payload['provenance']
                assert (prov['dataset'],prov['method'],prov['test_subject'],prov['target_mode'])==(ds,method,subject,'stimulus_residual')
                reference=dec.Reference.from_state(payload['outer_reference'])
                reference_check=dec.compute_reference(subjects,prov['split']['re_train_subjects'],payload['output_dim'])
                assert dec.reference_equal(reference,reference_check)
                model=dec.base.build_model(method,payload['input_dim'],payload['ordinal_model'],payload['output_dim'],argparse.Namespace(**payload['protocol']),device)
                model.load_state_dict(payload['model_state'],strict=True)
                x,y,records=dec.collect_residual(subjects,[subject],reference,method)
                p=dec.predict_raw(model,x,method,dec.base.state_to_standardizer(payload['x_scaler']),dec.base.state_to_standardizer(payload.get('indiv_scaler')),dec.base.state_to_standardizer(payload['y_scaler']),device)[method]
                for col,target in enumerate(config['datasets'][ds]['label_names']):
                    r=float(pearsonr(y[:,col],p[:,col]).statistic)
                    metrics.append(dict(dataset=ds,method='Ours' if method=='poe' else 'NeuroStorm',subject=subject,checkpoint=checkpoint,target=target,pearson_r=r))
                for record,truth,pred in zip([r for r in records if r['included']],y,p):
                    for col,target in enumerate(config['datasets'][ds]['label_names']):
                        predictions.append(dict(dataset=ds,method=method,subject=subject,checkpoint=checkpoint,trial_index=record['trial_index'],target=target,true=float(truth[col]),prediction=float(pred[col])))
            print('Replayed',ds,method,flush=True)
    pd.DataFrame(predictions).to_csv(OUT/'predictions.csv',index=False)
    metrics=pd.DataFrame(metrics)
    metrics.to_csv(OUT/'metrics_per_checkpoint.csv',index=False)
    checks=[]
    for row in table.itertuples():
        m=metrics[(metrics.dataset==row.dataset)&(metrics.method==row.method)&(metrics.subject==row.subject)&(metrics.target==row.target)]
        m=m[m.checkpoint.isin(row.checkpoints)]; assert len(m)==len(row.checkpoints)
        actual=float(m.pearson_r.mean())
        checks.append(dict(dataset=row.dataset,method=row.method,subject=row.subject,target=row.target,checkpoints=json.dumps(row.checkpoints),value=actual))
    checks=pd.DataFrame(checks)
    checks.to_csv(OUT/'metrics.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    targets=sorted(checks.target.unique());fig,axes=plt.subplots(1,len(targets),figsize=(4*len(targets),4),squeeze=False)
    for ax,target in zip(axes[0],targets):
        d=checks[checks.target==target];methods=sorted(d.method.unique());ax.boxplot([d[d.method==m].value for m in methods],labels=methods);ax.set_title(target)
    fig.tight_layout();fig.savefig(OUT/'G.pdf');fig.savefig(OUT/'G.png',dpi=200);plt.close(fig)
    print(checks.groupby(['target','method']).value.median().to_string())

if __name__=='__main__': main()
