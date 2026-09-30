

from pathlib import Path as _ReleasePath
import sys as _release_sys
_release_root = next(p for p in _ReleasePath(__file__).resolve().parents if (p/'asset_manifest.json').is_file())
_release_sys.path.insert(0, str(_release_root))
from release_checkpoints import resolve_weight as _weight_path

from pathlib import Path
import argparse
import hashlib
import json
import time
import tempfile
import importlib.metadata
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.linalg import orthogonal_procrustes
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
from numerics import KEYS,DOMAINS,PATHS,metrics,draw_indices,fast,stars
import replay

HERE=Path(__file__).resolve().parent
ROOT=HERE/'inputs'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8<<20),b''):
            h.update(part)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--mds-python',default=replay.MDS_PYTHON)
    args=parser.parse_args()
    replay.MDS_PYTHON=args.mds_python
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    out=args.output or Path(tempfile.mkdtemp(prefix='validation_',dir=HERE))
    if args.output:
        out.mkdir(parents=True,exist_ok=False)
    print(f'OUTPUT={out}',flush=True)
    t=time.monotonic()
    g=args.root/'guoyi_exp/geometric_add_exp'
    base=g/'R2_20260916'
    reference=base/'C1_NSDsub01_20260917'
    log=[]
    def register(path,role):
        path=Path(path)
        if role in ('model', 'patch_encoder'):
            packaged = _weight_path(HERE / 'weights' / path.relative_to(args.root))
            if packaged.is_file():
                path = packaged
        if not path.is_file():
            raise FileNotFoundError(path)
        log.append(dict(path=str(path),role=role,sha256=sha(path),size=path.stat().st_size))
        (out/'input_audit.json').write_text(json.dumps(log,indent=2))
        return path
    references=[g/'bcic_extended_models_20260917/BCIC_selected.npz',
                reference/'NSD_sub01_display_20260917.npz',
                base/'PastOther_gallery_20260917/cache_sub-07_seed05.npz']
    fullrows,bootrows,coordrows,statsrows,checks,indices_rows=[],[],[],[],[],[]
    for fn,ref in zip([replay.motor,replay.perception,replay.internal],references):
        print(f'REPLAY {fn.__name__}',flush=True)
        it=fn(args.root,g,torch.device(args.device),register)
        domain,y,ids=it['domain'],it['y'],it['ids']
        assert len(set(ids))==len(ids)
        tag=domain.replace(' ','_')
        np.savez_compressed(out/f'{tag}_fresh.npz',low_display=it['q'][0],high_display=it['q'][1],
                            low_input=it['low_input'],high_input=it['high_input'],high_native=it['high_native'],
                            labels=y,ids=ids,names=it['names'])
        original=np.stack([metrics(q,y) for q in it['q']])

        for k,p in enumerate(PATHS):
            fullrows.append(dict(domain=domain,pathway=p,subject=it['subject'],n=len(y),**dict(zip(KEYS,original[k]))))
            for i,point in enumerate(it['q'][k]):
                coordrows.append(dict(domain=domain,pathway=p,sample_id=ids[i],label=int(y[i]),
                                      condition=it['names'][int(y[i])],MDS1=point[0],MDS2=point[1],MDS3=point[2]))
        draw50=draw_indices(y,50)
        boot=np.array([[metrics(q[ix],y[ix]) for q in it['q']] for ix in draw50])
        for r,p in enumerate(PATHS):
            for b,v in enumerate(boot[:,r]):
                bootrows.append(dict(domain=domain,pathway=p,bootstrap_id=b,**dict(zip(KEYS,v))))
        for b,ix in enumerate(draw50):
            for j,i in enumerate(ix):
                indices_rows.append(dict(domain=domain,bootstrap_id=b,draw_position=j,source_index=int(i),sample_id=ids[i]))
        print(f'{domain}: fresh MDS and 50 plotted resamples done; checking 100000 test resamples',flush=True)
        ix=draw_indices(y,100000,20260917)
        counts=np.stack([(ix==i).sum(1) for i in range(len(y))],axis=1)
        v=np.stack([fast(q,y,counts) for q in it['q']],axis=1)
        for b in [0,1,17,555,99999]:
            for r,q in enumerate(it['q']):
                np.testing.assert_allclose(v[b,r],metrics(q[ix[b]],y[ix[b]]),atol=1e-10,rtol=0)
        np.savez_compressed(out/f'{tag}_bootstrap100000.npz',indices=ix,bootstrap_values=v,original_values=original,labels=y,ids=ids)
        delta=original[1]-original[0]
        for k,key in enumerate(KEYS):
            hits=int(np.sum(abs(v[:,1,k]-v[:,0,k]-delta[k])>=abs(delta[k])))
            statsrows.append(dict(domain=domain,metric=key,subject=it['subject'],n_subjects=1,
                                  n_samples=len(y),n_resamples=100000,delta_high_minus_low=delta[k],
                                  null_tail_count=hits,p_two_sided_approx=(1+hits)/100001))
        print(f'{domain} DONE elapsed={time.monotonic()-t:.1f}s',flush=True)
    full=pd.DataFrame(fullrows)
    boots=pd.DataFrame(bootrows)
    coords=pd.DataFrame(coordrows)
    statistics=pd.DataFrame(statsrows)
    statistics['p_Holm9_displayed']=multipletests(statistics.p_two_sided_approx,method='holm')[1]
    statistics['stars_Holm9_displayed']=statistics.p_Holm9_displayed.map(stars)
    full.to_csv(out/'fullsample_metrics.csv',index=False)
    boots.to_csv(out/'B_plot_data.csv',index=False)
    coords.to_csv(out/'A_plot_coordinates.csv',index=False)
    statistics.to_csv(out/'fresh_statistics.csv',index=False)
    pd.DataFrame(indices_rows).to_csv(out/'bootstrap50_indices.csv',index=False)
    summary=dict(status='COMPUTED',
                 n_mds_points=len(coords),n_bootstrap_rows=len(boots),n_bootstrap_scalar_values=len(boots)*3,
                 n_subjects_per_domain=1,n_test_resamples_per_domain=100000,
                 motor_mds_algorithm='vendored scikit-learn 1.3.2 SMACOF',motor_mds_python=args.mds_python,
                 actual_internal_contrast='Past / Other (theory of mind); screenshot Future label does not match source',
                 selection='Historical fixed outcome-selected examples. No new scan or retraining. Holm9 does not adjust for selection.',
                 elapsed_seconds=time.monotonic()-t)
    (out/'validation.json').write_text(json.dumps(summary,indent=2))
    (out/'runtime_versions.json').write_text(json.dumps({k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','torch','h5py','scikit-learn','matplotlib','statsmodels']},indent=2))
    (out/'code_sha256.json').write_text(json.dumps({p.name:sha(p) for p in HERE.glob('*.py')},indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    from plot import render
    render(out)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
