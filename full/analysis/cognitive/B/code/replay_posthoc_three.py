
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

ROOT = Path('/media/wsqlab/Expansion/meg_h5')
OUT = Path(__file__).resolve().parent / 'posthoc_three_subjects_240'
SELECTION = {'linguiyu':136, 'lizhuo':197, 'zhangchi':14}

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def rsa(neural, text, session, labels):
    arrays=[]
    for a in [neural,text]:
        a=np.asarray(a,dtype=np.float64)
        a=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
        for s in np.unique(session):
            mask=session==s
            a[mask]-=a[mask].mean(axis=0)
        centroids=np.stack([a[labels==c].mean(axis=0) for c in range(7)])
        arrays.append(pdist(centroids,metric='sqeuclidean'))
    return float(spearmanr(*arrays).statistic)

def export():
    OUT.mkdir(parents=True,exist_ok=True)
    archive=ROOT/'result/semantic_rsa_240trials_20260925'
    history=pd.read_csv(archive/'all_draws.csv')
    history.to_csv(OUT/'all_original_draws.csv',index=False)
    records=[]
    for subject,repeat in SELECTION.items():
        source=archive/f'{subject}_draws.npz'
        feature=ROOT/'result/semantic_multiverse_20260719/frozen_features'/f'{subject}.npz'
        meta=ROOT/'imagery_highlevel/feats'/f'{subject}.npz'
        with np.load(source,allow_pickle=True) as d, np.load(feature,allow_pickle=True) as f, np.load(meta,allow_pickle=True) as m:
            for key in ['session','sentences','bge']:
                np.testing.assert_array_equal(f[key],m[key])
            index=d['indices'][repeat].copy()
            assert len(index)==len(np.unique(index))==240
            col=list(d['configs']).index('ours_img_full')
            expected=float(d['observed'][repeat,col])
            h=history[(history.subject==subject)&(history.config=='ours_img_full')]
            np.testing.assert_allclose(expected,h[h['repeat']==repeat].rsa.iloc[0],atol=1e-12)
            np.testing.assert_allclose(expected,h.rsa.max(),atol=1e-12)
            frame=pd.DataFrame({'subject':subject,'repeat_index_0based':repeat,
                'trial_index_0based':index,'trial_index_1based':index+1,
                'session':f['session'][index],'category_label':m['label_id'][index],
                'sentence':f['sentences'][index].astype(str)})
            frame.to_csv(OUT/f'{subject}_selected_trials.csv',index=False)
            np.savez_compressed(OUT/f'{subject}_selected_features.npz',
                neural=f['ours_img_full'][index],text=f['bge'][index],
                session=f['session'][index],labels=m['label_id'][index],
                original_trial_indices=index,expected_rsa=expected)
        records.append({'subject':subject,'repeat_index_0based':repeat,'n_trials':240,
            'source_files':[{'path':str(p),'sha256':digest(p)} for p in [source,feature,meta]]})
    (OUT/'provenance.json').write_text(json.dumps({'window':'0-4s',
        'representation':'frozen group-eigenmode prior, not full PoE',
        'trial_indices':'rows of the recorded frozen feature files, not raw MEG event IDs',
        'all_draws_retained':True,'records':records},indent=2))

def verify():
    rows=[]
    for subject,repeat in SELECTION.items():
        with np.load(OUT/f'{subject}_selected_features.npz') as d:
            score=rsa(d['neural'],d['text'],d['session'],d['labels'])
            expected=float(d['expected_rsa'])
            np.testing.assert_allclose(score,expected,atol=1e-12,rtol=0)
            rows.append({'subject':subject,'repeat_index_0based':repeat,'n_trials':240,
                'historical_rsa':expected,'recomputed_rsa':score,'absolute_error':abs(score-expected)})
    results=pd.DataFrame(rows)
    results.to_csv(OUT/'verification.csv',index=False)
    print(results.to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if not args.verify_only: export()
    verify()
