from pathlib import Path
import argparse,sys,json
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'N/code'))
from decode_narratives_shortvideo_poe import GaussianPoERegressor
RATING_SETS={'five_happy_sad_afraid_disgusted_engaged':(1,2,3,4,6)}
RSA_SUBJECTS=[]
def select_exact30(
    data: dict[str, np.ndarray],
    exact35: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subjects = data["subject"].astype(str)
    sessions = data["session"].astype(int)
    objectives = data["objective"].astype(int)
    ratings = data["labels_subjective"].astype(np.float64)
    obs = data["obs"].astype(np.float64)
    obs35 = []
    rating35 = []
    for subject in RSA_SUBJECTS:
        subject_indices = np.flatnonzero(subjects == subject)
        lookup = {
            (int(sessions[index]), int(objectives[index])): int(index)
            for index in subject_indices
        }
        if len(lookup) != len(subject_indices):
            raise RuntimeError(f"{subject}: duplicate session x objective key")
        missing = [key for key in exact35 if key not in lookup]
        if missing:
            raise RuntimeError(f"{subject}: missing exact35 keys {missing}")
        indices = np.asarray([lookup[key] for key in exact35], dtype=int)
        obs35.append(obs[indices])
        rating35.append(ratings[indices])
    obs35_array = np.stack(obs35)
    rating35_array = np.stack(rating35)
    five = list(RATING_SETS["five_happy_sad_afraid_disgusted_engaged"])
    valid = np.all(
        np.isfinite(rating35_array[:, :, five])
        & (rating35_array[:, :, five] >= 0),
        axis=2,
    ).all(axis=0)
    if int(valid.sum()) != 30:
        raise RuntimeError(f"expected 30 common complete videos, got {valid.sum()}")
    return obs35_array[:, valid], rating35_array[:, valid], valid

def main():
    global RSA_SUBJECTS
    p=argparse.ArgumentParser();p.add_argument('--panel',choices=['B','N'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);rows=[]
    if a.panel=='B':
        for subject in ['S02','S04','S05']:
            with np.load(ROOT/'inputs/rsa'/('B_'+subject+'.npz')) as z:
                edges=[]
                for key in ['neural','text']:
                    x=z[key].astype(float);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
                    for session in np.unique(z['session']):
                        mask=z['session']==session;x[mask]-=x[mask].mean(0)
                    centers=np.stack([x[z['labels']==label].mean(0) for label in range(7)]);edges.append(pdist(centers,'sqeuclidean'))
                rows.append(dict(subject=subject,method='Ours',value=float(spearmanr(*edges).statistic)))
    else:
        selected=pd.DataFrame(json.loads((ROOT/'N/fixed_checkpoints.json').read_text())['rows']);RSA_SUBJECTS=selected.subject.tolist();v=pd.read_csv(ROOT/'N/videos.csv');videos=list(zip(v.session,v.objective_id))
        for epoch,level,key in [(13,'Meta-neural','highlevel_checkpoint'),(14,'Observation','lowlevel_checkpoint')]:
            with np.load(ROOT/'inputs/rsa'/f'epoch_{epoch:03d}.npz') as z:data={k:z[k] for k in z.files}
            obs,rating,valid=select_exact30(data,videos);indiv,_,_=select_exact30(dict(data,obs=data['indiv']),videos)
            for checkpoint in sorted(selected[key].unique()):
                ck=torch.load(ROOT/checkpoint,map_location='cpu',weights_only=False)
                cfg=ck['model_config'];model=GaussianPoERegressor(cfg['input_dim'],cfg['latent_dim'],cfg['hidden'],cfg['d_out'],cfg['dropout']).eval();model.load_state_dict(ck['model_state'],strict=True)
                xs=[]
                for x,k in [(obs,'obs_scaler'),(indiv,'indiv_scaler')]:
                    state=ck[k];xs.append(torch.tensor(((x.astype(np.float32)-np.asarray(state['mean'],np.float32))/np.asarray(state['std'],np.float32)).reshape(-1,x.shape[-1])))
                with torch.inference_mode():features=(model(*xs)['mu_poe'] if epoch==13 else model.obs_expert(xs[0])[0]).numpy().reshape(len(selected),30,-1)
                state=ck['rating_rdm_scalers']['five_happy_sad_afraid_disgusted_engaged']
                for i,row in enumerate(selected.itertuples()):
                    if getattr(row,key)!=checkpoint:continue
                    y=(rating[i][:,[1,2,3,4,6]]-np.asarray(state['mean']))/np.asarray(state['std'])
                    ne=pdist(features[i].astype(float),'correlation');be=pdist(y,'euclidean')
                    rows.append(dict(subject=row.subject,method=level,value=float(pearsonr(ne,be).statistic)))
                    np.savez_compressed(a.output/f'{level}_{row.subject}.npz',features=features[i],ratings=y,neural_distances=ne,rating_distances=be)
    frame=pd.DataFrame(rows);frame.to_csv(a.output/'metrics.csv',index=False)
    methods=sorted(frame.method.unique());fig,ax=plt.subplots();ax.boxplot([frame[frame.method==m].value for m in methods],labels=methods);ax.set_ylabel('RSA correlation');fig.savefig(a.output/(a.panel+'.pdf'));fig.savefig(a.output/(a.panel+'.png'),dpi=200);plt.close(fig)
    print(frame.groupby('method').value.agg(['count','mean','median']))
if __name__=='__main__':main()
