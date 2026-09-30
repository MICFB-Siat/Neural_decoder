from pathlib import Path
import argparse,sys
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist,squareform
from scipy.stats import rankdata,pearsonr
from sklearn.decomposition import PCA
from smacof_132 import _smacof_single
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'K/code'))
from decode_cross_subject_residual_5fold import build_model_from_checkpoint,ArrayScaler
def fit(x,y):
 valid=np.isfinite(y)&(y>=0)&np.isfinite(x).all(-1)
 xx=np.where(valid[:,:,None],x,0);yy=np.where(valid,y,0);count=valid.sum(0)[None,:]-valid
 rx=x-(xx.sum(0)[None]-xx)/np.maximum(count[:,:,None],1);ry=y-(yy.sum(0)[None]-yy)/np.maximum(count,1)
 valid &= count>=2
 beta=[];alpha=[];ns=[]
 sd=np.std(ry[valid]);ry=ry/sd
 for s in range(len(x)):
  m=valid[s];a=rx[s,m];r=ry[s,m];ns.append(m.sum())
  if m.sum()<5 or np.std(r)<1e-10:beta.append(np.full(x.shape[-1],np.nan));alpha.append(np.full(x.shape[-1],np.nan));continue
  b=((r-r.mean())[:,None]*(a-a.mean(0))).sum(0)/np.sum((r-r.mean())**2);beta.append(b);alpha.append(a.mean(0)-b*r.mean())
 b=np.array(beta);a=np.array(alpha);u=b/np.linalg.norm(b,axis=1,keepdims=True);mat=u@u.T
 return mat,b,a,np.array(ns)

def fractional_rank(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=np.float64), method="average")
    return (ranks - 1.0) / max(len(ranks) - 1.0, 1.0)

def participant_mean_rank_rdm(features: np.ndarray) -> np.ndarray:

    values = np.asarray(features, dtype=np.float64).transpose(1, 0, 2)
    values -= values.mean(axis=2, keepdims=True)
    norms = np.sqrt(np.sum(values**2, axis=2, keepdims=True))
    norms[norms <= 1e-12] = 1.0
    values /= norms
    distances = 1.0 - np.einsum("vsd,vtd->vst", values, values)
    n_subjects = features.shape[0]
    triangle = np.triu_indices(n_subjects, 1)
    edges = distances[:, triangle[0], triangle[1]]
    ranked = np.vstack([fractional_rank(row) for row in edges])
    mean_edge = ranked.mean(axis=0)
    rdm = np.zeros((n_subjects, n_subjects), dtype=np.float64)
    rdm[triangle] = mean_edge
    rdm[(triangle[1], triangle[0])] = mean_edge
    return rdm

def main():
    p=argparse.ArgumentParser();p.add_argument('--panels',nargs='+',choices=list('KLM'),default=list('KLM'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    folder=ROOT/'inputs/geometry';records={};keys=set()
    for f in sorted(folder.glob('sub-*.npz')):
        with np.load(f) as z:
            kk=list(zip(z['session'].astype(int).ravel(),z['objective'].astype(int).ravel()));records[f.stem]=(kk,z['obs'],z['indiv'],z['ratings']);keys.update(kk)
    subjects=sorted(records);keys=sorted(keys);lookup={k:i for i,k in enumerate(keys)};n=len(subjects)
    obs=np.full((n,len(keys),512),np.nan,np.float32);indiv=obs.copy();rating=np.full((n,len(keys),7),np.nan)
    for i,s in enumerate(subjects):
        kk,o,v,y=records[s];idx=[lookup[k] for k in kk];obs[i,idx]=o;indiv[i,idx]=v;rating[i,idx]=y
    present=np.isfinite(obs).all(-1);ck=torch.load(ROOT/'K/weights/fold25_model00_poe.pt',map_location='cpu',weights_only=False);model=build_model_from_checkpoint(ck,torch.device('cpu')).eval()
    x=[torch.tensor(ArrayScaler.from_state(st).transform(v[present]),dtype=torch.float32) for st,v in zip(ck['feature_scalers'],[obs,indiv])]
    with torch.inference_mode():
        mo,lo=model.obs_expert(x[0]);mi,li=model.indiv_expert(x[1]);po=torch.exp(-lo);pi=torch.exp(-li);mp=(po*mo+pi*mi)/(1+po+pi)
    high=np.full((n,len(keys),128),np.nan);mu=high.copy();high[present]=mp.numpy();mu[present]=mo.numpy()
    np.savez_compressed(a.output/'fresh_features.npz',Low=obs,ObsMu=mu,High=high,ratings=rating,subjects=subjects,videos=keys)
    with np.load(folder/'cohort.npz') as z:ids=[subjects.index(s) for s in z['subjects']];vid=[lookup[tuple(v)] for v in z['videos']];cohort=z['subjects']
    rr=rating[ids][:,vid];valid=np.all(np.isfinite(rr[:,:,[1,2,3,4,6]])&(rr[:,:,[1,2,3,4,6]]>=0),axis=(0,2));rr=rr[:,valid];vv=np.array(vid)[valid];assert len(vv)==30
    if 'K' in a.panels:
        r=rr[:,:,1];r=r-(r.sum(0,keepdims=True)-r)/16;r/=r.std();sel=[list(cohort).index(s) for s in ['sub-0034','sub-0060','sub-0004','sub-0099']];rows=[];fig=plt.figure(figsize=(9,4))
        for j,(name,source) in enumerate([('Observation',obs),('Meta-neural',high)]):
            xx=source[ids][:,vv].astype(float);xx=xx-(xx.sum(0,keepdims=True)-xx)/16;xx/=np.sqrt(np.mean(xx*xx));xc=xx-xx.mean(1,keepdims=True);rc=r-r.mean(1,keepdims=True);beta=np.einsum('sv,svf->sf',rc,xc)/(np.sum(rc*rc,axis=1)[:,None]+1e-12);alpha=xx.mean(1)-beta*r.mean(1)[:,None];ends=np.stack([alpha-beta,alpha+beta],axis=1);pts=ends[sel].reshape(8,-1)
            coords,stress,nit=_smacof_single(squareform(pdist(pts)),n_components=3,init=PCA(3).fit_transform(pts),max_iter=1500,eps=1e-9,random_state=12);coords=coords.reshape(4,2,3);ax=fig.add_subplot(1,2,j+1,projection='3d');ax.set_title(name)
            for sid,zz in zip(cohort[sel],coords):
                ax.plot(*zz.T,marker='o')
                for state,point in zip(['minus1SD','plus1SD'],zz):rows.append(dict(method=name,subject=sid,state=state,MDS1=point[0],MDS2=point[1],MDS3=point[2]))
        pd.DataFrame(rows).to_csv(a.output/'K_coordinates.csv',index=False);fig.savefig(a.output/'K.pdf');fig.savefig(a.output/'K.png',dpi=200);plt.close(fig)
    if 'L' in a.panels:
        rows=[]
        for name,source in [('Observation',obs),('Meta-neural',high)]:
            matrix,b,alpha,count=fit(source,rating[:,:,1]);np.fill_diagonal(matrix,np.nan)
            for sid,value in zip(subjects,np.nanmean(matrix,axis=1)):rows.append(dict(subject=sid,method=name,value=value))
        frame=pd.DataFrame(rows);frame.to_csv(a.output/'L_metrics.csv',index=False);fig,ax=plt.subplots();ax.boxplot([frame[frame.method==m].value for m in ['Observation','Meta-neural']],labels=['Observation','Meta-neural']);ax.set_ylabel('Mean parallelism to other participants');fig.savefig(a.output/'L.pdf');fig.savefig(a.output/'L.png',dpi=200);plt.close(fig)
    if 'M' in a.panels:
        raw=np.load(folder/'scaler_ratings37.npy')[:,[1,2,3,4,6]];raw=raw[np.all(np.isfinite(raw)&(raw>=0),axis=1)];center=raw.mean(0);scale=raw.std(0);scale[scale<=1e-12]=1;ratings=(rr[:,:,[1,2,3,4,6]]-center)/scale
        edges=np.mean([fractional_rank(pdist(v,'euclidean')) for v in ratings.transpose(1,0,2)],axis=0);tri=np.triu_indices(17,1);table={'rating_distance':edges};fig,axes=plt.subplots(1,2,figsize=(9,4))
        for ax,(name,source) in zip(axes,[('Observation',obs),('Ours',high)]):
            v=participant_mean_rank_rdm(source[ids][:,vv])[tri];table[name]=v;corr=float(pearsonr(v,edges).statistic);ax.scatter(v,edges,s=10);ax.set(title=f'{name}: r={corr:.6f}',xlabel='Neural distance',ylabel='Rating distance');print(name,corr)
        pd.DataFrame(table).to_csv(a.output/'M_distances.csv',index=False);fig.savefig(a.output/'M.pdf');fig.savefig(a.output/'M.png',dpi=200);plt.close(fig)
if __name__=='__main__':main()
