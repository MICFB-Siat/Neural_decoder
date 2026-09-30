from pathlib import Path
import numpy as np,pandas as pd,json,itertools
from scipy.stats import t
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist,squareform
P=Path(__file__).resolve().parent;d=np.load(P/'features30.npz');subjects=d['subjects'];dims={'happy':1,'sad':2,'afraid':3,'disgusted':4,'engaged':6};methods={'Observation':'Low','Meta-neural':'High'}
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
rows=[];pairs=[];coverage=[];jack=[];fits={}
for emotion,di in dims.items():
 vals={};loo={}
 for method,key in methods.items():
  mat,b,a,n=fit(d[key],d['ratings'][:,:,di]);fits[emotion,method]=(mat,b,a,n);v=mat[np.triu_indices(30,1)];v=v[np.isfinite(v)];vals[method]=v.mean();loo[method]=[]
  for i in range(30):
   coverage.append(dict(emotion=emotion,method=method,subject=subjects[i],valid_films=int(n[i]),estimable=bool(np.isfinite(b[i]).all())))
   for j in range(i+1,30):pairs.append(dict(emotion=emotion,method=method,subject_i=subjects[i],subject_j=subjects[j],PS=mat[i,j]))
   mask=np.arange(30)!=i;mm,*_=fit(d[key][mask],d['ratings'][mask,:,di]);vv=mm[np.triu_indices(29,1)];loo[method].append(np.nanmean(vv));jack.append(dict(emotion=emotion,method=method,deleted_subject=subjects[i],PS=loo[method][-1]))
  rows.append(dict(emotion=emotion,method=method,n_subjects=int(np.isfinite(b).all(1).sum()),n_pairs=len(v),mean_PS=v.mean(),median_PS=np.median(v),min_films=int(n.min()),max_films=int(n.max())))
 delta=vals['Meta-neural']-vals['Observation'];jj=np.array(loo['Meta-neural'])-loo['Observation'];se=np.sqrt(29/30*np.sum((jj-jj.mean())**2));pv=2*t.sf(abs(delta/se),29)
 for rr in rows[-2:]:rr.update(delta=delta,jackknife_SE=se,p_raw=pv)
pv=np.array([r['p_raw'] for r in rows[::2]]);order=np.argsort(pv);adj=np.empty(5);adj[order]=np.minimum(1,np.maximum.accumulate(pv[order]*(5-np.arange(5))))
for i,r in enumerate(rows):r['p_Holm']=adj[i//2]
for fn,rs in [('PS_summary.csv',rows),('PS_pairs.csv',pairs),('coverage.csv',coverage),('jackknife.csv',jack)]:pd.DataFrame(rs).to_csv(P/fn,index=False)

old=json.loads((P.parent/'R5_manifest.json').read_text());sel=[list(subjects).index(s) for s in old['chosen_subjects']];out={};points=[]
for method in methods:
 mat,b,a,n=fits['happy',method];e=np.stack([a-b,a+b],1);x=e[sel].reshape(8,-1);dist=squareform(pdist(x));z=MDS(n_components=3,metric=True,dissimilarity='precomputed',n_init=1,max_iter=1500,eps=1e-9,random_state=12).fit_transform(dist,init=PCA(3).fit_transform(x));out[method+'_MDS']=z.reshape(4,2,3)
 for s,zz in zip(subjects[sel],z.reshape(4,2,3)):
  for state,pt in zip(['minus1SD','plus1SD'],zz):points.append(dict(method=method,subject=s,state=state,MDS1=pt[0],MDS2=pt[1],MDS3=pt[2]))
 out[method+'_endpoints']=e
np.savez_compressed(P/'geometry.npz',**out);pd.DataFrame(points).to_csv(P/'geometry_coordinates.csv',index=False)
(P/'manifest.json').write_text(json.dumps(dict(subjects=subjects.tolist(),display_subjects=subjects[sel].tolist(),emotion='happy',missing='per emotion pairwise available other-subject film mean; at least two other raters; at least five valid films per subject',selection='same four people as previous17-person figure, no new search',limits='fixed checkpoint trained on29/30; variable film sets; conditional exploratory jackknife, not heldout generalization'),indent=2))
print(pd.DataFrame(rows).to_string(index=False))
