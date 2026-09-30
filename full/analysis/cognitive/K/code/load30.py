from pathlib import Path
import sys,os,importlib,json
import numpy as np,pandas as pd,torch
ROOT=Path('/home/guoyi/a800/code/Eigen_brain_decoding');OUT=Path(__file__).resolve().parent
D=ROOT/'lizhuo_exp/lizhuo_exp_NSD/test_class2_0715/44_alignvideo_continue20_loso80_supervisedpoe128___17x35_20260724'
sys.path[:0]=[str(D),str(ROOT/'lizhuo_exp/lizhuo_exp_class2/SKIP/code/enc_mod')]
for m in ['', '.multiarray','.numeric','.umath','.numerictypes','.overrides','._multiarray_umath']:
 try:sys.modules['numpy._core'+m]=importlib.import_module('numpy.core'+m)
 except ImportError:pass
import run_loso80_supervised_poe128__ as r
import compute_k_l_affect45_pearson_spearman as a
from decode_cross_subject_residual_5fold import ArrayScaler
torch.set_num_threads(4)
import h5py
subs=sorted(pd.read_csv(ROOT/'lizhuo_exp/lizhuo_exp_result/Exp_Fig6_Spacetop_Subjective_Ratings/result/paper_results/fig6j_five_emotion_residual_correlations.csv').participant.unique())
assert len(subs)==30
records=[];keys=set()
for sub in subs:
 with h5py.File(r.FEATURE_ROOT/f'{sub}.h5','r') as h:
  kk=list(zip(np.asarray(h['session'],int).tolist(),np.asarray(h['labels_objective'],int).tolist()));assert len(set(kk))==len(kk)
  records.append((kk,np.asarray(h['bci_obs'],np.float32).mean(1),np.asarray(h['bci_prior_indiv'],np.float32).mean(1),np.asarray(h['labels_subjective'],float)));keys.update(kk)
keys=sorted(keys);lookup={k:i for i,k in enumerate(keys)}
obs=np.full((30,len(keys),512),np.nan,np.float32);indiv=obs.copy();ratings=np.full((30,len(keys),7),np.nan)
for i,(kk,o,v,y) in enumerate(records):
 idx=[lookup[k] for k in kk];obs[i,idx]=o;indiv[i,idx]=v;ratings[i,idx]=y
present=np.isfinite(obs).all(-1)
ck=torch.load(r.checkpoint_path(25,0),map_location='cpu',weights_only=False);model=r.build_model_from_checkpoint(ck,torch.device('cpu'));model.eval();sc=[ArrayScaler.from_state(v) for v in ck['feature_scalers']]
x=[torch.tensor(sc[i].transform(a[present]),dtype=torch.float32) for i,a in enumerate([obs,indiv])]
with torch.inference_mode():
 ot=model.obs_expert.trunk(x[0]);it=model.indiv_expert.trunk(x[1]);mo=model.obs_expert.mu(ot);mi=model.indiv_expert.mu(it)
 po=torch.exp(-model.obs_expert.logvar(ot).clamp(-5,5));pi=torch.exp(-model.indiv_expert.logvar(it).clamp(-5,5));mp=(po*mo+pi*mi)/(1+po+pi)
high=np.full((30,len(keys),128),np.nan);high[present]=mp.numpy()
np.savez_compressed(OUT/'features30.npz',Low=obs,High=high,ratings=ratings,subjects=subs,videos=keys)
(OUT/'checkpoint.json').write_text(json.dumps(dict(checkpoint=str(r.checkpoint_path(25,0)),train_subjects=ck['outer_train_subjects'],test_subjects=ck['outer_test_subjects'],n_subjects=30,n_union_films=len(keys),n_trials=present.sum().item()),indent=2))
print('DONE',len(keys),present.sum(),flush=True)
