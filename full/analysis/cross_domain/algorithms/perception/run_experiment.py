
from pathlib import Path
import ast,json,sys,importlib.util
from collections import defaultdict
import numpy as np
import pandas as pd
import h5py
import torch
from pycocotools import mask as masks
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
NSD=Path('/media/wsqlab/nas2/Dataset/NSD-download')
RES=ROOT/'lizhuo_exp/lizhuo_exp_NSD/test_cifti_modes/result'
spec=importlib.util.spec_from_file_location('renderer',HERE.parent/'fig2_AB_NSD_source_verified_20260916/run.py')
plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)
plot.HERE=HERE
sys.path.insert(0,str(ROOT/'lizhuo_exp/lizhuo_exp_NSD/test_cifti_modes/code'))
import optimize_subject1_poe as model_code
GROUPS=plot.GROUPS

def paths(s):
 base=RES/('subject1_poe_mindeye2_1024' if s==1 else f'multisubject_poe/sub-{s:02d}/baseline')
 opt=RES/('subject1_poe_optimization' if s==1 else f'multisubject_poe/sub-{s:02d}/optimization')
 return base,opt

def extract_nsd():
 manifest=[]
 device=torch.device('cuda:1')
 for s in range(1,9):
  base,opt=paths(s);ck=opt/'checkpoints/r2_anchored_cont_stage2_best.pt'
  saved=torch.load(ck,map_location='cpu',mmap=True,weights_only=False)
  assert saved['metadata']['variant']=='anchored_poe'
  fusion=model_code.AnchoredPoE().to(device).eval()
  fusion.load_state_dict(saved['fusion'],strict=True)
  inputs=[np.load(opt/'cache/cifti_test.npy'),np.load(base/'cache/modes_test.npy')]
  split=np.load(base/'cache/split_and_normalization.npz')
  ids=split['test_stim_ids'];rows=split['test_rows']
  with h5py.File(NSD/f'preprocess/stage2/sub-{s:02d}.h5') as f:
   assert np.array_equal(f['stim_id'][rows],ids)
  assert len(inputs[0])==len(inputs[1])==len(ids)
  output=defaultdict(list)
  with torch.inference_mode():
   for start in range(0,len(ids),128):
    a,b=[torch.tensor(x[start:start+128],dtype=torch.float32,device=device) for x in inputs]
    obs=fusion.cifti_mean(a);prior=fusion.modes_mean(b)
    _,stats=fusion(a,b)
    expected=(obs*stats['cifti_precision']+prior*stats['modes_precision'])/(stats['cifti_precision']+stats['modes_precision'])
    assert torch.allclose(expected,stats['fused_mean'],rtol=1e-5,atol=1e-6)
    for k,v in [('observation',obs),('prior',prior),('posterior',stats['fused_mean']),('posterior_variance',stats['fused_variance'])]:
     output[k].append(v.cpu().numpy())
  result={k:np.concatenate(v) for k,v in output.items()}
  assert all(np.isfinite(v).all() for v in result.values())
  np.savez_compressed(HERE/f'NSD_sub-{s:02d}_experts.npz',**result,stim_ids=ids,h5_rows=rows)
  manifest.append(dict(subject=f'sub-{s:02d}',checkpoint=str(ck),n=len(ids),dimension=1024,metadata=saved['metadata']))
  print(f'NSD sub-{s:02d}: fresh Gaussian forward {len(ids)} images',flush=True)
  del saved,fusion
 (HERE/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2))

def crop_labels():
 info=pd.read_csv(NSD/'nsddata/experiments/nsd/nsd_stim_info_merged.csv').set_index('nsdId')
 ids=set()
 for s in range(1,9):ids.update(np.load(HERE/f'NSD_sub-{s:02d}_experts.npz')['stim_ids'].tolist())
 lookup={(row.cocoSplit,int(row.cocoId)):(int(nsdid),row) for nsdid,row in info.loc[np.array(sorted(ids))-1].iterrows()}
 selected={};crop_audit=[]
 for split in ['train2017','val2017']:
  data=json.loads((NSD/f'nsddata_stimuli/stimuli/annotations/instances_{split}.json').read_text())
  categories={c['id']:c for c in data['categories']}
  images={im['id']:im for im in data['images']}
  for ann in data['annotations']:
   key=(split,ann['image_id'])
   if key not in lookup:continue
   nsdid,row=lookup[key];im=images[ann['image_id']];height,width=im['height'],im['width']
   top,bottom,left,right=ast.literal_eval(row.cropBox)
   y0,y1=int(round(top*height)),int(round((1-bottom)*height))
   x0,x1=int(round(left*width)),int(round((1-right)*width))
   assert 0<=y0<y1<=height and 0<=x0<x1<=width
   assert abs((y1-y0)-(x1-x0))<=2, 'Crop convention must yield a square'
   seg=ann['segmentation']
   if isinstance(seg,list):rle=masks.merge(masks.frPyObjects(seg,height,width))
   elif isinstance(seg['counts'],list):rle=masks.frPyObjects(seg,height,width)
   else:rle=seg
   decoded=masks.decode(rle)
   area=int(decoded[y0:y1,x0:x1].sum())
   if area==0:continue
   cat=categories[ann['category_id']]
   record=dict(stim_id=nsdid+1,nsd_id=nsdid,coco_id=ann['image_id'],coco_split=split,
               category_id=ann['category_id'],category=cat['name'],group=cat['supercategory'],
               annotation_id=ann['id'],crop_area=area,original_area=ann['area'],
               crop_box=row.cropBox,crop_y0=y0,crop_y1=y1,crop_x0=x0,crop_x1=x1)
   if nsdid+1 not in selected or area>selected[nsdid+1]['crop_area']:selected[nsdid+1]=record
  del data
 for sid in sorted(ids):
  record=selected.get(sid,dict(stim_id=sid,nsd_id=sid-1,group='no_visible_annotation'))
  record['plot_code']=GROUPS.index(record['group']) if record['group'] in GROUPS else -1
  crop_audit.append(record)
 labels=pd.DataFrame(crop_audit);labels.to_csv(HERE/'crop_corrected_labels.csv',index=False)
 old=pd.read_csv(HERE.parent/'fig2_AB_NSD_source_verified_20260916/NSD_stimulus_category_mapping.csv').drop_duplicates('stim_id_1based')
 comparison=labels.merge(old[['stim_id_1based','plot_code']],left_on='stim_id',right_on='stim_id_1based',suffixes=('_cropped','_original'))
 comparison.to_csv(HERE/'crop_label_changes.csv',index=False)
 print('Crop labels:',labels.plot_code.value_counts().to_dict(),'changed:',int((comparison.plot_code_cropped!=comparison.plot_code_original).sum()),flush=True)

def sources():
 p=HERE.parent/'R2_multidataset_domain_geometry_20260915_gy_codex/features/eeg_all_subjects/BCIC/heldout_fold_geometry.npz'
 z=np.load(p)
 for s in np.unique(z['subject']):
  q=z['subject']==s
  yield 0,str(s),z['y'][q].astype(int),dict(observation=z['low'][q],prior=z['high'][q])
 labels=pd.read_csv(HERE/'crop_corrected_labels.csv').set_index('stim_id')
 counts=[]
 for s in range(1,9):
  z=np.load(HERE/f'NSD_sub-{s:02d}_experts.npz');y=labels.loc[z['stim_ids'],'plot_code'].to_numpy(int);take=y>=0
  for c,n in zip(*np.unique(y,return_counts=True)):counts.append(dict(subject=f'sub-{s:02d}',class_code=int(c),n=int(n)))
  yield 1,f'sub-{s:02d}',y[take],{k:z[k][take] for k in ['observation','prior','posterior']}
 pd.DataFrame(counts).to_csv(HERE/'NSD_subject_class_counts.csv',index=False)
 root=ROOT/'lizhuo_exp/lizhuo_exp_NSD/test_class2_0715/shared_model_tokenwise/features/ds000210'
 for p in sorted(root.glob('*.h5')):
  with h5py.File(p) as f:
   y=f['labels_objective'][:].reshape(-1).astype(int)
   assert np.array_equal(np.array(plot.LABELS[2])[y],np.array([v.decode() for v in f['trial_condition'][:]]))
   yield 2,p.stem,y,dict(observation=f['obs_expert_mu'][:],prior=f['indiv_expert_mu'][:],posterior=f['high'][:])

def analyse():
 rows=[];best={};coordinate_rows=[];selection=[]
 for d,s,y,features in sources():
  normalized={}
  for name,x in features.items():
   x=np.asarray(x,dtype=float);assert np.isfinite(x).all()
   raw=plot.h.metrics(x,y)
   centered=x-x.mean(0);scale=np.sqrt(np.mean(np.sum(centered**2,axis=1)))
   normalized[name]=centered/scale
   record=plot.h.metrics(normalized[name],y)
   rows.append(dict(domain=plot.DOMAINS[d],subject=s,representation=name,n=len(y),dimension=x.shape[1],rms_scale=scale,
                    **record,raw_within=raw['within'],raw_between=raw['between']))
  mm=[plot.h.metrics(normalized[k],y) for k in ['observation','prior']]
  gain=mm[1]['silhouette']-mm[0]['silhouette']
  selection.append(dict(domain=plot.DOMAINS[d],subject=s,silhouette_gain=gain,prior_silhouette=mm[1]['silhouette']))
  if d not in best or gain>best[d][0]:best[d]=(gain,s,y,normalized)
 table=pd.DataFrame(rows);table.to_csv(HERE/'all_subject_all_layers_metrics.csv',index=False)
 pd.DataFrame(selection).to_csv(HERE/'exemplar_selection_all_candidates.csv',index=False)
 summary=[]
 for domain in plot.DOMAINS:
  t=table[table.domain==domain]
  for rep in ['prior','posterior']:
   if rep not in t.representation.values:continue
   for metric in ['within','between','quality','silhouette']:
    v=t[t.representation.isin(['observation',rep])].pivot(index='subject',columns='representation',values=metric)
    delta=v[rep]-v.observation
    summary.append(dict(domain=domain,comparison=rep,metric=metric,n=len(v),observation_median=v.observation.median(),high_median=v[rep].median(),mean_delta=delta.mean(),improved=int(((delta<0) if metric=='within' else (delta>0)).sum())))
 pd.DataFrame(summary).to_csv(HERE/'cohort_summary.csv',index=False)
 arrays={};diag=[]
 for d,(_,s,y,features) in best.items():
  xx=[features[k] for k in ['observation','prior']]

  pca=PCA(3,svd_solver='full').fit(np.vstack(xx));aa=[pca.transform(x) for x in xx];bb=[]
  for m,x in enumerate(xx):
   q,rdm,stress=plot.h.mds(x);r,_=orthogonal_procrustes(q,aa[m]-aa[m].mean(0));bb.append(q@r)
   np.save(HERE/f'domain{d}_method{m}_RDM.npy',rdm)
   diag.append(dict(domain=plot.DOMAINS[d],subject=s,method=m,pca3_variance=pca.explained_variance_ratio_.sum(),mds_stress=stress))
  arrays[d]=(s,y,aa,bb)
  np.savez_compressed(HERE/f'domain{d}_selected_experts.npz',**features,labels=y,subject=s)
  for version,cc in [('A',aa),('B',bb)]:
   for m,q in enumerate(cc):
    for i,v in enumerate(q):coordinate_rows.append(dict(version=version,domain=plot.DOMAINS[d],subject=s,method=m,point=i,label=y[i],x=v[0],y=v[1],z=v[2]))
 pd.DataFrame(coordinate_rows).to_csv(HERE/'display_coordinates.csv',index=False)
 pd.DataFrame(diag).to_csv(HERE/'projection_diagnostics.csv',index=False)
 paired=table[table.representation.isin(['observation','prior'])].copy();paired['method']=paired.representation.map({'observation':0,'prior':1})
 for version in ['A','B']:
  plot.render(version,arrays,paired)
 print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__':
 extract_nsd()
 crop_labels()
 analyse()
