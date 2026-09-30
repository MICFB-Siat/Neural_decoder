
from pathlib import Path
import json, importlib.util
from collections import defaultdict
import numpy as np
import pandas as pd
import h5py
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                     'pdf.fonttype':42,'svg.fonttype':'none'})

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
NSD=Path('/media/wsqlab/nas2/Dataset/NSD-download')
RES=ROOT/'lizhuo_exp/lizhuo_exp_NSD/test_cifti_modes/result'
spec=importlib.util.spec_from_file_location('real',HERE.parent/'fig2_AB_real_verified_20260916/plot_real.py')
real=importlib.util.module_from_spec(spec);spec.loader.exec_module(real)
h=real.h
LABELS=[real.LABELS[0],['People','Animals','Vehicles','Food','Furniture'],real.LABELS[1]]
GROUPS=['person','animal','vehicle','food','furniture']
DOMAINS=['Motor','Perception','Internal mentation']

def nsd_sources():
 info=pd.read_csv(NSD/'nsddata/experiments/nsd/nsd_stim_info_merged.csv').set_index('nsdId')
 splits={}; required=set()
 for s in range(1,9):
  path=RES/('subject1_poe_mindeye2_1024' if s==1 else f'multisubject_poe/sub-{s:02d}/baseline')/'cache/split_and_normalization.npz'
  z=np.load(path); ids=z['test_stim_ids']; rows=z['test_rows']
  with h5py.File(NSD/f'preprocess/stage2/sub-{s:02d}.h5') as f:
   assert np.array_equal(f['stim_id'][rows],ids)

  meta=info.loc[ids-1];splits[s]=(ids,rows,meta)
  required.update(meta.cocoId.astype(int))
 dominant={}
 for split in ['train2017','val2017']:
  p=NSD/f'nsddata_stimuli/stimuli/annotations/instances_{split}.json'
  data=json.loads(p.read_text());cats={c['id']:c for c in data['categories']}
  for a in data['annotations']:
   iid=a['image_id']
   if iid not in required:continue
   key=(split,iid)
   if key not in dominant or a['area']>dominant[key]['area']:
    dominant[key]=dict(area=a['area'],annotation_id=a['id'],category_id=a['category_id'],
                        category=cats[a['category_id']]['name'],group=cats[a['category_id']]['supercategory'])
  del data
 audit=[]
 for s,(ids,rows,meta) in splits.items():
  y=[]
  for i,(sid,(_,m)) in enumerate(zip(ids,meta.iterrows())):
   a=dominant.get((m.cocoSplit,int(m.cocoId)),{})
   label=GROUPS.index(a.get('group')) if a.get('group') in GROUPS else -1
   y.append(label)
   audit.append(dict(subject=f'sub-{s:02d}',feature_row=i,h5_row=int(rows[i]),stim_id_1based=int(sid),
                     nsd_id_0based=int(sid)-1,coco_id=int(m.cocoId),coco_split=m.cocoSplit,
                     crop_box=m.cropBox,plot_code=label,included=label>=0,**a))
  y=np.array(y);take=y>=0
  lo=np.load(RES/f'cls_global_retrieval/sub-{s:02d}/brain_direct_mean_test.npy')
  hi=np.load(ROOT/f'lizhuo_exp/lizhuo_exp_result/Exp_Fig5_NSD_Retrieval_Reconstruction/result/fresh_retrieval_all_subjects/retrieval/sub-{s:02d}/predicted_global_embeddings.npy')
  assert len(lo)==len(hi)==len(ids)
  yield 1,f'sub-{s:02d}',y[take],[lo[take],hi[take]],'NSD retrieval cache; low=fused backbone, high=retrieval head'
 pd.DataFrame(audit).to_csv(HERE/'NSD_stimulus_category_mapping.csv',index=False)

def main():
 records=[];best={}
 def all_sources():
  for d,s,y,xx,source in real.sources():yield (0 if d==0 else 2),s,y,xx,source
  yield from nsd_sources()
 for d,s,y,xx,source in all_sources():
  assert all(np.isfinite(x).all() for x in xx)
  xx=[np.asarray(x,dtype=float) for x in xx]
  xx=[(x-x.mean(0))/np.sqrt(np.mean(np.sum((x-x.mean(0))**2,axis=1))) for x in xx]
  mm=[h.metrics(x,y) for x in xx]
  for m in [0,1]:records.append(dict(domain=DOMAINS[d],subject=s,method=m,n=len(y),source=source,**mm[m]))
  gain=mm[1]['silhouette']-mm[0]['silhouette']
  if d not in best or gain>best[d][0]:best[d]=(gain,s,y,xx)
 table=pd.DataFrame(records);table.to_csv(HERE/'all_subject_metrics.csv',index=False)
 coords=[];arrays={};diag=[]
 for d,(_,s,y,xx) in best.items():
  aa=[PCA(3,svd_solver='full').fit_transform(x) for x in xx]
  r,_=orthogonal_procrustes(aa[1],aa[0]);aa[1]=aa[1]@r
  bb=[]
  for m,x in enumerate(xx):
   q,rdm,stress=h.mds(x);r,_=orthogonal_procrustes(q,aa[m]);bb.append(q@r)
   diag.append(dict(domain=DOMAINS[d],subject=s,method=m,mds_stress=stress))
   np.save(HERE/f'domain{d}_method{m}_RDM.npy',rdm)
  arrays[d]=(s,y,aa,bb)
  np.savez_compressed(HERE/f'domain{d}_selected.npz',low=xx[0],high=xx[1],labels=y,subject=s)
  for version,cc in [('A',aa),('B',bb)]:
   for m,q in enumerate(cc):
    for i,v in enumerate(q):coords.append(dict(version=version,domain=DOMAINS[d],subject=s,method=m,point=i,label=y[i],x=v[0],y=v[1],z=v[2]))
 pd.DataFrame(coords).to_csv(HERE/'display_coordinates.csv',index=False)
 pd.DataFrame(diag).to_csv(HERE/'mds_diagnostics.csv',index=False)
 for version in ['A','B']:render(version,arrays,table)

def render(version,arrays,table,notes=None):
 fig=plt.figure(figsize=(11.8,6.8))
 fig.text(.5,.963,'Real-data geometry across cognitive domains',ha='center',fontsize=13,weight='bold')
 fig.text(.5,.928,f'{version} · '+('PCA point clouds' if version=='A' else 'Correlation RDM → metric MDS'),ha='center',fontsize=9)
 gs=fig.add_gridspec(2,3,left=.05,right=.76,bottom=.19,top=.82,wspace=.07,hspace=.15)
 rg=fig.add_gridspec(3,1,left=.815,right=.982,bottom=.23,top=.81,hspace=.65)
 for d in range(3):
  s,y,aa,bb=arrays[d];cc=aa if version=='A' else bb
  lim=np.abs(np.vstack(cc)).max()*1.06
  pos=gs[0,d].get_position(fig);mid=(pos.x0+pos.x1)/2
  fig.text(mid,.877,DOMAINS[d],ha='center',fontsize=11,weight='bold')
  fig.text(mid,.846,f'{["BCIC-IV-2a","NSD","ds000210"][d]} · {s} · n={len(y)}',ha='center',fontsize=7)
  for m,q in enumerate(cc):
   ax=fig.add_subplot(gs[m,d],projection='3d')
   for c in np.unique(y):
    cloud=q[y==c];ax.scatter(*cloud.T,s=8,color=h.COLORS[c],alpha=.6,depthshade=False,edgecolors='none',rasterized=True)
    ax.scatter(*cloud.mean(0),s=32,marker='D',color=h.COLORS[c],edgecolors='white',linewidths=.5,depthshade=False)
   ax.view_init(24,-58);ax.set_proj_type('ortho');ax.set_box_aspect((1,1,.9))
   ax.set(xlim=(-lim,lim),ylim=(-lim,lim),zlim=(-lim,lim))
   for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
    axis.set_ticks([-lim,0,lim]);axis.set_ticklabels([]);axis.set_pane_color((1,1,1,0));axis.line.set_color('#AAB5BD')
    axis._axinfo['grid'].update(color='#E4E9ED',linewidth=.5)
   prefix='PC' if version=='A' else 'MDS'
   ax.set_xlabel(prefix+'1',labelpad=-10,fontsize=6);ax.set_ylabel(prefix+'2',labelpad=-10,fontsize=6);ax.set_zlabel(prefix+'3',labelpad=-10,fontsize=6)
   ax.text2D(0,1,'abcdef'[3*m+d],transform=ax.transAxes,fontsize=11,weight='bold')
  handles=[h.Line2D([],[],marker='o',ls='',color=h.COLORS[c],label=l,markersize=4) for c,l in enumerate(LABELS[d])]
  fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(mid,.172),ncol=2,fontsize=6.5,frameon=False)
 fig.text(.018,.67,'Observation*',rotation=90,va='center',fontsize=10,color='#657F90')
 fig.text(.018,.35,'MN-SR*',rotation=90,va='center',fontsize=10,color='#B5756B')
 for j,(metric,title) in enumerate([('within','Normalized within-state dispersion'),('between','Normalized centroid separation'),('quality','Between / within')]):
  ax=fig.add_subplot(rg[j])
  for d in range(3):
   vals=table[table.domain==DOMAINS[d]].pivot(index='subject',columns='method',values=metric)
   for _,row in vals.iterrows():ax.plot([d-.16,d+.16],row,color='#AEB7BD',lw=.5,alpha=.5)
   for m in [0,1]:
    v=vals[m];p=d+[-.16,.16][m]
    box=ax.boxplot(v,positions=[p],widths=.22,patch_artist=True,showfliers=False,medianprops={'color':'#39464F'})
    box['boxes'][0].set_facecolor(h.METHOD_COLORS[m]);box['boxes'][0].set_alpha(.7)
    ax.scatter(np.full(len(v),p),v,s=5,color=h.METHOD_COLORS[m],zorder=3)
  ax.set_xticks([0,1,2]);ax.set_xticklabels(['Motor\nn=9','NSD\nn=8','Internal\nn=15'],fontsize=6)
  ax.tick_params(axis='y',labelsize=6);ax.set_title(title,fontsize=7.1,loc='left')
  ax.text(-.18,1.12,'ghi'[j],transform=ax.transAxes,fontsize=11,weight='bold')
 fig.legend(handles=[h.Line2D([],[],marker='s',ls='',color=c,label=l,markersize=4) for c,l in zip(h.METHOD_COLORS,['Observation*','MN-SR*'])],loc='upper center',bbox_to_anchor=(.90,.17),frameon=False,fontsize=7)
 notes=notes or ['*NSD: fused-backbone features → retrieval-head embeddings; these caches do not isolate Observation → MN-SR.',
                 'Exploratory, outcome-selected subjects. NSD colors: dominant COCO instance supercategory in the original image; crop not corrected.']
 fig.text(.5,.059,notes[0],ha='center',fontsize=7,color='#795B4D')
 fig.text(.5,.029,notes[1],ha='center',fontsize=6.7,color='#697780')
 fig.savefig(HERE/f'Fig2_{version}_REAL_NSD.png',dpi=360)
 fig.savefig(HERE/f'Fig2_{version}_REAL_NSD.pdf')
 fig.savefig(HERE/f'Fig2_{version}_REAL_NSD.svg')
 plt.close(fig)

if __name__=='__main__':main()
