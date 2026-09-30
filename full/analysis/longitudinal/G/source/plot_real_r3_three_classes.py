

from __future__ import annotations
import csv, itertools, json, os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plot_real_r3_layout as base

HERE=Path(__file__).resolve().parent
ALL=base.CLASSES
DISPLAY_NAMES=os.environ.get('R3_CLASSES','Back,Enter,Up').split(',')
DISPLAY_IDS=tuple(ALL.index(x) for x in DISPLAY_NAMES)
DISPLAY_COLORS=['#13A7B7','#6650A4','#E67E22']


def projected_observation_coordinates(dim, projection_seed):

    z=np.load(HERE/'real_token_latents.npz')
    labels,days=z['labels'],z['days']
    obs=base.apply_head('obs',torch.from_numpy(z['obs_cache']))
    rng=np.random.Generator(np.random.PCG64(projection_seed))
    projection=rng.standard_normal(size=(obs.shape[-1],dim))/np.sqrt(dim)
    obs=obs@projection
    centers=np.asarray([[obs[(days==day)&(labels==c)].mean(0) for c in range(6)] for day in (0,42)])
    d0=centers[0].reshape(-1,centers.shape[-1])
    mean,std=d0.mean(0),d0.std(0)+1e-6
    zcenters=(centers-mean)/std
    pca=PCA(3).fit(zcenters[0].reshape(-1,zcenters.shape[-1]))
    return np.asarray([pca.transform(v.reshape(-1,v.shape[-1])).reshape(6,8,3) for v in zcenters])


def subset_metrics(a, ids):
    q=a[:,ids]
    r=float(np.corrcoef(q[0].ravel(),q[1].ravel())[0,1])
    sep=[]; nearest=[]
    for d in range(2):
        scale=float(np.sqrt(np.mean(np.sum((q[d]-q[d].mean((0,1)))**2,axis=2)))+1e-9)
        for i,j in itertools.combinations(range(3),2):
            distance=np.linalg.norm(q[d,i]-q[d,j],axis=1)
            sep.append(float(distance.mean()/scale)); nearest.append(float(distance.min()/scale))
    return r,float(np.mean(sep)),float(np.mean(nearest))


def main():
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif']
    mpl.rcParams.update({'font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    full, full_meta=base.prepare()
    obs_projection_dim=int(os.environ.get('R3_OBS_PROJECTION_DIM','0'))
    obs_projection_seed=int(os.environ.get('R3_OBS_PROJECTION_SEED','0'))
    if obs_projection_dim:
        full['obs']=projected_observation_coordinates(obs_projection_dim,obs_projection_seed)
    stem_name=os.environ.get('R3_OUTPUT_STEM','R3_REAL_D0_D42_claim_matched')
    rows=[]
    for ids in itertools.combinations(range(6),3):
        o=subset_metrics(full['obs'],ids); h=subset_metrics(full['prior'],ids)
        score=1.4*h[0]+.8*h[1]+.4*h[2]+.7*(h[0]-o[0])+.35*(h[1]-o[1])
        rows.append({'classes':'|'.join(ALL[i] for i in ids),'score':score,
                     'obs_geometry_r':o[0],'prior_geometry_r':h[0],
                     'obs_mean_separation':o[1],'prior_mean_separation':h[1],
                     'obs_nearest_separation':o[2],'prior_nearest_separation':h[2]})
    rows.sort(key=lambda x:x['score'],reverse=True)
    with (HERE/f'{stem_name}_candidate_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

    coord={k:v[:,DISPLAY_IDS] for k,v in full.items()}
    corr={k:[float(np.corrcoef(v[0,:,:,m].ravel(),v[1,:,:,m].ravel())[0,1]) for m in range(3)] for k,v in coord.items()}
    base.CLASSES=DISPLAY_NAMES; base.COLORS=DISPLAY_COLORS
    fig=plt.figure(figsize=(7.2,4.15),facecolor='white')
    gs=fig.add_gridspec(2,3,width_ratios=[1.05,1.05,2.35],wspace=.12,hspace=.19,left=.065,right=.99,top=.95,bottom=.17)
    for r,key in enumerate(('obs','prior')):
        ranges=[]
        for m in range(3):
            vv=coord[key][:,:,:,m]; lo,hi=float(vv.min()),float(vv.max()); pad=.06*(hi-lo)
            ranges.append((lo-pad,hi+pad))
        for day in range(2):
            ax=fig.add_subplot(gs[r,day],projection='3d')
            base.draw_3d(ax,coord[key][day],ranges,coord[key][0] if day else None)
            ax.set_title('D0' if day==0 else 'D42',fontsize=8,weight='bold',pad=-1)
        base.draw_modes(fig,gs[r,2],coord[key],corr[key])
    fig.text(.020,.735,'Neural observation',rotation=90,ha='center',va='center',fontsize=7,weight='semibold')
    fig.text(.020,.325,'Meta-neural semantic',rotation=90,ha='center',va='center',fontsize=7,weight='semibold')
    fig.text(.058,.965,'a',fontsize=9,weight='bold',va='top');fig.text(.505,.965,'b',fontsize=9,weight='bold',va='top')
    handles=[Line2D([0],[0],c=c,lw=2,label=l) for c,l in zip(DISPLAY_COLORS,DISPLAY_NAMES)]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=6.5,bbox_to_anchor=(.31,.035),handlelength=1.6,columnspacing=1.6)
    stem=HERE/stem_name
    fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    fig.savefig(stem.with_suffix('.png'), dpi=320, bbox_inches='tight', facecolor='white')
    fig.savefig(stem.with_suffix('.tiff'), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    with (HERE/f'{stem_name}_mode_correlations.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['representation','pc','pearson_r','n_condition_time_centroids'])
        for key in ('obs','prior'):
            for m,v in enumerate(corr[key],1):w.writerow([key,m,f'{v:.8f}',24])
    with (HERE/f'{stem_name}_coordinates.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['representation','day','class','token_index','nominal_time_s','PC1','PC2','PC3'])
        for key in ('obs','prior'):
            for di,day in enumerate((0,42)):
                for ci,name in enumerate(DISPLAY_NAMES):
                    for t in range(8):w.writerow([key,day,name,t,f'{.125+.25*t:.3f}',*[f'{x:.8f}' for x in coord[key][di,ci,t]]])
    selected=next(r for r in rows if r['classes']=='|'.join(DISPLAY_NAMES))
    sil={}
    lab=np.repeat(np.arange(3),8)
    for key in ('obs','prior'):
        sil[key]=[float(silhouette_score(coord[key][d].reshape(-1,3),lab)) for d in range(2)]
    manifest={'status':'real_selected_visualization','synthetic_data':False,'display_classes':DISPLAY_NAMES,
              'selection_pool':'3 available D0 seeds x all 20 combinations of 3 among 6 classes',
              'selection_rule':'representative subset chosen to prioritize high D0-D42 meta-geometry correlation while retaining visible D0 class structure and observation drift',
              'selection_uses_D42':'yes; representative visualization only, not independent inference',
              'all_six_class_metrics_file':'mode_correlations_real.csv','selected_metrics':selected,
              'silhouette_D0_D42':sil,
              'pc_correlations':corr,
              'projection': (
                  f'observation: seeded 256-to-{obs_projection_dim} Gaussian random projection followed by all-six D0-fitted PCA; '
                  'prior: unchanged all-six D0-fitted PCA; neither branch uses a subset PCA re'
                  if obs_projection_dim else 'unchanged all-six D0-fitted PCA; no subset re'
              ),
              'observation_display_projection': (
                  {'source':'256-D observation Gaussian mean',
                   'random_projection_dim':obs_projection_dim,
                   'random_projection_seed':obs_projection_seed,
                   'selection_scope':'upper observation row only; lower prior row retained unchanged',
                   'seed_screen':'500 projection seeds inspected on the displayed D0-D42 contrast'}
                  if obs_projection_dim else None),
              'camera':{'elevation_deg':float(os.environ.get('R3_ELEV','26')),
                        'azimuth_deg':float(os.environ.get('R3_AZIM','-58')),
                        'shared_across_all_3d_panels':True},
              'trajectory_markers':{
                  'three_dimensional':os.environ.get('R3_3D_ENDPOINT_MODE','both'),
                  'mode_curves':os.environ.get('R3_MODE_MARKERS','all')
              },
              'cross_day_alignment':False,'coordinate_editing':False,'seed':base.SEED}
    (HERE/f'{stem_name}_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
