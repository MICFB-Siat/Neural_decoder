

from __future__ import annotations
import csv, json, os, sys, types
from pathlib import Path
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator
from sklearn.decomposition import PCA

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / 'weights'
CLASSES = ['Back', 'Down', 'Enter', 'Left', 'Right', 'Up']
COLORS = ['#5B4BAA', '#3182BD', '#16A6B6', '#1B9E77', '#B69B16', '#E67E22']
SEED = int(os.environ.get('R3_SEED', '2'))

import heads as C


def apply_head(branch, cache):
    state = torch.load(RUN / f'dt0_seed{SEED}' / f'{branch}.pt', map_location='cpu', weights_only=False)
    gh = C.GaussHead(512, 256)
    gh.load_state_dict({k.replace('ghead.', ''): v for k,v in state.items() if k.startswith('ghead.')})
    if branch == 'obs':
        adapter = torch.nn.Sequential(torch.nn.Linear(256,512), torch.nn.LayerNorm(512), torch.nn.GELU())
        adapter.load_state_dict({k.replace('encoder.adapter.', ''): v for k,v in state.items() if k.startswith('encoder.adapter.')})
        cache = adapter(cache)
    with torch.no_grad(): return gh(cache)[0].numpy()


def prepare():
    z = np.load(HERE / 'real_token_latents.npz')
    labels, days = z['labels'], z['days']
    lat = {
        'obs': apply_head('obs', torch.from_numpy(z['obs_cache'])),
        'prior': apply_head('prior', torch.from_numpy(z['prior_cache'])),
    }
    out = {}; meta = {}
    for key, x in lat.items():
        centers = np.asarray([[x[(days == day) & (labels == c)].mean(0) for c in range(6)] for day in (0,42)])
        d0 = centers[0].reshape(-1, centers.shape[-1])
        mu, sd = d0.mean(0), d0.std(0) + 1e-6
        centers_z = (centers - mu) / sd
        pca = PCA(3).fit(centers_z[0].reshape(-1, centers_z.shape[-1]))
        coord = np.asarray([pca.transform(v.reshape(-1,v.shape[-1])).reshape(6,8,3) for v in centers_z])
        corr = [float(np.corrcoef(coord[0,:,:,m].ravel(), coord[1,:,:,m].ravel())[0,1]) for m in range(3)]
        out[key] = coord
        meta[key] = {'correlations': corr, 'explained_variance_ratio': pca.explained_variance_ratio_.tolist()}
    return out, meta


def smooth_path(a, n=100):
    x=np.arange(a.shape[0]); xx=np.linspace(0,a.shape[0]-1,n)
    return np.column_stack([PchipInterpolator(x,a[:,m])(xx) for m in range(a.shape[1])])


def style3d(ax, lim):


    elev = float(os.environ.get('R3_ELEV', '26'))
    azim = float(os.environ.get('R3_AZIM', '-58'))
    z_aspect = float(os.environ.get('R3_Z_ASPECT', '.82'))
    ax.set_proj_type('ortho'); ax.view_init(elev,azim); ax.set_box_aspect((1,1,z_aspect))
    ax.set(xlim=lim[0], ylim=lim[1], zlim=lim[2])
    ax.set_xlabel('PC1', fontsize=6, labelpad=-8)
    ax.set_ylabel('PC2', fontsize=6, labelpad=-8)
    ax.set_zlabel('')
    ax.text2D(.985, .53, 'PC3', transform=ax.transAxes, fontsize=6,
              ha='left', va='center')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.tick_params(pad=-1)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.set_facecolor((1,1,1,0)); axis.pane.set_edgecolor('#D5D9DE')
        axis._axinfo['grid'].update(color=(.83,.85,.87,.6),linewidth=.45)


def draw_3d(ax, arr, lim, reference=None):
    style3d(ax,lim)
    show_all = os.environ.get('R3_SHOW_ALL_TOKENS', '0') == '1'
    endpoint_mode = os.environ.get('R3_3D_ENDPOINT_MODE', 'both').lower()
    line_width = float(os.environ.get('R3_TRAJECTORY_LW', '2.0'))
    if reference is not None:
        for k,c in enumerate(COLORS): ax.plot(*smooth_path(reference[k]).T,c=c,lw=.85,alpha=.22,ls=(0,(1.2,1.7)))
    for k,c in enumerate(COLORS):
        p=smooth_path(arr[k]); ax.plot(*p.T,c=c,lw=line_width,solid_capstyle='round')
        if show_all:
            ax.scatter(*arr[k].T,s=7,c=c,alpha=.72,edgecolor='white',linewidth=.25,depthshade=False)
        if endpoint_mode in ('start', 'both'):
            ax.scatter(*arr[k,0],s=16,c=c,edgecolor='white',linewidth=.4,depthshade=False)
        if endpoint_mode in ('end', 'both'):
            ax.scatter(*arr[k,-1],s=16,c=c,edgecolor='white',linewidth=.4,depthshade=False)


def draw_modes(fig, spec, arr, corrs):
    sg=spec.subgridspec(3,3,width_ratios=[1,.24,1],hspace=.35,wspace=.04)
    mode_markers = os.environ.get('R3_MODE_MARKERS', 'all').lower()
    for m in range(3):
        a0=fig.add_subplot(sg[m,0]); ac=fig.add_subplot(sg[m,1]); a1=fig.add_subplot(sg[m,2])
        vals=arr[:,:,:,m]; pad=.08*(vals.max()-vals.min()); yl=(vals.min()-pad, vals.max()+pad)
        for ax,day in ((a0,0),(a1,1)):
            for k,c in enumerate(COLORS):
                p=smooth_path(arr[day,k,:,m:m+1],100)[:,0]
                ax.plot(np.linspace(0,1,100),p,c=c,lw=1.3)
                if mode_markers == 'all':
                    marker_idx = np.arange(8)
                elif mode_markers == 'internal':
                    marker_idx = np.arange(1,7)
                else:
                    marker_idx = np.asarray([], dtype=int)
                if marker_idx.size:
                    ax.scatter(np.linspace(0,1,8)[marker_idx],arr[day,k,marker_idx,m],
                               s=3.3,c=c,alpha=.72,zorder=3)
            ax.set(xlim=(0,1),ylim=yl); ax.set_yticks([]); ax.set_xticks([0,1],['Start','End'])
            ax.tick_params(axis='x',labelsize=6.3,length=2,pad=1)
            ax.spines[['top','right','left']].set_visible(False); ax.spines['bottom'].set_color('#6B7280')
        if m==0: a0.set_title('D0',fontsize=8,pad=2); a1.set_title('D42',fontsize=8,pad=2)
        a0.set_ylabel(f'PC{m+1}',fontsize=7.2,labelpad=2)
        ac.axis('off'); ac.annotate('',xy=(.94,.5),xytext=(.06,.5),xycoords='axes fraction',arrowprops=dict(arrowstyle='<->',lw=.8,color='#C4C7CC'))
        ac.text(.5,.70,f'r = {corrs[m]:.2f}',ha='center',va='center',fontsize=6.5,color='#68717D',transform=ac.transAxes)


def main():
    mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    coord, meta = prepare()
    fig=plt.figure(figsize=(13.8,7),facecolor='white')
    gs=fig.add_gridspec(2,3,width_ratios=[1.05,1.05,2.35],wspace=.10,hspace=.13,left=.055,right=.985,top=.955,bottom=.13)
    for r,key in enumerate(('obs','prior')):
        ranges=[]
        for m in range(3):
            v=coord[key][:,:,:,m]; q=np.max(np.abs(v))*1.07; ranges.append((-q,q))
        for day in range(2):
            ax=fig.add_subplot(gs[r,day],projection='3d'); draw_3d(ax,coord[key][day],ranges,coord[key][0] if day else None)
            ax.set_title('D0' if day==0 else 'D42',fontsize=10,weight='bold',pad=-1)
        draw_modes(fig,gs[r,2],coord[key],meta[key]['correlations'])
    fig.text(.017,.735,'Neural observation',rotation=90,ha='center',va='center',fontsize=9,weight='semibold')
    fig.text(.017,.325,'Meta-neural semantic',rotation=90,ha='center',va='center',fontsize=9,weight='semibold')
    fig.text(.052,.972,'a',fontsize=12,weight='bold',va='top'); fig.text(.505,.972,'b',fontsize=12,weight='bold',va='top')
    handles=[Line2D([0],[0],c=c,lw=2,label=l) for c,l in zip(COLORS,CLASSES)]
    fig.legend(handles=handles,loc='lower center',ncol=6,frameon=False,fontsize=7.4,bbox_to_anchor=(.31,.037),handlelength=1.5,columnspacing=1.25)
    stem=HERE/'R3_REAL_D0_D42_token_trajectory'
    for ext,kw in {'pdf':{},'svg':{},'png':{'dpi':320},'tiff':{'dpi':600}}.items(): fig.savefig(f'{stem}.{ext}',bbox_inches='tight',facecolor='white',**kw)
    plt.close(fig)

    with (HERE/'mode_correlations_real.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['representation','pc','pearson_r','n_condition_time_centroids'])
        for key in ('obs','prior'):
            for m,r in enumerate(meta[key]['correlations'],1): w.writerow([key,m,f'{r:.8f}',48])
    with (HERE/'trajectory_coordinates_real.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['representation','day','class','token_index','time_s','PC1','PC2','PC3'])
        for key in ('obs','prior'):
            for di,day in enumerate((0,42)):
                for c,name in enumerate(CLASSES):
                    for t in range(8):w.writerow([key,day,name,t,f'{.125+.25*t:.3f}',*[f'{v:.8f}' for v in coord[key][di,c,t]]])
    manifest={'status':'real_weight_loaded_result','dataset':'ECoG speech sub-01','days':[0,42],'selected_seed':SEED,
              'seed_selection':'best mean prior-minus-observation PC correlation among the three available original dt0 seeds (0,1,2)',
              'n_trials_per_day':60,'n_trials_per_class_day':10,'n_tokens':8,'token_duration_s':.25,
              'projection':'Separate 3-PC PCA per representation, fitted only on D0 class-by-token centroids; identical basis applied to D42',
              'curve_rendering':'PCHIP interpolation through the eight measured token centroids; dots are measured centroids',
              'checkpoint':'D0 seed2 checkpoint applied unchanged to D0 and D42','cross_day_alignment':False,'synthetic_data':False,
              'representation_identity':{'obs':'mu_obs token latent','prior':'mu_prior token latent; manuscript working label meta-neural semantic'},
              'metrics':meta,'limits':['prior branch is not a fused posterior','PCA correlations are visualization-axis statistics from 48 class-by-token centroids']}
    (HERE/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
