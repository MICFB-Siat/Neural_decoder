
from pathlib import Path
import argparse
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numerics import DOMAINS,PATHS,KEYS

CLASS_COLORS=['#604584','#35AC8A']
METHOD_COLORS=['#EFF3BE','#66BFA0']


def envelope(q):
    center=q.mean(0)
    cov=np.cov(q,rowvar=False,ddof=1)
    vals,vec=np.linalg.eigh(cov)
    assert vals.min()>0
    transform=vec@np.diag(1.5*np.sqrt(vals))
    u,v=np.linspace(0,2*np.pi,49),np.linspace(0,np.pi,25)
    unit=np.stack([np.outer(np.cos(u),np.sin(v)),np.outer(np.sin(u),np.sin(v)),
                   np.outer(np.ones_like(u),np.cos(v))],axis=-1)
    e,a=np.deg2rad([45,125])
    view=np.array([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)])
    _,_,vh=np.linalg.svd(np.linalg.solve(transform,view)[None,:],full_matrices=True)
    t=np.linspace(0,2*np.pi,150)
    circle=np.cos(t)[:,None]*vh[1]+np.sin(t)[:,None]*vh[2]
    extent=1.5*np.sqrt(np.diag(cov))
    return unit@transform.T+center,circle@transform.T+center,center-extent,center+extent


def render(out):
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    out=Path(out)
    boots=pd.read_csv(out/'B_plot_data.csv')
    tests=pd.read_csv(out/'fresh_statistics.csv')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':6,'axes.labelsize':5.5,'axes.titlesize':7,
        'xtick.labelsize':5.5,'ytick.labelsize':5.5,'axes.linewidth':.6,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(7.2047,3.95))
    grid=fig.add_gridspec(2,3,left=.025,right=.665,top=.88,bottom=.18,wspace=.04,hspace=.11)
    layout=json.loads((out/'display_layout.json').read_text()) if (out/'display_layout.json').exists() else {}
    audit=[]
    for j,domain in enumerate(DOMAINS):
        with np.load(out/(domain.replace(' ','_')+'_fresh.npz')) as z:
            y=z['labels'];qs=[z['low_display'],z['high_display']];names=z['names']
        envs=[[envelope(q[y==c]) for c in [0,1]] for q in qs]
        edges=list(qs)
        for env in envs:
            for _,_,lo,hi in env:
                edges.extend([lo[None],hi[None]])
        edges=np.vstack(edges)
        center=(edges.min(0)+edges.max(0))/2
        half=np.ptp(edges,axis=0).max()*.55
        bounds=np.array(layout.get(domain,np.c_[center-half,center+half]))

        bounds[:,0]=np.minimum(bounds[:,0],edges.min(0))
        bounds[:,1]=np.maximum(bounds[:,1],edges.max(0))
        for r,q in enumerate(qs):
            ax=fig.add_subplot(grid[r,j],projection='3d',computed_zorder=False)
            for c,color in enumerate(CLASS_COLORS):
                surf,outline,_,_=envs[r][c]
                ax.plot_surface(*[surf[:,:,k] for k in range(3)],color=color,alpha=.09,
                                linewidth=0,shade=False,rcount=49,ccount=25,zorder=1)
                ax.plot(*outline.T,color=color,alpha=.35,lw=.4,zorder=2)
                ax.scatter(*q[y==c].T,c=color,s=3.4,alpha=.9,depthshade=False,edgecolors='none',zorder=5)
            ax.view_init(elev=45,azim=125)
            ax.set_proj_type('ortho')
            ax.set_box_aspect((1,1,1))
            for k,axis in enumerate([ax.xaxis,ax.yaxis,ax.zaxis]):
                [ax.set_xlim,ax.set_ylim,ax.set_zlim][k](*bounds[k])
                axis.set_ticks(np.linspace(*bounds[k],3));axis.set_ticklabels([])
                axis.set_pane_color((1,1,1,0));axis.line.set_color('#A3AFB9')
                axis._axinfo['grid'].update(color='#D7E0EC',linewidth=.45)
            ax.set_xlabel('MDS1',labelpad=-13,fontsize=5)
            ax.set_ylabel('MDS2',labelpad=-13,fontsize=5)
            ax.text2D(.99,.59,'MDS3',ha='right',transform=ax.transAxes,fontsize=5)
            if r==0:
                ax.set_title(domain,pad=9,fontsize=7)
            else:
                ax.legend(handles=[Line2D([],[],marker='o',ls='',color=c,markersize=2.5,label=n)
                           for c,n in zip(CLASS_COLORS,names)],loc='upper center',
                          bbox_to_anchor=(.5,-.045),ncol=2,fontsize=5.5,frameon=False,
                          handletextpad=.2,columnspacing=.6)
        audit.append(dict(domain=domain,n=len(y),bounds=bounds.tolist(),names=names.tolist()))
    titles=['Within-class dispersion (↓)','Between-class separation (↑)','Category-structure RSA (↑)']
    labels=['Normalized\nRMS radius','Normalized\ncentroid distance','RDM correlation']
    ylim=[(.4,1.12),(0,2.0),(-.2,1.1)]
    annotation_y=[1.065,1.96,1.04]
    for k,key in enumerate(KEYS):
        ax=fig.add_axes([.745,.72-k*.26,.245,.155])
        for j,domain in enumerate(DOMAINS):
            pair=boots[boots.domain==domain].pivot(index='bootstrap_id',columns='pathway',values=key)
            assert pair.shape==(50,2)
            for r,path in enumerate(PATHS):
                v=pair[path].to_numpy()
                x=j+[-.18,.18][r]
                body=ax.violinplot([v],positions=[x],widths=.30,points=160,
                                   bw_method='scott',showextrema=False)['bodies'][0]
                body.set(facecolor=METHOD_COLORS[r],edgecolor='black',alpha=1,linewidth=.5)
                for value in np.quantile(v,[.25,.5,.75]):
                    ax.hlines(value,x-.09,x+.09,color='black',ls=':',lw=.5)
            a=tests[(tests.domain==domain)&(tests.metric==key)].iloc[0]
            ax.plot([j-.18,j+.18],[annotation_y[k]]*2,color='black',lw=.5)
            ax.text(j,annotation_y[k],a.stars_Holm9_displayed,ha='center',va='bottom',fontsize=7,fontweight='bold')
        ax.set_ylim(*ylim[k]);ax.set_xlim(-.5,2.5)
        ax.set_xticks([0,1,2]);ax.set_xticklabels(['Motor','Perception','Internal mentation'],fontsize=5)
        ax.set_ylabel(labels[k],fontsize=5.5,labelpad=2)
        ax.set_title(titles[k],fontsize=6.5,pad=6)
        ax.tick_params(length=2,width=.6,pad=2)
    fig.text(.008,.93,'A',fontsize=13,fontweight='bold')
    fig.text(.69,.93,'B',fontsize=13,fontweight='bold')
    fig.text(.014,.70,'Low-level',rotation=90,va='center',fontsize=7)
    fig.text(.014,.37,'High-level',rotation=90,va='center',fontsize=7)
    fig.legend(handles=[Patch(facecolor=c,edgecolor='black',label=n) for c,n in zip(METHOD_COLORS,['Low-level','High-level'])],
               loc='lower center',bbox_to_anchor=(.862,.095),ncol=2,fontsize=5.5,frameon=False)
    fig.text(.5,.06,'One selected participant per domain; violins: 50 paired trial/image resamples.',ha='center',fontsize=6)
    fig.text(.5,.028,'Internal: Past / Other (theory of mind). Stars: conditional bootstrap + Holm9, not population inference.',ha='center',fontsize=5.5)
    fig.savefig(out/'S3_fresh.png',dpi=600,bbox_inches='tight')
    fig.savefig(out/'S3_fresh.pdf',bbox_inches='tight')
    fig.savefig(out/'S3_fresh.svg',bbox_inches='tight')
    plt.close(fig)
    (out/'figure_qa.json').write_text(json.dumps(dict(panels=audit,all_selected_points_shown=True,
        no_artificial_values=True,covariance='descriptive 1.5-SD, not confidence intervals',
        methods='Class-balanced normalized W/B; Spearman category RSA in fixed 3D',
        legend_correction='Past / Other, not screenshot Past / Future',
        population_claim=False),indent=2))
    print(f'PLOT={out/"S3_fresh.png"}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    render(p.parse_args().output)
