
from pathlib import Path
import argparse
import logging
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(out):
    out=Path(out)
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.labelsize':6,'xtick.labelsize':6,'ytick.labelsize':6,
        'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,
        'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(7.2047,4.7),gridspec_kw={'width_ratios':[1,1,.90]})
    fig.subplots_adjust(left=.075,right=.985,bottom=.16,top=.91,wspace=.60,hspace=.70)
    methods=['neurostorm','ours'];labels=['Observation MLP*','Ours']
    colors=['#42BFA5','#F07D7D']
    stats=pd.read_csv(out/'statistics.csv')
    for row,task,pk,pl in [(0,'narratives','A','B'),(1,'shortvideo','C','D')]:
        data=pd.read_csv(out/f'{pk}_plot_data.csv')
        for j,(method,label,color) in enumerate(zip(methods,labels,colors)):
            ax=axes[row,j]
            x=data[method].to_numpy();y=data.subjective_rating_distance.to_numpy()
            assert len(x)==136 and np.isfinite(x).all() and np.isfinite(y).all()
            ax.scatter(x,y,s=6,facecolors='none',edgecolors=color,linewidths=.55)
            line=np.array([.15,.85]);slope,intercept=np.polyfit(x,y,1)
            ax.plot(line,slope*line+intercept,color=color,lw=.9)
            ax.plot(line,line,color='#D8D8D8',ls=(0,(3,3)),lw=.7)
            ax.text(.06,.92,f'Pearson r = {pearsonr(x,y).statistic:.3f}',transform=ax.transAxes,color='#FF2D2D',fontsize=6.5)
            ax.set(xlim=(.15,.85),ylim=(.15,.85),xticks=[.2,.4,.6,.8],yticks=[.2,.4,.6,.8],
                   xlabel='Neural representational distance',ylabel='Subjective-rating distance',title=label)
        violin=pd.read_csv(out/f'{pl}_plot_data.csv')
        ax=axes[row,2]
        arrays=[violin[m].to_numpy() for m in methods]
        assert len(violin)==17
        v=ax.violinplot(arrays,positions=[1,2],widths=.62,showextrema=False,points=300,bw_method=.25)
        rng=np.random.default_rng(20260729)
        for i,(body,values,color) in enumerate(zip(v['bodies'],arrays,colors),1):
            body.set(facecolor=color,edgecolor='#222222',linewidth=.55,alpha=.85)
            ax.scatter(i+rng.uniform(-.055,.055,len(values)),values,s=3,color='#222222',zorder=3)
            for q in np.quantile(values,[.25,.5,.75]):
                ax.plot([i-.25,i+.25],[q,q],color='#222222',lw=.6,ls=(0,(2,2)))
        test=stats[(stats.panel==pl)&(stats.method=='paired')].iloc[0]
        ax.plot([1,2],[.565,.565],color='#222222',lw=.6)
        ax.text(1.5,.57,test.stars,ha='center',fontsize=9,fontweight='bold')
        ax.set(xlim=(.4,2.6),ylim=(0 if row==0 else -.2,.6),ylabel='RSA correlation',title=f'Spacetop {task}')
        ax.set_xticks([1,2]);ax.set_xticklabels(['Obs. MLP*','Ours'],fontsize=6)
        ax.set_yticks(np.arange(0 if row==0 else -.2,.61,.2))
        for axis,letter in [(axes[row,0],pk),(axes[row,2],pl)]:
            axis.text(-.38,1.13,letter,transform=axis.transAxes,fontsize=13,fontweight='bold')
    fig.text(.5,.067,'* Source labeled this branch NeuroSTORM; the loaded weights produce an observation-expert MLP.',ha='center',fontsize=5.5)
    fig.text(.5,.035,'Historical in-cohort, outcome-selected examples. A/C and B/D use different 17-participant subsets.',ha='center',fontsize=5.5)
    fig.savefig(out/'S4_fresh.png',dpi=600,bbox_inches='tight')
    fig.savefig(out/'S4_fresh.pdf',bbox_inches='tight')
    fig.savefig(out/'S4_fresh.svg',bbox_inches='tight')
    plt.close(fig)
    print(f'PLOT={out/"S4_fresh.png"}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    render(p.parse_args().output)
