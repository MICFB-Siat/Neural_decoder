import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.runtime import INPUTS,TEMPLATES,output,style,save_figure
from common.activity_flow import semantic_maps,local_mask,actual_and_predicted,pearson
from common import anatomy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    out=output('G');atlas=np.load(TEMPLATES/'activity_flow_atlas.npz')
    matrix,cent,nets=atlas['matrix'],atlas['centroids'],atlas['networks']
    local=local_mask(cent,40);bge=np.load(INPUTS/'LPPCHK/semantic.npz')['pooled']
    squared=[]
    for hemi in ['lh','rh']:
        b=np.asarray(anatomy.basis(hemi))[:,:1000][anatomy.cortex_mask(hemi)].astype(np.float32)**2
        b/=np.maximum(b.sum(1,keepdims=True),1e-12);squared.append(b)
    order=['all','SalVentAttn','Default','Cont','DorsAttn','SomMot','Limbic','Vis'];rows=[];audit=[]
    for i in range(1,13):
        sid=f'sub-HK{i:03d}';print('G',sid,flush=True);z=np.load(INPUTS/'LPPCHK'/f'{sid}.npz');count=len(z['cift_mean'])

        cogreader_map,prior_map=semantic_maps(z['cift_mean'],z['modes_mean'],bge[:count],min(40,max(6,count//9)),*squared)
        ts=z['parcel_time_series'];ts=(ts-ts.mean(1,keepdims=True))/(ts.std(1,keepdims=True)+1e-8)
        fc=(ts@ts.T)/ts.shape[1];fc*=local;np.fill_diagonal(fc,0)
        for level,value in [('low',cogreader_map),('high',prior_map)]:
            actual,pred=actual_and_predicted(value[:29696],fc,matrix)
            for region in order:
                mask=np.ones(len(nets),bool) if region=='all' else nets==region
                rows.append(dict(subject=sid,level=level,region=region,r=pearson(pred[mask],actual[mask])))
        audit.append(dict(subject=sid,n_trials=count,shared_label_match_fraction=float(z['label_exact_match'].mean())))
    frame=pd.DataFrame(rows);frame.to_csv(out/'participant_metrics.csv',index=False);pd.DataFrame(audit).to_csv(out/'label_alignment.csv',index=False)
    summary=frame.groupby(['region','level']).r.agg(mean='mean',sem=lambda x:np.std(x,ddof=0)/np.sqrt(len(x))).reset_index()
    summary.to_csv(out/'plot_points.csv',index=False);style();fig,ax=plt.subplots(figsize=(6,3.4))
    for level,color,offset,label in [('high','#ce5145',-.10,'High-level'),('low','#278672',.10,'Observation')]:
        t=summary[summary.level==level].set_index('region').loc[order]
        ax.errorbar(np.arange(8)+offset,t['mean'],yerr=t['sem'],fmt='o',color=color,ms=3,capsize=2,label=label,lw=.7)
        ax.annotate(f"{t.loc['all','mean']:.2f}",(offset,t.loc['all','mean']),xytext=(0,8 if level=='high' else -13),textcoords='offset points',ha='center',color=color)
    ax.axhline(0,color='gray',ls='--',lw=.6);ax.set_xticks(np.arange(8),['all','VentAttn','Default','Control','DorsAttn','SomMot','Limbic','Visual'],rotation=40)
    ax.set_ylabel('Global coefficient');ax.legend(frameon=False,ncol=2);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();save_figure(fig,out/'panel_G');print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
