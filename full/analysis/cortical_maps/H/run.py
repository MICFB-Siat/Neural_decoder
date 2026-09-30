import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.runtime import ROOT,output,style,save_figure
import runpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    source=ROOT/'E/outputs/maps.npz'
    if not source.exists():raise FileNotFoundError('Missing supplied source data: E/outputs/maps.npz.')
    out=output('H');ns=runpy.run_path(str(ROOT/'common/chord.py'),init_globals={'CACHE':str(source)})
    ns['chord'].__globals__['NODE_S']=2.
    pd.DataFrame({'node':np.arange(ns['P'])+1,'low':ns['io'],'high':ns['ip'],'x':ns['nx'],'y':ns['ny']}).to_csv(out/'nodes.csv',index=False)
    style();fig,axes=plt.subplots(1,2,figsize=(7.4,3.4));rows=[]
    for ax,key,k,title in zip(axes,['io','ip'],[90,180],['Observation (low-level)','Eigenmode prior']):
        ns['chord'](ax,ns[key],k,None,None);ax.set_title(title,fontsize=9)
        for text in ax.texts:
            x,y=text.get_position();text.set_position((x*1.12,y*1.12));text.set_fontweight('normal')
        ax.set_xlim(-1.8,1.8);ax.set_ylim(-1.8,1.6)
        w=np.outer(ns[key],ns[key]);w[np.tril_indices(len(w))]=0
        ii,jj=np.unravel_index(np.argsort(w.ravel())[::-1][:k],w.shape)
        rows.extend(dict(method=title,source=int(i+1),target=int(j+1),weight=float(w[i,j]),length_mm=float(ns['D'][i,j])) for i,j in zip(ii,jj))
    pd.DataFrame(rows).to_csv(out/'edges.csv',index=False)
    cax=fig.add_axes([.92,.34,.018,.40]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(20,100),cmap='RdYlBu'),cax=cax)
    cb.set_ticks([20,60,100]);cax.set_xlabel('Edge length\n(mm)',fontsize=7,labelpad=8)
    fig.subplots_adjust(left=.01,right=.90,bottom=.04,top=.90);save_figure(fig,out/'panel_H')

if __name__=='__main__':main()
