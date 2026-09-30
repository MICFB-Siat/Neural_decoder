import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.runtime import INPUTS,WEIGHTS,TEMPLATES,output,device,style,save_figure
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from common.numerics import DirectResidualMLP,predict_full_direct,ridge_affine,project_hemi,column_corr

def main():
    d=device();out=output('C');rows=[]
    for i in range(1,9):
        sid=f'sub-{i:02d}';print('C',sid,flush=True)
        z=np.load(INPUTS/'NSD'/f'{sid}.npz');m=np.load(WEIGHTS/'C'/f'{sid}_normalization.npz')
        x=((z['cls']-m['cls_mu'])/m['cls_sd']).astype(np.float32)
        p=np.load(WEIGHTS/'C'/f'{sid}_baseline_pca6.npz')
        reduced=(x-p['pca_mean'])@p['pca_components'].T
        pred=(reduced-p['ridge_x_mean'])@p['ridge_coef']+p['ridge_y_mean']
        low=column_corr(pred,(z['fmri']-m['cift_mu'])/m['cift_sd'],chunk=4096)
        ck=torch.load(WEIGHTS/'C'/f'{sid}_direct_residual_mlp.pt',map_location='cpu',weights_only=False)
        model=DirectResidualMLP(hidden=int(ck['hidden'])).to(d).eval();model.load_state_dict(ck['model_state'])
        coef,bias=ridge_affine(m,'baseline_cls_to_cifti')
        pred=predict_full_direct(model,x,torch.from_numpy(coef).to(d),torch.from_numpy(bias).to(d),d)
        raw=(pred*m['cift_sd']+m['cift_mu']).astype(np.float32)
        eig=np.load(TEMPLATES/f'{sid}_eigenbasis.npz');l=eig['mL'].astype(np.float32);r=eig['mR'].astype(np.float32)
        recon=np.concatenate([project_hemi(raw[:,:29696],l,d)@l.T,project_hemi(raw[:,29696:],r,d)@r.T],axis=1).astype(np.float32)
        high=column_corr((.8*raw+.2*recon).astype(np.float32),z['fmri'],chunk=4096)
        snr=np.maximum(z['ncsnr'],0)**2;nc=(snr/(snr+float(np.mean(1/z['n_rep'])))).astype(np.float32)
        row={'subject':sid}
        for name,corr in [('low',low),('high',high)]:
            valid=np.isfinite(corr)&np.isfinite(nc)&(nc>=.01)
            row[name]=float(np.mean(corr[valid].astype(float)**2/nc[valid].astype(float)))
        rows.append(row);np.savez(out/f'{sid}.npz',low=low,high=high,nc=nc)
        del model,ck,pred,raw,recon;torch.cuda.empty_cache()
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics.csv',index=False)
    style();fig,ax=plt.subplots(figsize=(3,3.4))
    boxes=ax.boxplot([frame.low,frame.high],widths=.45,showfliers=False)
    for i,color in enumerate(['#53bd99','#ff747d']):
        for key in ['boxes','medians']:boxes[key][i].set_color(color)
        for key in ['caps','whiskers']:
            for artist in boxes[key][2*i:2*i+2]:artist.set_color(color)
    p=float(wilcoxon(frame.high,frame.low).pvalue)
    ax.plot([1,2],[.57,.57],color='black',lw=.6);ax.text(1.5,.58,'**' if p<.01 else ('*' if p<.05 else 'ns'),ha='center')
    ax.set_xticks([1,2],['Stimulus-\nresponse','Cognitive-\ninferential']);ax.set_ylim(0,.62)
    ax.set_ylabel('Noise-ceiling-normalized\nsquared prediction correlation')
    ax.spines[['top','right']].set_visible(False);fig.tight_layout();save_figure(fig,out/'panel_C')
    print(frame.to_string(index=False),flush=True)

if __name__=='__main__':main()
