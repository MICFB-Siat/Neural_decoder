import sys
import argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.runtime import INPUTS,WEIGHTS,TEMPLATES,output,device,style,save_figure
from common.attribution import load_stage_a,StageAPipeline,FullPipelineWithModesEncoder,gradient_shap_per_trial,cos_alignment_score
from common.spectral_stats import hier
from models.prior.fmri_prior_eigenval_v2 import FMRIPriorEigenvalV2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--subjects',nargs='+',type=int,choices=range(1,13),default=list(range(1,13)))
    args=parser.parse_args()
    d=device();out=output('F')
    pack=load_stage_a(WEIGHTS/'F/alignment.pt',device=d);stage=StageAPipeline(pack,default_mode='joint')
    ck=torch.load(WEIGHTS/'F/prior_encoder.pt',map_location='cpu',weights_only=False)
    encoder=FMRIPriorEigenvalV2(n_modes=ck['n_modes'],n_groups=ck['n_groups'],d_model=ck['d_model'],d_inner=ck['d_inner'],
        n_heads=ck['args']['n_heads'],n_factor_layers=ck['n_factor_layers'],n_time_layers=ck['n_time_layers'],
        n_pool_queries=ck['n_pool_queries'],dropout=ck['args']['dropout'],use_mode_signature=True).to(d).eval()
    encoder.load_state_dict(ck['fmri_prior'])
    for parameter in encoder.parameters():parameter.requires_grad_(False)
    enc=dict(modes_enc=encoder,full_eigval=torch.as_tensor(ck['full_eigval'],device=d,dtype=torch.float32))
    pipe=FullPipelineWithModesEncoder(stage,enc,n_out_tokens=20);importance=[]
    for i in args.subjects:
        sid=f'sub-{i:02d}';z=np.load(INPUTS/'SMN4Lang'/f'{sid}.npz')
        modes=torch.tensor(z['test_modes'],device=d);baselines=torch.tensor(z['baseline_modes'],device=d)
        obs=torch.tensor(z['test_obs'],device=d);bge=torch.tensor(z['test_bge'],device=d)
        sids=torch.full((4,),pack['subject_ids'].index(sid),device=d,dtype=torch.long)
        rng=torch.Generator(device=d);rng.manual_seed(0)
        accum=np.zeros(2000,np.float64);signed=accum.copy();cos=np.zeros(len(modes),np.float32)
        for lo in range(0,len(modes),4):
            mb=modes[lo:lo+4];ob=obs[lo:lo+4];gb=bge[lo:lo+4];sb=sids[:len(mb)]
            with torch.no_grad():cos[lo:lo+4]=cos_alignment_score(pipe(mb,ob,sb,mode='joint'),gb,per_trial=True).cpu().numpy()
            def forward(m):return cos_alignment_score(pipe(m,ob,sb,mode='joint'),gb,per_trial=True).sum()
            shap=gradient_shap_per_trial(forward,mb,baselines,n_baselines=8,n_steps=8,rng=rng)
            accum+=shap.abs().sum(-1).sum(0).cpu().double().numpy();signed+=shap.sum(-1).sum(0).cpu().double().numpy()
            if lo%16==0:print('F',sid,lo,'/',len(modes),flush=True)
            del shap;torch.cuda.empty_cache()
        value=(accum/len(modes)).astype(np.float32);importance.append(value)
        np.savez(out/f'{sid}.npz',mode_importance=value,signed_importance=(signed/len(modes)).astype(np.float32),cos_joint=cos)
    if args.subjects!=list(range(1,13)):
        print('Attribution saved. Panel F requires all twelve participants.',flush=True)
        return
    imp=np.stack(importance);valid=np.array([r for r in range(950) if min(abs(r-250),abs(r-500),abs(r-750))>60]);bands=np.array_split(valid,12)
    values=np.array([[imp[s,b].sum()/imp[s,:1000].sum()*100 for b in bands] for s in range(12)])
    rho,rhoci,p,mean,ci=hier(values)
    pd.DataFrame({'point':np.arange(1,13),'mean_percent':mean,'ci_lower':ci[:,0],'ci_upper':ci[:,1]}).to_csv(out/'plot_points.csv',index=False)
    pd.DataFrame(values,index=[f'sub-{i:02d}' for i in range(1,13)]).to_csv(out/'participant_values.csv')
    style();fig=plt.figure(figsize=(8,3.2));grid=fig.add_gridspec(2,6,height_ratios=[1,1.4])
    g=np.load(TEMPLATES/'lh_surface.npz');fields=np.load(TEMPLATES/'eigenmode_fields.npz')['fields'].T
    coords=g['coords'];faces=g['faces'];faces=faces[coords[faces,0].mean(1)<np.median(coords[:,0])];tri=Triangulation(coords[:,1],coords[:,2],faces)
    for i,(rank,wave) in enumerate(zip([25,123,342,440,659,878],[84,36,21,19,15,13])):
        ax=fig.add_subplot(grid[0,i]);v=np.nan_to_num(fields[i]);limit=np.percentile(abs(v),97)
        ax.tripcolor(tri,v,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='gouraud',rasterized=True)
        ax.set_aspect('equal');ax.axis('off');ax.text(.5,-.10,f'λ = {wave} mm\nmode #{rank}',ha='center',va='top',transform=ax.transAxes,fontsize=7)
    ax=fig.add_subplot(grid[1,:]);x=np.arange(12);ax.fill_between(x,ci[:,0],ci[:,1],color='#bd73a4',alpha=.22)
    ax.plot(x,mean,color='#bd73a4',lw=1.8);ax.plot(x,np.polyval(np.polyfit(x,mean,1),x),color='#bd73a4',lw=.7,ls='--')
    ax.text(.62,.87,f'Spearman ρ = {rho:.2f}',transform=ax.transAxes)
    ax.set_ylabel('Semantic\nattribution (%)');ax.set_xticks([]);ax.spines[['top','right']].set_visible(False)
    fig.subplots_adjust(left=.10,right=.98,top=.98,bottom=.10,hspace=.62);save_figure(fig,out/'panel_F')

if __name__=='__main__':main()
