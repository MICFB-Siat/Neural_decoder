import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.runtime import INPUTS,WEIGHTS,TEMPLATES,output,device
import numpy as np
import torch
from common.numerics import column_corr,unpatch_cift
from models.encoders.fmri_cifti_encoder_xyz_v2 import CortexCIFTIMAEDecoderXYZV2

def ridge(x,m,prefix,d):
    with torch.no_grad():
        a=torch.as_tensor(x-m[prefix+'_x_mean'],dtype=torch.float32,device=d)
        y=a@torch.as_tensor(m[prefix+'_coef'],dtype=torch.float32,device=d)
        y+=torch.as_tensor(m[prefix+'_y_mean'],dtype=torch.float32,device=d)
        return y.cpu().numpy()

def main():
    d=device();out=output('B');eigen=np.load(TEMPLATES/'group_eigenbasis.npz')
    ck=torch.load(WEIGHTS/'B/cift_decoder.pt',map_location='cpu',weights_only=False)
    decoder=CortexCIFTIMAEDecoderXYZV2(max_group_size=7429,d_model=512,n_layers=2).to(d).eval()
    decoder.load_state_dict(ck['decoder'],strict=False)
    all_low=[];all_high=[]
    for i in range(1,9):
        sid=f'sub-{i:02d}';print('B',sid,flush=True)
        z=np.load(INPUTS/'NSD'/f'{sid}.npz');m=np.load(WEIGHTS/'B'/f'{sid}.npz')
        x=((z['semantic']-m['semantic_mu'])/m['semantic_sd']).astype(np.float32)
        poe=ridge(x,m,'caption_to_poe',d);poe/=np.linalg.norm(poe,axis=1,keepdims=True)+1e-8
        aug=np.concatenate([poe,np.sign(poe)*np.sqrt(np.abs(poe)+1e-8)],axis=1).astype(np.float32)
        modes=ridge(aug,m,'poe_to_modes',d)*m['modes_sd']+m['modes_mu']
        true=(z['fmri']-m['cift_mu'])/m['cift_sd']
        high=np.empty(59412,np.float32)
        with torch.no_grad():
            a=torch.tensor(aug-m['poe_to_cifti_x_mean'],device=d)
            ml=torch.tensor(modes[:,:1000],device=d);mr=torch.tensor(modes[:,1000:],device=d)
            left=eigen['mL'].astype(np.float32);right=eigen['mR'].astype(np.float32)
            coef=m['poe_to_cifti_coef'];bias=m['poe_to_cifti_y_mean'];nleft=len(left)
            for lo in range(0,59412,8192):
                hi=min(lo+8192,59412)
                direct=a@torch.tensor(coef[:,lo:hi],device=d)+torch.tensor(bias[lo:hi],device=d)
                chunks=[]
                if lo<nleft:chunks.append(ml@torch.tensor(left[lo:min(hi,nleft)].T,device=d))
                if hi>nleft:chunks.append(mr@torch.tensor(right[max(lo-nleft,0):hi-nleft].T,device=d))
                geom=torch.cat(chunks,dim=1)
                geom=(geom-torch.tensor(m['cift_mu'][lo:hi],device=d))/torch.tensor(m['cift_sd'][lo:hi],device=d)
                pred=((1-float(m['geometry_weight']))*direct+float(m['geometry_weight'])*geom).cpu().numpy()
                pc=pred-pred.mean(0);tc=true[:,lo:hi]-true[:,lo:hi].mean(0)
                high[lo:hi]=(pc*tc).sum(0)/(np.linalg.norm(pc,axis=0)*np.linalg.norm(tc,axis=0)+1e-8)
        p4=(x-m['pca4_mean'])@m['pca4_components'].T
        flat=(p4-m['baseline_pca4_to_zcift_x_mean'])@m['baseline_pca4_to_zcift_coef']+m['baseline_pca4_to_zcift_y_mean']
        tokens=flat.reshape(len(flat),1,8,512);pred=np.empty_like(true)
        with torch.no_grad():
            for lo in range(0,len(flat),16):
                pred[lo:lo+16]=unpatch_cift(decoder(torch.tensor(tokens[lo:lo+16],device=d)).cpu().numpy())
        low=column_corr(pred,true,chunk=4096)
        all_low.append(low);all_high.append(high)
        np.savez(out/f'{sid}.npz',low=low,high=high)
        torch.cuda.empty_cache()
    group=lambda a:np.tanh(np.arctanh(np.clip(np.stack(a),-.999999,.999999)).mean(0))
    np.savez(out/'group.npz',low=group(all_low),high=group(all_high))

if __name__=='__main__':main()
