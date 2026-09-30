import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.runtime import INPUTS,WEIGHTS,TEMPLATES,output,device
import numpy as np
import torch
from common.numerics import Bridge,rdm,unit_rank,batch_rsa,p_signflip,bh

def main():
    d=device();torch.set_num_threads(2);out=output('D');arrays=[]
    n=np.load(TEMPLATES/'neighborhoods.npz');ids,ptr=n['indices'],n['indptr']
    for i in range(1,9):
        sid=f'sub-{i:02d}';z=np.load(INPUTS/'NSD'/f'{sid}.npz')
        x=np.concatenate([z['image_raw'],z['bigg'],z['convnext']],axis=1)
        ck=torch.load(WEIGHTS/'D'/f'{sid}.pt',map_location='cpu',weights_only=False)
        model=Bridge(ck['nx'],ck['ny'],ck['state']['target_mean'],ck['width']).to(d)
        model.load_state_dict(ck['state']);model.eval()
        with torch.no_grad():pred=model(torch.tensor((x-ck['mean'])/ck['sd'],dtype=torch.float32,device=d)).cpu().numpy()
        refs=torch.tensor(np.stack([unit_rank(rdm(v)) for v in [z['image_raw'],pred]]).astype(np.float32),device=d)
        brain=torch.tensor(z['fmri'],device=d);result=np.empty((59412,2),np.float32)
        for lo in range(0,59412,48):
            result[lo:lo+48]=batch_rsa(brain,refs,np.arange(lo,min(lo+48,59412)),ids,ptr)
            if lo%9600==0:print('D',sid,lo,'/59412',flush=True)
        arrays.append(result);np.save(out/f'{sid}.npy',result)
        del brain,model,ck;torch.cuda.empty_cache()
    a=np.stack(arrays);effects=np.concatenate([a,(a[:,:,1]-a[:,:,0])[:,:,None]],axis=2)
    p=np.stack([p_signflip(effects[:,:,j]) for j in range(3)],1)
    q=np.column_stack([bh(p[:,:2].ravel()).reshape(59412,2),bh(p[:,2])])
    np.savez(out/'group.npz',participant_rsa=a,mean=effects.mean(0),p=p,q=q)

if __name__=='__main__':main()
