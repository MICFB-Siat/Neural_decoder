
from low_search_common_20260923 import *
import argparse,sys,traceback
import h5py
import torch
import torch.nn.functional as F

CODE=REPO/'lizhuo_exp/lizhuo_exp_class2/SKIP/code/enc_mod'
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(CODE))
from models.encoders.fmri_cifti_encoder_xyz_v2 import CortexCIFTIEncoderXYZV2
import continue_pretrain_cifti_skip as trainer

CURRENT=Path('/media/wsqlab/nas2/Dataset/ds005256-download/experiments/alignvideo_selected30_continue20_fourtask_lr3e5_loso80_20260723/mae_continue20/checkpoints_obs/cifti_encoder_skip_continue.pt')
TRAJ=Path('/media/wsqlab/nas2/Dataset/ds005256-download/experiments/alignvideo_mae_epochwise_rsa17_two_init_20260724/mae/crossdataset/obs/epoch_checkpoints')
RAW=REPO/'lizhuo_exp/lizhuo_exp_data/SKIP/preprocess/stage2/alignvideo'
RAW_FULL=Path('/media/wsqlab/nas2/Dataset/ds005256-download/preprocess/stage2/alignvideo')


def build(ckpt,device):
    c=torch.load(ckpt,map_location='cpu',weights_only=False)
    enc=CortexCIFTIEncoderXYZV2(n_lh=c['n_lh'],n_rh=c['n_rh'],n_groups_per_hemi=c['n_groups_per_hemi'],
        d_model=c['d_model'],n_heads=c['n_heads'],n_factor_layers=c['n_factor_layers'],n_time_layers=c['n_time_layers'],
        n_pool_queries=c['n_pool_queries'],dropout=c['args']['dropout'],use_xyz_pos=c['use_xyz_pos']).to(device)
    enc.set_group_centroids(torch.as_tensor(c['group_centroids'],dtype=torch.float32,device=device))
    enc.load_state_dict(c['cifti_encoder'],strict=True);enc.eval().requires_grad_(False)
    return enc,c


@torch.inference_mode()
def layers(enc,x):
    tokens,_=enc._embed(x,None)
    out={'embed':tokens.mean((1,2)).float().cpu().numpy()}
    tokens=tokens.permute(0,2,1,3).contiguous()
    for i,blk in enumerate(enc.factor_blocks,1):
        tokens=blk(tokens);out[f'factor{i}']=tokens.mean((1,2)).float().cpu().numpy()
    z=enc.g_pool(tokens);out['gpool']=z.mean(1).float().cpu().numpy()
    for i,blk in enumerate(enc.time_blocks,1):
        z=blk(z);out[f'time{i}']=z.mean(1).float().cpu().numpy()
    out['norm']=enc.norm(z).mean(1).float().cpu().numpy()
    return out


def cache_inputs(device):
    dest=OUT/'internal/input_cache';dest.mkdir(parents=True,exist_ok=True)
    fixed=load_fixed('internal');enc,c=build(CURRENT,device)
    replay=[]
    for sub,f in fixed.items():
        path=dest/f'{sub}.npy'
        if not path.exists():
            source=RAW/f'{sub}.h5'
            if not source.exists():source=RAW_FULL/f'{sub}.h5'
            with h5py.File(source) as h:x=h['fmri_cift'][:,:enc.n_cortex,:].astype(np.float32)
            mu=x.mean((0,2),keepdims=True);sd=x.std((0,2),keepdims=True)+1e-8
            x=((x-mu)/sd)[f['trials']].astype(np.float16)
            np.save(path,x);del x,mu,sd
        x=np.load(path,mmap_mode='r')
        assert len(x)==len(f['cells'])

        got=[]
        for start in range(0,len(x),2):
            got.append(layers(enc,torch.as_tensor(np.asarray(x[start:start+2]),dtype=torch.float32,device=device))['norm'])
        got=np.concatenate(got)
        err=float(np.max(np.abs(got-f['low'])));replay.append(dict(subject=sub,max_abs_error=err))
        assert err<.025,(sub,err)
        print('internal input cache',sub,'replay',err,flush=True)
    write_json(OUT/'internal/input_replay.json',replay)


def extract_checkpoint(name,path,device):
    dest=OUT/'internal';fixed=load_fixed('internal')
    enc,c=build(path,device);allf={}
    for sub in fixed:
        x=np.load(dest/f'input_cache/{sub}.npy',mmap_mode='r');acc={}
        for start in range(0,len(x),2):
            value=layers(enc,torch.as_tensor(np.asarray(x[start:start+2]),dtype=torch.float32,device=device))
            for key,v in value.items():acc.setdefault(key,[]).append(v)
        for layer,values in acc.items():allf.setdefault(layer,{})[sub]=np.concatenate(values)
    for layer,features in allf.items():
        candidate=f'{name}_{layer}'
        if not (dest/f'metrics/{candidate}.csv').exists():
            evaluate('internal',candidate,features,dict(arm=name,epoch=int(c.get('epoch',-1)),layer=layer,
                checkpoint=str(path),preprocessing='original full-subject vertex z-score; fixed shared 30 videos; all-token mean',
                low_training='self-supervised CIFTI MAE; subjective labels not used'))
    aggregate('internal');root_index()


def main(device_name):
    torch.set_num_threads(4);device=torch.device(device_name)
    for d in ('features','centroids','metrics','figures','metadata'):(OUT/'internal'/d).mkdir(parents=True,exist_ok=True)
    status('internal','caching normalized selected trials',device=device_name)
    cache_inputs(device)

    paths=[('selected30_epoch20',CURRENT)]+[(f'crossdataset_epoch{i:03d}',TRAJ/f'epoch_{i:03d}.pt') for i in range(21)]
    for index,(name,path) in enumerate(paths,1):
        status('internal','extracting saved pretraining epochs and layers',candidate=name,index=index,total=len(paths))
        extract_checkpoint(name,path,device)
        print('Internal extracted',index,'/',len(paths),name,flush=True)
    status('internal','saved-trajectory scan complete',candidates=len(paths),high_fixed='epoch12_obs_mu')
    aggregate('internal');root_index()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:1');args=ap.parse_args()
    try:main(args.device)
    except Exception:
        status('internal','failed',traceback=traceback.format_exc());root_index();raise
