
from low_search_common_20260923 import *
import argparse,sys,hashlib,traceback
import h5py
import torch
import torch.nn.functional as F

BOLD_ARMS=[
    ('bold_lr3e4_wd1e4',3e-4,1e-4),
    ('bold_lr3e4_wd1e2',3e-4,1e-2),
    ('bold_lr3e4_wd1e1',3e-4,1e-1),
    ('bold_lr1e3_wd1e3',1e-3,1e-3),
    ('bold_lr1e3_wd1e1',1e-3,1e-1),
    ('bold_lr3e3_wd1e2',3e-3,1e-2),
]


def main(device_name,epochs,arm_set):
    torch.set_num_threads(4)
    dev=torch.device(device_name)
    dest=OUT/'motor'
    for d in ('token_cache','checkpoints','features','metrics'):(dest/d).mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(REPO/'lizhuo_exp/lizhuo_exp_result/_common'))
    import poe_eeg_runtime as R
    module=R._load_canonical()
    model,_=R._load_brainomni(module,dev)
    base=Path(module.TINY_CKPT_DIR)/'BrainOmni.pt'
    fixed=load_fixed('motor')
    if not (dest/'metrics/original_baseline.csv').exists():
        evaluate('motor','original_baseline',{s:f['low'] for s,f in fixed.items()},dict(layer='original_block11',epoch=0,arm='baseline'))
    aggregate('motor');root_index()
    rawroot=REPO/'data_check_20260507/full_run_5fold/stage2_new/MOTOR'
    status('motor','caching frozen tokenizer outputs',device=device_name)
    token_paths=[];replay=[]
    for path in sorted(rawroot.glob('sub-*.h5')):
        sub=path.stem;cache=dest/f'token_cache/{sub}.pt';token_paths.append(cache)
        if cache.exists():continue
        with h5py.File(path) as h:
            raw=torch.tensor(h['eeg_raw'][:,:,:2560],dtype=torch.float32)
            pos=torch.tensor(h['eeg_pos'][:],dtype=torch.float32)
            stp=torch.tensor(h['eeg_sensor_type'][:],dtype=torch.long)
            labels=np.array(h['labels'][:])
        chunks=[];indices=[]
        with torch.inference_mode():
            for start in range(0,len(raw),16):
                xr=raw[start:start+16].to(dev);po=pos.to(dev).unsqueeze(0).expand(len(xr),-1,-1);st=stp.to(dev).unsqueeze(0).expand(len(xr),-1)



                t,lab=model.tokenizer.tokenize(xr,po,st,model.overlap_ratio)
                if start==0:
                    reference=model.encode(xr,po,st)
                    reference=F.adaptive_avg_pool1d(reference.float().mean(1).permute(0,2,1),40).permute(0,2,1).mean(1)
                chunks.append(t.cpu());indices.append(lab.cpu())
        neuro=model.tokenizer.encoder.neuros.detach().cpu()
        payload=dict(tokens=torch.cat(chunks),indices=torch.cat(indices),neuro=neuro,subject=sub,labels=labels)
        torch.save(payload,cache)
        if sub in fixed:
            old=np.load(BASE/f'v2/lioi_pretrained/{sub}.npz')
            assert np.array_equal(labels,old['y_true'])
            err=float(np.max(np.abs(reference.cpu().numpy()-old['o'][:len(reference)])))
            replay.append(dict(subject=sub,max_feature_abs_error=err))
            assert err<.015,(sub,err)
        print('token cache',sub,tuple(payload['tokens'].shape),flush=True)
    write_json(dest/'tokenizer_replay.json',replay)
    original={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('tokenizer.')}

    @torch.inference_mode()
    def extract(arm,epoch):
        model.eval()
        all_features={f'block{i:02d}':{} for i in range(1,len(model.blocks)+1)}
        for sub,f in fixed.items():
            data=torch.load(dest/f'token_cache/{sub}.pt',map_location='cpu',weights_only=False)
            acc={k:[] for k in all_features}
            for start in range(0,len(data['tokens']),16):
                t=data['tokens'][start:start+16].to(dev)
                neuro=data['neuro'].to(device=dev,dtype=t.dtype).view(1,t.shape[1],1,-1)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    x=model.projection(t+neuro)
                    for i,block in enumerate(model.blocks,1):
                        x=block(x)
                        z=F.normalize(x,p=2.,dim=-1,eps=1e-6).float().mean(1).permute(0,2,1)
                        z=F.adaptive_avg_pool1d(z,40).permute(0,2,1).mean(1)
                        acc[f'block{i:02d}'].append(z.cpu().numpy())
            for layer in acc:all_features[layer][sub]=np.concatenate(acc[layer])[f['take']]
        if epoch==0 and arm=='pretrained':
            checks=[]
            for sub,f in fixed.items():
                err=float(np.max(np.abs(all_features['block11'][sub]-f['low'])))
                checks.append(dict(subject=sub,max_feature_abs_error=err));assert err<.02,(sub,err)
            write_json(dest/'encoder_replay.json',checks)
        for layer,feats in all_features.items():
            name=f'{arm}_epoch{epoch:03d}_{layer}'
            if (dest/f'metrics/{name}.csv').exists():continue
            evaluate('motor',name,feats,dict(arm=arm,epoch=epoch,layer=layer,
                checkpoint=str(dest/f'checkpoints/{arm}/epoch_{epoch:03d}.pt') if epoch else str(base),
                preprocessing='original raw EEG, frozen tokenizer, L2 token norm, channel mean, adaptive40 token mean',
                low_training='none' if epoch==0 else 'masked token self-supervised continuation on all ten subjects; labels not used; low is transductive to unlabeled EEG'))
        aggregate('motor');root_index()

    extract('pretrained',0)

    for n,p in model.named_parameters():p.requires_grad_(not n.startswith('tokenizer.'))
    search_arms=ARMS if arm_set=='standard' else BOLD_ARMS
    for arm,lr,wd in search_arms:
        checkpoint_dir=dest/f'checkpoints/{arm}';checkpoint_dir.mkdir(parents=True,exist_ok=True)
        model.load_state_dict(original,strict=False)
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        rng=np.random.default_rng(42)
        optimizer=torch.optim.AdamW(model.get_parameters_groups(lr,wd))
        history=[]
        for epoch in range(1,epochs+1):
            status('motor','self-supervised continuation',arm=arm,epoch=epoch,epochs=epochs,lr=lr,weight_decay=wd)
            model.train();model.tokenizer.eval();losses=[]
            for pi in rng.permutation(len(token_paths)):
                data=torch.load(token_paths[pi],map_location='cpu',weights_only=False)
                for ids in np.array_split(rng.permutation(len(data['tokens'])),max(1,int(np.ceil(len(data['tokens'])/16)))):
                    t=data['tokens'][ids].to(dev);target=data['indices'][ids].to(dev)
                    B,C,W,D=t.shape
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        preserve=torch.rand((B,C,W),device=dev)>model.mask_ratio
                        random_tokens=t.reshape(-1,D)[torch.randperm(B*C*W,device=dev)].reshape(B,C,W,D)
                        x=torch.where(preserve.unsqueeze(-1),t,random_tokens)
                        keep=((preserve.float()+torch.rand((B,C,W),device=dev))>.8).unsqueeze(-1).type_as(x)
                        x=x*keep+model.mask_token.type_as(x)*(1-keep)
                        x=x+data['neuro'].to(device=dev,dtype=x.dtype).view(1,C,1,-1)
                        x=model.projection(x)
                        for block in model.blocks:x=block(x)
                        logits=model.predict_head(x).reshape(B,C,W,model.num_quantizers_used,-1)
                        loss,_=model.compute_cross_entropy(logits,target,preserve)
                    if not torch.isfinite(loss):raise RuntimeError(f'nonfinite {arm}/{epoch}')
                    loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                    optimizer.step();losses.append(float(loss.detach()))
            history.append(dict(epoch=epoch,loss=float(np.mean(losses))))
            pd.DataFrame(history).to_csv(checkpoint_dir/'training_history.csv',index=False)
            torch.save(dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('tokenizer.')},
                frozen_tokenizer_checkpoint=str(base),epoch=epoch,arm=arm,lr=lr,weight_decay=wd,optimizer_state=optimizer.state_dict(),
                rng_state=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),checkpoint_dir/f'epoch_{epoch:03d}.pt')
            extract(arm,epoch)
    status('motor','complete',arm_set=arm_set,epochs_per_arm=epochs,arms=search_arms,high_fixed='seed1_mu_mean')
    aggregate('motor');root_index()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');ap.add_argument('--epochs',type=int,default=13)
    ap.add_argument('--arm-set',choices=('standard','bold'),default='standard')
    args=ap.parse_args()
    try:main(args.device,args.epochs,args.arm_set)
    except Exception:
        status('motor','failed',traceback=traceback.format_exc());root_index();raise
