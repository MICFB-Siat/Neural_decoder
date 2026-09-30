

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve()
EXP = HERE.parents[1]
BASE = HERE.parents[2]
CLASSIFIER = BASE.parent / "Exp_Classification" / "code" / "classify_baseline_v2.py"
INIT = BASE.parents[2] / "checkpoints" / "unified_v3" / "prior_unified_v3_20260514_020311.pt"

spec = importlib.util.spec_from_file_location("fourier_classifier", CLASSIFIER)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--experiment-dir", default=str(EXP))
    ap.add_argument("--mode-name", default="fourier")
    args = ap.parse_args()
    exp = Path(args.experiment_dir).resolve()
    out = exp / "pretrain"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "adapt_fourier.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)], force=True)
    log = logging.getLogger("adapt")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    eigval = torch.from_numpy(np.load(exp / "data" / "native_eigval_K30.npy")).float()
    ds = mod._load_modes_hetero(str(exp / "data"), "eeg", device, override_eigval=eigval)
    train_chunks=[]; val_chunks=[]; names=[]; evs=[]
    g = torch.Generator(device=device).manual_seed(args.seed)
    for x,nm,ev in zip(ds.chunks, ds.names, ds.eigvals):
        p=torch.randperm(len(x), generator=g, device=device); nv=max(1, round(len(x)*0.1))
        val_chunks.append(x[p[:nv]]); train_chunks.append(x[p[nv:]]); names.append(nm); evs.append(ev)
    train=mod._HeteroKDataset(train_chunks,names,evs); val=mod._HeteroKDataset(val_chunks,names,evs)
    ck=torch.load(INIT,map_location=device,weights_only=False)
    enc=mod.EEGMEGPriorEigenval3(patch_size=mod.PATCH_SIZE,d_model=ck["d_model"],n_heads=8,
        n_factor_layers=ck.get("n_factor_layers",2),n_time_layers=ck.get("n_time_layers",4)).to(device)
    state={k.replace("_orig_mod.","",1) if k.startswith("_orig_mod.") else k:v for k,v in ck["encoder"].items()}
    missing,unexpected=enc.load_state_dict(state,strict=False)
    if missing or unexpected: raise RuntimeError(f"Initialization mismatch missing={missing}, unexpected={unexpected}")
    dec=mod.EEGMEGMAEDecoderEigenval3(d_model=ck["d_model"],d_dec=ck["d_model"]//2,n_heads=4,n_layers=2,patch_size=mod.PATCH_SIZE).to(device)
    params=list(enc.parameters())+list(dec.parameters())
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=.05,betas=(.9,.95))
    warm=max(1,int(args.epochs*.05))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda ep: ep/warm if ep<warm else .5*(1+np.cos(np.pi*(ep-warm)/max(1,args.epochs-warm))))
    scaler=torch.amp.GradScaler("cuda"); rng=np.random.default_rng(args.seed); best=float("inf"); best_ep=0
    ckpt_path=out/f"prior_{args.mode_name}_seedv_mae50.pt"; NP=ds.T//mod.PATCH_SIZE
    log.info(f"Initialized from {INIT}; N={ds.N}, K={ds.Ks}, T={ds.T}, position range={eigval.min():.6g}..{eigval.max():.6g}")
    for ep in range(1,args.epochs+1):
        enc.train(); dec.train(); total=n=0
        for x,_,ev in train.iter_batches(args.batch_size,rng=rng):
            B,K,_=x.shape; mask=mod.brainomni_2d_mask(B,K,NP,.5,device)
            with torch.amp.autocast("cuda",dtype=torch.bfloat16):
                tgt=mod.patchify(x,mod.PATCH_SIZE); pred=dec(enc.encode_full_grid(x,ev,mask)); m=mask.float().unsqueeze(-1)
                loss=((pred.float()-tgt.float())**2*m).sum()/(m.sum()*mod.PATCH_SIZE+1e-8)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params,1.0); scaler.step(opt); scaler.update(); total+=loss.item()*B; n+=B
        sched.step(); enc.eval(); dec.eval(); vt=vn=0
        with torch.no_grad():
            for x,_,ev in val.iter_batches(args.batch_size,shuffle=False):
                B,K,_=x.shape; mask=mod.brainomni_2d_mask(B,K,NP,.5,device)
                with torch.amp.autocast("cuda",dtype=torch.bfloat16):
                    tgt=mod.patchify(x,mod.PATCH_SIZE); pred=dec(enc.encode_full_grid(x,ev,mask)); m=mask.float().unsqueeze(-1)
                    loss=((pred.float()-tgt.float())**2*m).sum()/(m.sum()*mod.PATCH_SIZE+1e-8)
                vt+=loss.item()*B; vn+=B
        vl=vt/vn
        if vl<best:
            best=vl; best_ep=ep
            torch.save({"encoder":enc.state_dict(),"epoch":ep,"val_loss":vl,"n_modes":30,"n_samples":ds.T,
                "d_model":ck["d_model"],"n_factor_layers":ck.get("n_factor_layers",2),"n_time_layers":ck.get("n_time_layers",4),
                "patch_size":mod.PATCH_SIZE,"mode_signature":f"{args.mode_name} native lambda + spectral gap","adapted_from":str(INIT)},ckpt_path)
        log.info(f"epoch {ep:02d}/{args.epochs} train={total/n:.6f} val={vl:.6f} best={best:.6f}@{best_ep}")
        if not np.isfinite(vl): raise FloatingPointError(f"non-finite validation loss at epoch {ep}")
    log.info(f"DONE checkpoint={ckpt_path}")

if __name__ == "__main__": main()
