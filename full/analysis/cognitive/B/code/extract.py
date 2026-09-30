














import os, sys, pickle, numpy as np, h5py, torch, torch.nn.functional as F

PROJ = "/media/wsqlab/data/gy/code/Eigen_brain_decoding"
sys.path.insert(0, PROJ)
from models.prior.eeg_meg_prior_eigenval3 import EEGMEGPriorEigenval3

DATA = "/media/wsqlab/Expansion/meg_h5/data"
BGE  = "/media/wsqlab/Expansion/meg_h5/cache/bge_sent_emb.pkl"
CKPT = os.path.join(PROJ, "checkpoints/stage1_language/prior_20260523_000520/mae_prior_meg_libribrain_smn4lang.pt")
OUT  = "/media/wsqlab/Expansion/meg_h5/imagery_highlevel/feats"
SUBS = ["guoyi", "linguiyu", "linjunhao", "lizhuo", "zhangchi", "zhanghongfei"]
DEV  = torch.device("cuda:1")
os.makedirs(OUT, exist_ok=True)


def load_prior():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    enc = EEGMEGPriorEigenval3(
        patch_size=int(ck.get("patch_size", 64)),
        d_model=int(ck["d_model"]),
        n_heads=int(ck.get("args", {}).get("n_heads", 8)),
        n_factor_layers=int(ck["n_factor_layers"]),
        n_time_layers=int(ck["n_time_layers"]),
        dropout=0.0,
    ).to(DEV).eval()
    state = ck["encoder"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    enc.load_state_dict(state, strict=True)
    for p in enc.parameters():
        p.requires_grad = False
    print(f"[prior] d_model={ck['d_model']} patch={ck.get('patch_size',64)} "
          f"epoch={ck.get('epoch')} val={ck.get('val_loss')}", flush=True)
    return enc


@torch.no_grad()
def hi(enc, modes, sigma, batch=64):

    ev = torch.from_numpy(sigma.astype(np.float32)).to(DEV)
    out = []
    N = modes.shape[0]
    x_all = torch.from_numpy(modes.astype(np.float32))
    for i in range(0, N, batch):
        x = x_all[i:i+batch].to(DEV)
        y = enc.encode_pooled(x, eigval=ev)
        out.append(y.mean(1).float().cpu().numpy())
    return np.concatenate(out, 0)


def lo(modes, B=8):

    N, K, T = modes.shape
    e = np.linspace(0, T, B+1).astype(int)
    fs = []
    for b in range(B):
        seg = modes[:, :, e[b]:e[b+1]]
        fs.append(seg.mean(2)); fs.append(seg.std(2))
    return np.concatenate(fs, 1).astype(np.float32)


def main():
    emb = pickle.load(open(BGE, "rb"))
    enc = load_prior()
    for s in SUBS:
        with h5py.File(f"{DATA}/{s}.h5", "r") as f:
            mr  = f["meg_modes_reading"][:]
            mi  = f["meg_modes_imagine"][:]
            mir = f["meg_modes_indiv_reading"][:]
            mii = f["meg_modes_indiv_imagine"][:]
            sig  = f["meg_sigma"][:]
            sigi = f["meg_sigma_indiv"][:]
            sents = [x.decode() if isinstance(x, bytes) else str(x) for x in f["label_sentence"][:]]
            lid  = f["label_id"][:]
            l3   = f["label3"][:]
            ses  = f["session"][:]

        def rms(*arrs):
            cat = np.concatenate([a.reshape(a.shape[0], -1) for a in arrs], 1)
            return float(np.sqrt((cat**2).mean()) + 1e-12)
        rg = rms(mr, mi); ri = rms(mir, mii)
        mr, mi   = mr/rg,  mi/rg
        mir, mii = mir/ri, mii/ri

        miL, miE = mi[:, :, 512:1024], mi[:, :, 0:512]
        miiL = mii[:, :, 512:1024]

        bge = np.stack([emb[t] for t in sents]).astype(np.float32)

        d = dict(
            sentences=np.array(sents, object), label_id=lid, label3=l3, session=ses, bge=bge,

            hi_grp_read=hi(enc, mr, sig),  hi_grp_img=hi(enc, mi, sig),
            hi_grp_imgL=hi(enc, miL, sig), hi_grp_imgE=hi(enc, miE, sig),

            hi_ind_read=hi(enc, mir, sigi), hi_ind_img=hi(enc, mii, sigi),
            hi_ind_imgL=hi(enc, miiL, sigi),

            lo_grp_read=lo(mr), lo_grp_img=lo(mi), lo_grp_imgL=lo(miL), lo_grp_imgE=lo(miE),
        )
        np.savez(f"{OUT}/{s}.npz", **d)
        print(f"[{s}] N={len(sents)} hi_grp_imgL std={d['hi_grp_imgL'].std():.3f} "
              f"lo_grp_imgL dim={d['lo_grp_imgL'].shape[1]} saved", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
