










from __future__ import annotations
import os, sys, json, re, argparse
from types import SimpleNamespace
from pathlib import Path
import numpy as np, torch
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = "/home/guoyi/a800/share/code/Eigen_brain_decoding"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "guoyi_exp/Exp1"))
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True); torch.backends.cudnn.enabled = False
import classify_baseline_v2 as C
HERE = Path(__file__).resolve().parent


@torch.no_grad()
def predict_all(model, prior_tok, device, bs=128):
    model.eval(); out = []
    for s in range(0, len(prior_tok), bs):
        pt = prior_tok[s:s+bs].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _, _ = model.cached_forward(pt)
        out.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--subject", default="sub-01")
    ap.add_argument("--n_classes", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run_dir if os.path.isabs(args.run_dir) else os.path.join(HERE, args.run_dir))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(run_dir, "encoder_prior"); os.makedirs(out_dir, exist_ok=True)
    C._setup_logging(out_dir); log = C.log
    split = json.load(open(os.path.join(run_dir, "split.json")))["split"]

    prior_enc = C.load_prior_from_ckpt(args.ckpt, device, trainable=False, n_samples_override=None)
    n_tok = prior_enc.expected_n_samples // C.PATCH_SIZE
    d_model = prior_enc.d_model_dim
    bundle = C.load_subject(run_dir, args.subject, prior_enc.expected_n_samples, "eeg", "labels", override_eigval=None)
    eigval = bundle["eigval"].to(device)
    modes = bundle["modes"]; y = bundle["labels"].numpy()
    N = len(y)
    log.info(f"loaded {N} trials  modes={tuple(modes.shape)}  n_tok={n_tok} d_model={d_model}")


    prior_enc.eval(); toks = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, N, 64):
            xm = modes[s:s+64].to(device)
            toks.append(prior_enc.encode_pooled(xm, eigval=eigval).float().cpu())
    prior_tok = torch.cat(toks)
    log.info(f"prior_tok={tuple(prior_tok.shape)}")


    base = SimpleNamespace(lat_dim=256, dropout=0.3, kl_weight=1e-3, n_classes=args.n_classes)
    tr_all = np.array(split["train"])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    cr, vr = next(sss.split(tr_all, y[tr_all])); tr, va = tr_all[cr], tr_all[vr]
    of_dummy_tr = torch.zeros(len(tr), n_tok, d_model)
    of_dummy_va = torch.zeros(len(va), n_tok, d_model)
    tr_ds = C.CachedGPUDataset(of_dummy_tr, prior_tok[tr], bundle["labels"][tr], device)
    va_ds = C.CachedGPUDataset(of_dummy_va, prior_tok[va], bundle["labels"][va], device)
    pri_ghead = C.GaussHead(d_model, base.lat_dim).to(device)
    pri_cls = C.ClsHead(base.lat_dim, n_classes=args.n_classes, dropout=base.dropout).to(device)
    pri_model = C.ExpertModel(prior_enc, pri_ghead, pri_cls, is_obs=False, encoder_trainable=False).to(device)
    pri_model.set_eigval(eigval)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    acc, state = C.train_expert_cached(pri_model, tr_ds, va_ds, branch="prior",
                                       epochs=args.epochs, batch_size=64,
                                       lr_adapter=1e-3, lr_head=1e-3, patience=40,
                                       kl_weight=base.kl_weight, label="geomprior")
    pri_model.load_state_dict(state, strict=False)
    log.info(f"prior heads trained: val_acc={acc:.3f}")

    p_prior = predict_all(pri_model, prior_tok, device)
    np.save(os.path.join(out_dir, "p_prior.npy"), p_prior.astype(np.float32))
    np.save(os.path.join(out_dir, "labels.npy"), y)

    test_segs = sorted(((k, int(re.search(r"dt(\d+)", k).group(1))) for k in split if k.startswith("test")),
                       key=lambda x: x[1])
    rows = {}; print("\n  Δt   n   enc_prior_acc")
    for seg_key, dt in test_segs:
        idx = np.array(split[seg_key]); a = float((p_prior[idx].argmax(1) == y[idx]).mean())
        rows[dt] = {"acc": a, "n": int(len(idx))}; print(f"  {dt:3d} {len(idx):3d}   {a*100:5.1f}")
    mean = np.mean([rows[d]["acc"] for d in rows])
    lng = np.mean([rows[d]["acc"] for d in rows if d >= 204])
    print(f"  MEAN={mean*100:.1f}  long(>=204)={lng*100:.1f}")
    json.dump({"ckpt": args.ckpt, "per_dt": rows, "mean": mean, "long": lng},
              open(os.path.join(out_dir, "encoder_prior_summary.json"), "w"), indent=2)
    print(f"[saved] {out_dir}/p_prior.npy")


if __name__ == "__main__":
    main()
