












from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

ROOT = str(_LOCAL_SOURCE / '')
HERE = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.join(ROOT, "data_check_20260507/full_run_5fold"))
sys.path.insert(0, os.path.join(ROOT, "weights/BrainOmni/BrainOmni-main"))
sys.path.insert(0, ROOT)

import classify_seedv_exp3_v2 as C

C.PROJECT_ROOT  = ROOT
C.BRAINOMNI_SRC = os.path.join(ROOT, "weights/BrainOmni/BrainOmni-main")
C.TINY_CKPT_DIR = os.path.join(ROOT, "weights/BrainOmni/BrainOmni/tiny")

RUN          = os.path.join(HERE, "run_robust_20260514_143330")
ENCODER_CKPT = os.path.join(HERE, "ckpts/prior_exp3_uncertain_20260514_091235.pt")
CONDS        = ["clean"]
N_CLASSES    = 5
LAT_DIM      = 256
DROPOUT      = 0.3
DEVICE       = torch.device("cuda:0")
BATCH        = 128

AGG = json.load(open(os.path.join(RUN, "aggregate.json")))


def main():
    print("Loading K-free prior encoder + BrainOmni-tiny backbone (frozen) …")
    prior_enc = C.load_prior_from_ckpt(ENCODER_CKPT, DEVICE, trainable=False)
    backbone, lm_dim = C.load_brainomni_tiny(DEVICE, finetune=False)
    n_tok   = prior_enc.expected_n_samples // C.PATCH_SIZE
    d_model = prior_enc.d_model_dim
    n_samples = prior_enc.expected_n_samples
    print(f"  lm_dim={lm_dim} d_model={d_model} n_tok={n_tok} n_samples={n_samples}")

    out = {}
    for cond in CONDS:
        h5_dir = os.path.join(HERE, "data", cond)
        run_dir = os.path.join(RUN, "runs", f"{cond}_pooled_run_robust_20260514_143330")
        subjects = sorted(f[:-3] for f in os.listdir(h5_dir)
                          if f.startswith("sub-") and f.endswith(".h5"))
        print(f"\n===== {cond}  ({len(subjects)} subjects) =====")
        items = [C.load_subject(h5_dir, s, n_samples) for s in subjects]
        bundle = C.stack_subjects(items)
        subj_ids = bundle["subj_ids"]
        n_per = bundle["N_per_subject"]; n_subj = bundle["n_subjects"]
        splits = C.load_splits(h5_dir, "pooled")

        correct = {m: {s: 0 for s in subjects} for m in ("obs", "prior", "poe_arith")}
        total = {s: 0 for s in subjects}
        fold_obs_g, fold_arith_g = [], []

        for fr in splits["folds"]:
            fold = fr["fold"]
            _, te_idx = C.pooled_indices(fr["train_idx"], fr["test_idx"], n_subj, n_per)

            obs_te, pri_te = C.precompute_features(
                bundle, te_idx, backbone, prior_enc, lm_dim, n_tok, DEVICE, batch_size=BATCH)
            y_te = bundle["labels"][te_idx]
            subj_te = subj_ids[te_idx]


            fold_dir = os.path.join(run_dir, f"fold_{fold:02d}")
            obs_model = C._ObsCachedBranch(lm_dim, d_model, LAT_DIM, N_CLASSES, DROPOUT).to(DEVICE)
            pri_model = C._PriorCachedBranch(d_model, LAT_DIM, N_CLASSES, DROPOUT).to(DEVICE)
            obs_model.load_state_dict(torch.load(os.path.join(fold_dir, "obs.pt"), map_location=DEVICE))
            pri_model.load_state_dict(torch.load(os.path.join(fold_dir, "prior.pt"), map_location=DEVICE))
            obs_model.eval(); pri_model.eval()

            zo, zp = [], []
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for s in range(0, len(y_te), BATCH):
                    ob = obs_te[s:s+BATCH].to(DEVICE)
                    pb = pri_te[s:s+BATCH].to(DEVICE)
                    lo, _, _ = obs_model(ob)
                    lp, _, _ = pri_model(pb)
                    zo.append(lo.float().cpu()); zp.append(lp.float().cpu())
            z_obs = torch.cat(zo); z_pri = torch.cat(zp)
            y = y_te.numpy()

            obs_pred = z_obs.argmax(1).numpy()
            pri_pred = z_pri.argmax(1).numpy()
            prob_obs = F.log_softmax(z_obs, -1).exp()
            prob_pri = F.log_softmax(z_pri, -1).exp()
            arith_pred = (prob_obs + prob_pri).argmax(1).numpy()

            fold_obs_g.append((obs_pred == y).mean())
            fold_arith_g.append((arith_pred == y).mean())

            for i, s in enumerate(subj_te):
                total[s] += 1
                if obs_pred[i] == y[i]:   correct["obs"][s] += 1
                if pri_pred[i] == y[i]:   correct["prior"][s] += 1
                if arith_pred[i] == y[i]: correct["poe_arith"][s] += 1


        ref_obs = np.array(AGG[cond]["obs_acc"]["values"])
        ref_ari = np.array(AGG[cond]["poe_arith"]["values"])
        d_obs = np.abs(np.array(fold_obs_g) - ref_obs).max()
        d_ari = np.abs(np.array(fold_arith_g) - ref_ari).max()
        flag = "OK" if (d_obs < 1e-3 and d_ari < 1e-3) else "CHECK"
        print(f"  validate vs aggregate.json: max|d_obs|={d_obs:.2e} max|d_arith|={d_ari:.2e} [{flag}]")

        cond_out = {}
        for s in subjects:
            cond_out[s] = {
                "obs_acc":   correct["obs"][s] / total[s],
                "prior_acc": correct["prior"][s] / total[s],
                "poe_arith": correct["poe_arith"][s] / total[s],
                "n_trials":  total[s],
            }
        out[cond] = cond_out


        print(f"  {'subject':9s} {'n':>5s} {'obs%':>8s} {'prior%':>8s} {'PoE_arith%':>11s}")
        for s in subjects:
            r = cond_out[s]
            print(f"  {s:9s} {r['n_trials']:5d} {r['obs_acc']*100:8.2f} "
                  f"{r['prior_acc']*100:8.2f} {r['poe_arith']*100:11.2f}")
        mo = np.mean([cond_out[s]["obs_acc"] for s in subjects]) * 100
        mp = np.mean([cond_out[s]["prior_acc"] for s in subjects]) * 100
        ma = np.mean([cond_out[s]["poe_arith"] for s in subjects]) * 100
        so = np.std([cond_out[s]["obs_acc"] for s in subjects]) * 100
        spr = np.std([cond_out[s]["prior_acc"] for s in subjects]) * 100
        sa = np.std([cond_out[s]["poe_arith"] for s in subjects]) * 100
        print(f"  {'MEAN±SD':9s} {'':>5s} {mo:6.2f}±{so:.2f} {mp:6.2f}±{spr:.2f} {ma:6.2f}±{sa:.2f}")

    with open(os.path.join(HERE, "per_subject_acc_robust_clean.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved -> {os.path.join(HERE, 'per_subject_acc_robust_clean.json')}")


if __name__ == "__main__":
    main()
