

import argparse, copy, hashlib, importlib.util, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parents[1]
REF_ROOT = EXP_ROOT / "Exp_Classification" / "result"
BASELINE = EXP_ROOT / "Exp_Classification" / "code" / "classify_baseline_v2.py"
RESULT_ROOT = EXP_ROOT / "Exp_correction" / "result"

spec = importlib.util.spec_from_file_location("baseline_v2", BASELINE)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

DATASETS = ("BCIC", "FACED", "MOTOR", "SEEDV")
RATIOS = (0.10, 0.30, 0.50)
NAS_DATA = Path.home() / "nas/ugreen/NoLanguageData"
FS32K = Path.home() / "nas/ugreen/eigenmode_test/fs32k"
NAS_INPUTS = {
    "BCIC": (NAS_DATA / "BCIC_IV_2a_Raw/preprocess/stage2",
             NAS_DATA / "BCIC_IV_2a_Raw/preprocess/stage2/cache/d_template_eeg_17_5e253942b967.npy"),
    "FACED": (NAS_DATA / "FACED_Raw/preprocess/stage2",
              NAS_DATA / "FACED_Raw/preprocess/stage2/cache/d_template_eeg_25.npy"),
    "MOTOR": (NAS_DATA / "24+EEG-fMRI Motor Imagery/preprocess/stage2",
              NAS_DATA / "24+EEG-fMRI Motor Imagery/preprocess/stage2/cache/eeg_template/D_6d3a70b33fed.npy"),
    "SEEDV": (NAS_DATA / "SEED-V/preprocess/stage2",
              NAS_DATA / "SEED-V/preprocess/stage2/sensor_geom_eigenmode_template_60x48.npy"),
}


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def seed0_run(dataset):
    roots = sorted((REF_ROOT / dataset / "BT-ND+BrainOmni_cross5fold").glob("*seed0*"))
    if not roots: raise FileNotFoundError(f"seed0 reference run missing: {dataset}")
    return roots[0]


def resolve_path(old, suffix_hint=None):
    p = Path(old)
    if p.exists(): return str(p)
    marker = "data_check_20260507/"
    if marker in old:
        q = EXP_ROOT.parent / old.split(marker, 1)[1]
        if q.exists(): return str(q)
    if suffix_hint:
        q = EXP_ROOT.parent / suffix_hint
        if q.exists(): return str(q)
    if p.name.startswith("prior_unified_v3"):
        bundled = EXP_ROOT / "checkpoints" / p.name
        if bundled.exists(): return str(bundled)
    raise FileNotFoundError(old)


def stratified_nested(y, seed):

    rng = np.random.default_rng(seed)
    cal_order, test = [], []
    for c in sorted(np.unique(y).tolist()):
        ix = np.flatnonzero(y == c); rng.shuffle(ix)
        n_cal = len(ix) // 2
        cal_order.extend(ix[:n_cal].tolist()); test.extend(ix[n_cal:].tolist())

    cal_order = np.asarray(cal_order); rng.shuffle(cal_order)
    test = np.asarray(test); rng.shuffle(test)
    out = {}
    n = len(y)
    for ratio in RATIOS:
        target = min(len(cal_order), int(round(n * ratio)))


        out[str(int(ratio * 100))] = np.sort(cal_order[:target]).tolist()
    return out, np.sort(test).tolist()


def metrics(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(y, pred))}


def build_models(cfg, device, prior, backbone, lm_dim, fold_dir):
    d_model = prior.d_model_dim; n_tok = prior.expected_n_samples // base.PATCH_SIZE
    obs = base.ExpertModel(base.ObsEncoder(backbone, lm_dim, n_tok, d_model, False),
                           base.GaussHead(d_model, cfg["lat_dim"]),
                           base.ClsHead(cfg["lat_dim"], n_classes=cfg["n_classes"], dropout=cfg["dropout"]),
                           True, False).to(device)
    pri = base.ExpertModel(prior, base.GaussHead(d_model, cfg["lat_dim"]),
                           base.ClsHead(cfg["lat_dim"], n_classes=cfg["n_classes"], dropout=cfg["dropout"]),
                           False, False).to(device)
    obs.load_state_dict(torch.load(fold_dir / "obs.pt", map_location=device), strict=False)
    pri.load_state_dict(torch.load(fold_dir / "prior.pt", map_location=device), strict=False)
    for model in (obs, pri):
        for p in model.parameters(): p.requires_grad = False
        for p in model.cls_head.parameters(): p.requires_grad = True
    return obs, pri


@torch.no_grad()
def predict(model, feat, branch, batch=256):
    model.eval(); out = []
    for s in range(0, len(feat), batch):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            z, _, _ = model.cached_forward(feat[s:s+batch])
        out.append(z.argmax(1).cpu())
    return torch.cat(out).numpy()


def train_head(model, feat, y, seed, epochs, batch, lr):
    seed_all(seed)
    model.train()

    model.encoder.eval(); model.ghead.eval(); model.cls_head.train()
    opt = torch.optim.AdamW(model.cls_head.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda")
    for _ in range(epochs):
        order = torch.randperm(len(y), device=y.device)
        for s in range(0, len(y), batch):
            ix = order[s:s+batch]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _, _ = model.cached_forward(feat[ix])
                loss = criterion(logits, y[ix])
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.cls_head.parameters(), 1.0)
            scaler.step(opt); scaler.update()


def state_hash(module):
    h = hashlib.sha256()
    for k, v in module.state_dict().items(): h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run_dataset(dataset, device, epochs, only_seed=None, only_fold=None, only_subject=None):
    ref = seed0_run(dataset); cfg = json.loads((ref / "args.json").read_text())
    nas_h5, nas_d = NAS_INPUTS[dataset]
    try: h5_dir = resolve_path(cfg["h5_dir"])
    except FileNotFoundError: h5_dir = str(nas_h5)
    enc_ckpt = resolve_path(cfg["encoder_ckpt"])
    d_path = str(nas_d) if nas_d.exists() else resolve_path(cfg["d_path"])
    D = np.load(d_path)
    k = D.shape[1]
    lh = np.load(FS32K / "fsLR_32k_lh_eval_1024.npy")[:(k + 1)//2]
    rh = np.load(FS32K / "fsLR_32k_rh_eval_1024.npy")[:k//2]
    lam = np.concatenate([lh, rh])
    ev, _ = base.compute_eigval_from_D(D, lam_cortex=lam, ratio=cfg.get("ratio", .01))
    override = torch.from_numpy(ev)
    prior = base.load_prior_from_ckpt(enc_ckpt, device, trainable=False, n_samples_override=cfg.get("n_samples"))
    backbone, lm_dim = base.load_brainomni_tiny(device, finetune=False)
    subjects = base.discover_subjects(h5_dir)
    split_rows = json.loads((ref / "splits.json").read_text())
    for row in split_rows:
        fold = int(row["fold"])
        if only_fold is not None and fold != only_fold: continue
        targets = row.get("val_subjects") or row.get("test_subjects")
        for subj in targets:
            if only_subject and subj != only_subject: continue
            if only_seed is None:
                done = sum(1 for _ in (RESULT_ROOT / dataset).glob(
                    f"*/fold_{fold:02d}/{subj}/seed_*/ratio_*/completed.flag"))
                if done >= 400:
                    continue
            item = base.load_subject(h5_dir, subj, prior.expected_n_samples, cfg.get("modality", "eeg"),
                                     cfg.get("label_key", "labels"), override_eigval=override)
            if item is None: continue
            bundle = base.stack_subjects([item]); idx = np.arange(len(item["labels"]))
            n_tok = prior.expected_n_samples // base.PATCH_SIZE
            of, pt, yy = base.precompute_features(bundle, idx, backbone, prior, n_tok,
                                                   item["eigval"], device, cfg.get("batch_size", 32))
            of, pt, yy = of.to(device), pt.to(device), yy.to(device)
            fold_dir = ref / f"fold_{fold:02d}"
            obs0, pri0 = build_models(cfg, device, prior, backbone, lm_dim, fold_dir)
            initial_heads = {
                "BrainOmni": {k: v.detach().clone() for k, v in obs0.cls_head.state_dict().items()},
                "ours": {k: v.detach().clone() for k, v in pri0.cls_head.state_dict().items()},
            }
            frozen_hash = {"BrainOmni": state_hash(obs0.encoder) + state_hash(obs0.ghead),
                           "ours": state_hash(pri0.encoder) + state_hash(pri0.ghead)}
            seeds = [only_seed] if only_seed is not None else range(50)
            y_np = yy.cpu().numpy()
            for seed in seeds:
                cal, test = stratified_nested(y_np, seed)
                split_obj = {"dataset": dataset, "fold": fold, "target_subject": subj, "seed": seed,
                             "n_total": len(y_np), "test_indices": test, "calibration": cal,
                             "class_counts": {"all": np.bincount(y_np, minlength=cfg["n_classes"]).tolist(),
                                              "test": np.bincount(y_np[test], minlength=cfg["n_classes"]).tolist()}}
                split_path = RESULT_ROOT / dataset / "splits" / f"fold_{fold:02d}" / subj / f"seed_{seed:02d}.json"
                atomic_json(split_path, split_obj)
                te = torch.as_tensor(test, device=device)
                for method, template, feat in (("ours", pri0, pt), ("BrainOmni", obs0, of)):
                    template.cls_head.load_state_dict(initial_heads[method])
                    zero_dir = RESULT_ROOT / dataset / method / f"fold_{fold:02d}" / subj / f"seed_{seed:02d}" / "ratio_000"
                    if not (zero_dir / "completed.flag").exists():
                        pred = predict(template, feat[te], method)
                        atomic_json(zero_dir / "metrics.json", {**metrics(y_np[test], pred), "n_test": len(test)})
                        (zero_dir / "completed.flag").write_text("ok\n")
                    for ratio in (10, 30, 50):
                        out = RESULT_ROOT / dataset / method / f"fold_{fold:02d}" / subj / f"seed_{seed:02d}" / f"ratio_{ratio:03d}"
                        if (out / "completed.flag").exists(): continue
                        template.cls_head.load_state_dict(initial_heads[method])
                        model = template; tr = torch.as_tensor(cal[str(ratio)], device=device)
                        before = frozen_hash[method]
                        train_head(model, feat[tr], yy[tr], seed, epochs, min(cfg.get("batch_size",32), len(tr)), cfg.get("lr_head",1e-3))
                        after = (state_hash(model.encoder) + state_hash(model.ghead))
                        if before != after: raise RuntimeError("Frozen parameters changed")
                        pred = predict(model, feat[te], method)
                        out.mkdir(parents=True, exist_ok=True)
                        torch.save(model.cls_head.state_dict(), out / "head.pth")
                        atomic_json(out / "metrics.json", {**metrics(y_np[test], pred), "n_calibration": len(tr),
                                    "n_test": len(test), "ratio": ratio/100, "seed": seed,
                                    "frozen_hash_verified": True})
                        (out / "completed.flag").write_text("ok\n")
            del of, pt, yy, obs0, pri0; torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int); ap.add_argument("--fold", type=int); ap.add_argument("--subject")
    args = ap.parse_args(); device = torch.device(args.device)
    for d in args.datasets: run_dataset(d, device, args.epochs, args.seed, args.fold, args.subject)

if __name__ == "__main__": main()
