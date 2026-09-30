












import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR    = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT  = os.path.normpath(os.path.join(SCRIPT_DIR, "../../"))
BRAINOMNI_SRC = os.path.join(PROJECT_ROOT, "weights/BrainOmni/BrainOmni-main")
TINY_CKPT_DIR = os.path.join(PROJECT_ROOT, "weights/BrainOmni/BrainOmni/tiny")

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, BRAINOMNI_SRC)
from eeg_mae_model import EEGMAEEncoder, PATCH_SIZE

RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_DIR = SCRIPT_DIR
LOG_PATH = os.path.join(_LOG_DIR, f"classify_poe_v5_kfold_{RUN_ID}.log")

def _setup_logging(out_dir: str):
    global LOG_PATH
    LOG_PATH = os.path.join(out_dir, f"classify_poe_v5_kfold_{RUN_ID}.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(sh)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_frozen_mae(ckpt_path: str, device: torch.device) -> EEGMAEEncoder:
    ckpt = torch.load(ckpt_path, map_location=device)
    enc  = EEGMAEEncoder(
        n_modes    = ckpt["n_modes"],
        patch_size = ckpt.get("patch_size", PATCH_SIZE),
        n_samples  = ckpt.get("n_samples", 1024),
        d_model    = ckpt["d_model"],
        n_heads    = ckpt.get("n_heads", 8),
        n_layers   = ckpt.get("n_layers", 6),
        dropout    = 0.0,
    ).to(device)

    state = ckpt["encoder"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    enc.load_state_dict(state)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False

    enc.expected_n_modes   = ckpt["n_modes"]
    enc.expected_n_samples = ckpt.get("n_samples", 1024)
    enc.d_model_dim        = ckpt["d_model"]
    log.info(f"  MAE encoder loaded & frozen: n_modes={ckpt['n_modes']}"
             f"  n_samples={enc.expected_n_samples}  d_model={ckpt['d_model']}"
             f"  epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss',float('nan')):.4f}")
    return enc


def load_brainomni_tiny(device: torch.device):
    from brainomni.model import BrainOmni
    with open(os.path.join(TINY_CKPT_DIR, "model_cfg.json")) as f:
        cfg = json.load(f)
    model = BrainOmni(**cfg)
    ckpt  = torch.load(os.path.join(TINY_CKPT_DIR, "BrainOmni.pt"), map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    lm_dim = cfg["lm_dim"]
    log.info(f"  BrainOmni-tiny loaded & frozen  lm_dim={lm_dim}")
    return model, lm_dim


class ObsEncoder(nn.Module):
    def __init__(self, backbone, lm_dim: int, n_tok: int, d_model: int):
        super().__init__()
        self.backbone = backbone
        self.n_tok    = n_tok
        self.adapter  = nn.Sequential(
            nn.Linear(lm_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x_raw, eeg_pos, sensor_type):
        with torch.no_grad():
            enc = self.backbone.encode(x_raw, eeg_pos, sensor_type)
        feat = enc.float().mean(dim=1)
        feat = feat.permute(0, 2, 1)
        feat = F.adaptive_avg_pool1d(feat, self.n_tok)
        feat = feat.permute(0, 2, 1)
        return self.adapter(feat)


class GaussHead(nn.Module):
    def __init__(self, d_in: int, lat_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden, lat_dim)
        self.lv_head = nn.Linear(hidden, lat_dim)

    def forward(self, x):
        h  = self.net(x)
        mu = self.mu_head(h)
        lv = self.lv_head(h).clamp(-8, 4)
        return mu, lv


class ClsHead(nn.Module):
    def __init__(self, lat_dim: int, hidden: int = 256,
                 n_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(lat_dim),
            nn.Linear(lat_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, mu):
        return self.net(mu.mean(dim=1))


class ExpertModel(nn.Module):
    def __init__(self, encoder, ghead: GaussHead, cls_head: ClsHead,
                 is_obs: bool = False):
        super().__init__()
        self.encoder  = encoder
        self.ghead    = ghead
        self.cls_head = cls_head
        self.is_obs   = is_obs

    def encode(self, x_modes=None, x_raw=None, eeg_pos=None, sensor_type=None):
        if self.is_obs:
            tok = self.encoder(x_raw, eeg_pos, sensor_type)
        else:
            tok = self.encoder(x_modes)
        mu, lv = self.ghead(tok)
        return mu, lv

    def forward(self, x_modes=None, x_raw=None, eeg_pos=None, sensor_type=None):
        mu, lv = self.encode(x_modes, x_raw, eeg_pos, sensor_type)
        logits = self.cls_head(mu)
        return logits, mu, lv


class GPUDataset:
    def __init__(self, X_group, X_indiv, X_raw, eeg_pos, sensor_type, y, device=None):
        self.X_group     = X_group.cpu()
        self.X_indiv     = X_indiv.cpu()
        self.X_raw       = X_raw.cpu()
        self.eeg_pos     = eeg_pos.cpu()
        self.sensor_type = sensor_type.cpu()
        self.y           = y.cpu()
        self.N           = len(y)
        self.device      = device or (y.device if y.is_cuda else torch.device("cpu"))

    def iter_batches(self, batch_size: int, shuffle: bool = True):
        dev = self.device
        idx = torch.randperm(self.N) if shuffle else torch.arange(self.N)
        for s in range(0, self.N, batch_size):
            b = idx[s: s + batch_size]
            yield (self.X_group[b].to(dev),     self.X_indiv[b].to(dev),
                   self.X_raw[b].to(dev),        self.eeg_pos[b].to(dev),
                   self.sensor_type[b].to(dev),  self.y[b].to(dev))


def train_expert(model: ExpertModel, train_ds: GPUDataset, val_ds: GPUDataset,
                 epochs: int, batch_size: int, lr_adapter: float, lr_head: float,
                 patience: int, kl_weight: float, label: str,
                 alpha: float = 0.0) -> float:
    trainable = []
    if model.is_obs:
        trainable.append({"params": model.encoder.adapter.parameters(),
                          "lr": lr_adapter, "weight_decay": 1e-3})
    trainable.append({"params": list(model.ghead.parameters()) +
                                list(model.cls_head.parameters()),
                      "lr": lr_head, "weight_decay": 1e-3})

    opt = torch.optim.AdamW(trainable)
    steps_per_epoch = max(1, (train_ds.N + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in trainable],
        steps_per_epoch=steps_per_epoch,
        epochs=epochs, pct_start=0.1, anneal_strategy="cos",
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler    = torch.amp.GradScaler("cuda")

    best_acc   = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        if model.is_obs:
            model.encoder.backbone.eval()
        else:
            model.encoder.eval()

        c = n = 0
        for xg, xi, xr, pos, stype, yb in train_ds.iter_batches(batch_size):
            xm = xg

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, mu, lv = model(xm, xr, pos, stype)
                kl   = 0.5 * (lv.exp() + mu.pow(2) - 1 - lv).mean()
                loss = criterion(logits, yb) + kl_weight * kl
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            c += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
        train_acc = c / n

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            for xg, xi, xr, pos, stype, yb in val_ds.iter_batches(batch_size, shuffle=False):
                vl, _, _ = model(xg, xr, pos, stype)
                val_correct += (vl.argmax(1) == yb).sum().item()
                val_total   += len(yb)
        val_acc = val_correct / max(val_total, 1)

        if val_acc > best_acc + 1e-4:
            best_acc   = val_acc
            best_state = deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1 or epoch == epochs or no_improve == patience:
            log.info(f"    [{label}] Epoch {epoch:4d}/{epochs}"
                     f"  train={train_acc:.4f}  val={val_acc:.4f}"
                     f"  best={best_acc:.4f}")
        if no_improve >= patience:
            log.info(f"    [{label}] Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                               for k, v in best_state.items()})
    return best_acc


def eval_ensemble(obs_model: ExpertModel, prior_model: ExpertModel,
                  val_ds: GPUDataset, n_classes: int, batch_size: int = 64) -> dict:
    obs_model.eval()
    prior_model.eval()

    all_log_obs, all_log_pri, all_y = [], [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for xg, xi, xr, pos, stype, yb in val_ds.iter_batches(batch_size, shuffle=False):
            lo, _, _ = obs_model(xg, xr, pos, stype)
            lp, _, _ = prior_model(xg, xr, pos, stype)
            all_log_obs.append(F.log_softmax(lo.float(), dim=-1))
            all_log_pri.append(F.log_softmax(lp.float(), dim=-1))
            all_y.append(yb)

    log_obs = torch.cat(all_log_obs, dim=0)
    log_pri = torch.cat(all_log_pri, dim=0)
    yv      = torch.cat(all_y, dim=0)

    prob_obs = log_obs.exp()
    prob_pri = log_pri.exp()
    chance   = 1.0 / n_classes

    w_obs = (prob_obs.max(1).values - chance).clamp(min=1e-4)
    w_pri = (prob_pri.max(1).values - chance).clamp(min=0.0)
    w_sum = (w_obs + w_pri).clamp(min=1e-8)

    wo = (w_obs / w_sum).unsqueeze(1)
    wp = (w_pri / w_sum).unsqueeze(1)

    log_fused = wo * log_obs + wp * log_pri
    acc_poe   = (log_fused.argmax(1) == yv).float().mean().item()

    prob_arith = (prob_obs + prob_pri) / 2
    acc_arith  = (prob_arith.argmax(1) == yv).float().mean().item()

    avg_w_obs = w_obs.mean().item()
    avg_w_pri = w_pri.mean().item()
    log.info(f"    Avg confidence margin — obs: {avg_w_obs:.3f}  prior: {avg_w_pri:.3f}"
             f"  (prior weight share: {avg_w_pri/(avg_w_obs+avg_w_pri+1e-8):.2%})")
    log.info(f"    Arithmetic avg acc: {acc_arith:.4f}  |  Conf-weighted PoE: {acc_poe:.4f}")

    return {"poe": acc_poe, "arith": acc_arith}


def load_all_subjects(h5_dir: str, subjects: list, n_modes: int, n_samples: int,
                      modality: str = "eeg"):
    all_modes, all_raw, all_pos, all_stype, all_labels = [], [], [], [], []
    for subj in subjects:
        h5_path = os.path.join(h5_dir, f"{subj}.h5")
        if not os.path.isfile(h5_path):
            log.warning(f"  {subj}: not found — skip")
            continue
        with h5py.File(h5_path, "r") as f:
            modes  = torch.tensor(f[f"{modality}_modes"][:],       dtype=torch.float32)
            raw    = torch.tensor(f[f"{modality}_raw"][:],          dtype=torch.float32)
            labels = torch.tensor(f["labels"][:],                   dtype=torch.long)
            pos    = torch.tensor(f[f"{modality}_pos"][:],          dtype=torch.float32)
            stype  = torch.tensor(f[f"{modality}_sensor_type"][:],  dtype=torch.long)
        modes = modes[:, :n_modes, :n_samples]
        raw   = raw[:,   :,         :n_samples]
        N = len(labels)
        all_modes.append(modes)
        all_raw.append(raw)
        all_labels.append(labels)
        all_pos.append(pos.unsqueeze(0).expand(N, -1, -1))
        all_stype.append(stype.unsqueeze(0).expand(N, -1))
        log.info(f"  {subj}: {N} trials loaded")

    modes  = torch.cat(all_modes,  dim=0)
    raw    = torch.cat(all_raw,    dim=0)
    labels = torch.cat(all_labels, dim=0)
    pos_n  = torch.cat(all_pos,    dim=0)
    stype_n= torch.cat(all_stype,  dim=0)

    log.info(f"  Total pooled: {len(labels)} trials")
    return modes, modes, raw, pos_n, stype_n, labels


def run_kfold(h5_dir, subjects, mae_enc, backbone, lm_dim, device, args):
    log.info(f"\n{'─'*60}")
    log.info(f"  K-fold CV with all {len(subjects)} subjects pooled, n_folds={args.n_folds}")
    log.info(f"{'─'*60}")

    n_modes   = mae_enc.expected_n_modes
    n_samples = mae_enc.expected_n_samples
    d_model   = mae_enc.d_model_dim
    n_tok     = n_samples // PATCH_SIZE

    mg, mi, raw, pos_n, stype_n, labels = load_all_subjects(
        h5_dir, subjects, n_modes, n_samples, modality=args.modality)
    N = len(labels)
    log.info(f"  Total N={N}  n_tok={n_tok}  d_model={d_model}  K={n_modes}")
    log.info(f"  Class dist (all): {torch.bincount(labels).tolist()}")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(skf.split(np.zeros(N), labels.numpy()))

    obs_accs, prior_accs, poe_accs, arith_accs = [], [], [], []

    for fold_id, (tr_idx_np, te_idx_np) in enumerate(folds):
        if args.fold is not None and fold_id != args.fold:
            continue
        tr_idx = torch.tensor(tr_idx_np, dtype=torch.long)
        te_idx = torch.tensor(te_idx_np, dtype=torch.long)
        log.info(f"\n========== FOLD {fold_id+1}/{args.n_folds} ==========")
        log.info(f"  train={len(tr_idx)}  test={len(te_idx)}")

        train_ds = GPUDataset(mg[tr_idx], mi[tr_idx], raw[tr_idx],
                              pos_n[tr_idx], stype_n[tr_idx], labels[tr_idx], device=device)
        test_ds  = GPUDataset(mg[te_idx], mi[te_idx], raw[te_idx],
                              pos_n[te_idx], stype_n[te_idx], labels[te_idx], device=device)


        log.info(f"  [obs] training")
        obs_enc   = ObsEncoder(backbone, lm_dim, n_tok, d_model).to(device)
        obs_ghead = GaussHead(d_model, args.lat_dim).to(device)
        obs_cls   = ClsHead(args.lat_dim, n_classes=args.n_classes, dropout=args.dropout).to(device)
        obs_model = ExpertModel(obs_enc, obs_ghead, obs_cls, is_obs=True).to(device)
        obs_best = train_expert(obs_model, train_ds, test_ds,
                                args.epochs, args.batch_size,
                                args.lr_adapter, args.lr_head,
                                args.patience, args.kl_weight, f"obs.f{fold_id}")

        log.info(f"  [prior] training")
        prior_ghead = GaussHead(d_model, args.lat_dim).to(device)
        prior_cls   = ClsHead(args.lat_dim, n_classes=args.n_classes, dropout=args.dropout).to(device)
        prior_model = ExpertModel(mae_enc, prior_ghead, prior_cls, is_obs=False).to(device)
        prior_best  = train_expert(prior_model, train_ds, test_ds,
                                   args.epochs, args.batch_size,
                                   args.lr_adapter, args.lr_head,
                                   args.patience, args.kl_weight, f"pri.f{fold_id}")

        fuse_res = eval_ensemble(obs_model, prior_model, test_ds, args.n_classes)

        log.info(f"  FOLD {fold_id+1}: obs={obs_best:.4f}  prior={prior_best:.4f}  "
                 f"poe(arith)={fuse_res['arith']:.4f}  poe(conf)={fuse_res['poe']:.4f}")
        obs_accs.append(obs_best)
        prior_accs.append(prior_best)
        arith_accs.append(fuse_res["arith"])
        poe_accs.append(fuse_res["poe"])


        del obs_model, prior_model, train_ds, test_ds
        torch.cuda.empty_cache()

    def stat(v):
        a = np.array(v)
        return {"mean": float(a.mean()), "std": float(a.std()), "values": [float(x) for x in v]}

    summary = {
        "obs_only":  stat(obs_accs),
        "prior_only": stat(prior_accs),
        "poe_arith": stat(arith_accs),
        "poe_full":  stat(poe_accs),
        "n_folds":   len(obs_accs),
        "K_modes":   int(n_modes),
        "N_total":   int(N),
        "n_classes": int(args.n_classes),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_dir",       type=str, required=True)
    parser.add_argument("--encoder_ckpt", type=str, required=True)
    parser.add_argument("--subjects",     type=str, nargs="+",
                        default=[f"sub-{i:03d}" for i in range(1, 10)])
    parser.add_argument("--train_ratio",  type=float, default=0.8)
    parser.add_argument("--epochs",       type=int,   default=300)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr_adapter",   type=float, default=5e-4)
    parser.add_argument("--lr_head",      type=float, default=1e-3)
    parser.add_argument("--lat_dim",      type=int,   default=256)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--patience",     type=int,   default=80)
    parser.add_argument("--kl_weight",    type=float, default=1e-3)
    parser.add_argument("--n_classes",    type=int,   default=4)
    parser.add_argument("--out_dir",      type=str,   default=None)
    parser.add_argument("--device",       type=str,   default="cuda:0")
    parser.add_argument("--n_folds",      type=int,   default=5)
    parser.add_argument("--fold",         type=int,   default=None,
                        help="If set, run only this fold index (0..n_folds-1).")
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--tag",          type=str,   default="all")
    parser.add_argument("--results_json", type=str,   default=None,
                        help="Path to write per-fold result JSON.")
    parser.add_argument("--modality",     type=str,   default="eeg",
                        choices=["eeg", "meg"],
                        help="Reads {modality}_modes / {modality}_raw / {modality}_pos / {modality}_sensor_type from H5.")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir else SCRIPT_DIR
    os.makedirs(out_dir, exist_ok=True)
    _setup_logging(out_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    log.info(f"Run ID       : {RUN_ID}")
    log.info(f"Log          : {LOG_PATH}")
    log.info(f"Device       : {device}")
    log.info(f"Subjects     : {args.subjects}")
    log.info(f"Encoder ckpt : {args.encoder_ckpt}")
    log.info(f"n_classes    : {args.n_classes}")
    log.info(f"train_ratio  : {args.train_ratio}")
    log.info(f"Fusion       : confidence-weighted log-product PoE")

    log.info("\nLoading frozen MAE encoder …")
    mae_enc = load_frozen_mae(args.encoder_ckpt, device)

    log.info("Loading frozen BrainOmni-tiny …")
    backbone, lm_dim = load_brainomni_tiny(device)

    res = run_kfold(
        args.h5_dir, args.subjects, mae_enc, backbone, lm_dim, device, args
    )

    log.info("\n" + "=" * 75)
    log.info(f"FINAL SUMMARY [{args.tag}]  ({res['n_folds']}-fold CV, conf-weighted PoE)")
    log.info("=" * 75)
    for k in ("obs_only", "prior_only", "poe_arith", "poe_full"):
        s = res[k]
        log.info(f"  {k:<12s} = {s['mean']*100:.2f}% ± {s['std']*100:.2f}%   "
                 f"folds={[f'{v*100:.1f}' for v in s['values']]}")
    log.info(f"  Chance level = {100/args.n_classes:.1f}%")

    if args.results_json is None:
        args.results_json = os.path.join(out_dir, f"results_{args.tag}_{RUN_ID}.json")
    res["meta"] = {"tag": args.tag, "h5_dir": args.h5_dir,
                   "encoder_ckpt": args.encoder_ckpt, "subjects": args.subjects,
                   "RUN_ID": RUN_ID, "epochs": args.epochs, "seed": args.seed}
    with open(args.results_json, "w") as f:
        json.dump(res, f, indent=2)
    log.info(f"\nResults JSON → {args.results_json}")
    log.info(f"Log saved   → {LOG_PATH}")


if __name__ == "__main__":
    main()
