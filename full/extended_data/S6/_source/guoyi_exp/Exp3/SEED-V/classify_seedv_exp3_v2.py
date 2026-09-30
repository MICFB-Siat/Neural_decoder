































from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))
import argparse
import copy
import glob
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
from sklearn.metrics import balanced_accuracy_score, recall_score

SCRIPT_DIR    = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT  = str(_LOCAL_SOURCE / '')
BRAINOMNI_SRC = os.path.join(PROJECT_ROOT, "weights/BrainOmni/BrainOmni-main")
TINY_CKPT_DIR = os.path.join(PROJECT_ROOT, "weights/BrainOmni/BrainOmni/tiny")
FULLRUN_DIR   = os.path.join(PROJECT_ROOT, "data_check_20260507/full_run_5fold")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, BRAINOMNI_SRC)
sys.path.insert(0, FULLRUN_DIR)

from models.prior.eeg_meg_prior_eigenval3 import EEGMEGPriorEigenval3
from models.prior.eeg_meg_prior_eigenval import PATCH_SIZE

RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(SCRIPT_DIR, f"classify_seedv_exp3_{RUN_ID}.log")


def _setup_logging(out_dir: str):
    global LOG_PATH
    LOG_PATH = os.path.join(out_dir, "log.txt")
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





class _HeteroKDataset:
    def __init__(self, chunks, names):
        self.chunks = chunks
        self.names  = names
        self.N      = sum(c.shape[0] for c in chunks)
        self.Ks     = [c.shape[1] for c in chunks]
        self.T      = chunks[0].shape[2] if chunks else 0

    def iter_batches(self, batch_size, shuffle=True, rng=None):
        rng = rng or np.random.default_rng()
        order = rng.permutation(len(self.chunks)) if shuffle else np.arange(len(self.chunks))
        for ci in order:
            c = self.chunks[ci]
            n = c.shape[0]
            sub_idx = torch.randperm(n, device=c.device) if shuffle else torch.arange(n, device=c.device)
            for s in range(0, n, batch_size):
                yield c[sub_idx[s: s + batch_size]], ci


def _load_modes_hetero(h5_dir, modality, device):
    key = f"{modality}_modes"
    files = sorted(glob.glob(os.path.join(h5_dir, "sub-*.h5")))
    chunks, names = [], []
    for fp in files:
        with h5py.File(fp, "r") as f:
            if key not in f:
                continue
            arr = f[key][:].astype(np.float32)
        chunks.append(torch.from_numpy(arr).to(device))
        names.append(os.path.splitext(os.path.basename(fp))[0])
    if not chunks:
        raise RuntimeError(f"No '{key}' arrays found in {h5_dir}")
    return _HeteroKDataset(chunks, names)


def auto_pretrain_prior(h5_dir, modality, out_dir, device, hp, tag):






    raise NotImplementedError(
        "auto_pretrain disabled for v3 encoder; pretrain externally with "
        "pretrain_v3_eegmeg.py and pass --encoder_ckpt.")

    from eeg_mae_model_v3final import (EEGMAEEncoderV3Final, EEGMAEDecoderV3Final,
                                        brainomni_2d_mask, patchify)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mae_kfree_{tag}_{modality}.pt")

    torch.manual_seed(hp["seed"]); np.random.seed(hp["seed"])
    rng = np.random.default_rng(hp["seed"])

    log.info("\n" + "─" * 75)
    log.info(f"AUTO-PRETRAIN  →  {out_path}")
    log.info("─" * 75)
    ds_all = _load_modes_hetero(h5_dir, modality, device)
    log.info(f"  Total trials = {ds_all.N}  Ks = {ds_all.Ks}  T = {ds_all.T}")

    train_chunks, val_chunks, names = [], [], []
    for c, nm in zip(ds_all.chunks, ds_all.names):
        n = c.shape[0]
        n_val = max(1, int(round(n * hp["val_ratio"])))
        perm = torch.randperm(n, device=c.device)
        val_chunks.append(c[perm[:n_val]])
        train_chunks.append(c[perm[n_val:]])
        names.append(nm)
    train_ds = _HeteroKDataset(train_chunks, names)
    val_ds   = _HeteroKDataset(val_chunks,  names)

    T  = ds_all.T
    NP = T // PATCH_SIZE
    log.info(f"  NP per trial = {NP}  patch_size = {PATCH_SIZE}")

    encoder = EEGMAEEncoderV3Final(
        patch_size=PATCH_SIZE, d_model=hp["d_model"], n_heads=8,
        n_factor_layers=hp["n_factor_layers"],
        n_time_layers=hp["n_time_layers"],
        use_mode_signature=False,
    ).to(device)
    decoder = EEGMAEDecoderV3Final(
        d_model=hp["d_model"], d_dec=hp["d_model"] // 2,
        n_heads=4, n_layers=hp["n_dec_layers"],
        patch_size=PATCH_SIZE,
    ).to(device)
    log.info(f"  Encoder {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M  "
             f"Decoder {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M")

    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=0.05, betas=(0.9, 0.95))
    warmup = max(1, int(hp["epochs"] * 0.05))

    def lr_lambda(ep):
        if ep < warmup:
            return ep / warmup
        t = (ep - warmup) / max(1, hp["epochs"] - warmup)
        return 0.5 * (1 + np.cos(np.pi * t))

    sched  = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    best, best_epoch = float("inf"), 0
    for epoch in range(1, hp["epochs"] + 1):
        encoder.train(); decoder.train()
        tl = tn = 0
        for x, _ in train_ds.iter_batches(hp["batch_size"], rng=rng):
            B, K, _ = x.shape
            mask = brainomni_2d_mask(B, K, NP, hp["mask_ratio"], device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                tgt  = patchify(x, PATCH_SIZE)
                enc  = encoder.encode_full_grid(x, None, mask)
                pred = decoder(enc)
                m = mask.float().unsqueeze(-1)
                loss = ((pred.float() - tgt.float()) ** 2 * m).sum() / (m.sum() * PATCH_SIZE + 1e-8)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
            tl += loss.item() * B; tn += B
        sched.step()
        train_loss = tl / max(tn, 1)

        encoder.eval(); decoder.eval()
        vl = vn = 0
        with torch.no_grad():
            for x, _ in val_ds.iter_batches(hp["batch_size"], shuffle=False):
                B, K, _ = x.shape
                mask = brainomni_2d_mask(B, K, NP, hp["mask_ratio"], device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    tgt  = patchify(x, PATCH_SIZE)
                    enc  = encoder.encode_full_grid(x, None, mask)
                    pred = decoder(enc)
                    m = mask.float().unsqueeze(-1)
                    loss = ((pred.float() - tgt.float()) ** 2 * m).sum() / (m.sum() * PATCH_SIZE + 1e-8)
                vl += loss.item() * B; vn += B
        val_loss = vl / max(vn, 1)

        if val_loss < best:
            best, best_epoch = val_loss, epoch
            torch.save({
                "encoder": encoder.state_dict(),
                "epoch": epoch, "val_loss": val_loss, "run_id": RUN_ID,
                "n_modes": min(ds_all.Ks),
                "n_samples": T, "d_model": hp["d_model"],
                "n_factor_layers": hp["n_factor_layers"],
                "n_time_layers": hp["n_time_layers"],
                "patch_size": PATCH_SIZE,
                "mode_signature": "none",
                "modality": modality, "tag": tag,
            }, out_path)
        if epoch % hp["log_every"] == 0 or epoch == 1 or epoch == hp["epochs"]:
            log.info(f"  PRE epoch {epoch:4d}/{hp['epochs']}  train={train_loss:.6f}  "
                     f"val={val_loss:.6f}  best={best:.6f}@{best_epoch}")

    log.info(f"\nAuto-pretrain done. best_val={best:.6f} @ epoch {best_epoch}")
    log.info("─" * 75 + "\n")
    del encoder, decoder, train_ds, val_ds, ds_all
    torch.cuda.empty_cache()
    return out_path





def load_prior_from_ckpt(ckpt_path, device, trainable):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = EEGMEGPriorEigenval3(
        patch_size      = ckpt.get("patch_size", PATCH_SIZE),
        d_model         = ckpt["d_model"],
        n_heads         = 8,
        n_factor_layers = ckpt.get("n_factor_layers", 2),
        n_time_layers   = ckpt.get("n_time_layers", 4),
        dropout         = 0.0,
    ).to(device)
    state = ckpt["encoder"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if missing or unexpected:
        log.warning(f"  load_state_dict — missing={len(missing)}  unexpected={len(unexpected)}")
    for p in enc.parameters():
        p.requires_grad = bool(trainable)
    enc.eval()

    n_samples = ckpt.get("n_samples")
    if n_samples is None:
        dt = ckpt.get("dataset_T", {})
        n_samples = dt.get("SEED-V", dt.get(next(iter(dt), None), 2560)) if dt else 2560
    enc.expected_n_samples = int(n_samples)
    enc.d_model_dim        = ckpt["d_model"]
    log.info(f"  prior loaded ({'trainable' if trainable else 'frozen'}): "
             f"d_model={ckpt['d_model']}  n_samples={enc.expected_n_samples}  "
             f"epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss', float('nan'))}")
    return enc


def load_brainomni_tiny(device, finetune):
    from brainomni.model import BrainOmni
    with open(os.path.join(TINY_CKPT_DIR, "model_cfg.json")) as f:
        cfg = json.load(f)
    model = BrainOmni(**cfg)
    ckpt  = torch.load(os.path.join(TINY_CKPT_DIR, "BrainOmni.pt"), map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device)
    if finetune:
        for p in model.parameters():
            p.requires_grad = True
        for p in model.tokenizer.parameters():
            p.requires_grad = False
        model.tokenizer.eval()
    else:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
    log.info(f"  BrainOmni-tiny loaded  finetune={'ON' if finetune else 'OFF'}  lm_dim={cfg['lm_dim']}")
    return model, cfg["lm_dim"]





class ObsEncoder(nn.Module):
    def __init__(self, backbone, lm_dim, n_tok, d_model, backbone_trainable):
        super().__init__()
        self.backbone = backbone
        self.n_tok    = n_tok
        self.backbone_trainable = backbone_trainable
        self.adapter  = nn.Sequential(
            nn.Linear(lm_dim, d_model), nn.LayerNorm(d_model), nn.GELU())

    def forward(self, x_raw, eeg_pos, sensor_type):
        if self.backbone_trainable:
            enc = self.backbone.encode(x_raw, eeg_pos, sensor_type)
        else:
            with torch.no_grad():
                enc = self.backbone.encode(x_raw, eeg_pos, sensor_type)
        feat = enc.float().mean(dim=1)
        feat = feat.permute(0, 2, 1)
        feat = F.adaptive_avg_pool1d(feat, self.n_tok)
        feat = feat.permute(0, 2, 1)
        return self.adapter(feat)


class GaussHead(nn.Module):
    def __init__(self, d_in, lat_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        self.mu_head = nn.Linear(hidden, lat_dim)
        self.lv_head = nn.Linear(hidden, lat_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu_head(h), self.lv_head(h).clamp(-8, 4)


class ClsHead(nn.Module):
    def __init__(self, lat_dim, hidden=256, n_classes=5, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(lat_dim),
            nn.Linear(lat_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes))

    def forward(self, mu):
        return self.net(mu.mean(dim=1))


class ExpertModel(nn.Module):
    def __init__(self, encoder, ghead, cls_head, is_obs, encoder_trainable):
        super().__init__()
        self.encoder, self.ghead, self.cls_head = encoder, ghead, cls_head
        self.is_obs            = is_obs
        self.encoder_trainable = encoder_trainable

        self.register_buffer("eigval", torch.zeros(1), persistent=False)

    def set_eigval(self, eigval: torch.Tensor):

        self.eigval = eigval.to(next(self.parameters()).device)

    def encode(self, x_modes=None, x_raw=None, eeg_pos=None, sensor_type=None):
        if self.is_obs:
            tok = self.encoder(x_raw, eeg_pos, sensor_type)
        elif self.encoder_trainable:
            tok = self.encoder.encode_pooled(x_modes, self.eigval)
        else:
            with torch.no_grad():
                tok = self.encoder.encode_pooled(x_modes, self.eigval)
        return self.ghead(tok)

    def forward(self, x_modes=None, x_raw=None, eeg_pos=None, sensor_type=None):
        mu, lv = self.encode(x_modes, x_raw, eeg_pos, sensor_type)
        return self.cls_head(mu), mu, lv


class GPUDataset:
    def __init__(self, X_modes, X_raw, eeg_pos, sensor_type, y, device=None):
        self.X_modes     = X_modes.cpu()
        self.X_raw       = X_raw.cpu()
        self.eeg_pos     = eeg_pos.cpu()
        self.sensor_type = sensor_type.cpu()
        self.y           = y.cpu()
        self.N           = len(y)
        self.device      = device or torch.device("cpu")

    def iter_batches(self, batch_size, shuffle=True):
        dev = self.device
        idx = torch.randperm(self.N) if shuffle else torch.arange(self.N)
        for s in range(0, self.N, batch_size):
            b = idx[s: s + batch_size]
            yield (self.X_modes[b].to(dev), self.X_raw[b].to(dev),
                   self.eeg_pos[b].to(dev), self.sensor_type[b].to(dev),
                   self.y[b].to(dev))





def _set_train_modes(model):
    model.train()
    if model.is_obs:
        if model.encoder.backbone_trainable:
            model.encoder.backbone.train()
            model.encoder.backbone.tokenizer.eval()
        else:
            model.encoder.backbone.eval()
    else:
        if model.encoder_trainable:
            model.encoder.train()
        else:
            model.encoder.eval()


def train_expert(model, train_ds, val_ds, epochs, batch_size,
                 lr_adapter, lr_head, lr_backbone, patience, kl_weight, label):
    trainable = []
    if model.is_obs:
        trainable.append({"params": model.encoder.adapter.parameters(),
                          "lr": lr_adapter, "weight_decay": 1e-3})
        if model.encoder.backbone_trainable:
            bb = [p for p in model.encoder.backbone.parameters() if p.requires_grad]
            if bb: trainable.append({"params": bb, "lr": lr_backbone, "weight_decay": 1e-4})
    else:
        if model.encoder_trainable:
            ep = [p for p in model.encoder.parameters() if p.requires_grad]
            if ep: trainable.append({"params": ep, "lr": lr_backbone, "weight_decay": 1e-4})
    trainable.append({"params": list(model.ghead.parameters()) +
                                list(model.cls_head.parameters()),
                      "lr": lr_head, "weight_decay": 1e-3})

    opt = torch.optim.AdamW(trainable)
    steps_per_epoch = max(1, (train_ds.N + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in trainable],
        steps_per_epoch=steps_per_epoch, epochs=epochs,
        pct_start=0.1, anneal_strategy="cos")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler    = torch.amp.GradScaler("cuda")

    best_acc, best_state, no_improve = 0.0, None, 0
    for epoch in range(1, epochs + 1):
        _set_train_modes(model)
        c = n = 0
        for xm, xr, pos, stype, yb in train_ds.iter_batches(batch_size):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, mu, lv = model(xm, xr, pos, stype)
                kl   = 0.5 * (lv.exp() + mu.pow(2) - 1 - lv).mean()
                loss = criterion(logits, yb) + kl_weight * kl
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for g in trainable for p in g["params"]], 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            c += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
        train_acc = c / max(n, 1)

        model.eval()
        vc = vt = 0
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for xm, xr, pos, stype, yb in val_ds.iter_batches(batch_size, shuffle=False):
                vl, _, _ = model(xm, xr, pos, stype)
                vc += (vl.argmax(1) == yb).sum().item()
                vt += len(yb)
        val_acc = vc / max(vt, 1)

        if val_acc > best_acc + 1e-4:
            best_acc   = val_acc
            best_state = deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1 or epoch == epochs or no_improve == patience:
            log.info(f"    [{label}] Epoch {epoch:4d}/{epochs}"
                     f"  train={train_acc:.4f}  val={val_acc:.4f}  best={best_acc:.4f}")
        if no_improve >= patience:
            log.info(f"    [{label}] Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                               for k, v in best_state.items()})
    return best_acc, (best_state or {})


def _per_class_recall(yt, yp, n_classes):
    rec = recall_score(yt, yp, labels=list(range(n_classes)),
                       average=None, zero_division=0).tolist()
    return rec, "  ".join([f"c{i}={r:.3f}" for i, r in enumerate(rec)])


def eval_branch_detailed(model, ds, batch_size, n_classes, label):
    model.eval()
    preds, ys = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for xm, xr, pos, stype, yb in ds.iter_batches(batch_size, shuffle=False):
            logits, _, _ = model(xm, xr, pos, stype)
            preds.append(logits.argmax(1).cpu())
            ys.append(yb.cpu())
    yp = torch.cat(preds).numpy()
    yt = torch.cat(ys).numpy()
    acc    = float((yp == yt).mean())
    balacc = float(balanced_accuracy_score(yt, yp))
    rec, rec_line = _per_class_recall(yt, yp, n_classes)
    log.info(f"    [{label}] acc={acc:.4f}  BalAcc={balacc:.4f}  per-class: {rec_line}")
    return {"acc": acc, "balacc": balacc, "per_class_recall": rec}


def eval_ensemble(obs_model, prior_model, val_ds, n_classes, batch_size=64,
                   save_logits_to: str = None):
    obs_model.eval(); prior_model.eval()
    all_z_obs, all_z_pri, all_y = [], [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for xm, xr, pos, stype, yb in val_ds.iter_batches(batch_size, shuffle=False):
            lo, _, _ = obs_model  (xm, xr, pos, stype)
            lp, _, _ = prior_model(xm, xr, pos, stype)
            all_z_obs.append(lo.float().cpu())
            all_z_pri.append(lp.float().cpu())
            all_y.append(yb.cpu())
    z_obs_t = torch.cat(all_z_obs); z_pri_t = torch.cat(all_z_pri); yv = torch.cat(all_y)
    log_obs = F.log_softmax(z_obs_t, dim=-1)
    log_pri = F.log_softmax(z_pri_t, dim=-1)
    prob_obs, prob_pri = log_obs.exp(), log_pri.exp()
    chance = 1.0 / n_classes
    w_obs = (prob_obs.max(1).values - chance).clamp(min=1e-4)
    w_pri = (prob_pri.max(1).values - chance).clamp(min=0.0)
    w_sum = (w_obs + w_pri).clamp(min=1e-8)
    log_fused = (w_obs / w_sum).unsqueeze(1) * log_obs + (w_pri / w_sum).unsqueeze(1) * log_pri
    yp_poe   = log_fused.argmax(1).cpu().numpy()
    yp_arith = (prob_obs + prob_pri).argmax(1).cpu().numpy()
    H_max = float(np.log(n_classes))
    H_obs = -(prob_obs * log_obs).sum(-1).clamp(min=0.0)
    H_pri = -(prob_pri * log_pri).sum(-1).clamp(min=0.0)
    we_obs = (H_max - H_obs).clamp(min=1e-4)
    we_pri = (H_max - H_pri).clamp(min=0.0)
    we_sum = (we_obs + we_pri).clamp(min=1e-8)
    log_fused_e = (we_obs / we_sum).unsqueeze(1) * log_obs + (we_pri / we_sum).unsqueeze(1) * log_pri
    yp_ent = log_fused_e.argmax(1).cpu().numpy()
    yt_np   = yv.cpu().numpy()
    acc_poe   = float((yp_poe   == yt_np).mean())
    acc_arith = float((yp_arith == yt_np).mean())
    acc_ent   = float((yp_ent   == yt_np).mean())
    bal_poe   = float(balanced_accuracy_score(yt_np, yp_poe))
    bal_arith = float(balanced_accuracy_score(yt_np, yp_arith))
    bal_ent   = float(balanced_accuracy_score(yt_np, yp_ent))
    rec_poe,   line_poe   = _per_class_recall(yt_np, yp_poe,   n_classes)
    rec_arith, line_arith = _per_class_recall(yt_np, yp_arith, n_classes)
    rec_ent,   line_ent   = _per_class_recall(yt_np, yp_ent,   n_classes)
    log.info(f"    Conf-PoE acc={acc_poe:.4f}  BalAcc={bal_poe:.4f}  per-class: {line_poe}")
    log.info(f"    Arith    acc={acc_arith:.4f}  BalAcc={bal_arith:.4f}  per-class: {line_arith}")
    log.info(f"    Ent-PoE  acc={acc_ent:.4f}  BalAcc={bal_ent:.4f}  per-class: {line_ent}")

    z_obs_np = z_obs_t.numpy()
    z_pri_np = z_pri_t.numpy()
    temp_info = _temp_scaled_fusion(z_obs_np, z_pri_np, yt_np, n_classes, seed=0)
    rec_line_temp = "  ".join([f"c{i}={r:.3f}" for i, r in enumerate(temp_info['poe_temp_recall'])])
    log.info(f"    Temp-PoE acc={temp_info['poe_temp']:.4f}  "
             f"BalAcc={temp_info['poe_temp_balacc']:.4f}  per-class: {rec_line_temp}")
    log.info(f"      T_obs={temp_info['temp_T_obs_mean']:.3f}  "
             f"T_pri={temp_info['temp_T_pri_mean']:.3f}")
    if save_logits_to is not None:
        torch.save({"z_obs": z_obs_t, "z_pri": z_pri_t, "y": yv}, save_logits_to)

    return {
        "poe_conf": acc_poe, "poe_arith": acc_arith, "poe_entropy": acc_ent,
        "poe_conf_balacc":  bal_poe,
        "poe_arith_balacc": bal_arith,
        "poe_entropy_balacc": bal_ent,
        "poe_conf_recall":  rec_poe,
        "poe_arith_recall": rec_arith,
        "poe_entropy_recall": rec_ent,
        **temp_info,
    }





def discover_subjects(h5_dir):
    files = sorted(f for f in os.listdir(h5_dir) if f.startswith("sub-") and f.endswith(".h5"))
    return [os.path.splitext(f)[0] for f in files]


def load_subject(h5_dir, subj, n_samples, modality="eeg", label_key="labels"):
    fp = os.path.join(h5_dir, f"{subj}.h5")
    if not os.path.isfile(fp):
        log.warning(f"  {subj}: not found — skip")
        return None
    with h5py.File(fp, "r") as f:
        modes  = torch.tensor(f[f"{modality}_modes"][:],      dtype=torch.float32)
        raw    = torch.tensor(f[f"{modality}_raw"][:],         dtype=torch.float32)
        labels = torch.tensor(f[label_key][:],                  dtype=torch.long)
        pos    = torch.tensor(f[f"{modality}_pos"][:],         dtype=torch.float32)
        stype  = torch.tensor(f[f"{modality}_sensor_type"][:], dtype=torch.long)
        eigkey = f"{modality}_mode_eigval"
        if eigkey not in f:
            raise KeyError(f"{fp}: missing '{eigkey}' — run patch_eigval.py first")
        eigval = torch.tensor(f[eigkey][:], dtype=torch.float32)
    modes = modes[:, :, :n_samples]
    raw   = raw[:,   :, :n_samples]
    N = len(labels)
    return {
        "modes":  modes, "raw": raw, "labels": labels,
        "pos":    pos.unsqueeze(0).expand(N, -1, -1).contiguous(),
        "stype":  stype.unsqueeze(0).expand(N, -1).contiguous(),
        "eigval": eigval,
        "subj":   subj, "K": modes.shape[1],
    }


def stack_subjects(items):
    Ks = {it["K"] for it in items}
    if len(Ks) != 1:
        raise ValueError(f"Pooled CV needs uniform K across subjects, got {Ks}.")
    Ns = {len(it["labels"]) for it in items}
    if len(Ns) != 1:
        raise ValueError(f"Pooled-trial CV needs identical N_trials across subjects, got {Ns}.")

    ref_eigval = items[0]["eigval"]
    for it in items[1:]:
        if not torch.allclose(it["eigval"], ref_eigval):
            log.warning(f"  eigval differs for {it['subj']} — using sub-01's vector for prior_enc")
    return {
        "modes":  torch.cat([it["modes"]  for it in items], dim=0),
        "raw":    torch.cat([it["raw"]    for it in items], dim=0),
        "labels": torch.cat([it["labels"] for it in items], dim=0),
        "pos":    torch.cat([it["pos"]    for it in items], dim=0),
        "stype":  torch.cat([it["stype"]  for it in items], dim=0),
        "eigval": ref_eigval,
        "subj_ids": np.concatenate([np.array([it["subj"]] * len(it["labels"])) for it in items]),
        "K":      items[0]["K"],
        "N_per_subject": list(Ns)[0],
        "n_subjects":    len(items),
    }


def load_splits(h5_dir, cv_mode: str = "pooled"):





    fname = "splits_crosssub.json" if cv_mode == "cross" else "splits.json"
    sp = os.path.join(h5_dir, fname)
    if not os.path.isfile(sp):
        raise FileNotFoundError(
            f"Expected {fname} at {sp} — build with "
            f"{'build_splits_crosssub.py' if cv_mode == 'cross' else 'build_data.py'} first.")
    with open(sp) as f:
        return json.load(f)


def pooled_indices(template_train, template_test, n_subjects, n_per_subject):

    tr_template = np.asarray(template_train, dtype=np.int64)
    te_template = np.asarray(template_test,  dtype=np.int64)
    offsets = np.arange(n_subjects, dtype=np.int64) * n_per_subject
    tr_idx = (offsets[:, None] + tr_template[None, :]).reshape(-1)
    te_idx = (offsets[:, None] + te_template[None, :]).reshape(-1)
    return np.sort(tr_idx), np.sort(te_idx)


def crosssub_indices(train_subjects, test_subjects, all_subjects, n_per_subject):






    pos = {s: i for i, s in enumerate(all_subjects)}
    missing = [s for s in (*train_subjects, *test_subjects) if s not in pos]
    if missing:
        raise ValueError(f"crosssub split references unknown subjects: {missing}")
    tr_idx = np.concatenate([
        np.arange(pos[s] * n_per_subject, (pos[s] + 1) * n_per_subject, dtype=np.int64)
        for s in train_subjects
    ])
    te_idx = np.concatenate([
        np.arange(pos[s] * n_per_subject, (pos[s] + 1) * n_per_subject, dtype=np.int64)
        for s in test_subjects
    ])
    return np.sort(tr_idx), np.sort(te_idx)





def _make_prior_for_fold(prior_template, prior_init_state, trainable, device):
    if not trainable:
        return prior_template
    enc = copy.deepcopy(prior_template).to(device)
    enc.load_state_dict(prior_init_state)
    for p in enc.parameters():
        p.requires_grad = True
    enc.expected_n_samples = prior_template.expected_n_samples
    enc.d_model_dim        = prior_template.d_model_dim
    return enc


def _make_backbone_for_fold(backbone_template, backbone_init_state, finetune, device):
    if not finetune:
        return backbone_template
    bb = copy.deepcopy(backbone_template).to(device)
    if backbone_init_state is not None:
        bb.load_state_dict(backbone_init_state)
    for p in bb.parameters():
        p.requires_grad = True
    for p in bb.tokenizer.parameters():
        p.requires_grad = False
    bb.tokenizer.eval()
    return bb



class TokenDataset:








    def __init__(self, obs_tok, pri_tok, y, device=None, keep_on_cpu=False):
        target = device or torch.device("cpu")
        if keep_on_cpu or target.type == "cpu":
            self.obs_tok = obs_tok.cpu()
            self.pri_tok = pri_tok.cpu()
            self.y       = y.cpu()
        else:
            self.obs_tok = obs_tok.to(target, non_blocking=True).contiguous()
            self.pri_tok = pri_tok.to(target, non_blocking=True).contiguous()
            self.y       = y.to(target, non_blocking=True).contiguous()
        self.N      = len(y)
        self.device = target

    def iter_batches(self, batch_size, shuffle=True):
        if shuffle:
            idx = torch.randperm(self.N, device=self.obs_tok.device)
        else:
            idx = torch.arange(self.N, device=self.obs_tok.device)
        for s in range(0, self.N, batch_size):
            b = idx[s: s + batch_size]
            yield (self.obs_tok[b], self.pri_tok[b], self.y[b])


@torch.no_grad()
def precompute_features(bundle, idx, backbone, prior_enc, lm_dim, n_tok,
                         device, batch_size: int = 64):




    backbone.eval(); prior_enc.eval()
    eigval = bundle["eigval"].to(device)
    obs_list, pri_list = [], []
    n = len(idx)
    for s in range(0, n, batch_size):
        b = idx[s: s + batch_size]
        xr  = bundle["raw"][b].to(device)
        xm  = bundle["modes"][b].to(device)
        pos = bundle["pos"][b].to(device)
        stype = bundle["stype"][b].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            enc = backbone.encode(xr, pos, stype)
            feat = enc.float().mean(dim=1)
            feat = feat.permute(0, 2, 1)
            feat = F.adaptive_avg_pool1d(feat, n_tok)
            feat = feat.permute(0, 2, 1).contiguous()
            ptok = prior_enc.encode_pooled(xm, eigval)
        obs_list.append(feat.float().cpu())
        pri_list.append(ptok.float().cpu())
    return torch.cat(obs_list), torch.cat(pri_list)


class _ObsCachedBranch(nn.Module):


    def __init__(self, lm_dim, d_model, lat_dim, n_classes, dropout):
        super().__init__()
        class _AdapterOnly(nn.Module):
            def __init__(self, lm_dim, d_model):
                super().__init__()
                self.adapter = nn.Sequential(
                    nn.Linear(lm_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.encoder  = _AdapterOnly(lm_dim, d_model)
        self.ghead    = GaussHead(d_model, lat_dim)
        self.cls_head = ClsHead(lat_dim, n_classes=n_classes, dropout=dropout)

    def forward(self, obs_tok):
        feat = self.encoder.adapter(obs_tok)
        mu, lv = self.ghead(feat)
        return self.cls_head(mu), mu, lv


class _PriorCachedBranch(nn.Module):

    def __init__(self, d_model, lat_dim, n_classes, dropout):
        super().__init__()
        self.ghead    = GaussHead(d_model, lat_dim)
        self.cls_head = ClsHead(lat_dim, n_classes=n_classes, dropout=dropout)

    def forward(self, pri_tok):
        mu, lv = self.ghead(pri_tok)
        return self.cls_head(mu), mu, lv


def _train_cached_branch(model, train_ds, val_ds, branch, args, label):

    trainable = []
    if branch == "obs":
        trainable.append({"params": model.encoder.adapter.parameters(),
                          "lr": args.lr_adapter, "weight_decay": 1e-3})
    trainable.append({"params": list(model.ghead.parameters()) +
                                list(model.cls_head.parameters()),
                      "lr": args.lr_head, "weight_decay": 1e-3})
    opt = torch.optim.AdamW(trainable)
    steps = max(1, (train_ds.N + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in trainable],
        steps_per_epoch=steps, epochs=args.epochs,
        pct_start=0.1, anneal_strategy="cos")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler    = torch.amp.GradScaler("cuda")

    best_acc, best_state, no_improve = 0.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        c = n = 0
        for ob_tok, pri_tok, yb in train_ds.iter_batches(args.batch_size):
            x = ob_tok if branch == "obs" else pri_tok
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, mu, lv = model(x)
                kl   = 0.5 * (lv.exp() + mu.pow(2) - 1 - lv).mean()
                loss = criterion(logits, yb) + args.kl_weight * kl
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for g in trainable for p in g["params"]], 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            c += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
        train_acc = c / max(n, 1)

        model.eval()
        vc = vt = 0
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for ob_tok, pri_tok, yb in val_ds.iter_batches(args.batch_size, shuffle=False):
                x = ob_tok if branch == "obs" else pri_tok
                logits, _, _ = model(x)
                vc += (logits.argmax(1) == yb).sum().item()
                vt += len(yb)
        val_acc = vc / max(vt, 1)

        if val_acc > best_acc + 1e-4:
            best_acc   = val_acc
            best_state = deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            no_improve = 0
        else:
            no_improve += 1
        if epoch % 50 == 0 or epoch == 1 or epoch == args.epochs or no_improve == args.patience:
            log.info(f"    [{label}] Epoch {epoch:4d}/{args.epochs}"
                     f"  train={train_acc:.4f}  val={val_acc:.4f}  best={best_acc:.4f}")
        if no_improve >= args.patience:
            log.info(f"    [{label}] Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                               for k, v in best_state.items()})
    return best_acc, (best_state or {})


def _eval_cached_branch(model, ds, branch, batch_size, n_classes, label):
    model.eval()
    preds, ys = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for ob_tok, pri_tok, yb in ds.iter_batches(batch_size, shuffle=False):
            x = ob_tok if branch == "obs" else pri_tok
            logits, _, _ = model(x)
            preds.append(logits.argmax(1).cpu())
            ys.append(yb.cpu())
    yp = torch.cat(preds).numpy()
    yt = torch.cat(ys).numpy()
    acc    = float((yp == yt).mean())
    balacc = float(balanced_accuracy_score(yt, yp))
    rec, rec_line = _per_class_recall(yt, yp, n_classes)
    log.info(f"    [{label}] acc={acc:.4f}  BalAcc={balacc:.4f}  per-class: {rec_line}")
    return {"acc": acc, "balacc": balacc, "per_class_recall": rec}



def _fit_temperature_nll(z: np.ndarray, y: np.ndarray, T_grid: np.ndarray) -> float:

    zt = torch.from_numpy(z).float()
    yt = torch.from_numpy(y).long()
    best_T, best_nll = 1.0, float("inf")
    arange = torch.arange(len(yt))
    for T in T_grid:
        if T <= 0:
            continue
        lp = F.log_softmax(zt / float(T), dim=-1)
        nll = float(-lp[arange, yt].mean().item())
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def _stratified_half_split(y: np.ndarray, n_classes: int, seed: int = 0):

    rng = np.random.default_rng(seed)
    A, B = [], []
    for c in range(n_classes):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        mid = len(idx_c) // 2
        A.extend(idx_c[:mid].tolist())
        B.extend(idx_c[mid:].tolist())
    return np.array(sorted(A)), np.array(sorted(B))


def _temp_scaled_fusion(z_obs: np.ndarray, z_pri: np.ndarray, y: np.ndarray,
                         n_classes: int, seed: int = 0):








    T_grid = np.linspace(0.05, 10.0, 200)
    A, B = _stratified_half_split(y, n_classes, seed=seed)
    T_obs_A = _fit_temperature_nll(z_obs[A], y[A], T_grid)
    T_pri_A = _fit_temperature_nll(z_pri[A], y[A], T_grid)
    T_obs_B = _fit_temperature_nll(z_obs[B], y[B], T_grid)
    T_pri_B = _fit_temperature_nll(z_pri[B], y[B], T_grid)
    preds = np.empty(len(y), dtype=np.int64)

    for half_idx, T_obs, T_pri in [(B, T_obs_A, T_pri_A), (A, T_obs_B, T_pri_B)]:
        zo = torch.from_numpy(z_obs[half_idx]).float() / float(T_obs)
        zp = torch.from_numpy(z_pri[half_idx]).float() / float(T_pri)
        fused = F.log_softmax(zo, dim=-1) + F.log_softmax(zp, dim=-1)
        preds[half_idx] = fused.argmax(-1).numpy()
    acc    = float((preds == y).mean())
    balacc = float(balanced_accuracy_score(y, preds))
    rec, _ = _per_class_recall(y, preds, n_classes)
    return {
        "poe_temp":         acc,
        "poe_temp_balacc":  balacc,
        "poe_temp_recall":  rec,
        "temp_T_obs_A":     T_obs_A,
        "temp_T_pri_A":     T_pri_A,
        "temp_T_obs_B":     T_obs_B,
        "temp_T_pri_B":     T_pri_B,
        "temp_T_obs_mean":  float(0.5 * (T_obs_A + T_obs_B)),
        "temp_T_pri_mean":  float(0.5 * (T_pri_A + T_pri_B)),
    }


def _eval_cached_ensemble(obs_model, pri_model, ds, n_classes, batch_size,
                           save_logits_to: str = None):
    obs_model.eval(); pri_model.eval()
    all_z_obs, all_z_pri, all_y = [], [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for ob_tok, pri_tok, yb in ds.iter_batches(batch_size, shuffle=False):
            lo, _, _ = obs_model(ob_tok)
            lp, _, _ = pri_model(pri_tok)
            all_z_obs.append(lo.float().cpu())
            all_z_pri.append(lp.float().cpu())
            all_y.append(yb.cpu())
    z_obs_t = torch.cat(all_z_obs)
    z_pri_t = torch.cat(all_z_pri)
    yv      = torch.cat(all_y)
    log_obs = F.log_softmax(z_obs_t, dim=-1)
    log_pri = F.log_softmax(z_pri_t, dim=-1)
    prob_obs, prob_pri = log_obs.exp(), log_pri.exp()
    chance = 1.0 / n_classes
    w_obs = (prob_obs.max(1).values - chance).clamp(min=1e-4)
    w_pri = (prob_pri.max(1).values - chance).clamp(min=0.0)
    w_sum = (w_obs + w_pri).clamp(min=1e-8)
    log_fused = (w_obs / w_sum).unsqueeze(1) * log_obs + (w_pri / w_sum).unsqueeze(1) * log_pri
    yp_poe   = log_fused.argmax(1).cpu().numpy()
    yp_arith = (prob_obs + prob_pri).argmax(1).cpu().numpy()
    H_max = float(np.log(n_classes))
    H_obs = -(prob_obs * log_obs).sum(-1).clamp(min=0.0)
    H_pri = -(prob_pri * log_pri).sum(-1).clamp(min=0.0)
    we_obs = (H_max - H_obs).clamp(min=1e-4)
    we_pri = (H_max - H_pri).clamp(min=0.0)
    we_sum = (we_obs + we_pri).clamp(min=1e-8)
    log_fused_e = (we_obs / we_sum).unsqueeze(1) * log_obs + (we_pri / we_sum).unsqueeze(1) * log_pri
    yp_ent = log_fused_e.argmax(1).cpu().numpy()
    yt_np   = yv.cpu().numpy()
    acc_poe   = float((yp_poe   == yt_np).mean())
    acc_arith = float((yp_arith == yt_np).mean())
    acc_ent   = float((yp_ent   == yt_np).mean())
    bal_poe   = float(balanced_accuracy_score(yt_np, yp_poe))
    bal_arith = float(balanced_accuracy_score(yt_np, yp_arith))
    bal_ent   = float(balanced_accuracy_score(yt_np, yp_ent))
    rec_poe,   line_poe   = _per_class_recall(yt_np, yp_poe,   n_classes)
    rec_arith, line_arith = _per_class_recall(yt_np, yp_arith, n_classes)
    rec_ent,   line_ent   = _per_class_recall(yt_np, yp_ent,   n_classes)
    log.info(f"    [cached] Conf-PoE acc={acc_poe:.4f}  BalAcc={bal_poe:.4f}  per-class: {line_poe}")
    log.info(f"    [cached] Arith    acc={acc_arith:.4f}  BalAcc={bal_arith:.4f}  per-class: {line_arith}")
    log.info(f"    [cached] Ent-PoE  acc={acc_ent:.4f}  BalAcc={bal_ent:.4f}  per-class: {line_ent}")


    z_obs_np = z_obs_t.numpy()
    z_pri_np = z_pri_t.numpy()
    temp_info = _temp_scaled_fusion(z_obs_np, z_pri_np, yt_np, n_classes, seed=0)
    rec_line_temp = "  ".join([f"c{i}={r:.3f}" for i, r in enumerate(temp_info['poe_temp_recall'])])
    log.info(f"    [cached] Temp-PoE acc={temp_info['poe_temp']:.4f}  "
             f"BalAcc={temp_info['poe_temp_balacc']:.4f}  per-class: {rec_line_temp}")
    log.info(f"    [cached]   T_obs={temp_info['temp_T_obs_mean']:.3f}  "
             f"T_pri={temp_info['temp_T_pri_mean']:.3f}")


    if save_logits_to is not None:
        torch.save({"z_obs": z_obs_t, "z_pri": z_pri_t, "y": yv}, save_logits_to)

    return {
        "poe_conf": acc_poe, "poe_arith": acc_arith, "poe_entropy": acc_ent,
        "poe_conf_balacc":  bal_poe,
        "poe_arith_balacc": bal_arith,
        "poe_entropy_balacc": bal_ent,
        "poe_conf_recall":  rec_poe,
        "poe_arith_recall": rec_arith,
        "poe_entropy_recall": rec_ent,
        **temp_info,
    }


def run_one_fold_cached(bundle, tr_idx, te_idx,
                         prior_template, backbone_template, lm_dim,
                         device, args, fold_dir, label,
                         cache_dir=None):






    n_tok   = prior_template.expected_n_samples // PATCH_SIZE
    d_model = prior_template.d_model_dim


    def _maybe_cached(idx, tag):
        if cache_dir is None:
            return precompute_features(bundle, idx, backbone_template, prior_template,
                                       lm_dim, n_tok, device,
                                       batch_size=max(args.batch_size, 64))
        import hashlib
        h = hashlib.md5(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()[:12]
        cp = os.path.join(cache_dir, f"feat_{tag}_{h}.pt")
        if os.path.isfile(cp):
            log.info(f"  [{label}] cache HIT  {tag} → {cp}")
            d = torch.load(cp, map_location="cpu", weights_only=False)
            return d["obs"], d["pri"]
        log.info(f"  [{label}] cache MISS {tag} → precomputing …")
        ob, pr = precompute_features(bundle, idx, backbone_template, prior_template,
                                     lm_dim, n_tok, device,
                                     batch_size=max(args.batch_size, 64))
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({"obs": ob, "pri": pr,
                    "idx_md5": h, "n": len(idx)}, cp)
        return ob, pr

    log.info(f"  [{label}] precomputing features  train={len(tr_idx)}  test={len(te_idx)}")
    t0 = __import__("time").time()
    obs_tr, pri_tr = _maybe_cached(tr_idx, "tr")
    obs_te, pri_te = _maybe_cached(te_idx, "te")
    log.info(f"  [{label}] features ready in {(__import__('time').time()-t0):.1f}s  "
             f"obs={tuple(obs_tr.shape)}  pri={tuple(pri_tr.shape)}")

    y_tr = bundle["labels"][tr_idx]
    y_te = bundle["labels"][te_idx]
    tr_ds = TokenDataset(obs_tr, pri_tr, y_tr, device=device)
    te_ds = TokenDataset(obs_te, pri_te, y_te, device=device)

    log.info(f"  [{label}] train={tr_ds.N}  test={te_ds.N}  K={bundle['K']}  "
             f"n_tok={n_tok}  d_model={d_model}  train_mode=frozen[CACHED]")
    log.info(f"  [{label}] train class dist: {torch.bincount(y_tr, minlength=args.n_classes).tolist()}")
    log.info(f"  [{label}] test  class dist: {torch.bincount(y_te, minlength=args.n_classes).tolist()}")


    obs_model = _ObsCachedBranch(lm_dim, d_model, args.lat_dim,
                                  args.n_classes, args.dropout).to(device)
    obs_acc, obs_state = _train_cached_branch(obs_model, tr_ds, te_ds, "obs", args,
                                               label=f"{label}.obs")


    pri_model = _PriorCachedBranch(d_model, args.lat_dim,
                                    args.n_classes, args.dropout).to(device)
    pri_acc, pri_state = _train_cached_branch(pri_model, tr_ds, te_ds, "prior", args,
                                               label=f"{label}.pri")

    log.info(f"  [{label}] VAL fusion eval:")
    fuse = _eval_cached_ensemble(obs_model, pri_model, te_ds, args.n_classes, args.batch_size,
                                  save_logits_to=os.path.join(fold_dir, "logits_test.pt"))
    val_obs_detail = _eval_cached_branch(obs_model, te_ds, "obs", args.batch_size,
                                          args.n_classes, f"{label}.val.obs")
    val_pri_detail = _eval_cached_branch(pri_model, te_ds, "prior", args.batch_size,
                                          args.n_classes, f"{label}.val.pri")

    os.makedirs(fold_dir, exist_ok=True)
    torch.save(obs_state, os.path.join(fold_dir, "obs.pt"))
    torch.save(pri_state, os.path.join(fold_dir, "prior.pt"))

    metrics = {
        "label":     label,
        "train_mode": "frozen[cached]",
        "obs_acc":   obs_acc,
        "prior_acc": pri_acc,
        **fuse,
        "val_obs_balacc": val_obs_detail["balacc"],
        "val_pri_balacc": val_pri_detail["balacc"],
        "val_obs_recall": val_obs_detail["per_class_recall"],
        "val_pri_recall": val_pri_detail["per_class_recall"],
        "n_train":  int(tr_ds.N),
        "n_test":   int(te_ds.N),
    }
    with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    del obs_model, pri_model, tr_ds, te_ds, obs_tr, pri_tr, obs_te, pri_te
    torch.cuda.empty_cache()
    return metrics


def run_one_fold(bundle, tr_idx, te_idx,
                 prior_template, prior_init_state,
                 backbone_template, backbone_init_state, lm_dim,
                 device, args, fold_dir, label):
    n_tok   = prior_template.expected_n_samples // PATCH_SIZE
    d_model = prior_template.d_model_dim
    finetune = (args.train_mode == "finetune")

    tr = GPUDataset(bundle["modes"][tr_idx], bundle["raw"][tr_idx],
                    bundle["pos"][tr_idx],   bundle["stype"][tr_idx],
                    bundle["labels"][tr_idx], device=device)
    te = GPUDataset(bundle["modes"][te_idx], bundle["raw"][te_idx],
                    bundle["pos"][te_idx],   bundle["stype"][te_idx],
                    bundle["labels"][te_idx], device=device)

    log.info(f"  [{label}] train={tr.N}  test={te.N}  K={bundle['K']}  "
             f"n_tok={n_tok}  d_model={d_model}  train_mode={args.train_mode}")
    log.info(f"  [{label}] train class dist: {torch.bincount(bundle['labels'][tr_idx], minlength=args.n_classes).tolist()}")
    log.info(f"  [{label}] test  class dist: {torch.bincount(bundle['labels'][te_idx], minlength=args.n_classes).tolist()}")

    backbone = _make_backbone_for_fold(backbone_template, backbone_init_state, finetune, device)
    obs_enc   = ObsEncoder(backbone, lm_dim, n_tok, d_model, backbone_trainable=finetune).to(device)
    obs_ghead = GaussHead(d_model, args.lat_dim).to(device)
    obs_cls   = ClsHead(args.lat_dim, n_classes=args.n_classes, dropout=args.dropout).to(device)
    obs_model = ExpertModel(obs_enc, obs_ghead, obs_cls, is_obs=True, encoder_trainable=False).to(device)
    obs_acc, obs_state = train_expert(obs_model, tr, te,
                                      args.epochs, args.batch_size,
                                      args.lr_adapter, args.lr_head, args.lr_backbone,
                                      args.patience, args.kl_weight, f"{label}.obs")

    prior_enc = _make_prior_for_fold(prior_template, prior_init_state, finetune, device)
    pri_ghead = GaussHead(d_model, args.lat_dim).to(device)
    pri_cls   = ClsHead(args.lat_dim, n_classes=args.n_classes, dropout=args.dropout).to(device)
    pri_model = ExpertModel(prior_enc, pri_ghead, pri_cls, is_obs=False, encoder_trainable=finetune).to(device)
    pri_model.set_eigval(bundle["eigval"])
    pri_acc, pri_state = train_expert(pri_model, tr, te,
                                      args.epochs, args.batch_size,
                                      args.lr_adapter, args.lr_head, args.lr_backbone,
                                      args.patience, args.kl_weight, f"{label}.pri")

    log.info(f"  [{label}] VAL fusion eval:")
    fuse = eval_ensemble(obs_model, pri_model, te, args.n_classes, args.batch_size,
                          save_logits_to=os.path.join(fold_dir, "logits_test.pt"))
    val_obs_detail = eval_branch_detailed(obs_model, te, args.batch_size, args.n_classes, f"{label}.val.obs")
    val_pri_detail = eval_branch_detailed(pri_model, te, args.batch_size, args.n_classes, f"{label}.val.pri")

    os.makedirs(fold_dir, exist_ok=True)
    if finetune:
        obs_state_to_save = obs_state
    else:
        obs_state_to_save = {k: v for k, v in obs_state.items() if not k.startswith("encoder.backbone.")}
    torch.save(obs_state_to_save, os.path.join(fold_dir, "obs.pt"))
    if finetune:
        pri_state_to_save = pri_state
    else:
        pri_state_to_save = {k: v for k, v in pri_state.items() if not k.startswith("encoder.")}
    torch.save(pri_state_to_save, os.path.join(fold_dir, "prior.pt"))

    metrics = {
        "label":     label,
        "train_mode": args.train_mode,
        "obs_acc":   obs_acc,
        "prior_acc": pri_acc,
        **fuse,
        "val_obs_balacc": val_obs_detail["balacc"],
        "val_pri_balacc": val_pri_detail["balacc"],
        "val_obs_recall": val_obs_detail["per_class_recall"],
        "val_pri_recall": val_pri_detail["per_class_recall"],
        "n_train":  int(tr.N),
        "n_test":   int(te.N),
    }
    with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if finetune:
        del backbone, prior_enc
    del obs_model, pri_model, tr, te
    torch.cuda.empty_cache()
    return metrics





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_dir",       type=str, required=True,
                    help="Dataset dir containing sub-*.h5 AND splits[_crosssub].json")
    ap.add_argument("--encoder_ckpt", type=str, default=None,
                    help="K-free MAE ckpt. If omitted, runs auto-pretrain on --h5_dir first.")
    ap.add_argument("--train_mode",   type=str, choices=["frozen", "finetune"], default="frozen")
    ap.add_argument("--cv_mode",      type=str, choices=["pooled", "cross"], default="pooled",
                    help="pooled = within-subject 5-fold over trial positions (splits.json); "
                         "cross  = subject-disjoint 5-fold (splits_crosssub.json).")
    ap.add_argument("--modality",     type=str, choices=["eeg", "meg"], default="eeg")
    ap.add_argument("--label_key",    type=str, default="labels")
    ap.add_argument("--subjects",     type=str, nargs="+", default=None)
    ap.add_argument("--fold",         type=int, default=None,
                    help="Run only this fold (0..n_folds-1). Default: run all 5.")
    ap.add_argument("--epochs",       type=int,   default=50)
    ap.add_argument("--batch_size",   type=int,   default=32)
    ap.add_argument("--lr_adapter",   type=float, default=1e-3)
    ap.add_argument("--lr_head",      type=float, default=1e-3)
    ap.add_argument("--lr_backbone",  type=float, default=1e-4)
    ap.add_argument("--lat_dim",      type=int,   default=256)
    ap.add_argument("--dropout",      type=float, default=0.3)
    ap.add_argument("--patience",     type=int,   default=10)
    ap.add_argument("--kl_weight",    type=float, default=1e-3)
    ap.add_argument("--n_classes",    type=int,   default=5)
    ap.add_argument("--device",       type=str,   default="cuda:0")
    ap.add_argument("--seed",         type=int,   default=0)
    ap.add_argument("--tag",          type=str,   default=None)
    ap.add_argument("--out_dir",      type=str,   required=True,
                    help="Per-run root dir. <out_dir>/<tag>_pooled_<run_id>/fold_NN/...")
    ap.add_argument("--run_id",       type=str,   default=None)
    ap.add_argument("--cache_features", action="store_true",
                    help="Frozen-mode fast path: precompute obs (pre-adapter) and prior tokens "
                         "once per fold, then train adapter+heads on cached features. "
                         "Ignored when --train_mode=finetune.")
    ap.add_argument("--cache_dir",     type=str, default=None,
                    help="Optional dir to persist token caches across folds within an experiment.")


    ap.add_argument("--pretrain_epochs",     type=int,   default=50)
    ap.add_argument("--pretrain_batch_size", type=int,   default=64)
    ap.add_argument("--pretrain_lr",         type=float, default=3e-4)
    ap.add_argument("--pretrain_d_model",    type=int,   default=512)
    ap.add_argument("--pretrain_n_factor_layers", type=int, default=2)
    ap.add_argument("--pretrain_n_time_layers",   type=int, default=4)
    ap.add_argument("--pretrain_n_dec_layers",    type=int, default=2)
    ap.add_argument("--pretrain_mask_ratio", type=float, default=0.5)
    ap.add_argument("--pretrain_val_ratio",  type=float, default=0.1)
    ap.add_argument("--pretrain_log_every",  type=int,   default=5)

    args = ap.parse_args()

    global RUN_ID
    if args.run_id:
        RUN_ID = args.run_id
    tag = args.tag or os.path.basename(args.h5_dir.rstrip("/\\")) or "data"
    cv_tag = "cross" if args.cv_mode == "cross" else "pooled"
    root_dir = os.path.join(args.out_dir, f"{tag}_{cv_tag}_{RUN_ID}")
    os.makedirs(root_dir, exist_ok=True)
    _setup_logging(root_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(os.path.join(root_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    log.info(f"Run ID       : {RUN_ID}")
    log.info(f"Out root     : {root_dir}")
    log.info(f"Device       : {device}")
    if args.cv_mode == "cross":
        log.info(f"CV mode      : cross-subject 5-fold (subject-disjoint)")
    else:
        log.info(f"CV mode      : pooled-trial 5-fold (shared template, within-subject)")
    log.info(f"Train mode   : {args.train_mode}")
    log.info(f"H5 dir       : {args.h5_dir}")


    if not args.encoder_ckpt:
        log.info("\nNo --encoder_ckpt supplied → auto-pretrain on --h5_dir")
        pre_dir = os.path.join(root_dir, "pretrain")
        hp = {
            "epochs": args.pretrain_epochs, "batch_size": args.pretrain_batch_size,
            "lr": args.pretrain_lr, "d_model": args.pretrain_d_model,
            "n_factor_layers": args.pretrain_n_factor_layers,
            "n_time_layers":   args.pretrain_n_time_layers,
            "n_dec_layers":    args.pretrain_n_dec_layers,
            "mask_ratio": args.pretrain_mask_ratio,
            "val_ratio":  args.pretrain_val_ratio,
            "log_every":  args.pretrain_log_every,
            "seed": args.seed,
        }
        args.encoder_ckpt = auto_pretrain_prior(args.h5_dir, args.modality, pre_dir,
                                                device, hp, tag)
    log.info(f"Encoder ckpt : {args.encoder_ckpt}")


    splits = load_splits(args.h5_dir, cv_mode=args.cv_mode)
    n_folds = splits["n_folds"]
    folds_template = splits["folds"]
    splits_cache_name = "splits_crosssub.json" if args.cv_mode == "cross" else "splits.json"
    with open(os.path.join(root_dir, splits_cache_name), "w") as f:
        json.dump(splits, f, indent=2)
    if args.cv_mode == "cross":
        log.info(f"Splits       : {n_folds} folds over {splits['n_subjects']} subjects "
                 f"(sizes={splits.get('fold_sizes')})")
    else:
        log.info(f"Splits       : {n_folds} folds, template trial count = {splits['n_trials']}")


    finetune = (args.train_mode == "finetune")
    log.info("\nLoading K-free MAE prior …")
    prior_enc = load_prior_from_ckpt(args.encoder_ckpt, device, trainable=finetune)
    prior_init_state = {k: v.detach().clone() for k, v in prior_enc.state_dict().items()}
    log.info("Loading BrainOmni-tiny …")
    backbone, lm_dim = load_brainomni_tiny(device, finetune=finetune)
    backbone_init_state = ({k: v.detach().clone() for k, v in backbone.state_dict().items()}
                            if finetune else None)


    subjects = args.subjects or discover_subjects(args.h5_dir)
    log.info(f"Subjects ({len(subjects)}): {subjects}")
    n_samples = prior_enc.expected_n_samples
    items = []
    for s in subjects:
        it = load_subject(args.h5_dir, s, n_samples, args.modality, args.label_key)
        if it is not None:
            items.append(it)
    bundle = stack_subjects(items)
    log.info(f"\n  Pooled N={len(bundle['labels'])} trials across {bundle['n_subjects']} subjects "
             f"(K={bundle['K']}, N_per_subject={bundle['N_per_subject']})")
    log.info(f"  Class dist (pooled): {torch.bincount(bundle['labels']).tolist()}")


    keep_keys_pooled = ("fold", "train_idx",      "test_idx",      "n_train",      "n_test")
    keep_keys_cross  = ("fold", "train_subjects", "test_subjects", "n_train_subj", "n_test_subj")
    keep_keys = keep_keys_cross if args.cv_mode == "cross" else keep_keys_pooled
    for entry in folds_template:
        fid = entry["fold"]
        fold_dir = os.path.join(root_dir, f"fold_{fid:02d}")
        os.makedirs(fold_dir, exist_ok=True)

        with open(os.path.join(fold_dir, "split.json"), "w") as f:
            json.dump({k: v for k, v in entry.items() if k in keep_keys}, f, indent=2)




    if args.cv_mode == "cross":
        split_subjects = splits["subjects"]
        if split_subjects != subjects:


            log.warning("subjects order in splits_crosssub.json differs from loaded order — "
                        "using loaded order for index mapping.")

    fold_metrics = []
    for entry in folds_template:
        fid = entry["fold"]
        if args.fold is not None and fid != args.fold:
            continue
        fold_dir = os.path.join(root_dir, f"fold_{fid:02d}")
        metrics_path = os.path.join(fold_dir, "metrics.json")
        if os.path.isfile(metrics_path):
            log.info(f"\n========== FOLD {fid+1}/{n_folds}  RESUME (metrics.json exists) ==========")
            with open(metrics_path) as f:
                fold_metrics.append(json.load(f))
            continue

        if args.cv_mode == "cross":
            tr_idx, te_idx = crosssub_indices(
                entry["train_subjects"], entry["test_subjects"],
                subjects, bundle["N_per_subject"])
            log.info(f"\n========== FOLD {fid+1}/{n_folds}  CROSS-SUB  "
                     f"train_subj={len(entry['train_subjects'])}  "
                     f"test_subj={len(entry['test_subjects'])} ({entry['test_subjects']})  "
                     f"pooled_train={len(tr_idx)}  pooled_test={len(te_idx)} ==========")
        else:
            tr_idx, te_idx = pooled_indices(entry["train_idx"], entry["test_idx"],
                                             bundle["n_subjects"], bundle["N_per_subject"])
            log.info(f"\n========== FOLD {fid+1}/{n_folds}  template_train={len(entry['train_idx'])}  "
                     f"template_test={len(entry['test_idx'])}  pooled_train={len(tr_idx)}  pooled_test={len(te_idx)} ==========")
        if args.cache_features and not finetune:
            m = run_one_fold_cached(bundle, tr_idx, te_idx,
                                     prior_enc, backbone, lm_dim,
                                     device, args, fold_dir, label=f"f{fid}",
                                     cache_dir=args.cache_dir)
        else:
            m = run_one_fold(bundle, tr_idx, te_idx,
                             prior_enc, prior_init_state,
                             backbone, backbone_init_state, lm_dim,
                             device, args, fold_dir, label=f"f{fid}")
        fold_metrics.append(m)


    if args.fold is None and fold_metrics:
        def stat(v):
            a = np.array(v)
            return {"mean": float(a.mean()), "std": float(a.std()),
                    "values": [float(x) for x in v]}
        summary = {}
        for k in ("obs_acc", "prior_acc", "poe_arith", "poe_conf", "poe_entropy",
                  "poe_arith_balacc", "poe_conf_balacc", "poe_entropy_balacc",
                  "val_obs_balacc", "val_pri_balacc"):
            vals = [m.get(k) for m in fold_metrics if m.get(k) is not None]
            if vals:
                summary[k] = stat(vals)
        summary["meta"] = {
            "tag": tag, "h5_dir": args.h5_dir, "encoder_ckpt": args.encoder_ckpt,
            "subjects": subjects,
            "cv_mode": "crosssub" if args.cv_mode == "cross" else "pooled_trial",
            "n_folds": n_folds,
            "train_mode": args.train_mode, "seed": args.seed, "RUN_ID": RUN_ID,
            "n_classes": args.n_classes,
        }
        with open(os.path.join(root_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        log.info("\n" + "=" * 75)
        cv_label = "cross-subject" if args.cv_mode == "cross" else "pooled-trial"
        log.info(f"FINAL SUMMARY [{tag} / {cv_label} / {args.train_mode}]  ({n_folds}-fold)")
        log.info("=" * 75)
        for k in ("obs_acc", "prior_acc", "poe_arith", "poe_conf", "poe_entropy",
                  "poe_arith_balacc"):
            if k in summary:
                s = summary[k]
                log.info(f"  {k:<20s} = {s['mean']*100:6.2f}% ± {s['std']*100:.2f}%   "
                         f"folds={[f'{v*100:.1f}' for v in s['values']]}")
        log.info(f"  Chance level = {100/args.n_classes:.2f}%")
    log.info(f"\nArtifacts → {root_dir}")


if __name__ == "__main__":
    main()
