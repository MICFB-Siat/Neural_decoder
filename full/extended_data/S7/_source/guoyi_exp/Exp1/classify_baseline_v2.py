
















































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
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import balanced_accuracy_score, recall_score

SCRIPT_DIR    = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT  = os.path.normpath(os.path.join(SCRIPT_DIR, "../../"))
BRAINOMNI_SRC = os.path.join(PROJECT_ROOT, "weights/BrainOmni/BrainOmni-main")
TINY_CKPT_DIR = os.path.join(PROJECT_ROOT, "weights/BrainOmni/BrainOmni/tiny")
FULLRUN_DIR   = os.path.join(PROJECT_ROOT, "data_check_20260507/full_run_5fold")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, BRAINOMNI_SRC)
sys.path.insert(0, FULLRUN_DIR)

from models.prior.eeg_meg_prior_eigenval4 import (
    EEGMEGPriorEigenval4, compute_eigval_from_D,
)
from models.prior.eeg_meg_prior_eigenval3 import (
    EEGMEGPriorEigenval3, EEGMEGMAEDecoderEigenval3,
)
from models.prior.eeg_meg_prior_eigenval import (
    PATCH_SIZE, brainomni_2d_mask, patchify,
)

RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(SCRIPT_DIR, f"classify_baseline_{RUN_ID}.log")


def resolve_shared_eigval(d_path: str | None,
                          lam_cortex_path: str | None,
                          ratio: float,
                          h5_sample_path: str | None,
                          modality: str = "eeg") -> tuple[torch.Tensor | None, str]:











    if d_path is not None:
        D = np.load(d_path)
        lam = np.load(lam_cortex_path) if lam_cortex_path else None
        ev_np, K_eff = compute_eigval_from_D(D, lam_cortex=lam, ratio=ratio)
        log.info(f"  Eigval from D ({d_path})  ratio={ratio:g}  K_eff={K_eff}  "
                 f"range=[{ev_np.min():.3g}..{ev_np.max():.3g}]  "
                 f"lam_cortex={'individual' if lam is not None else 'group fs32k'}")
        return torch.from_numpy(ev_np), "D"

    if h5_sample_path is None or not os.path.isfile(h5_sample_path):
        return None, "deferred"
    ev_key = f"{modality}_mode_eigval"
    with h5py.File(h5_sample_path, "r") as f:
        if ev_key not in f:
            raise KeyError(
                f"No --d_path supplied AND {os.path.basename(h5_sample_path)} "
                f"has no '{ev_key}' — provide --d_path or pre-patch h5 with eigval."
            )
        ev_np = f[ev_key][:].astype(np.float32)
    log.info(f"  Eigval from h5 [{ev_key}]  K={ev_np.shape[0]}  "
             f"range=[{ev_np.min():.3g}..{ev_np.max():.3g}]")
    return torch.from_numpy(ev_np), "h5"


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



    def __init__(self, chunks: list[torch.Tensor],
                 names: list[str],
                 eigvals: list[torch.Tensor]):
        assert len(chunks) == len(names) == len(eigvals)
        self.chunks  = chunks
        self.names   = names
        self.eigvals = eigvals
        self.sizes   = [c.shape[0] for c in chunks]
        self.N       = sum(self.sizes)
        self.Ks      = [c.shape[1] for c in chunks]
        self.T       = chunks[0].shape[2] if chunks else 0
        for c, ev, K in zip(chunks, eigvals, self.Ks):
            assert c.shape[2] == self.T, "All subjects must share T"
            assert ev.shape == (K,), f"eigval shape {tuple(ev.shape)} ≠ K={K}"

    def iter_batches(self, batch_size: int, shuffle: bool = True,
                     rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        order = rng.permutation(len(self.chunks)) if shuffle else np.arange(len(self.chunks))
        for ci in order:
            c   = self.chunks[ci]
            ev  = self.eigvals[ci]
            n   = c.shape[0]
            sub_idx = (torch.randperm(n, device=c.device)
                       if shuffle else torch.arange(n, device=c.device))
            for s in range(0, n, batch_size):
                yield c[sub_idx[s: s + batch_size]], ci, ev


def _load_modes_hetero(h5_dir: str, modality: str, device: torch.device,
                       override_eigval: torch.Tensor | None = None) -> _HeteroKDataset:






    key = f"{modality}_modes"
    ev_key = f"{modality}_mode_eigval"
    files = sorted(glob.glob(os.path.join(h5_dir, "sub-*.h5")))
    chunks, names, eigvals = [], [], []
    for fp in files:
        with h5py.File(fp, "r") as f:
            if key not in f:
                log.warning(f"  {os.path.basename(fp)}: missing {key} — skip")
                continue
            arr = f[key][:].astype(np.float32)
            if override_eigval is None:
                if ev_key not in f:
                    raise KeyError(
                        f"{os.path.basename(fp)}: missing '{ev_key}'. "
                        f"Provide --d_path or pre-patch h5 with eigval."
                    )
                ev = f[ev_key][:].astype(np.float32)
            else:
                K_in = arr.shape[1]
                if override_eigval.shape[0] != K_in:
                    raise ValueError(
                        f"{os.path.basename(fp)}: modes K={K_in} ≠ "
                        f"--d_path eigval K={override_eigval.shape[0]}"
                    )
                ev = override_eigval.detach().cpu().numpy().astype(np.float32)
        chunks.append(torch.from_numpy(arr).to(device))
        eigvals.append(torch.from_numpy(ev).to(device))
        names.append(os.path.splitext(os.path.basename(fp))[0])
        log.info(f"  {names[-1]}: {key} {arr.shape}  eigval[{ev.shape[0]}] "
                 f"range=[{ev.min():.3g}..{ev.max():.3g}]")
    if not chunks:
        raise RuntimeError(f"No '{key}' arrays found in {h5_dir}")
    return _HeteroKDataset(chunks, names, eigvals)


def auto_pretrain_prior(h5_dir: str, modality: str, out_dir: str,
                        device: torch.device, hp: dict, tag: str,
                        override_eigval: torch.Tensor | None = None) -> str:



    pre_dir = out_dir
    os.makedirs(pre_dir, exist_ok=True)
    out_path = os.path.join(pre_dir, f"mae_kfree_v3_{tag}_{modality}.pt")

    torch.manual_seed(hp["seed"]); np.random.seed(hp["seed"])
    rng = np.random.default_rng(hp["seed"])

    log.info("\n" + "─" * 75)
    log.info(f"AUTO-PRETRAIN (v3 + E[λ])  →  {out_path}")
    log.info("─" * 75)
    log.info(f"Loading {h5_dir} (modality={modality}) …")
    ds_all = _load_modes_hetero(h5_dir, modality, device, override_eigval=override_eigval)
    log.info(f"  Total trials = {ds_all.N}  Ks per subject = {ds_all.Ks}  T = {ds_all.T}")

    train_chunks, val_chunks, train_evs, val_evs, names = [], [], [], [], []
    for c, ev, nm in zip(ds_all.chunks, ds_all.eigvals, ds_all.names):
        n = c.shape[0]
        n_val = max(1, int(round(n * hp["val_ratio"])))
        perm = torch.randperm(n, device=c.device)
        val_chunks.append(c[perm[:n_val]])
        train_chunks.append(c[perm[n_val:]])
        train_evs.append(ev); val_evs.append(ev)
        names.append(nm)
        log.info(f"  {nm}: train={n - n_val}  val={n_val}")
    train_ds = _HeteroKDataset(train_chunks, names, train_evs)
    val_ds   = _HeteroKDataset(val_chunks,   names, val_evs)

    T  = ds_all.T
    NP = T // PATCH_SIZE
    log.info(f"  NP per trial = {NP}  patch_size = {PATCH_SIZE}")

    encoder = EEGMEGPriorEigenval3(
        patch_size=PATCH_SIZE, d_model=hp["d_model"], n_heads=8,
        n_factor_layers=hp["n_factor_layers"],
        n_time_layers=hp["n_time_layers"],
    ).to(device)
    decoder = EEGMEGMAEDecoderEigenval3(
        d_model=hp["d_model"], d_dec=hp["d_model"] // 2,
        n_heads=4, n_layers=hp["n_dec_layers"],
        patch_size=PATCH_SIZE,
    ).to(device)
    n_enc = sum(p.numel() for p in encoder.parameters()) / 1e6
    n_dec = sum(p.numel() for p in decoder.parameters()) / 1e6
    log.info(f"  Encoder {n_enc:.2f}M  Decoder {n_dec:.2f}M  mode_signature=Elambda")

    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=0.05, betas=(0.9, 0.95))
    warmup = max(1, int(hp["epochs"] * 0.05))

    def lr_lambda(ep: int) -> float:
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
        for x, _, ev in train_ds.iter_batches(hp["batch_size"], rng=rng):
            B, K, _ = x.shape
            mask = brainomni_2d_mask(B, K, NP, hp["mask_ratio"], device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                tgt  = patchify(x, PATCH_SIZE)
                enc  = encoder.encode_full_grid(x, ev, mask)
                pred = decoder(enc)
                m = mask.float().unsqueeze(-1)
                loss = ((pred.float() - tgt.float()) ** 2 * m).sum() / (m.sum() * PATCH_SIZE + 1e-8)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
            tl += loss.item() * B
            tn += B
        sched.step()
        train_loss = tl / max(tn, 1)

        encoder.eval(); decoder.eval()
        vl = vn = 0
        with torch.no_grad():
            for x, _, ev in val_ds.iter_batches(hp["batch_size"], shuffle=False):
                B, K, _ = x.shape
                mask = brainomni_2d_mask(B, K, NP, hp["mask_ratio"], device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    tgt  = patchify(x, PATCH_SIZE)
                    enc  = encoder.encode_full_grid(x, ev, mask)
                    pred = decoder(enc)
                    m = mask.float().unsqueeze(-1)
                    loss = ((pred.float() - tgt.float()) ** 2 * m).sum() / (m.sum() * PATCH_SIZE + 1e-8)
                vl += loss.item() * B
                vn += B
        val_loss = vl / max(vn, 1)

        improved = val_loss < best
        if improved:
            best       = val_loss
            best_epoch = epoch
            torch.save({
                "encoder": encoder.state_dict(),
                "epoch": epoch, "val_loss": val_loss, "run_id": RUN_ID,
                "n_modes": min(ds_all.Ks),
                "n_modes_per_subject": ds_all.Ks,
                "n_samples": T, "d_model": hp["d_model"],
                "n_factor_layers": hp["n_factor_layers"],
                "n_time_layers": hp["n_time_layers"],
                "patch_size": PATCH_SIZE,
                "mode_signature": "Elambda",
                "modality": modality, "tag": tag,
            }, out_path)
        if epoch % hp["log_every"] == 0 or epoch == 1 or epoch == hp["epochs"]:
            mk = " ✓saved" if improved else ""
            log.info(f"  PRE epoch {epoch:4d}/{hp['epochs']}  train={train_loss:.6f}  "
                     f"val={val_loss:.6f}  best={best:.6f}@{best_epoch}{mk}")

    log.info(f"\nAuto-pretrain done. best_val={best:.6f} @ epoch {best_epoch}")
    log.info(f"Checkpoint:  {out_path}")
    log.info("─" * 75 + "\n")

    del encoder, decoder, train_ds, val_ds, ds_all
    torch.cuda.empty_cache()
    return out_path





def load_prior_from_ckpt(ckpt_path: str, device: torch.device,
                         trainable: bool,
                         n_samples_override: int | None = None) -> EEGMEGPriorEigenval4:






    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = EEGMEGPriorEigenval4(
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
        log.warning(f"  load_state_dict — missing={missing}  unexpected={unexpected}")
    for p in enc.parameters():
        p.requires_grad = bool(trainable)
    enc.eval()

    enc.expected_n_samples = (n_samples_override if n_samples_override is not None
                              else ckpt.get("n_samples", 2560))
    enc.d_model_dim        = ckpt["d_model"]
    log.info(f"  prior loaded ({'trainable' if trainable else 'frozen'}): "
             f"d_model={ckpt['d_model']}  "
             f"pretrain K={ckpt.get('n_modes','?')}  "
             f"mode_sig={ckpt.get('mode_signature','?')}  "
             f"n_samples={enc.expected_n_samples}  "
             f"epoch={ckpt.get('epoch','?')}  "
             f"val_loss={ckpt.get('val_loss', float('nan')):.4f}")
    return enc


def load_brainomni_tiny(device: torch.device, finetune: bool):
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
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        log.info(f"  BrainOmni-tiny loaded  finetune=ON  (tokenizer frozen)  "
                 f"lm_dim={cfg['lm_dim']}  trainable={n_tr:.2f}M")
    else:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        log.info(f"  BrainOmni-tiny loaded & FROZEN  lm_dim={cfg['lm_dim']}")
    return model, cfg["lm_dim"]





class ObsEncoder(nn.Module):





    def __init__(self, backbone, lm_dim: int, n_tok: int, d_model: int,
                 backbone_trainable: bool):
        super().__init__()
        self.backbone = backbone
        self.n_tok    = n_tok
        self.backbone_trainable = backbone_trainable
        self.adapter  = nn.Sequential(
            nn.Linear(lm_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def encode_raw(self, x_raw, eeg_pos, sensor_type) -> torch.Tensor:

        if self.backbone_trainable:
            enc = self.backbone.encode(x_raw, eeg_pos, sensor_type)
        else:
            with torch.no_grad():
                enc = self.backbone.encode(x_raw, eeg_pos, sensor_type)
        feat = enc.float().mean(dim=1)
        feat = feat.permute(0, 2, 1)
        feat = F.adaptive_avg_pool1d(feat, self.n_tok)
        return feat.permute(0, 2, 1)

    def apply_adapter(self, feat: torch.Tensor) -> torch.Tensor:
        return self.adapter(feat)

    def forward(self, x_raw, eeg_pos, sensor_type):
        return self.apply_adapter(self.encode_raw(x_raw, eeg_pos, sensor_type))


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
        h = self.net(x)
        return self.mu_head(h), self.lv_head(h).clamp(-8, 4)


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
                 is_obs: bool, encoder_trainable: bool):
        super().__init__()
        self.encoder, self.ghead, self.cls_head = encoder, ghead, cls_head
        self.is_obs            = is_obs
        self.encoder_trainable = encoder_trainable

        self.eigval: torch.Tensor | None = None

    def set_eigval(self, eigval: torch.Tensor | None):
        self.eigval = eigval

    def encode(self, x_modes=None, x_raw=None, eeg_pos=None, sensor_type=None):
        if self.is_obs:
            tok = self.encoder(x_raw, eeg_pos, sensor_type)
        elif self.encoder_trainable:
            tok = self.encoder.encode_pooled(x_modes, eigval=self.eigval)
        else:
            with torch.no_grad():
                tok = self.encoder.encode_pooled(x_modes, eigval=self.eigval)
        mu, lv = self.ghead(tok)
        return mu, lv

    def forward(self, x_modes=None, x_raw=None, eeg_pos=None, sensor_type=None):
        mu, lv = self.encode(x_modes, x_raw, eeg_pos, sensor_type)
        return self.cls_head(mu), mu, lv

    def cached_forward(self, cached: torch.Tensor):

        if self.is_obs:
            tok = self.encoder.apply_adapter(cached)
        else:
            tok = cached
        mu, lv = self.ghead(tok)
        return self.cls_head(mu), mu, lv





class GPUDataset:

    def __init__(self, X_modes, X_raw, eeg_pos, sensor_type, y, device=None):
        self.X_modes     = X_modes.cpu()
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
            yield (self.X_modes[b].to(dev),  self.X_raw[b].to(dev),
                   self.eeg_pos[b].to(dev),  self.sensor_type[b].to(dev),
                   self.y[b].to(dev))


class CachedGPUDataset:

    def __init__(self, obs_feat: torch.Tensor, prior_tok: torch.Tensor,
                 y: torch.Tensor, device: torch.device):
        self.obs_feat  = obs_feat.to(device)
        self.prior_tok = prior_tok.to(device)
        self.y         = y.to(device)
        self.N         = len(y)
        self.device    = device

    def iter_batches(self, batch_size: int, shuffle: bool = True):
        if shuffle:
            idx = torch.randperm(self.N, device=self.device)
        else:
            idx = torch.arange(self.N, device=self.device)
        for s in range(0, self.N, batch_size):
            b = idx[s: s + batch_size]
            yield self.obs_feat[b], self.prior_tok[b], self.y[b]


def precompute_features(bundle: dict, idx: np.ndarray,
                        backbone, prior_enc: EEGMEGPriorEigenval4,
                        n_tok: int, eigval: torch.Tensor | None,
                        device: torch.device, batch_size: int = 64
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:



    backbone.eval(); prior_enc.eval()
    obs_feats, prior_toks, ys = [], [], []
    N = len(idx)
    idx_t = torch.as_tensor(idx, dtype=torch.long)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, N, batch_size):
            sub = idx_t[s: s + batch_size]
            xm  = bundle["modes"][sub].to(device)
            xr  = bundle["raw"][sub].to(device)
            pos = bundle["pos"][sub].to(device)
            stp = bundle["stype"][sub].to(device)
            yb  = bundle["labels"][sub]


            enc  = backbone.encode(xr, pos, stp)
            feat = enc.float().mean(dim=1).permute(0, 2, 1)
            feat = F.adaptive_avg_pool1d(feat, n_tok).permute(0, 2, 1)
            obs_feats.append(feat.cpu())


            tok = prior_enc.encode_pooled(xm, eigval=eigval)
            prior_toks.append(tok.float().cpu())

            ys.append(yb)

    return torch.cat(obs_feats), torch.cat(prior_toks), torch.cat(ys)





def _set_train_modes(model: ExpertModel):
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


def train_expert(model: ExpertModel, train_ds: GPUDataset, val_ds: GPUDataset,
                 epochs: int, batch_size: int,
                 lr_adapter: float, lr_head: float, lr_backbone: float,
                 patience: int, kl_weight: float, label: str) -> tuple[float, dict]:
    trainable = []
    if model.is_obs:
        trainable.append({"params": model.encoder.adapter.parameters(),
                          "lr": lr_adapter, "weight_decay": 1e-3})
        if model.encoder.backbone_trainable:
            bb_params = [p for p in model.encoder.backbone.parameters() if p.requires_grad]
            if bb_params:
                trainable.append({"params": bb_params,
                                  "lr": lr_backbone, "weight_decay": 1e-4})
    else:
        if model.encoder_trainable:
            enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
            if enc_params:
                trainable.append({"params": enc_params,
                                  "lr": lr_backbone, "weight_decay": 1e-4})
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
            nn.utils.clip_grad_norm_(
                [p for g in trainable for p in g["params"]], 1.0)
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





def train_expert_cached(model: ExpertModel,
                        train_ds: CachedGPUDataset, val_ds: CachedGPUDataset,
                        branch: str, epochs: int, batch_size: int,
                        lr_adapter: float, lr_head: float,
                        patience: int, kl_weight: float, label: str
                        ) -> tuple[float, dict]:

    assert branch in ("obs", "prior")
    trainable = []
    if branch == "obs":
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

    best_acc, best_state, no_improve = 0.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        c = n = 0
        for of, pt, yb in train_ds.iter_batches(batch_size):
            cached = of if branch == "obs" else pt
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, mu, lv = model.cached_forward(cached)
                kl   = 0.5 * (lv.exp() + mu.pow(2) - 1 - lv).mean()
                loss = criterion(logits, yb) + kl_weight * kl
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(
                [p for g in trainable for p in g["params"]], 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            c += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
        train_acc = c / max(n, 1)

        model.eval()
        vc = vt = 0
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for of, pt, yb in val_ds.iter_batches(batch_size, shuffle=False):
                cached = of if branch == "obs" else pt
                vl, _, _ = model.cached_forward(cached)
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
        "temp_T_obs_A":     T_obs_A, "temp_T_pri_A": T_pri_A,
        "temp_T_obs_B":     T_obs_B, "temp_T_pri_B": T_pri_B,
        "temp_T_obs_mean":  float(0.5 * (T_obs_A + T_obs_B)),
        "temp_T_pri_mean":  float(0.5 * (T_pri_A + T_pri_B)),
    }


def _ensemble_from_logs(log_obs, log_pri, yv, n_classes, z_obs=None, z_pri=None):






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
    out = {
        "poe_conf": acc_poe, "poe_arith": acc_arith, "poe_entropy": acc_ent,
        "poe_conf_balacc":  bal_poe,
        "poe_arith_balacc": bal_arith,
        "poe_entropy_balacc": bal_ent,
        "poe_conf_recall":  rec_poe,
        "poe_arith_recall": rec_arith,
        "poe_entropy_recall": rec_ent,
    }
    if z_obs is not None and z_pri is not None:
        z_obs_np = z_obs.cpu().numpy() if torch.is_tensor(z_obs) else z_obs
        z_pri_np = z_pri.cpu().numpy() if torch.is_tensor(z_pri) else z_pri
        temp_info = _temp_scaled_fusion(z_obs_np, z_pri_np, yt_np, n_classes, seed=0)
        rec_line_temp = "  ".join([f"c{i}={r:.3f}"
                                    for i, r in enumerate(temp_info['poe_temp_recall'])])
        log.info(f"    Temp-PoE acc={temp_info['poe_temp']:.4f}  "
                 f"BalAcc={temp_info['poe_temp_balacc']:.4f}  per-class: {rec_line_temp}")
        log.info(f"      T_obs={temp_info['temp_T_obs_mean']:.3f}  "
                 f"T_pri={temp_info['temp_T_pri_mean']:.3f}")
        out.update(temp_info)
    return out


def eval_branch_detailed(model: ExpertModel, ds: GPUDataset, batch_size: int,
                         n_classes: int, label: str) -> dict:
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


def eval_branch_cached(model: ExpertModel, ds: CachedGPUDataset, branch: str,
                       batch_size: int, n_classes: int, label: str) -> dict:
    assert branch in ("obs", "prior")
    model.eval()
    preds, ys = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for of, pt, yb in ds.iter_batches(batch_size, shuffle=False):
            cached = of if branch == "obs" else pt
            logits, _, _ = model.cached_forward(cached)
            preds.append(logits.argmax(1).cpu())
            ys.append(yb.cpu())
    yp = torch.cat(preds).numpy()
    yt = torch.cat(ys).numpy()
    acc    = float((yp == yt).mean())
    balacc = float(balanced_accuracy_score(yt, yp))
    rec, rec_line = _per_class_recall(yt, yp, n_classes)
    log.info(f"    [{label}] acc={acc:.4f}  BalAcc={balacc:.4f}  per-class: {rec_line}")
    return {"acc": acc, "balacc": balacc, "per_class_recall": rec}


def eval_ensemble(obs_model: ExpertModel, prior_model: ExpertModel,
                  val_ds: GPUDataset, n_classes: int, batch_size: int = 64) -> dict:
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
    return _ensemble_from_logs(F.log_softmax(z_obs_t, dim=-1),
                                F.log_softmax(z_pri_t, dim=-1),
                                yv, n_classes,
                                z_obs=z_obs_t, z_pri=z_pri_t)


def eval_ensemble_cached(obs_model: ExpertModel, prior_model: ExpertModel,
                         val_ds: CachedGPUDataset, n_classes: int,
                         batch_size: int = 64,
                         return_logits: bool = False) -> dict:
    obs_model.eval(); prior_model.eval()
    all_z_obs, all_z_pri, all_y = [], [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for of, pt, yb in val_ds.iter_batches(batch_size, shuffle=False):
            lo, _, _ = obs_model.cached_forward(of)
            lp, _, _ = prior_model.cached_forward(pt)
            all_z_obs.append(lo.float().cpu())
            all_z_pri.append(lp.float().cpu())
            all_y.append(yb.cpu())
    z_obs_t = torch.cat(all_z_obs); z_pri_t = torch.cat(all_z_pri); yv = torch.cat(all_y)
    out = _ensemble_from_logs(F.log_softmax(z_obs_t, dim=-1),
                               F.log_softmax(z_pri_t, dim=-1),
                               yv, n_classes,
                               z_obs=z_obs_t, z_pri=z_pri_t)
    if return_logits:
        return out, z_obs_t, z_pri_t, yv
    return out





def discover_subjects(h5_dir: str) -> list[str]:
    files = sorted(f for f in os.listdir(h5_dir) if f.startswith("sub-") and f.endswith(".h5"))
    return [os.path.splitext(f)[0] for f in files]


def load_subject(h5_dir: str, subj: str, n_samples: int, modality: str = "eeg",
                 label_key: str = "labels",
                 override_eigval: torch.Tensor | None = None):


    h5_path = os.path.join(h5_dir, f"{subj}.h5")
    if not os.path.isfile(h5_path):
        log.warning(f"  {subj}: not found — skip")
        return None
    ev_key = f"{modality}_mode_eigval"
    with h5py.File(h5_path, "r") as f:
        modes  = torch.tensor(f[f"{modality}_modes"][:],      dtype=torch.float32)
        raw    = torch.tensor(f[f"{modality}_raw"][:],         dtype=torch.float32)
        labels = torch.tensor(f[label_key][:],                  dtype=torch.long)
        pos    = torch.tensor(f[f"{modality}_pos"][:],         dtype=torch.float32)
        stype  = torch.tensor(f[f"{modality}_sensor_type"][:], dtype=torch.long)
        if override_eigval is None:
            if ev_key not in f:
                raise KeyError(
                    f"{h5_path}: missing '{ev_key}' — provide --d_path or pre-patch h5."
                )
            eigval = torch.tensor(f[ev_key][:], dtype=torch.float32)
        else:
            K_in = modes.shape[1]
            if override_eigval.shape[0] != K_in:
                raise ValueError(
                    f"{subj}: modes K={K_in} ≠ --d_path eigval K={override_eigval.shape[0]}"
                )
            eigval = override_eigval.detach().cpu().float()
    modes = modes[:, :, :n_samples]
    raw   = raw[:,   :, :n_samples]
    N = len(labels)
    return {
        "modes":  modes,
        "raw":    raw,
        "labels": labels,
        "pos":    pos.unsqueeze(0).expand(N, -1, -1).contiguous(),
        "stype":  stype.unsqueeze(0).expand(N, -1).contiguous(),
        "eigval": eigval,
        "subj":   subj,
        "K":      modes.shape[1],
    }


def stack_subjects(items: list[dict]) -> dict:
    Ks = {it["K"] for it in items}
    if len(Ks) != 1:
        raise ValueError(f"Cross-subject pooling needs uniform K across subjects, got {Ks}. "
                         f"Either re-extract modes with consistent K or fall back to --cv_mode within.")

    ev0 = items[0]["eigval"]
    for it in items[1:]:
        if it["eigval"].shape != ev0.shape:
            raise ValueError(f"eigval shape mismatch: {it['subj']}={tuple(it['eigval'].shape)} "
                             f"vs ref={tuple(ev0.shape)}")
        max_diff = (it["eigval"] - ev0).abs().max().item()
        rel = max_diff / (ev0.abs().max().item() + 1e-12)
        if rel > 1e-4:
            log.warning(f"  eigval mismatch: {it['subj']} vs ref  max|Δ|={max_diff:.3g}  rel={rel:.3g}")
    return {
        "modes":  torch.cat([it["modes"]  for it in items], dim=0),
        "raw":    torch.cat([it["raw"]    for it in items], dim=0),
        "labels": torch.cat([it["labels"] for it in items], dim=0),
        "pos":    torch.cat([it["pos"]    for it in items], dim=0),
        "stype":  torch.cat([it["stype"]  for it in items], dim=0),
        "subj_ids": np.concatenate([np.array([it["subj"]] * len(it["labels"])) for it in items]),
        "eigval": ev0,
        "K":      items[0]["K"],
    }





def _make_prior_for_fold(prior_template: EEGMEGPriorEigenval4,
                         prior_init_state: dict,
                         trainable: bool, device: torch.device) -> EEGMEGPriorEigenval4:


    if not trainable:
        return prior_template
    enc = copy.deepcopy(prior_template).to(device)
    enc.load_state_dict(prior_init_state)
    for p in enc.parameters():
        p.requires_grad = True
    enc.expected_n_samples = prior_template.expected_n_samples
    enc.d_model_dim        = prior_template.d_model_dim
    return enc


def _make_backbone_for_fold(backbone_template, backbone_init_state: dict | None,
                            finetune: bool, device: torch.device):
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


def run_one_fold(bundle: dict, tr_idx: np.ndarray, te_idx: np.ndarray,
                 prior_template: EEGMEGPriorEigenval4, prior_init_state: dict,
                 backbone_template, backbone_init_state: dict | None, lm_dim: int,
                 device: torch.device, args, fold_dir: str, label: str,
                 test_bundle: dict | None = None) -> dict:
    n_tok   = prior_template.expected_n_samples // PATCH_SIZE
    d_model = prior_template.d_model_dim
    finetune = (args.train_mode == "finetune")
    use_cache = bool(args.cache_features) and not finetune

    log.info(f"  [{label}] train={len(tr_idx)}  test={len(te_idx)}  K={bundle['K']}  "
             f"n_tok={n_tok}  d_model={d_model}  "
             f"train_mode={args.train_mode}  cache={'on' if use_cache else 'off'}")

    eigval = bundle["eigval"].to(device)

    if use_cache:

        log.info(f"  [{label}] precomputing features …")
        of_tr, pt_tr, y_tr = precompute_features(
            bundle, tr_idx, backbone_template, prior_template, n_tok,
            eigval, device, args.batch_size,
        )
        of_te, pt_te, y_te = precompute_features(
            bundle, te_idx, backbone_template, prior_template, n_tok,
            eigval, device, args.batch_size,
        )
        log.info(f"  [{label}] cached  obs_feat={tuple(of_tr.shape)}  "
                 f"prior_tok={tuple(pt_tr.shape)}")
        tr_ds = CachedGPUDataset(of_tr, pt_tr, y_tr, device)
        te_ds = CachedGPUDataset(of_te, pt_te, y_te, device)


        obs_enc   = ObsEncoder(backbone_template, lm_dim, n_tok, d_model,
                               backbone_trainable=False).to(device)
        obs_ghead = GaussHead(d_model, args.lat_dim).to(device)
        obs_cls   = ClsHead(args.lat_dim, n_classes=args.n_classes, dropout=args.dropout).to(device)
        obs_model = ExpertModel(obs_enc, obs_ghead, obs_cls,
                                is_obs=True, encoder_trainable=False).to(device)
        obs_acc, obs_state = train_expert_cached(
            obs_model, tr_ds, te_ds, branch="obs",
            epochs=args.epochs, batch_size=args.batch_size,
            lr_adapter=args.lr_adapter, lr_head=args.lr_head,
            patience=args.patience, kl_weight=args.kl_weight,
            label=f"{label}.obs",
        )

        pri_ghead = GaussHead(d_model, args.lat_dim).to(device)
        pri_cls   = ClsHead(args.lat_dim, n_classes=args.n_classes, dropout=args.dropout).to(device)
        pri_model = ExpertModel(prior_template, pri_ghead, pri_cls,
                                is_obs=False, encoder_trainable=False).to(device)
        pri_model.set_eigval(eigval)
        pri_acc, pri_state = train_expert_cached(
            pri_model, tr_ds, te_ds, branch="prior",
            epochs=args.epochs, batch_size=args.batch_size,
            lr_adapter=args.lr_adapter, lr_head=args.lr_head,
            patience=args.patience, kl_weight=args.kl_weight,
            label=f"{label}.pri",
        )

        log.info(f"  [{label}] VAL fusion eval (cached):")
        fuse, _z_obs_v, _z_pri_v, _y_v = eval_ensemble_cached(
            obs_model, pri_model, te_ds,
            args.n_classes, args.batch_size, return_logits=True)
        val_obs_detail = eval_branch_cached(obs_model, te_ds, "obs",
                                             args.batch_size, args.n_classes,
                                             f"{label}.val.obs")
        val_pri_detail = eval_branch_cached(pri_model, te_ds, "prior",
                                             args.batch_size, args.n_classes,
                                             f"{label}.val.pri")


        os.makedirs(fold_dir, exist_ok=True)
        np.savez(os.path.join(fold_dir, "holdout_preds.npz"),
                 z_obs=_z_obs_v.numpy(), z_pri=_z_pri_v.numpy(), y=_y_v.numpy(),
                 test_idx=np.asarray(te_idx, dtype=np.int64))
        torch.save({k: v for k, v in obs_state.items()
                    if not k.startswith("encoder.backbone.")},
                   os.path.join(fold_dir, "obs.pt"))
        torch.save({k: v for k, v in pri_state.items()
                    if not k.startswith("encoder.")},
                   os.path.join(fold_dir, "prior.pt"))

        metrics = {
            "label":    label,
            "train_mode": args.train_mode,
            "cache_features": True,
            "obs_acc":  obs_acc,
            "prior_acc": pri_acc,
            **fuse,
            "val_obs_balacc": val_obs_detail["balacc"],
            "val_pri_balacc": val_pri_detail["balacc"],
            "val_obs_recall": val_obs_detail["per_class_recall"],
            "val_pri_recall": val_pri_detail["per_class_recall"],
            "train_idx": [int(i) for i in tr_idx.tolist()],
            "test_idx":  [int(i) for i in te_idx.tolist()],
            "n_train":  int(tr_ds.N),
            "n_test":   int(te_ds.N),
        }

        if test_bundle is not None:
            log.info(f"  [{label}] HELD-OUT TEST eval on {len(test_bundle['labels'])} trials (cached):")
            te_eigval = test_bundle["eigval"].to(device)
            of_ts, pt_ts, y_ts = precompute_features(
                test_bundle, np.arange(len(test_bundle["labels"])),
                backbone_template, prior_template, n_tok,
                te_eigval, device, args.batch_size,
            )
            test_ds = CachedGPUDataset(of_ts, pt_ts, y_ts, device)
            test_obs_detail = eval_branch_cached(obs_model, test_ds, "obs",
                                                  args.batch_size, args.n_classes,
                                                  f"{label}.test.obs")
            test_pri_detail = eval_branch_cached(pri_model, test_ds, "prior",
                                                  args.batch_size, args.n_classes,
                                                  f"{label}.test.pri")
            test_fuse = eval_ensemble_cached(obs_model, pri_model, test_ds,
                                              args.n_classes, args.batch_size)
            metrics["test_obs"]   = test_obs_detail
            metrics["test_prior"] = test_pri_detail
            metrics["test_fuse"]  = test_fuse
            del test_ds

        del obs_model, pri_model, tr_ds, te_ds, of_tr, pt_tr, of_te, pt_te, y_tr, y_te
        torch.cuda.empty_cache()

    else:

        tr = GPUDataset(bundle["modes"][tr_idx], bundle["raw"][tr_idx],
                        bundle["pos"][tr_idx],   bundle["stype"][tr_idx],
                        bundle["labels"][tr_idx], device=device)
        te = GPUDataset(bundle["modes"][te_idx], bundle["raw"][te_idx],
                        bundle["pos"][te_idx],   bundle["stype"][te_idx],
                        bundle["labels"][te_idx], device=device)

        backbone = _make_backbone_for_fold(backbone_template, backbone_init_state,
                                           finetune, device)
        obs_enc   = ObsEncoder(backbone, lm_dim, n_tok, d_model,
                               backbone_trainable=finetune).to(device)
        obs_ghead = GaussHead(d_model, args.lat_dim).to(device)
        obs_cls   = ClsHead(args.lat_dim, n_classes=args.n_classes, dropout=args.dropout).to(device)
        obs_model = ExpertModel(obs_enc, obs_ghead, obs_cls,
                                is_obs=True, encoder_trainable=False).to(device)
        obs_acc, obs_state = train_expert(obs_model, tr, te,
                                          args.epochs, args.batch_size,
                                          args.lr_adapter, args.lr_head, args.lr_backbone,
                                          args.patience, args.kl_weight, f"{label}.obs")

        prior_enc = _make_prior_for_fold(prior_template, prior_init_state, finetune, device)
        pri_ghead = GaussHead(d_model, args.lat_dim).to(device)
        pri_cls   = ClsHead(args.lat_dim, n_classes=args.n_classes, dropout=args.dropout).to(device)
        pri_model = ExpertModel(prior_enc, pri_ghead, pri_cls,
                                is_obs=False, encoder_trainable=finetune).to(device)
        pri_model.set_eigval(eigval)
        pri_acc, pri_state = train_expert(pri_model, tr, te,
                                          args.epochs, args.batch_size,
                                          args.lr_adapter, args.lr_head, args.lr_backbone,
                                          args.patience, args.kl_weight, f"{label}.pri")

        log.info(f"  [{label}] VAL fusion eval:")
        fuse = eval_ensemble(obs_model, pri_model, te, args.n_classes, args.batch_size)
        val_obs_detail = eval_branch_detailed(obs_model, te, args.batch_size,
                                              args.n_classes, f"{label}.val.obs")
        val_pri_detail = eval_branch_detailed(pri_model, te, args.batch_size,
                                              args.n_classes, f"{label}.val.pri")

        os.makedirs(fold_dir, exist_ok=True)
        if finetune:
            obs_state_to_save = obs_state
        else:
            obs_state_to_save = {k: v for k, v in obs_state.items()
                                 if not k.startswith("encoder.backbone.")}
        torch.save(obs_state_to_save, os.path.join(fold_dir, "obs.pt"))
        if finetune:
            pri_state_to_save = pri_state
        else:
            pri_state_to_save = {k: v for k, v in pri_state.items()
                                 if not k.startswith("encoder.")}
        torch.save(pri_state_to_save, os.path.join(fold_dir, "prior.pt"))

        metrics = {
            "label":    label,
            "train_mode": args.train_mode,
            "cache_features": False,
            "obs_acc":  obs_acc,
            "prior_acc": pri_acc,
            **fuse,
            "val_obs_balacc": val_obs_detail["balacc"],
            "val_pri_balacc": val_pri_detail["balacc"],
            "val_obs_recall": val_obs_detail["per_class_recall"],
            "val_pri_recall": val_pri_detail["per_class_recall"],
            "train_idx": [int(i) for i in tr_idx.tolist()],
            "test_idx":  [int(i) for i in te_idx.tolist()],
            "n_train":  int(tr.N),
            "n_test":   int(te.N),
        }

        if test_bundle is not None:
            log.info(f"  [{label}] HELD-OUT TEST eval on {len(test_bundle['labels'])} trials:")
            pri_model.set_eigval(test_bundle["eigval"].to(device))
            test_ds = GPUDataset(test_bundle["modes"], test_bundle["raw"],
                                  test_bundle["pos"], test_bundle["stype"],
                                  test_bundle["labels"], device=device)
            test_obs_detail = eval_branch_detailed(obs_model, test_ds, args.batch_size,
                                                    args.n_classes, f"{label}.test.obs")
            test_pri_detail = eval_branch_detailed(pri_model, test_ds, args.batch_size,
                                                    args.n_classes, f"{label}.test.pri")
            test_fuse = eval_ensemble(obs_model, pri_model, test_ds,
                                       args.n_classes, args.batch_size)
            metrics["test_obs"]   = test_obs_detail
            metrics["test_prior"] = test_pri_detail
            metrics["test_fuse"]  = test_fuse
            pri_model.set_eigval(eigval)
            del test_ds

        if finetune:
            del backbone, prior_enc
        del obs_model, pri_model, tr, te
        torch.cuda.empty_cache()

    with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics





def _prepare_within_splits(h5_dir: str, subjects: list[str], n_samples: int,
                           args, root_dir: str,
                           override_eigval: torch.Tensor | None = None) -> dict:
    out = {}
    splits_log = {}
    for subj in subjects:
        it = load_subject(h5_dir, subj, n_samples, args.modality, args.label_key,
                          override_eigval=override_eigval)
        if it is None or len(it["labels"]) < args.n_folds:
            log.warning(f"  {subj}: skipped (insufficient trials)")
            continue
        N = len(it["labels"])
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        folds = list(skf.split(np.zeros(N), it["labels"].numpy()))
        out[subj] = (it, folds)
        splits_log[subj] = []

        subj_dir = os.path.join(root_dir, subj)
        os.makedirs(subj_dir, exist_ok=True)
        for fid, (tr_idx, te_idx) in enumerate(folds):
            fold_dir = os.path.join(subj_dir, f"fold_{fid:02d}")
            os.makedirs(fold_dir, exist_ok=True)
            split_d = {"train_idx": tr_idx.tolist(),
                       "test_idx":  te_idx.tolist(),
                       "n_train":   int(len(tr_idx)),
                       "n_test":    int(len(te_idx)),
                       "subject":   subj,
                       "fold":      fid}
            with open(os.path.join(fold_dir, "split.json"), "w") as f:
                json.dump(split_d, f, indent=2)
            splits_log[subj].append({"train": tr_idx.tolist(), "test": te_idx.tolist()})
        log.info(f"  {subj}: N={N}  K={it['K']}  folds reserved → {subj_dir}/fold_*/")

    with open(os.path.join(root_dir, "splits.json"), "w") as f:
        json.dump(splits_log, f, indent=2)
    return out


def run_within(h5_dir: str, subjects: list[str], prior_template, prior_init_state,
               backbone_template, backbone_init_state, lm_dim,
               device: torch.device, args, root_dir: str,
               override_eigval: torch.Tensor | None = None) -> dict:
    n_samples = prior_template.expected_n_samples
    plan = _prepare_within_splits(h5_dir, subjects, n_samples, args, root_dir,
                                   override_eigval=override_eigval)

    splits_log = {}
    per_subj   = {}
    for subj, (it, folds) in plan.items():
        log.info(f"\n========== SUBJECT {subj}  N={len(it['labels'])}  K={it['K']} ==========")
        splits_log[subj] = [{"train": tr.tolist(), "test": te.tolist()} for tr, te in folds]


        bundle = {
            "modes":  it["modes"],
            "raw":    it["raw"],
            "labels": it["labels"],
            "pos":    it["pos"],
            "stype":  it["stype"],
            "eigval": it["eigval"],
            "K":      it["K"],
        }

        subj_dir = os.path.join(root_dir, subj)
        subj_metrics = []
        for fid, (tr_idx, te_idx) in enumerate(folds):
            if args.fold is not None and fid != args.fold:
                continue
            fold_dir = os.path.join(subj_dir, f"fold_{fid:02d}")
            metrics_path = os.path.join(fold_dir, "metrics.json")
            if os.path.isfile(metrics_path):
                with open(metrics_path) as f:
                    m = json.load(f)
                log.info(f"  [{subj}.f{fid}] RESUME — metrics.json exists, skip training")
                subj_metrics.append(m)
                continue
            m = run_one_fold(bundle, tr_idx, te_idx,
                             prior_template, prior_init_state,
                             backbone_template, backbone_init_state, lm_dim,
                             device, args, fold_dir, label=f"{subj}.f{fid}")
            subj_metrics.append(m)
        per_subj[subj] = subj_metrics

    return {"per_subject": per_subj, "splits": splits_log}


def _prepare_cross_splits(subjects: list[str], bundle: dict, args,
                          root_dir: str, test_subjects: list[str]) -> list[dict]:
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_subjs = list(kf.split(subjects))
    splits_log = []
    for fid, (tr_subj_idx, te_subj_idx) in enumerate(fold_subjs):
        tr_subj = {subjects[i] for i in tr_subj_idx}
        te_subj = {subjects[i] for i in te_subj_idx}
        is_tr   = np.array([s in tr_subj for s in bundle["subj_ids"]])
        tr_idx  = np.where(is_tr)[0]
        te_idx  = np.where(~is_tr)[0]
        entry = {
            "fold": fid,
            "train_subjects": sorted(tr_subj),
            "val_subjects":   sorted(te_subj),
            "n_train": int(len(tr_idx)),
            "n_val":   int(len(te_idx)),
            "test_subjects": sorted(test_subjects) if test_subjects else None,
            "_tr_idx": tr_idx, "_te_idx": te_idx,
        }
        splits_log.append(entry)

        fold_dir = os.path.join(root_dir, f"fold_{fid:02d}")
        os.makedirs(fold_dir, exist_ok=True)
        with open(os.path.join(fold_dir, "split.json"), "w") as f:
            json.dump({k: v for k, v in entry.items() if not k.startswith("_")}, f, indent=2)

    with open(os.path.join(root_dir, "splits.json"), "w") as f:
        json.dump([{k: v for k, v in e.items() if not k.startswith("_")} for e in splits_log],
                  f, indent=2)
    return splits_log


def run_cross(h5_dir: str, subjects: list[str], prior_template, prior_init_state,
              backbone_template, backbone_init_state, lm_dim,
              device: torch.device, args, root_dir: str,
              override_eigval: torch.Tensor | None = None) -> dict:
    n_samples = prior_template.expected_n_samples
    test_prefix = args.test_subject_prefix or ""

    cv_subjects, test_subjects = [], []
    for subj in subjects:
        if test_prefix and subj.startswith(test_prefix):
            test_subjects.append(subj)
        else:
            cv_subjects.append(subj)

    log.info(f"  CV pool  : {len(cv_subjects)} subjects (prefix-excluded)")
    if test_prefix:
        log.info(f"  Held-out test : {len(test_subjects)} subjects (prefix='{test_prefix}')")
    if len(cv_subjects) < args.n_folds:
        raise ValueError(f"Need ≥ n_folds={args.n_folds} subjects for cross-subject CV, "
                         f"got {len(cv_subjects)} after prefix filtering")

    items = []
    for subj in cv_subjects:
        it = load_subject(h5_dir, subj, n_samples, args.modality, args.label_key,
                          override_eigval=override_eigval)
        if it is not None:
            items.append(it)

    bundle = stack_subjects(items)
    N = len(bundle["labels"])
    log.info(f"\n  Pooled N={N} trials across {len(items)} CV subjects (uniform K={bundle['K']})")
    log.info(f"  CV class dist: {torch.bincount(bundle['labels']).tolist()}")

    test_bundle = None
    if test_subjects:
        log.info(f"  Loading held-out test subjects …")
        test_items = []
        for subj in test_subjects:
            it = load_subject(h5_dir, subj, n_samples, args.modality, args.label_key)
            if it is not None:
                test_items.append(it)
        if test_items:
            test_bundle = stack_subjects(test_items)
            if test_bundle["K"] != bundle["K"]:
                raise ValueError(f"CV/test K mismatch: CV={bundle['K']} vs test={test_bundle['K']}")
            log.info(f"  Held-out test  N={len(test_bundle['labels'])} trials  "
                     f"K={test_bundle['K']}")
            log.info(f"  Test class dist: {torch.bincount(test_bundle['labels']).tolist()}")

    subj_list = [it["subj"] for it in items]
    splits_log = _prepare_cross_splits(subj_list, bundle, args, root_dir, test_subjects)

    fold_metrics = []
    for entry in splits_log:
        fid = entry["fold"]
        if args.fold is not None and fid != args.fold:
            continue
        log.info(f"\n========== FOLD {fid+1}/{args.n_folds}  "
                 f"train={entry['n_train']} trials  val={entry['n_val']} trials"
                 + (f"  test={len(test_subjects)} subj" if test_bundle is not None else "")
                 + " ==========")
        fold_dir = os.path.join(root_dir, f"fold_{fid:02d}")
        metrics_path = os.path.join(fold_dir, "metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            log.info(f"  [f{fid}] RESUME — metrics.json exists, skip training")
            fold_metrics.append(m)
            continue
        m = run_one_fold(bundle, entry["_tr_idx"], entry["_te_idx"],
                         prior_template, prior_init_state,
                         backbone_template, backbone_init_state, lm_dim,
                         device, args, fold_dir, label=f"f{fid}",
                         test_bundle=test_bundle)

        if "subj_ids" in bundle:
            np.save(os.path.join(fold_dir, "holdout_subj_ids.npy"),
                    np.asarray(bundle["subj_ids"])[entry["_te_idx"]])
        m["train_subjects"] = entry["train_subjects"]
        m["val_subjects"]   = entry["val_subjects"]
        m["test_subjects"]  = entry["test_subjects"]
        fold_metrics.append(m)

    public_splits = [{k: v for k, v in e.items() if not k.startswith("_")} for e in splits_log]
    return {"per_fold": fold_metrics, "splits": public_splits}





def summarize(results: dict, mode: str) -> dict:
    def stat(v):
        a = np.array(v)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "values": [float(x) for x in v]}

    keys = ("obs_acc", "prior_acc", "poe_arith", "poe_conf", "poe_entropy", "poe_temp")
    def stat_opt(vals):
        v = [x for x in vals if x is not None]
        return stat(v) if v else None
    if mode == "cross":
        rows = results["per_fold"]
        out = {k: stat_opt([r.get(k) for r in rows]) for k in keys}
        if rows and rows[0].get("test_fuse") is not None:
            out["test_obs_acc"]    = stat([r["test_obs"]["acc"]    for r in rows])
            out["test_obs_balacc"] = stat([r["test_obs"]["balacc"] for r in rows])
            out["test_pri_acc"]    = stat([r["test_prior"]["acc"]    for r in rows])
            out["test_pri_balacc"] = stat([r["test_prior"]["balacc"] for r in rows])
            out["test_poe_arith_acc"]    = stat([r["test_fuse"]["poe_arith"]     for r in rows])
            out["test_poe_arith_balacc"] = stat([r["test_fuse"]["poe_arith_balacc"] for r in rows])
            out["test_poe_conf_acc"]     = stat([r["test_fuse"]["poe_conf"]      for r in rows])
            out["test_poe_conf_balacc"]  = stat([r["test_fuse"]["poe_conf_balacc"]  for r in rows])
            out["test_poe_ent_acc"]      = stat([r["test_fuse"]["poe_entropy"]   for r in rows])
            out["test_poe_ent_balacc"]   = stat([r["test_fuse"]["poe_entropy_balacc"] for r in rows])
            if "poe_temp" in rows[0]["test_fuse"]:
                out["test_poe_temp_acc"]    = stat([r["test_fuse"]["poe_temp"]        for r in rows])
                out["test_poe_temp_balacc"] = stat([r["test_fuse"]["poe_temp_balacc"] for r in rows])
        return out
    else:
        all_folds = [m for subj_rows in results["per_subject"].values() for m in subj_rows]
        overall = {k: stat_opt([m.get(k) for m in all_folds]) for k in keys}
        per_subj = {
            subj: {k: stat_opt([m.get(k) for m in subj_rows]) for k in keys}
            for subj, subj_rows in results["per_subject"].items()
        }
        return {"overall": overall, "per_subject": per_subj}





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_dir",       type=str, required=True)
    ap.add_argument("--encoder_ckpt", type=str, default=None,
                    help="K-free MAE ckpt (v3/v4 compatible). If omitted, auto-pretrain "
                         "runs on --h5_dir with E[λ] mode signature.")
    ap.add_argument("--train_mode",   type=str, choices=["frozen", "finetune"], default="frozen",
                    help="frozen: only adapter+heads train. "
                         "finetune: BrainOmni (except tokenizer) and prior encoder also train.")

    ap.add_argument("--cache_features", dest="cache_features", action="store_true",
                    help="Precompute backbone outputs once per fold and train only "
                         "adapter+heads on cached features (frozen-mode only).")
    ap.add_argument("--no_cache_features", dest="cache_features", action="store_false",
                    help="Disable feature caching (full backbone forward each batch).")
    ap.set_defaults(cache_features=True)

    ap.add_argument("--cv_mode",      type=str, choices=["within", "cross"], default="within")
    ap.add_argument("--modality",     type=str, choices=["eeg", "meg", "fnirs"], default="eeg")
    ap.add_argument("--label_key",    type=str, default="labels")
    ap.add_argument("--subjects",     type=str, nargs="+", default=None)
    ap.add_argument("--n_folds",      type=int,   default=5)
    ap.add_argument("--fold",         type=int,   default=None)
    ap.add_argument("--epochs",       type=int,   default=300)
    ap.add_argument("--batch_size",   type=int,   default=32)
    ap.add_argument("--lr_adapter",   type=float, default=1e-3)
    ap.add_argument("--lr_head",      type=float, default=1e-3)
    ap.add_argument("--lr_backbone",  type=float, default=1e-4,
                    help="Applied to prior encoder + BrainOmni (non-tokenizer). "
                         "Used only when --train_mode=finetune.")
    ap.add_argument("--lat_dim",      type=int,   default=256)
    ap.add_argument("--dropout",      type=float, default=0.3)
    ap.add_argument("--patience",     type=int,   default=80)
    ap.add_argument("--kl_weight",    type=float, default=1e-3)
    ap.add_argument("--n_classes",    type=int,   default=4)
    ap.add_argument("--device",       type=str,   default="cuda:0")
    ap.add_argument("--seed",         type=int,   default=0)
    ap.add_argument("--tag",          type=str,   default=None)
    ap.add_argument("--out_dir",      type=str,   default=os.path.join(SCRIPT_DIR, "runs"))
    ap.add_argument("--run_id",       type=str,   default=None)
    ap.add_argument("--single_subject", action="store_true")
    ap.add_argument("--test_subject_prefix", type=str, default="")


    ap.add_argument("--d_path",          type=str, default=None,
                    help="Path to D = Φ_sensor .npy. If set, eigval is computed via "
                         "compute_eigval_from_D and used for every subject (overrides h5).")
    ap.add_argument("--lam_cortex_path", type=str, default=None,
                    help="Optional individual λ_cortex .npy used together with --d_path; "
                         "default is fs32k group from eeg_meg_prior_eigenval4.")
    ap.add_argument("--ratio",           type=float, default=1e-3,
                    help="σ-cutoff ratio for D SVD truncation (only with --d_path).")
    ap.add_argument("--n_samples",       type=int, default=None,
                    help="Override the prior encoder's expected samples-per-trial "
                         "(default: read from ckpt, fallback 2560). Required when "
                         "the dataset's trial length differs from the pretrain "
                         "default (e.g. somatomotor = 512).")


    ap.add_argument("--pretrain_epochs",     type=int,   default=200)
    ap.add_argument("--pretrain_batch_size", type=int,   default=64)
    ap.add_argument("--pretrain_lr",         type=float, default=3e-4)
    ap.add_argument("--pretrain_d_model",    type=int,   default=512)
    ap.add_argument("--pretrain_n_factor_layers", type=int, default=2)
    ap.add_argument("--pretrain_n_time_layers",   type=int, default=4)
    ap.add_argument("--pretrain_n_dec_layers",    type=int, default=2)
    ap.add_argument("--pretrain_mask_ratio", type=float, default=0.5)
    ap.add_argument("--pretrain_val_ratio",  type=float, default=0.1)
    ap.add_argument("--pretrain_log_every",  type=int,   default=10)

    args = ap.parse_args()


    if args.train_mode == "finetune" and args.cache_features:
        log.info("  --train_mode=finetune → --cache_features auto-disabled.")
        args.cache_features = False

    global RUN_ID
    if args.run_id:
        RUN_ID = args.run_id
    tag = args.tag or os.path.basename(args.h5_dir.rstrip("/\\")) or "data"
    cache_str = "cache" if args.cache_features else "nocache"
    root_dir = os.path.join(args.out_dir,
                            f"{tag}_{args.modality}_{args.cv_mode}_{args.train_mode}_{cache_str}_{RUN_ID}")
    os.makedirs(root_dir, exist_ok=True)

    if args.single_subject and args.subjects and len(args.subjects) == 1:
        subj_dir = os.path.join(root_dir, args.subjects[0])
        os.makedirs(subj_dir, exist_ok=True)
        _setup_logging(subj_dir)
    else:
        _setup_logging(root_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not args.single_subject:
        with open(os.path.join(root_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    log.info(f"Run ID       : {RUN_ID}")
    log.info(f"Out root     : {root_dir}")
    log.info(f"Device       : {device}")
    log.info(f"CV mode      : {args.cv_mode}  ({args.n_folds}-fold)")
    log.info(f"Train mode   : {args.train_mode}  cache_features={args.cache_features}")
    log.info(f"H5 dir       : {args.h5_dir}")


    sample_h5 = None
    subj_files = sorted(glob.glob(os.path.join(args.h5_dir, "sub-*.h5")))
    if subj_files:
        sample_h5 = subj_files[0]
    log.info("\nResolving eigval …")
    override_eigval, ev_src = resolve_shared_eigval(
        args.d_path, args.lam_cortex_path, args.ratio,
        sample_h5, args.modality,
    )
    log.info(f"  eigval source = {ev_src}"
             + (f"  (D path = {args.d_path})" if ev_src == "D" else ""))


    if not args.encoder_ckpt:
        log.info("\nNo --encoder_ckpt supplied → running auto-pretrain (v3 + E[λ]) on --h5_dir")
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
        args.encoder_ckpt = auto_pretrain_prior(
            args.h5_dir, args.modality, pre_dir, device, hp, tag,
            override_eigval=override_eigval,
        )
    log.info(f"Encoder ckpt : {args.encoder_ckpt}")

    subjects = args.subjects or discover_subjects(args.h5_dir)
    log.info(f"Subjects ({len(subjects)}): {subjects}")


    finetune = (args.train_mode == "finetune")

    log.info("\nLoading K-free MAE prior (v4) …")
    prior_enc = load_prior_from_ckpt(args.encoder_ckpt, device, trainable=finetune,
                                     n_samples_override=args.n_samples)
    prior_init_state = {k: v.detach().clone() for k, v in prior_enc.state_dict().items()}

    log.info("Loading BrainOmni-tiny …")
    backbone, lm_dim = load_brainomni_tiny(device, finetune=finetune)
    backbone_init_state = ({k: v.detach().clone() for k, v in backbone.state_dict().items()}
                            if finetune else None)


    if args.cv_mode == "within":
        results = run_within(args.h5_dir, subjects,
                             prior_enc, prior_init_state,
                             backbone, backbone_init_state, lm_dim,
                             device, args, root_dir,
                             override_eigval=override_eigval)
    else:
        results = run_cross(args.h5_dir, subjects,
                            prior_enc, prior_init_state,
                            backbone, backbone_init_state, lm_dim,
                            device, args, root_dir,
                            override_eigval=override_eigval)

    if not args.single_subject:
        with open(os.path.join(root_dir, "splits.json"), "w") as f:
            json.dump(results["splits"], f, indent=2)

        summary = summarize(results, args.cv_mode)
        summary["meta"] = {
            "tag": tag, "h5_dir": args.h5_dir, "encoder_ckpt": args.encoder_ckpt,
            "subjects": subjects, "cv_mode": args.cv_mode, "n_folds": args.n_folds,
            "train_mode": args.train_mode, "cache_features": args.cache_features,
            "seed": args.seed, "RUN_ID": RUN_ID,
            "n_classes": args.n_classes,
            "d_path": args.d_path, "lam_cortex_path": args.lam_cortex_path,
            "ratio": args.ratio, "eigval_source": ev_src,
        }
        with open(os.path.join(root_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    else:
        subj = args.subjects[0]
        if subj in results.get("splits", {}):
            with open(os.path.join(root_dir, subj, "splits.json"), "w") as f:
                json.dump(results["splits"][subj], f, indent=2)
        summary = None

    if summary is not None:
        log.info("\n" + "=" * 75)
        log.info(f"FINAL SUMMARY [{tag} / {args.cv_mode} / {args.train_mode} / "
                 f"{'cache' if args.cache_features else 'nocache'}]  ({args.n_folds}-fold CV)")
        log.info("=" * 75)
        rows = summary["overall"] if args.cv_mode == "within" else summary
        log.info("  --- val (within-CV held-out fold) ---")
        for k in ("obs_acc", "prior_acc", "poe_arith", "poe_conf", "poe_entropy", "poe_temp"):
            if k in rows and rows[k] is not None:
                s = rows[k]
                log.info(f"  {k:<12s} = {s['mean']*100:.2f}% ± {s['std']*100:.2f}%   "
                         f"folds={[f'{v*100:.1f}' for v in s['values']]}")
        if "test_obs_balacc" in rows:
            log.info("  --- held-out TEST ---")
            for k in ("test_obs_acc", "test_obs_balacc", "test_pri_acc", "test_pri_balacc",
                      "test_poe_arith_acc", "test_poe_arith_balacc",
                      "test_poe_conf_acc",  "test_poe_conf_balacc",
                      "test_poe_ent_acc",   "test_poe_ent_balacc",
                      "test_poe_temp_acc",  "test_poe_temp_balacc"):
                if k in rows:
                    s = rows[k]
                    log.info(f"  {k:<22s} = {s['mean']*100:.2f}% ± {s['std']*100:.2f}%   "
                             f"folds={[f'{v*100:.1f}' for v in s['values']]}")
        log.info(f"  Chance level = {100/args.n_classes:.1f}%")
    log.info(f"\nArtifacts → {root_dir}")


if __name__ == "__main__":
    main()
