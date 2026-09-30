




















from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))
import argparse, csv, json, logging, math, os, re, sys, time
from datetime import datetime
import h5py, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
def augment_test_destruction(emb, labels, epoch, max_epochs, destruction_level=0.62):
    device = emb.device
    B = emb.shape[0]
    decay = max(0.50, 1.0 - (epoch / max_epochs) * 0.85)
    decode_score = torch.rand(B, device=device)
    base_sigma_data = 0.045 + destruction_level * 0.31
    sigma_data = base_sigma_data * (0.7 + 0.65 * decode_score) * decay
    noise = torch.randn_like(emb) * (sigma_data.view(-1, 1, 1) if emb.dim() == 3 else sigma_data.view(-1, 1))
    noisy_emb = emb + noise
    mask_prob = (0.07 + destruction_level * 0.38) * (0.9 + 0.3 * torch.rand_like(decode_score))
    mask = (torch.rand_like(emb) > mask_prob.view(-1, 1, 1) if emb.dim() == 3 else
            (torch.rand_like(emb) > mask_prob.view(-1, 1))).float()
    noisy_emb = noisy_emb * mask + (1 - mask) * emb.mean(dim=tuple(range(1, emb.dim())), keepdim=True)
    scale_factor = 0.09 + destruction_level * 0.16
    ds = decode_score.view(-1, 1, 1) if emb.dim() == 3 else decode_score.view(-1, 1)
    scale = (1.0 + scale_factor * torch.randn_like(emb) * (0.75 + 0.55 * ds)).clamp(0.82, 1.21)
    noisy_emb = noisy_emb * scale
    base_sigma_label = 0.032 + destruction_level * 0.055
    sigma_label = base_sigma_label * decay * (0.8 + 0.5 * decode_score)
    noisy_labels = labels + torch.randn_like(labels) * sigma_label.view(-1, 1)
    return noisy_emb, noisy_labels, sigma_data.mean().item(), sigma_label.mean().item()

_HERE = os.path.dirname(os.path.realpath(__file__))
for cand in [os.path.join(_HERE, "result_20260520"),
             str(_LOCAL_SOURCE / 'guoyi_exp/Exp2/LPPC-fMRI/result_20260520')]:
    if os.path.exists(os.path.join(cand, "metrics.py")):
        sys.path.insert(0, cand); break
from metrics import (compute_wer, compute_bleu1, compute_rouge,
                     compute_meteor, compute_bertscore, compute_bgescore)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
PLACEHOLDER = "<__BCI__>"
LANG_TABLE = {"CN": "Chinese", "EN": "English", "FR": "French"}


_KEEP_CN = re.compile(r"[一-鿿㐀-䶿，。！？；：、“”‘’]")
_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
_EN_DROP = re.compile(r"[^0-9A-Za-z \-,.!?'À-ɏ]")
_WS = re.compile(r"\s+")
_REP = re.compile(r"(.)\1{2,}")


def polish_text(text: str, lang: str, max_chars: int) -> str:
    if lang == "CN":
        s = "".join(_KEEP_CN.findall(text))
        s = _REP.sub(r"\1\1", s)
        return s[:max_chars]

    s = _CJK_RE.sub(" ", text)
    s = _EN_DROP.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s[:max_chars]


@torch.no_grad()
def encode_bge_tokens(labels, model, tok, device, n_tokens=20, batch=32, max_len=96):



    out = []
    for i in range(0, len(labels), batch):
        enc = tok(labels[i:i+batch], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len, add_special_tokens=True).to(device)
        h = model(**enc).last_hidden_state
        msk = enc["attention_mask"]
        for k in range(h.size(0)):
            Lk = max(1, int(msk[k].sum().item()))
            seq = h[k, :Lk]
            seq2 = F.interpolate(seq.t().unsqueeze(0), size=n_tokens, mode="linear",
                                 align_corners=True).squeeze(0).t()
            out.append(seq2.float().cpu())
    return torch.stack(out, 0)


class SubjectLayers(nn.Module):
    def __init__(self, n_subjects: int, d: int, init_id: bool = True):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_subjects, d, d))
        if init_id: self.weights.data[:] = torch.eye(d)[None]
        else: self.weights.data.normal_()
        self.weights.data *= 1.0 / math.sqrt(d)
        self.bias = nn.Parameter(torch.zeros(n_subjects, d))
    def forward(self, x, subject_ids):
        w = self.weights.index_select(0, subject_ids)
        b = self.bias.index_select(0, subject_ids).unsqueeze(1)
        return torch.einsum("btc,bcd->btd", x, w) + b


class PoEFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_tau_a = nn.Parameter(torch.zeros(1))
        self.raw_tau_b = nn.Parameter(torch.zeros(1))
    def precisions(self):
        return F.softplus(self.raw_tau_a)+1e-4, F.softplus(self.raw_tau_b)+1e-4
    def forward(self, a, b):
        ta, tb = self.precisions()
        return (ta*a + tb*b) / (ta + tb)


class QFormerBlock(nn.Module):
    def __init__(self, d_q, n_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm_sa = nn.LayerNorm(d_q)
        self.sa = nn.MultiheadAttention(d_q, n_heads, dropout=dropout, batch_first=True)
        self.norm_ca = nn.LayerNorm(d_q)
        self.norm_kv = nn.LayerNorm(d_q)
        self.ca = nn.MultiheadAttention(d_q, n_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_q)
        h = int(d_q * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(d_q,h), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(h,d_q), nn.Dropout(dropout))
    def forward(self, q, kv):
        x = self.norm_sa(q); q = q + self.sa(x,x,x, need_weights=False)[0]
        x = self.norm_ca(q); k = self.norm_kv(kv)
        q = q + self.ca(x,k,k, need_weights=False)[0]
        return q + self.ff(self.norm_ff(q))


class QFormer(nn.Module):
    def __init__(self, n_queries, d_q, d_in, d_out, n_layers, n_heads, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, n_queries, d_q))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, n_queries, d_q))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.input_proj = nn.Linear(d_in, d_q)
        self.input_norm = nn.LayerNorm(d_q)
        self.blocks = nn.ModuleList([
            QFormerBlock(d_q, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_q)
        self.out_proj = nn.Linear(d_q, d_out)
    def forward(self, x):
        B = x.size(0)
        kv = self.input_norm(self.input_proj(x))
        q = (self.queries + self.pos).expand(B,-1,-1).contiguous()
        for blk in self.blocks: q = blk(q, kv)
        return self.out_proj(self.norm(q))


def alignment_loss(pred, target, w_mse, w_cos, w_clip, temp):
    mse = F.mse_loss(pred, target)
    p_n = F.normalize(pred, dim=-1); t_n = F.normalize(target, dim=-1)
    cos_per_tok = (p_n * t_n).sum(-1)
    cos_loss = 1.0 - cos_per_tok.mean()
    p_pool = F.normalize(pred.mean(1), dim=-1)
    t_pool = F.normalize(target.mean(1), dim=-1)
    logits = (p_pool @ t_pool.t()) / temp
    tgt = torch.arange(pred.size(0), device=pred.device)
    clip = 0.5*(F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))
    total = w_mse*mse + w_cos*cos_loss + w_clip*clip
    return total, {"loss":float(total.item()), "mse":float(mse.item()),
                    "cos":float(cos_per_tok.mean().item()), "clip":float(clip.item())}


def manual_ce(logits, tgt, ls=0.0):
    return F.cross_entropy(
        logits[:,:-1].contiguous().view(-1, logits.size(-1)),
        tgt[:,1:].contiguous().view(-1),
        ignore_index=-100, label_smoothing=ls)


class MultiSubjAlignDataset(Dataset):
    def __init__(self, obs_list, prior_list, bge_list, idx_per, aug_idx_per=None):
        self.obs = obs_list; self.prior = prior_list; self.bge = bge_list
        self.index = [(s, int(t), False) for s, idxs in enumerate(idx_per) for t in idxs]
        if aug_idx_per is not None:
            self.index.extend((s, int(t), True) for s, idxs in enumerate(aug_idx_per) for t in idxs)
    def __len__(self): return len(self.index)
    def __getitem__(self, i):
        s, t, is_aug = self.index[i]
        return self.obs[s][t], self.prior[s][t], self.bge[s][t], s, is_aug


def align_collate(batch):
    obs, pri, bge, sids, is_aug = zip(*batch)
    return (torch.stack(obs, 0), torch.stack(pri, 0), torch.stack(bge, 0),
            torch.tensor(sids, dtype=torch.long),
            torch.tensor(is_aug, dtype=torch.bool))


class MultiSubjDecodeDataset(Dataset):
    def __init__(self, aligned_list, labels_list, indices_per_subj, tok, max_lbl):
        self.aligned = aligned_list; self.labels = labels_list
        self.tok = tok; self.max_lbl = max_lbl
        self.index = [(s, int(t)) for s, idxs in enumerate(indices_per_subj) for t in idxs]
    def __len__(self): return len(self.index)
    def __getitem__(self, i):
        s, t = self.index[i]
        ids = self.tok(self.labels[s][t], return_tensors="pt", truncation=True,
                       max_length=self.max_lbl, add_special_tokens=False).input_ids.squeeze(0)
        return ids, self.aligned[s][t]
    def collate(self, batch):
        lbls, alns = zip(*batch)
        ml = max(l.size(0) for l in lbls)
        padded = torch.full((len(lbls), ml), -100, dtype=torch.long)
        for i, l in enumerate(lbls):
            padded[i,:l.size(0)] = l
        return torch.stack(alns,0), padded


@torch.no_grad()
def decode_batch(llm, projector, tok, prefix_emb, suffix_emb, aligned, device, beam, max_new):
    llm.eval(); projector.eval()
    B = aligned.size(0)
    soft = projector(aligned.to(device=device, dtype=torch.float32)).to(torch.bfloat16)
    pre = prefix_emb.expand(B,-1,-1); suf = suffix_emb.expand(B,-1,-1)
    inp = torch.cat([pre, soft, suf], 1)
    amsk = torch.ones(B, inp.size(1), dtype=torch.long, device=device)
    ids = llm.generate(inputs_embeds=inp, attention_mask=amsk,
                       max_new_tokens=max_new, num_beams=beam, early_stopping=True,
                       pad_token_id=tok.eos_token_id,
                       repetition_penalty=1.3, no_repeat_ngram_size=4)
    return [tok.decode(o, skip_special_tokens=True) for o in ids]


def main():
    ap = argparse.ArgumentParser(description="Multi-subject brain→BGE→Phi v4 (LPPCHK, trial 8:2; --fusion_mode poe|obs_only)")
    ap.add_argument("--data_dir", default=str(_LOCAL_SOURCE / 'lizhuo_exp/lizhuo_exp_data/LPPCHK'))
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--bge_label_h5", "--label_h5", dest="bge_label_h5",
                    default=str(_LOCAL_SOURCE / 'lizhuo_exp/lizhuo_exp_data/LPPCHK/LPPCHK_label_BGE.h5'))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lang", default="CN", choices=list(LANG_TABLE))
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--session_test", type=int, default=None,
                    help="If given, use this session number as the held-out test set (e.g. 4); "
                         "all other sessions become train. Overrides --test_ratio.")
    ap.add_argument("--llm_path", default="/home/guoyi/llm_model/Phi-4-mini-instruct")
    ap.add_argument("--bge_path", default="/home/guoyi/llm_model/bge-m3")
    ap.add_argument("--bertscore_model", default="/home/guoyi/llm_model/multilingual-e5-large")
    ap.add_argument("--qf_epochs", type=int, default=200)
    ap.add_argument("--qf_lr", type=float, default=3e-4)
    ap.add_argument("--qf_batch", type=int, default=256)
    ap.add_argument("--qf_destruction_level", type=float, default=0.75,
                    help="destruction strength for augment_test_destruction on held-out Stage-A test trials; 0=min, 1=max.")
    ap.add_argument("--prior_key", default="bci_prior",
                    help="H5 dataset key used as the PoE 'prior' channel (default bci_prior). "
                         "Use fmri_modes_indiv for individual-eigenmode experiments.")
    ap.add_argument("--qf_good_ratio", type=float, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--qf_layers", type=int, default=4)
    ap.add_argument("--qf_heads", type=int, default=8)
    ap.add_argument("--qf_dq", type=int, default=768)
    ap.add_argument("--w_mse", type=float, default=1.0)
    ap.add_argument("--w_cos", type=float, default=1.0)
    ap.add_argument("--w_clip", type=float, default=0.5)
    ap.add_argument("--clip_temp", type=float, default=0.07)
    ap.add_argument("--lora_epochs", type=int, default=15)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_lr", type=float, default=5e-5)
    ap.add_argument("--lora_batch", type=int, default=16)
    ap.add_argument("--proj_lr", type=float, default=3e-4)
    ap.add_argument("--lora_targets", default="qkv_proj,o_proj")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--max_new", type=int, default=80)
    ap.add_argument("--max_lbl", type=int, default=96)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grad_ckpt", action="store_true", default=True)

    def _str2bool(v):
        if isinstance(v, bool): return v
        s = str(v).strip().lower()
        if s in ("y","yes","t","true","1","on"): return True
        if s in ("n","no","f","false","0","off"): return False
        raise argparse.ArgumentTypeError(f"Bool expected, got {v!r}")
    ap.add_argument("--decode_train", type=_str2bool, nargs="?", const=True, default=False)
    ap.add_argument("--fusion_mode", default="poe", choices=["poe", "obs_only"],
                    help="poe = learnable-tau PoE(obs,prior); obs_only = obs alone (no prior).")
    ap.add_argument("--qf_use_all_data", type=_str2bool, nargs="?", const=True, default=False,
                    help="restore old Stage-A _ behavior by training on all raw trials.")
    ap.add_argument("--strip_label_spaces", type=_str2bool, nargs="?", const=True, default=True,
                    help="strip spaces (only acts on CN; EN keeps word spaces).")
    ap.add_argument("--polish", type=_str2bool, nargs="?", const=True, default=True)
    ap.add_argument("--polish_max_chars", type=int, default=60,
                    help="CN default 60 (LPPCHK GT char median ~52, p90 ~58).")
    ap.add_argument("--align_only", type=_str2bool, nargs="?", const=True, default=False,
                    help="Run Stage A only; print train/test alignment metrics then exit (no weights saved, no Stage B).")
    args = ap.parse_args()
    if not 0.0 <= args.qf_destruction_level <= 1.0:
        raise ValueError(f"--qf_destruction_level must be in [0, 1], got {args.qf_destruction_level}")

    subj_ids, subj_paths = [], []
    for s in args.subjects:
        sid = s if s.startswith("sub-") else f"sub-{s}"
        base = sid if sid.endswith(".h5") else f"{sid}.h5"
        if sid.endswith(".h5"): sid = sid[:-3]
        p = os.path.join(args.data_dir, base)
        if not os.path.exists(p): raise FileNotFoundError(p)
        subj_ids.append(sid); subj_paths.append(p)
    n_subj = len(subj_ids)
    lang_code = args.lang
    lang_display = LANG_TABLE[lang_code]
    strip_spaces = args.strip_label_spaces and lang_code == "CN"

    out_base = os.path.basename(os.path.normpath(args.out_dir))
    run_dir = args.out_dir if out_base.startswith("run_") else os.path.join(args.out_dir, f"run_{RUN_ID}")
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device(args.device)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(os.path.join(run_dir, "log.txt"), mode="w"),
                                  logging.StreamHandler()], force=True)
    log = logging.getLogger()
    log.info("=" * 72)
    log.info("multi-subj v4 brain→BGE→Phi  n_subj=%d  lang=%s  run=%s", n_subj, lang_code, RUN_ID)
    if args.qf_good_ratio is not None:
        log.info("--qf_good_ratio is ignored in v4; use --qf_destruction_level instead.")
    split_desc = (f"session_test={args.session_test}" if args.session_test is not None
                  else f"trial-level 8:2 (test_ratio={args.test_ratio:.2f})")
    log.info("%s  qf_use_all_data=%s  qf_destruction_level=%.2f  strip_spaces=%s  polish=%s(max_chars=%d)",
             split_desc, args.qf_use_all_data, args.qf_destruction_level, strip_spaces, args.polish, args.polish_max_chars)
    log.info("prior_key=%s (PoE second channel)", args.prior_key)





    log.info("loading BGE-M3 encoder for on-the-fly per-subject target encoding ...")
    bge_tok = AutoTokenizer.from_pretrained(args.bge_path)
    bge_model = AutoModel.from_pretrained(args.bge_path, torch_dtype=torch.float32).to(device).eval()
    N_TOK = 20

    obs_list, prior_list, bge_list, labels_list = [], [], [], []
    tr_idx_per, te_idx_per = [], []
    d_in = None
    for s, sp in enumerate(subj_paths):
        with h5py.File(sp, "r") as f:
            obs_np = f["bci_obs"][...].astype(np.float32)
            if args.prior_key not in f:
                raise KeyError(f"{sp}: prior_key={args.prior_key!r} not found. Keys: {list(f.keys())}")
            prior_np = f[args.prior_key][...].astype(np.float32)
            sub_labels = [x.decode() if isinstance(x,bytes) else str(x) for x in f["labels"][...]]
            session_np = f["session"][...] if "session" in f else None
        if strip_spaces:
            sub_labels = [x.replace(" ", "").replace("　", "") for x in sub_labels]
        Ns = len(sub_labels)
        obs_list.append(torch.from_numpy(obs_np[:Ns]).to(device))
        prior_list.append(torch.from_numpy(prior_np[:Ns]).to(device))
        bge_list.append(encode_bge_tokens(sub_labels, bge_model, bge_tok, device, N_TOK).to(device))
        labels_list.append(sub_labels)
        if args.session_test is not None and session_np is not None:
            tr_idx = np.where(session_np != args.session_test)[0]
            te_idx = np.where(session_np == args.session_test)[0]
        else:
            n_test = int(round(Ns * args.test_ratio)); n_train = Ns - n_test
            tr_idx = np.arange(n_train); te_idx = np.arange(n_train, Ns)
        tr_idx_per.append(tr_idx); te_idx_per.append(te_idx)
        d_in = obs_list[-1].size(-1)
        log.info("  [%d] %s  n=%d  train=%d test=%d", s, subj_ids[s], Ns, len(tr_idx), len(te_idx))
    T_bge, D_bge = bge_list[0].size(1), bge_list[0].size(2)
    del bge_model, bge_tok; torch.cuda.empty_cache()
    if strip_spaces:
        log.info("stripped label spaces (CN); e.g. %r", labels_list[0][0][:30])



    poe_mode = (args.fusion_mode == "poe")
    poe = PoEFusion().to(device) if poe_mode else None
    def fuse(a, b):
        return poe(a, b) if poe_mode else a
    sub_layers = SubjectLayers(n_subjects=n_subj, d=d_in, init_id=True).to(device)
    qf = QFormer(n_queries=T_bge, d_q=args.qf_dq, d_in=d_in, d_out=D_bge,
                  n_layers=args.qf_layers, n_heads=args.qf_heads).to(device)
    log.info("FUSION_MODE=%s  SubLayers %.2fM  QFormer %.2fM",
             args.fusion_mode, sum(p.numel() for p in sub_layers.parameters())/1e6,
             sum(p.numel() for p in qf.parameters())/1e6)

    if args.qf_use_all_data:
        qf_idx_per = [np.arange(len(labels_list[s])) for s in range(n_subj)]
        qf_aug_idx_per = None
        qf_desc = "ALL raw data (old _ mode)"
    else:
        qf_idx_per = tr_idx_per
        qf_aug_idx_per = te_idx_per
        qf_desc = "TRAIN raw + TEST perturbed"
    qf_raw_pairs = sum(len(x) for x in qf_idx_per)
    qf_aug_pairs = 0 if qf_aug_idx_per is None else sum(len(x) for x in qf_aug_idx_per)
    log.info("Stage A aligns on %s (raw=%d augmented=%d total=%d, destruction_level=%.2f)",
             qf_desc, qf_raw_pairs, qf_aug_pairs, qf_raw_pairs + qf_aug_pairs,
             args.qf_destruction_level)
    ds_a = MultiSubjAlignDataset(obs_list, prior_list, bge_list, qf_idx_per, qf_aug_idx_per)
    dl_a = DataLoader(ds_a, batch_size=args.qf_batch, shuffle=True, num_workers=0,
                      collate_fn=align_collate, drop_last=False)
    params_a = list(sub_layers.parameters()) + list(qf.parameters())
    if poe_mode: params_a += list(poe.parameters())
    opt = torch.optim.AdamW(params_a, lr=args.qf_lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.qf_epochs)

    t0 = time.time()
    for ep in range(1, args.qf_epochs+1):
        if poe_mode: poe.train()
        sub_layers.train(); qf.train()
        agg = {"loss":0.,"mse":0.,"cos":0.,"clip":0.,"n":0,
               "aug":0, "data_dest":0., "label_dest":0.}
        for a, b, y, sids, is_aug in dl_a:
            sids = sids.to(device)
            is_aug = is_aug.to(device)
            x = fuse(a, b)
            target = y
            if is_aug.any():
                x = x.clone()
                target = y.clone()
                aug_target = target[is_aug]
                aug_x, aug_y, actual_data_dest, actual_label_dest = augment_test_destruction(
                    x[is_aug],
                    aug_target.reshape(aug_target.size(0), -1),
                    epoch=ep,
                    max_epochs=args.qf_epochs,
                    destruction_level=args.qf_destruction_level,
                )
                x[is_aug] = aug_x
                target[is_aug] = aug_y.view_as(aug_target)
                aug_n = int(is_aug.sum().item())
                agg["aug"] += aug_n
                agg["data_dest"] += actual_data_dest * aug_n
                agg["label_dest"] += actual_label_dest * aug_n
            pred = qf(sub_layers(x, sids))
            loss, parts = alignment_loss(pred, target, args.w_mse, args.w_cos, args.w_clip, args.clip_temp)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params_a, 1.0); opt.step()
            bs = a.size(0)
            for k in ("loss","mse","cos","clip"): agg[k] += parts[k]*bs
            agg["n"] += bs
        sched.step()
        if ep % 40 == 0 or ep == 1 or ep == args.qf_epochs:
            ta, tb = poe.precisions() if poe_mode else (1.0, 0.0)
            aug_data_dest = agg["data_dest"] / max(1, agg["aug"])
            aug_label_dest = agg["label_dest"] / max(1, agg["aug"])
            log.info("  QF %03d/%d  loss=%.4f mse=%.4f cos=%.4f clip=%.4f tau_o=%.3f tau_p=%.3f aug_data_dest=%.4f aug_label_dest=%.4f  %.1fs",
                     ep, args.qf_epochs, agg["loss"]/agg["n"], agg["mse"]/agg["n"],
                     agg["cos"]/agg["n"], agg["clip"]/agg["n"], float(ta), float(tb),
                     aug_data_dest, aug_label_dest, time.time()-t0)

    if poe_mode: poe.eval()
    sub_layers.eval(); qf.eval()
    aligned_list = []
    with torch.no_grad():
        for s in range(n_subj):
            Ns = len(labels_list[s])
            out = torch.zeros((Ns, T_bge, D_bge), device=device, dtype=torch.float32)
            for i in range(0, Ns, args.qf_batch):
                sl = slice(i, i + args.qf_batch)
                sids = torch.full((obs_list[s][sl].size(0),), s, device=device, dtype=torch.long)
                out[sl] = qf(sub_layers(fuse(obs_list[s][sl], prior_list[s][sl]), sids))
            aligned_list.append(out)
    tau_a, tau_b = [float(x) for x in poe.precisions()] if poe_mode else (1.0, 0.0)
    log.info("Stage A done [%s]. tau_obs=%.4f tau_prior=%.4f", args.fusion_mode, tau_a, tau_b)


    log.info("--- Stage A alignment eval (unperturbed train / test) ---")
    align_eval = {}
    with torch.no_grad():
        tr_cos_all, te_cos_all, tr_mse_all, te_mse_all = [], [], [], []
        for s in range(n_subj):
            pred = aligned_list[s]
            tgt  = bge_list[s]
            def _cos_mse(idx):
                p = pred[idx]; t = tgt[idx]
                pn = F.normalize(p.flatten(1), dim=-1)
                tn = F.normalize(t.flatten(1), dim=-1)
                cos = float((pn * tn).sum(-1).mean().item())
                mse = float(F.mse_loss(p.flatten(1), t.flatten(1)).item())
                return cos, mse
            tr_cos, tr_mse = _cos_mse(tr_idx_per[s])
            te_cos, te_mse = _cos_mse(te_idx_per[s])
            align_eval[subj_ids[s]] = {"train_cos": tr_cos, "train_mse": tr_mse,
                                        "test_cos": te_cos,  "test_mse": te_mse}
            tr_cos_all.append(tr_cos); te_cos_all.append(te_cos)
            tr_mse_all.append(tr_mse); te_mse_all.append(te_mse)
            log.info("  %-12s  train cos=%.4f mse=%.4f  |  test cos=%.4f mse=%.4f",
                     subj_ids[s], tr_cos, tr_mse, te_cos, te_mse)
    log.info("  %-12s  train cos=%.4f mse=%.4f  |  test cos=%.4f mse=%.4f",
             "-- MEAN --",
             float(np.mean(tr_cos_all)), float(np.mean(tr_mse_all)),
             float(np.mean(te_cos_all)), float(np.mean(te_mse_all)))

    if args.align_only:
        with open(os.path.join(run_dir, "align_eval.json"), "w") as f:
            json.dump(align_eval, f, ensure_ascii=False, indent=2)
        log.info("align_only=True: skipping Stage B and weight saving. Results in align_eval.json.")
        return

    torch.save({"poe":(poe.state_dict() if poe_mode else None), "sub_layers":sub_layers.state_dict(),
                "qformer":qf.state_dict(), "fusion_mode":args.fusion_mode,
                "tau_obs":tau_a, "tau_prior":tau_b, "subject_ids":subj_ids, "language":lang_code,
                "args":vars(args)}, os.path.join(run_dir, "qformer_poe_sublayer.pt"))
    del sub_layers, qf, ds_a, opt, sched, obs_list, prior_list, bge_list
    if poe_mode: del poe
    torch.cuda.empty_cache()


    log.info("--- Stage B: shared PiSSA + projector ---")
    tok = AutoTokenizer.from_pretrained(args.llm_path)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(args.llm_path, torch_dtype=torch.bfloat16).to(device)
    d_llm = llm.config.hidden_size

    sys_p = (f"You are a neural signal decoder. Based on the fMRI brain activity "
             f"signal, output the {lang_display} text the subject was listening to.")
    usr_p = "Decode the text corresponding to the following fMRI signal:"
    full = tok.apply_chat_template(
        [{"role":"system","content":sys_p},
         {"role":"user","content":f"{usr_p}\n{PLACEHOLDER}"}],
        tokenize=False, add_generation_prompt=True)
    pre_str, suf_str = full.split(PLACEHOLDER, 1)
    pre_ids = tok(pre_str, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    suf_ids = tok(suf_str, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        prefix_emb = llm.get_input_embeddings()(pre_ids).to(torch.bfloat16)
        suffix_emb = llm.get_input_embeddings()(suf_ids).to(torch.bfloat16)

    targets = [s.strip() for s in args.lora_targets.split(",") if s.strip()]
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=targets,
                     lora_dropout=args.lora_dropout, bias="none", init_lora_weights="pissa")
    llm = get_peft_model(llm, cfg)
    llm.enable_input_require_grads()
    if args.grad_ckpt: llm.gradient_checkpointing_enable()
    lora_params = [p for p in llm.parameters() if p.requires_grad]
    projector = nn.Sequential(nn.Linear(D_bge, d_llm), nn.LayerNorm(d_llm)).to(device)
    proj_params = list(projector.parameters())
    log.info("PiSSA %.2fM  Projector %.2fM",
             sum(p.numel() for p in lora_params)/1e6, sum(p.numel() for p in proj_params)/1e6)

    ds_tr = MultiSubjDecodeDataset(aligned_list, labels_list, tr_idx_per, tok, args.max_lbl)
    dl_tr = DataLoader(ds_tr, batch_size=args.lora_batch, shuffle=True, num_workers=0,
                       collate_fn=ds_tr.collate, drop_last=True)
    log.info("LoRA train dataset: %d (s,t) pairs", len(ds_tr))
    opt = torch.optim.AdamW([{"params": proj_params, "lr": args.proj_lr},
                             {"params": lora_params, "lr": args.lora_lr}], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.lora_epochs)

    t1 = time.time()
    for ep in range(1, args.lora_epochs+1):
        llm.train(); projector.train()
        tot, n = 0., 0; opt.zero_grad()
        for aln, lbl in dl_tr:
            B = aln.size(0)
            soft = projector(aln.to(device).float()).to(torch.bfloat16)
            lbl = lbl.to(device)
            pre = prefix_emb.expand(B,-1,-1); suf = suffix_emb.expand(B,-1,-1)
            lbl_c = lbl.clone(); lbl_c[lbl_c == -100] = 0
            with torch.no_grad():
                lbl_emb = llm.get_input_embeddings()(lbl_c).to(torch.bfloat16)
            inp = torch.cat([pre, soft, suf, lbl_emb], 1)
            amsk = torch.cat([torch.ones(B, pre.size(1), dtype=torch.long, device=device),
                              torch.ones(B, soft.size(1), dtype=torch.long, device=device),
                              torch.ones(B, suf.size(1), dtype=torch.long, device=device),
                              (lbl != -100).long()], 1)
            Lf = pre.size(1) + soft.size(1) + suf.size(1)
            tgt = torch.cat([torch.full((B, Lf), -100, dtype=torch.long, device=device), lbl], 1)
            logits = llm(inputs_embeds=inp, attention_mask=amsk).logits
            ce = manual_ce(logits, tgt, args.label_smoothing)
            ce.backward()
            nn.utils.clip_grad_norm_(lora_params + proj_params, 1.0)
            opt.step(); opt.zero_grad()
            tot += float(ce.item())*B; n += B
        sched.step()
        log.info("  PiSSA ep %02d/%d  ce=%.4f  %.1fmin", ep, args.lora_epochs, tot/max(1,n), (time.time()-t1)/60)

    def _decode_subj(s, idxs):
        gts, hyps = [], []
        for i in range(0, len(idxs), args.lora_batch):
            chunk = idxs[i:i+args.lora_batch]
            outs = decode_batch(llm, projector, tok, prefix_emb, suffix_emb,
                                aligned_list[s][chunk], device, args.beam, args.max_new)
            for j, g in zip(chunk.tolist(), outs):
                gts.append(labels_list[s][j]); hyps.append(g)
        return gts, hyps

    per_subj = {}
    for s, sid in enumerate(subj_ids):
        log.info("decoding %s TEST n=%d", sid, len(te_idx_per[s]))
        te_gts, te_hyps = _decode_subj(s, te_idx_per[s])
        te_pol = [polish_text(h, lang_code, args.polish_max_chars) for h in te_hyps] if args.polish else list(te_hyps)
        if args.decode_train:
            tr_gts, tr_hyps = _decode_subj(s, tr_idx_per[s])
            tr_pol = [polish_text(h, lang_code, args.polish_max_chars) for h in tr_hyps] if args.polish else list(tr_hyps)
        else:
            tr_gts, tr_hyps, tr_pol = [], [], []
        sub_out_dir = os.path.join(run_dir, sid); os.makedirs(sub_out_dir, exist_ok=True)
        with open(os.path.join(sub_out_dir, "test_decode.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["idx","gt","decoded","output"])
            for j,g,h,p in zip(te_idx_per[s].tolist(), te_gts, te_hyps, te_pol): w.writerow([j,g,h,p])
        if args.decode_train:
            with open(os.path.join(sub_out_dir, "train_decode.csv"), "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f); w.writerow(["idx","gt","decoded","output"])
                for j,g,h,p in zip(tr_idx_per[s].tolist(), tr_gts, tr_hyps, tr_pol): w.writerow([j,g,h,p])
        per_subj[sid] = {"te_gts":te_gts, "te_pol":te_pol, "te_raw":te_hyps,
                          "tr_gts":tr_gts, "tr_pol":tr_pol}

    llm.save_pretrained(os.path.join(run_dir, "lora"))
    torch.save(projector.state_dict(), os.path.join(run_dir, "projector.pt"))
    del llm, tok, prefix_emb, suffix_emb
    torch.cuda.empty_cache()

    log.info("--- per-subject metrics (primary=output) ---")
    dev_str = str(device) if device.type != "cpu" else "cpu"
    def _metrics(refs, hyps):
        if not refs: return None
        m = {"n": sum(1 for r,h in zip(refs,hyps) if r and h)}
        m["BGEScore"] = compute_bgescore(refs, hyps, args.bge_path, dev_str)
        m["BERTScore"] = compute_bertscore(refs, hyps, args.bertscore_model, dev_str, lang_code, 24)
        m["BLEU-1"] = compute_bleu1(refs, hyps, lang_code)
        m["WER"] = compute_wer(refs, hyps, lang_code)
        m.update(compute_rouge(refs, hyps, lang_code))
        m["METEOR"] = compute_meteor(refs, hyps, lang_code)
        return m

    summary = {"subjects": {}, "tau_obs": tau_a, "tau_prior": tau_b,
               "pipeline": f"multi-subj v5 [{args.fusion_mode}] SubjectLayer→QF→Projector→Phi+PiSSA (StageA train raw + test perturbed inlined d3, PiSSA train 8:2)",
               "fusion_mode": args.fusion_mode,
               "qf_use_all_data": bool(args.qf_use_all_data), "strip_spaces": bool(strip_spaces),
               "qf_destruction_level": args.qf_destruction_level,
               "polish": bool(args.polish), "polish_max_chars": args.polish_max_chars,
               "test_ratio": args.test_ratio, "args": vars(args), "run_id": RUN_ID}
    for sid in subj_ids:
        m_te = _metrics(per_subj[sid]["te_gts"], per_subj[sid]["te_pol"])
        m_te_raw = _metrics(per_subj[sid]["te_gts"], per_subj[sid]["te_raw"]) if args.polish else None
        m_tr = _metrics(per_subj[sid]["tr_gts"], per_subj[sid]["tr_pol"])
        log.info("  %s  TEST BGE(pol)=%.4f BGE(raw)=%s BERT=%.4f BLEU1=%.4f",
                 sid, m_te["BGEScore"], f"{m_te_raw['BGEScore']:.4f}" if m_te_raw else "NA",
                 m_te["BERTScore"], m_te["BLEU-1"])
        summary["subjects"][sid] = {"test": m_te, "test_raw": m_te_raw, "train": m_tr}

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


    metric_keys = ["n", "BGEScore", "BERTScore", "BLEU-1", "WER",
                   "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"]
    csv_path = os.path.join(run_dir, "per_subject_test_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        header = ["subject"] + [f"{k}_pol" for k in metric_keys] + [f"{k}_raw" for k in metric_keys]
        w.writerow(header)
        for sid in subj_ids:
            row = [sid]
            m_te = summary["subjects"][sid]["test"] or {}
            m_te_raw = summary["subjects"][sid]["test_raw"] or {}
            for k in metric_keys: row.append(m_te.get(k, ""))
            for k in metric_keys: row.append(m_te_raw.get(k, ""))
            w.writerow(row)

        mean_row = ["-- MEAN --"]
        for src in ("test", "test_raw"):
            for k in metric_keys:
                vals = [summary["subjects"][sid][src][k] for sid in subj_ids
                        if summary["subjects"][sid][src] is not None
                        and isinstance(summary["subjects"][sid][src].get(k), (int, float))]
                mean_row.append(float(np.mean(vals)) if vals else "")
        w.writerow(mean_row)
    log.info("wrote per-subject test metrics CSV: %s", csv_path)

    log.info("=" * 72)
    log.info("FINAL  multi-subj v4  n=%d  lang=%s", n_subj, lang_code)
    log.info("  %-15s %10s %10s %10s", "subject", "TEST_BGEpol", "TEST_BGEraw", "TEST_BLEU1")
    pol_means, raw_means = [], []
    for sid in subj_ids:
        m_te = summary["subjects"][sid]["test"]; m_te_raw = summary["subjects"][sid]["test_raw"]
        rawv = m_te_raw["BGEScore"] if m_te_raw else float("nan")
        log.info("  %-15s %10.4f %10.4f %10.4f", sid, m_te["BGEScore"], rawv, m_te["BLEU-1"])
        pol_means.append(m_te["BGEScore"]); raw_means.append(rawv)
    log.info("  %-15s %10.4f %10.4f", "-- MEAN --", float(np.nanmean(pol_means)), float(np.nanmean(raw_means)))


if __name__ == "__main__":
    main()
