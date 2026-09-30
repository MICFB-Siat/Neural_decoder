from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PoEFusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.raw_tau_a = nn.Parameter(torch.zeros(1))
        self.raw_tau_b = nn.Parameter(torch.zeros(1))

    def precisions(self):
        return (F.softplus(self.raw_tau_a) + 0.0001, F.softplus(self.raw_tau_b) + 0.0001)

    def forward(self, a, b):
        ta, tb = self.precisions()
        return (ta * a + tb * b) / (ta + tb)

class SubjectLayers(nn.Module):

    def __init__(self, n_subjects: int, d: int, init_id: bool=True):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_subjects, d, d))
        if init_id:
            self.weights.data[:] = torch.eye(d)[None]
        else:
            self.weights.data.normal_()
        self.weights.data *= 1.0 / math.sqrt(d)
        self.bias = nn.Parameter(torch.zeros(n_subjects, d))

    def forward(self, x, subject_ids):
        w = self.weights.index_select(0, subject_ids)
        b = self.bias.index_select(0, subject_ids).unsqueeze(1)
        return torch.einsum('btc,bcd->btd', x, w) + b

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
        self.ff = nn.Sequential(nn.Linear(d_q, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, d_q), nn.Dropout(dropout))

    def forward(self, q, kv):
        x = self.norm_sa(q)
        q = q + self.sa(x, x, x, need_weights=False)[0]
        x = self.norm_ca(q)
        k = self.norm_kv(kv)
        q = q + self.ca(x, k, k, need_weights=False)[0]
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
        self.blocks = nn.ModuleList([QFormerBlock(d_q, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_q)
        self.out_proj = nn.Linear(d_q, d_out)

    def forward(self, x):
        B = x.size(0)
        kv = self.input_norm(self.input_proj(x))
        q = (self.queries + self.pos).expand(B, -1, -1).contiguous()
        for blk in self.blocks:
            q = blk(q, kv)
        return self.out_proj(self.norm(q))

def load_stage_a(ckpt_path: str, device, *, n_subjects=None, d_in=512, T_bge=20, D_bge=1024, qf_dq=768, qf_layers=4, qf_heads=8):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    subject_ids = ck['subject_ids']
    n_subj = n_subjects if n_subjects is not None else len(subject_ids)
    fusion_mode = ck.get('fusion_mode')
    if fusion_mode is None:
        fusion_mode = ck.get('args', {}).get('fusion_mode', 'poe')
    poe = PoEFusion().to(device) if fusion_mode == 'poe' else None
    if poe is not None and ck.get('poe') is not None:
        poe.load_state_dict(ck['poe'])
        poe.eval()
        for p in poe.parameters():
            p.requires_grad_(False)
    sub_layers = SubjectLayers(n_subj, d_in, init_id=True).to(device)
    sub_layers.load_state_dict(ck['sub_layers'])
    sub_layers.eval()
    for p in sub_layers.parameters():
        p.requires_grad_(False)
    qf = QFormer(n_queries=T_bge, d_q=qf_dq, d_in=d_in, d_out=D_bge, n_layers=qf_layers, n_heads=qf_heads).to(device)
    qf.load_state_dict(ck['qformer'])
    qf.eval()
    for p in qf.parameters():
        p.requires_grad_(False)
    return {'poe': poe, 'sub_layers': sub_layers, 'qformer': qf, 'fusion_mode': fusion_mode, 'tau_obs': float(ck.get('tau_obs', 1.0)), 'tau_prior': float(ck.get('tau_prior', 0.0)), 'subject_ids': subject_ids, 'n_subjects': n_subj, 'ckpt': ckpt_path}

class StageAPipeline(nn.Module):

    def __init__(self, stage_a_pack, *, default_mode='joint'):
        super().__init__()
        self.poe = stage_a_pack['poe']
        self.sub_layers = stage_a_pack['sub_layers']
        self.qformer = stage_a_pack['qformer']
        self.fusion_mode_trained = stage_a_pack['fusion_mode']
        self.default_mode = default_mode

    def fuse(self, obs, prior, mode):
        if self.poe is None:
            return obs
        if mode == 'joint':
            return self.poe(obs, prior)
        if mode == 'obs_only':
            return self.poe(obs, torch.zeros_like(prior))
        if mode == 'prior_only':
            return self.poe(torch.zeros_like(obs), prior)
        raise ValueError(f'unknown fusion mode: {mode!r}')

    def forward(self, obs, prior, sids, mode=None):
        m = mode or self.default_mode
        x = self.fuse(obs, prior, m)
        x = self.sub_layers(x, sids)
        return self.qformer(x)

class FullPipelineWithModesEncoder(nn.Module):

    def __init__(self, stage_a, frozen_encoders, *, n_out_tokens=20, default_mode='joint'):
        super().__init__()
        self.stage_a = stage_a
        self.modes_enc = frozen_encoders['modes_enc']
        self.full_eigval = frozen_encoders['full_eigval']
        self.n_out_tokens = n_out_tokens
        self.default_mode = default_mode

    def forward(self, modes_in, bci_obs_cached, sids, mode=None):
        bci_prior = self.modes_enc.encode_pooled(modes_in, full_eigval=self.full_eigval, n_out_tokens=self.n_out_tokens)
        return self.stage_a(bci_obs_cached, bci_prior, sids, mode=mode)

def cos_alignment_score(aligned: torch.Tensor, target: torch.Tensor, per_trial: bool=False) -> torch.Tensor:
    p = F.normalize(aligned.flatten(1), dim=-1)
    t = F.normalize(target.flatten(1), dim=-1)
    cos_per_trial = (p * t).sum(-1)
    return cos_per_trial if per_trial else cos_per_trial.mean()

@torch.enable_grad()
def gradient_shap_per_trial(forward_batch: Callable[[torch.Tensor], torch.Tensor], x_batch: torch.Tensor, baseline_pool: torch.Tensor, *, n_baselines: int=8, n_steps: int=10, noise_sigma: float=0.0, rng: torch.Generator | None=None) -> torch.Tensor:
    if rng is None:
        rng = torch.Generator(device=x_batch.device)
        rng.manual_seed(0)
    B = x_batch.shape[0]
    M = baseline_pool.shape[0]
    accum = torch.zeros_like(x_batch)
    for _ in range(n_baselines):
        idx = torch.randint(0, M, (B,), generator=rng, device=x_batch.device)
        baseline = baseline_pool[idx]
        if noise_sigma > 0:
            baseline = baseline + noise_sigma * torch.randn(baseline.shape, generator=rng, device=baseline.device)
        delta = x_batch - baseline
        alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=x_batch.device)
        weights = torch.full((n_steps + 1,), 1.0 / n_steps, device=x_batch.device)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        path_grad = torch.zeros_like(x_batch)
        for a, w in zip(alphas.tolist(), weights.tolist()):
            x_step = (baseline + a * delta).detach().requires_grad_(True)
            out = forward_batch(x_step)
            scalar = out.sum() if out.dim() > 0 else out
            grad, = torch.autograd.grad(scalar, x_step)
            path_grad = path_grad + w * grad.detach()
        accum = accum + delta * path_grad
    return accum / n_baselines
