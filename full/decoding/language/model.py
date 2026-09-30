import re, math
import torch
from torch import nn
from torch.nn import functional as F
PLACEHOLDER = "<__BCI__>"
def polish_text(text: str, lang: str, max_chars: int) -> str:
    _KEEP_CN = re.compile("[一-鿿㐀-䶿，。！？；：、“”‘’]")
    _CJK_RE = re.compile('[一-鿿㐀-䶿]')
    _EN_DROP = re.compile("[^0-9A-Za-z \\-,.!?'À-ɏ]")
    _WS = re.compile('\\s+')
    _REP = re.compile('(.)\\1{2,}')
    if lang == 'CN':
        s = ''.join(_KEEP_CN.findall(text))
        s = _REP.sub('\\1\\1', s)
        return s[:max_chars]
    s = _CJK_RE.sub(' ', text)
    s = _EN_DROP.sub(' ', s)
    s = _WS.sub(' ', s).strip()
    return s[:max_chars]

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

@torch.no_grad()
def decode_batch(llm, projector, tok, prefix_emb, suffix_emb, aligned, device, beam, max_new):
    llm.eval()
    projector.eval()
    B = aligned.size(0)
    soft = projector(aligned.to(device=device, dtype=torch.float32)).to(torch.bfloat16)
    pre = prefix_emb.expand(B, -1, -1)
    suf = suffix_emb.expand(B, -1, -1)
    inp = torch.cat([pre, soft, suf], 1)
    amsk = torch.ones(B, inp.size(1), dtype=torch.long, device=device)
    ids = llm.generate(inputs_embeds=inp, attention_mask=amsk, max_new_tokens=max_new, num_beams=beam, early_stopping=True, pad_token_id=tok.eos_token_id, repetition_penalty=1.3, no_repeat_ngram_size=4)
    return [tok.decode(o, skip_special_tokens=True) for o in ids]
