


from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))
import re, sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(_LOCAL_SOURCE / ''))
from utils.metrics_eval import _BgeDense, DEFAULT_BGE_PATH

ROOT = Path(str(_LOCAL_SOURCE / ''))

def clean(t):
    t = str(t)
    t = re.sub(r"_+\s*x[0-9A-Fa-f]+\s*_+", "", t)
    t = t.replace("\xa0", "").replace(" ", "")
    return t.strip()

o = pd.read_csv(ROOT / "_bge_per_trial_ours_random/lppchk_bge_per_trial.csv").dropna(subset=["BGEScore"])
o = o[o.method == "ours"].copy()
o["gt_c"] = o["gt"].map(clean); o["hyp_c"] = o["hypothesis"].map(clean)
o = o[o.gt_c.str.len() > 8]

c = pd.read_csv("/media/wsqlab/Expansion/LPPC_HK_processed/results/"
                "phase3_lppc_hk/predictions_redecode_test_bge_per_trial.csv").dropna(subset=["BGEScore"])
c["ref_c"] = c.reference.map(clean); c["hyp_c"] = c.hypothesis.map(clean)
c = c[c.ref_c.str.len() > 8]

bge = _BgeDense(DEFAULT_BGE_PATH, None, batch_size=256, verbose=True)
ov = bge._encode(o.gt_c.tolist())
cv = bge._encode(c.ref_c.tolist())
sim = cv @ ov.T

rows = []
for i in range(len(c)):
    j = int(sim[i].argmax())
    rows.append((i, j, float(sim[i, j])))
m = pd.DataFrame(rows, columns=["ci", "oi", "match"])
m["o_bge"] = o.iloc[m.oi].BGEScore.values
m["c_bge"] = c.iloc[m.ci].BGEScore.values
m["gt_len"] = o.iloc[m.oi].gt_c.str.len().values

cand = m[(m.match > 0.90) & (m.o_bge > 0.60) & (m.c_bge < 0.55)].copy()
cand["gap"] = cand.o_bge - cand.c_bge

cand = cand.sort_values("c_bge").drop_duplicates("oi")
cand = cand.sort_values(["gap"], ascending=False)
print(f"[candidates] {len(cand)} (match>0.92 近乎同段, ours>0.62, cog<0.55)\n")
for _, r in cand.head(8).iterrows():
    orow = o.iloc[int(r.oi)]; crow = c.iloc[int(r.ci)]
    print("=" * 90)
    print(f"match(ref-ref)={r.match:.3f}  gt_len={int(r.gt_len)}  ours_BGE={r.o_bge:.3f}  cog_BGE={r.c_bge:.3f}")
    print(f"[GT(ours {orow.subject} idx{orow.idx})] {orow.gt_c}")
    print(f"[OURS decode  ] {orow.hyp_c}")
    print(f"[GT(CogReader {crow.subject_name} c{crow.chunk_idx})] {crow.ref_c}")
    print(f"[CogReader dec] {crow.hyp_c}")
    print()
