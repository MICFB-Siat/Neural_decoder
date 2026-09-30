









from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))
import sys, glob, os, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = str(_LOCAL_SOURCE / '')
sys.path.insert(0, ROOT)
from utils.metrics_eval import _BgeDense, clean_for_embedding, DEFAULT_BGE_PATH

NOISE = Path(ROOT) / "lizhuo_exp/lizhuo_exp_onlynoise"
OUT = Path(ROOT) / "Results汇总" / "Noise_BGE_20260611"
DEVICE = "cuda:0"


DATASETS = [
    ("LPPCHK",        "LPPCHK",        "zh"),
    ("smn4lang",      "SMN4Lang",      "zh"),
    ("LPPC-fMRI_CN",  "LPPC_CN",       "zh"),
    ("LPPC-fMRI_EN",  "LPPC_EN",       "en"),
    ("LPPC-fMRI_FR",  "LPPC_FR",       "fr"),
    ("Narratives2",   "Narratives",    "en"),
    ("Alice",         "Alice",         "en"),
]
VERSIONS = [("LoRA_100", "_lora50"), ("NoLoRA_100", "_base20_noLoRA")]

def load_reps(version_root, safe):

    reps = {}
    pat = str(NOISE / version_root / safe / "r*" / "run_*" / "rep*_decode.csv")
    for cf in glob.glob(pat):
        rep = int(re.search(r"rep(\d+)_decode", os.path.basename(cf)).group(1))
        df = pd.read_csv(cf, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        rows = []
        for _, r in df.iterrows():
            rows.append(dict(subj=r.get("subj",""), idx=r.get("idx",""),
                             gt=(r.get("gt","") or "").strip(),
                             output=(r.get("output","") or "").strip()))
        reps[rep] = rows
    return reps


print("[1] 收集 rep CSV ...", flush=True)
all_rows = []
counts = []
for vname, vroot in VERSIONS:
    for safe, disp, lang in DATASETS:
        reps = load_reps(vroot, safe)
        nrep = len(reps); ntr = sum(len(v) for v in reps.values())
        counts.append((vname, disp, nrep, ntr))
        for rep, rows in reps.items():
            for r in rows:
                r.update(version=vname, dataset=disp, lang=lang, rep=rep)
                all_rows.append(r)
        print(f"    {vname:11s} {disp:11s} reps={nrep:3d} trials={ntr}", flush=True)
print(f"[1] 共 {len(all_rows)} trial-rows", flush=True)


print("[2] _BgeDense 打分 gt-vs-output ...", flush=True)
bge = _BgeDense(DEFAULT_BGE_PATH, DEVICE, batch_size=256, verbose=True)
valid = [r for r in all_rows if r["gt"] and r["output"]]
refs = [clean_for_embedding(r["gt"], r["lang"]) for r in valid]
hyps = [clean_for_embedding(r["output"], r["lang"]) for r in valid]
sc = bge.cos_pairs(hyps, refs)
for r, v in zip(valid, sc): r["BGEScore"] = float(v)
for r in all_rows: r.setdefault("BGEScore", float("nan"))


df = pd.DataFrame(all_rows)
OUT.mkdir(parents=True, exist_ok=True)
summ = []
for (vname, disp), g in df.groupby(["version", "dataset"], sort=False):
    d = OUT / vname / disp; d.mkdir(parents=True, exist_ok=True)
    g[["rep","subj","idx","lang","BGEScore","gt","output"]].to_csv(
        d/"per_trial.csv", index=False, encoding="utf-8-sig")
    per_rep = g.dropna(subset=["BGEScore"]).groupby("rep")["BGEScore"].mean()
    per_rep.to_csv(d/"per_rep_mean.csv", encoding="utf-8-sig")
    summ.append(dict(dataset=disp, version=vname, lang=g["lang"].iloc[0],
        n_reps=int(per_rep.size), n_trial=int(g["BGEScore"].notna().sum()),
        BGE_mean=round(float(per_rep.mean()),4), BGE_std=round(float(per_rep.std(ddof=0)),4),
        BGE_min=round(float(per_rep.min()),4), BGE_max=round(float(per_rep.max()),4)))
    print(f"    [out] {vname}/{disp}: mean={summ[-1]['BGE_mean']:.4f} "
          f"std={summ[-1]['BGE_std']:.4f} reps={per_rep.size}", flush=True)

sm = pd.DataFrame(summ)

wide = sm.pivot(index="dataset", columns="version",
                values=["BGE_mean","BGE_std","n_reps"]).reset_index()
sm.to_csv(OUT/"SUMMARY_noise_100.csv", index=False, encoding="utf-8-sig")
wide.to_csv(OUT/"SUMMARY_noise_100_wide.csv", index=False, encoding="utf-8-sig")
print("\n[3] SUMMARY:")
print(sm.to_string(index=False))
print(f"\n[done] -> {OUT}")
