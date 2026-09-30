






from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))
import sys, glob, os, re
from pathlib import Path
import pandas as pd, numpy as np

ROOT = str(_LOCAL_SOURCE / '')
sys.path.insert(0, ROOT)
from utils.metrics_eval import _BgeDense, clean_for_embedding, DEFAULT_BGE_PATH

XF = Path(ROOT) / "lizhuo_exp/overnight_0606/results/xfold"
OUT = Path(ROOT) / "Results汇总" / "Ours_BGE_20260611" / "群体跨被试20260612"
DEVICE = "cuda:0"

DS = [("smn","SMN4Lang","zh",0.5896), ("lppc_cn","LPPC_CN","zh",0.6024),
      ("lppc_en","LPPC_EN","en",0.6112), ("lppc_fr","LPPC_FR","fr",0.6042),
      ("alice","Alice","en",0.5566), ("lppchk","LPPCHK","zh",0.6172)]

def latest_run(fold_dir):
    rs = sorted(glob.glob(os.path.join(fold_dir, "run_*")))
    return rs[-1] if rs else None


print("[1] 收集 xfold 逐 trial ...", flush=True)
all_rows = []
for name, disp, lang, _ in DS:
    seen_sub = set()
    for fd in sorted(glob.glob(str(XF / name / "fold*"))):
        run = latest_run(fd)
        if not run: continue
        for cf in sorted(glob.glob(os.path.join(run, "sub-*", "test_decode.csv"))):
            subj = Path(cf).parent.name
            if subj in seen_sub: continue
            seen_sub.add(subj)
            df = pd.read_csv(cf, encoding="utf-8-sig", dtype=str, keep_default_na=False)
            cols = {c.strip().lstrip("﻿"): c for c in df.columns}
            for _, r in df.iterrows():
                all_rows.append(dict(dataset=disp, lang=lang, subject=subj,
                    idx=r.get(cols.get("idx",""),""),
                    gt=(r.get(cols.get("gt",""),"") or "").strip(),
                    output=(r.get(cols.get("output",""),"") or "").strip()))
    print(f"    {disp:10s} subjects={len(seen_sub)} trials={sum(1 for x in all_rows if x['dataset']==disp)}", flush=True)
print(f"[1] 共 {len(all_rows)} trials", flush=True)


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
for name, disp, lang, old in DS:
    g = df[df.dataset == disp]
    if g.empty: continue
    d = OUT / disp; d.mkdir(parents=True, exist_ok=True)
    g[["subject","idx","lang","BGEScore","gt","output"]].to_csv(
        d/"per_trial.csv", index=False, encoding="utf-8-sig")
    ps = g.dropna(subset=["BGEScore"]).groupby("subject")["BGEScore"].mean().reset_index()
    ps.to_csv(d/"per_subject.csv", index=False, encoding="utf-8-sig")
    m, s, n = ps.BGEScore.mean(), ps.BGEScore.std(ddof=0), len(ps)
    (d/"summary.txt").write_text(
        f"{disp} 全被试 cross-subject (K折覆盖全部被试, _BgeDense重算)\n"
        f"lang={lang}  n_subjects={n}  n_trials={int(g.BGEScore.notna().sum())}\n"
        f"BGEScore: mean={m:.4f} std={s:.4f} min={ps.BGEScore.min():.4f} max={ps.BGEScore.max():.4f}\n"
        f"旧版(只评最后一批): {old:.4f}\n", encoding="utf-8")
    summ.append(dict(dataset=disp, lang=lang, n_subj=n,
        n_trial=int(g.BGEScore.notna().sum()),
        allsubj_mean=round(m,4), allsubj_std=round(s,4),
        old_lastblock=old, delta=round(m-old,4)))
    print(f"    {disp:10s} 全被试 mean={m:.4f}±{s:.4f} (n={n})  旧版={old:.4f}  Δ={m-old:+.4f}", flush=True)

sm = pd.DataFrame(summ)
sm.to_csv(OUT/"SUMMARY_allsubj.csv", index=False, encoding="utf-8-sig")
print("\n[done] ->", OUT)
print(sm.to_string(index=False))
