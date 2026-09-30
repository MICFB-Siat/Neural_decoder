




from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))
import glob, os, json
from pathlib import Path
import pandas as pd, numpy as np

ROOT = str(_LOCAL_SOURCE / '')
XF = Path(ROOT) / "lizhuo_exp/overnight_0606/results/xfold"
OUT = Path(ROOT) / "Results汇总" / "Ours_xsubj_allsubj_20260611"

OLD = {
 "smn":     ("SMN4Lang", 0.5896, 0.52),
 "lppc_cn": ("LPPC_CN",  0.6024, 0.68),
 "lppc_en": ("LPPC_EN",  0.6112, 0.83),
 "lppc_fr": ("LPPC_FR",  0.6042, 1.0),
 "alice":   ("Alice",    0.5566, 1.0),
 "lppchk":  ("LPPCHK",   0.6172, 1.0),
}

OUT.mkdir(parents=True, exist_ok=True)
rows = []
for name, (disp, old_bge, frac) in OLD.items():

    fold_dirs = sorted(glob.glob(str(XF / name / "fold*")))
    per_subj = []
    nfold = 0
    for fd in fold_dirs:
        runs = sorted(glob.glob(os.path.join(fd, "run_*")))
        if not runs: continue
        sj = os.path.join(runs[-1], "summary.json")
        if not os.path.exists(sj): continue
        s = json.load(open(sj))
        recs = [{"subject": sid, "BGEScore": d["test"]["BGEScore"]}
                for sid, d in s.get("subjects", {}).items()
                if d.get("test") and d["test"].get("BGEScore") is not None]
        if recs:
            per_subj.append(pd.DataFrame(recs)); nfold += 1
    if not per_subj:
        print(f"{disp}: 无fold结果"); continue
    allsub = pd.concat(per_subj, ignore_index=True).drop_duplicates("subject")
    d = OUT / disp; d.mkdir(parents=True, exist_ok=True)
    allsub.sort_values("subject").to_csv(d / "per_subject.csv", index=False, encoding="utf-8-sig")
    m, s, n = allsub["BGEScore"].mean(), allsub["BGEScore"].std(ddof=0), len(allsub)
    (d / "summary.txt").write_text(
        f"{disp} 全被试 cross-subject (K折覆盖全部被试)\n"
        f"frac={frac}  n_folds={nfold}  n_subjects={n}\n"
        f"BGEScore: mean={m:.4f}  std={s:.4f}  min={allsub.BGEScore.min():.4f}  max={allsub.BGEScore.max():.4f}\n"
        f"旧版(只评最后一批): {old_bge:.4f}\n", encoding="utf-8")
    rows.append(dict(dataset=disp, frac=frac, n_folds=nfold, n_subj_all=n,
                     allsubj_mean=round(m, 4), allsubj_std=round(s, 4),
                     old_lastblock=old_bge, delta=round(m - old_bge, 4)))
    print(f"{disp:10s} 全被试 mean={m:.4f}±{s:.4f} (n={n}, {nfold}折)  旧版={old_bge:.4f}  Δ={m-old_bge:+.4f}")

summ = pd.DataFrame(rows)
summ.to_csv(OUT / "SUMMARY_allsubj_vs_lastblock.csv", index=False, encoding="utf-8-sig")
print(f"\n[done] -> {OUT}\n")
print(summ.to_string(index=False))
