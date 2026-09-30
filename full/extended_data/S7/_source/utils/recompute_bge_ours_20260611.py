








from pathlib import Path as _Path
import sys as _sys
_LOCAL_SOURCE = next(p for p in _Path(__file__).resolve().parents if (p / '.submission_source').is_file())
_sys.path.insert(0, str(_LOCAL_SOURCE))
import sys, glob, math, csv as _csv
from pathlib import Path
import pandas as pd, numpy as np

ROOT = str(_LOCAL_SOURCE / '')
sys.path.insert(0, ROOT)
from utils.metrics_eval import _BgeDense, clean_for_embedding, DEFAULT_BGE_PATH

LE = Path(ROOT) / "lizhuo_exp"
OUT = Path(ROOT) / "Results汇总" / "Ours_BGE_20260611"
DEVICE = "cuda:0"


def g(*parts): return str(LE / Path(*parts))
ENTRIES = [

 ("群体","SMN4Lang","zh",[g("lizhuo_exp_language/smn4lang/code/results_v9_targets/smn4lang/run_20260601_005528/sub-*/test_decode.csv")]),
 ("群体","LPPC_CN","zh",[g("lizhuo_exp_lowdata/LPPC-fMRI/outputs_verify_256_16/CN/drop00/run_*/sub-*/test_decode.csv")]),
 ("群体","LPPC_EN","en",[g("lizhuo_exp_lowdata/LPPC-fMRI/outputs_verify_256_16/EN/drop00/run_*/sub-*/test_decode.csv")]),
 ("群体","LPPC_FR","fr",[g("lizhuo_exp_lowdata/LPPC-fMRI/outputs_verify_256_16/FR/drop00/run_*/sub-*/test_decode.csv")]),
 ("群体","Narratives","en",[g("lizhuo_exp_language/Narratives2/run_20260522_162255/sub-*/test_decode.csv")]),
 ("群体","Alice","en",[g("lizhuo_exp_language/Alice/run_20260521_231244/sub-*/test_decode.csv")]),
 ("群体","LPPCHK","zh",[g("lizhuo_exp_language/LPPCHK/run_20260523_104130/sub-*/test_decode.csv")]),

 ("个体","SMN4Lang","zh",[g("../lizhuo_exp/overnight_0606/results/smn_indiv_v2/run_20260606_103253/sub-*/test_decode.csv")]),
 ("个体","LPPC_CN","zh",[g("lizhuo_exp_lowdata_sub/LPPCfmri/outputs_128_16/CN/drop00/run_*/sub-*/test_decode.csv")]),
 ("个体","LPPC_EN","en",[g("lizhuo_exp_lowdata_sub/LPPCfmri/outputs_128_16/EN/drop00/run_*/sub-*/test_decode.csv")]),
 ("个体","LPPC_FR","fr",[g("lizhuo_exp_lowdata_sub/LPPCfmri/outputs_128_16/FR/drop00/run_*/sub-*/test_decode.csv")]),
 ("个体","Narratives","en",[g("lizhuo_exp_sublanguage/narratives/run_20260524_130227/sub-*/test_decode.csv")]),
 ("个体","Alice","en",[g("lizhuo_exp_sublanguage/Alice/v5_indiv_prior_multisub_d05/run_20260523_172843/sub-*/test_decode.csv")]),
 ("个体","LPPCHK","zh",[g("lizhuo_exp_language/LPPCHK/run_20260523_134259_indiv_d0_sess4/sub-*/test_decode.csv")]),

 ("群体跨被试","SMN4Lang","zh",[g("overnight_0606/results/smn_xsubj_f2/run_20260608_064948/sub-*/test_decode.csv")]),
 ("群体跨被试","LPPC_CN","zh",[g("overnight_0606/results/lppc_cn_f1/run_20260608_044416/sub-*/test_decode.csv")]),
 ("群体跨被试","LPPC_EN","en",[g("overnight_0606/results/lppc_en_f2/run_20260608_075352/sub-*/test_decode.csv")]),
 ("群体跨被试","LPPC_FR","fr",[g("overnight_0606/results/lppc_fr_xsubj/run_20260606_090859/sub-*/test_decode.csv")]),
 ("群体跨被试","Narratives","en",[g("lizhuo_exp_language/Narratives2_xsubj_partial/run_20260608_064932/sub-*/test_decode.csv")]),
 ("群体跨被试","Alice","en",[g("overnight_0606/results/alice_sweep_d05/run_20260606_205800/sub-*/test_decode.csv")]),
 ("群体跨被试","LPPCHK","zh",[g("overnight_0606/results/lppchk_xsubj/run_20260606_084358/sub-*/test_decode.csv")]),
]

def load_entry(globs):
    files = []
    for pat in globs: files += glob.glob(pat)
    files = sorted(files)
    recs = []
    for fp in files:
        subj = Path(fp).parent.name
        df = pd.read_csv(fp, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        cols = {c.strip().lstrip("﻿"): c for c in df.columns}
        gtc, dec, pol = cols.get("gt"), cols.get("decoded"), cols.get("output")
        idc = cols.get("idx")
        for _, r in df.iterrows():
            recs.append(dict(subject=subj, idx=(r[idc] if idc else ""),
                             gt=(r[gtc] if gtc else "").strip(),
                             decoded=(r[dec] if dec else "").strip(),
                             output=(r[pol] if pol else "").strip()))
    return recs, len(files)


print("[1] 收集 trial ...", flush=True)
all_recs = []
for cat, name, lang, globs in ENTRIES:
    recs, nf = load_entry(globs)
    for r in recs: r.update(category=cat, name=name, lang=lang)
    all_recs += recs
    print(f"    {cat:8s} {name:11s} lang={lang} files={nf:3d} trials={len(recs)}", flush=True)
print(f"[1] 共 {len(all_recs)} trials", flush=True)


bge = _BgeDense(DEFAULT_BGE_PATH, DEVICE, batch_size=256, verbose=True)
def score(refkey, hypkey):
    valid = [r for r in all_recs if r["gt"] and r[hypkey]]
    refs = [clean_for_embedding(r["gt"], r["lang"]) for r in valid]
    hyps = [clean_for_embedding(r[hypkey], r["lang"]) for r in valid]
    s = bge.cos_pairs(hyps, refs)
    for r, v in zip(valid, s): r[refkey] = float(v)
    for r in all_recs: r.setdefault(refkey, float("nan"))
print("[2] 打分 gt-vs-output ...", flush=True); score("BGEScore", "output")
print("[2] 打分 gt-vs-decoded(raw) ...", flush=True); score("BGEScore_raw", "decoded")


df = pd.DataFrame(all_recs)
OUT.mkdir(parents=True, exist_ok=True)
summ_rows = []
for (cat, name), gdf in df.groupby(["category", "name"], sort=False):
    d = OUT / cat / name; d.mkdir(parents=True, exist_ok=True)
    pt = gdf[["subject","idx","lang","BGEScore","BGEScore_raw","gt","output","decoded"]].copy()
    pt.to_csv(d/"per_trial.csv", index=False, encoding="utf-8-sig")

    per_subj = gdf.dropna(subset=["BGEScore"]).groupby("subject").agg(
        n=("BGEScore","size"), BGEScore=("BGEScore","mean"),
        BGEScore_raw=("BGEScore_raw","mean")).reset_index()
    per_subj.to_csv(d/"summary.csv", index=False, encoding="utf-8-sig")
    v_subj = per_subj["BGEScore"].values
    v_all  = gdf["BGEScore"].dropna().values
    summ_rows.append(dict(category=cat, name=name, lang=gdf["lang"].iloc[0],
        n_subj=len(per_subj), n_trial=int(len(v_all)),
        BGE_subjmean=round(float(np.mean(v_subj)),4), BGE_subjstd=round(float(np.std(v_subj)),4),
        BGE_trialmean=round(float(np.mean(v_all)),4),
        BGE_raw_subjmean=round(float(per_subj["BGEScore_raw"].mean()),4)))
    print(f"    [out] {cat}/{name}: subjmean={summ_rows[-1]['BGE_subjmean']:.4f} "
          f"std={summ_rows[-1]['BGE_subjstd']:.4f} n_subj={len(per_subj)}", flush=True)

sm = pd.DataFrame(summ_rows)
sm.to_csv(OUT/"SUMMARY_ALL.csv", index=False, encoding="utf-8-sig")
print("\n[3] 总表 SUMMARY_ALL.csv:")
print(sm.to_string(index=False))
print(f"\n[done] -> {OUT}")
