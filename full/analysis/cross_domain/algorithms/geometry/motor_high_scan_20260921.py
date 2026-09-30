












from __future__ import annotations

import hashlib
import json
import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_motor_high_scan")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
COMMON = REPO / "lizhuo_exp/lizhuo_exp_result/_common"
EXTRACT = REPO / "lizhuo_exp/lizhuo_exp_result/20260910_geometry_obs_vs_mnsr"
OUT = BASE / "v2/motor_high_scan_20260921"
CACHE = OUT / "features"
MODEL_ROOT = REPO / "data_check_20260507/Exp_tyf_Results/Exp_Classification/result/MOTOR/BT-ND+BrainOmni_cross5fold"
META_PATH = REPO / "figures/compositional_geometry_experiment/three_domains_real/motor_trial_context.csv"
LOW_ROOT = BASE / "v2/lioi_pretrained"
SEED_DIRS = sorted(MODEL_ROOT.glob("MOTOR_cross_unifiedv3_seed*_eeg_cross_frozen_cache_*"))
LAYERS = ("h1_mean", "h2_mean", "mu_mean", "h1_flat", "h2_flat", "mu_flat")

sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(EXTRACT / "code"))
import poe_eeg_runtime as R
from run_extract_multi import CFG


def ps(c: np.ndarray) -> float:
    u, v = c[1] - c[0], c[3] - c[2]
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def geometry(c: np.ndarray) -> dict[str, float]:
    u, v = c[1] - c[0], c[3] - c[2]
    cosine = ps(c)
    mid = (c[2] + c[3] - c[0] - c[1]) / 2
    direction = u / np.linalg.norm(u) + v / np.linalg.norm(v)
    direction /= np.linalg.norm(direction)
    orth = mid - (mid @ direction) * direction
    scale = np.mean([np.linalg.norm(c[i] - c[j]) for i, j in combinations(range(4), 2)])
    return {
        "PS": cosine,
        "angle_deg": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
        "midline_sep_norm": float(np.linalg.norm(mid) / scale),
        "orthogonal_sep_norm": float(np.linalg.norm(orth) / scale),
    }


def exact3(c: np.ndarray) -> tuple[np.ndarray, float, float]:
    x = c.astype(float) - c.mean(0)
    ex = x[1] - x[0]
    ex /= np.linalg.norm(ex)
    ey = x[3] - x[2]
    ey -= (ey @ ex) * ex
    if np.linalg.norm(ey) < 1e-12:
        _, _, vh = np.linalg.svd(x, full_matrices=False)
        ey = vh[1] - (vh[1] @ ex) * ex
    ey /= np.linalg.norm(ey)
    ez = x[2] - x[0]
    ez -= (ez @ ex) * ex + (ez @ ey) * ey
    if np.linalg.norm(ez) < 1e-12:
        _, _, vh = np.linalg.svd(x, full_matrices=False)
        ez = vh[2] - (vh[2] @ ex) * ex - (vh[2] @ ey) * ey
    ez /= np.linalg.norm(ez)
    p = x @ np.stack([ex, ey, ez], axis=1)
    if p[2, 2] < 0:
        p[:, 2] *= -1
    d0 = np.array([np.linalg.norm(x[i] - x[j]) for i, j in combinations(range(4), 2)])
    d1 = np.array([np.linalg.norm(p[i] - p[j]) for i, j in combinations(range(4), 2)])
    return p / d1.mean(), float(np.max(np.abs(d1 - d0)) / d0.max()), abs(ps(p) - ps(c))


def fold_map(root: Path) -> dict[str, int]:
    ans = {}
    for fold in range(5):
        split = R._read_split(root / f"fold_{fold:02d}")
        for sub in split.get("val_subjects", []):
            ans[sub] = fold
    return ans


def extract_features(device: torch.device, subjects: list[str]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    if all((CACHE / f"seed{seed}_{sub}.npz").exists() for seed in range(5) for sub in subjects):
        print("feature cache complete; extraction skipped", flush=True)
        return
    module = R._load_canonical()
    cfg = CFG["MOTOR"]
    maps = {seed: fold_map(root) for seed, root in enumerate(SEED_DIRS)}


    missing = [(seed, sub) for seed, fmap in maps.items() for sub in subjects if sub not in fmap]
    if missing:
        raise RuntimeError(f"participants missing from held-out folds: {missing}")
    first_fold = SEED_DIRS[0] / "fold_00"
    args = R._checkpoint_args(first_fold)
    n_samples = int(args.get("n_samples") or 2560)
    prior_encoder = R._load_prior(module, R.PRIOR_CKPT, device, n_samples)
    backbone, lm_dim = R._load_brainomni(module, device)
    _, base_prior, _, n_tok = R._build_models(
        module, first_fold, prior_encoder, backbone, lm_dim, n_samples, device)
    raw = {}
    for sub in subjects:
        h5 = cfg["h5"] / f"{sub}.h5"
        eigval = R._resolve_eigval(module, h5, cfg["d"], cfg["ratio"], cfg["lam"], "eeg")
        bundle = R._load_single_subject(
            module, cfg["h5"], sub, h5, n_samples, "eeg", cfg["label_key"], 0,
            cfg["n_classes"], eigval)
        idx = np.arange(len(bundle["labels"]), dtype=np.int64)
        _, pf, labels = module.precompute_features(
            bundle, idx, backbone, prior_encoder, n_tok, eigval.to(device), device, 32)
        raw[sub] = (pf, labels.numpy().astype(np.int64))
        print("precomputed", sub, tuple(pf.shape), flush=True)
    del base_prior, backbone, prior_encoder
    torch.cuda.empty_cache()

    for seed, root in enumerate(SEED_DIRS):
        for fold in range(5):
            relevant = [s for s in subjects if maps[seed][s] == fold]
            if not relevant:
                continue
            fold_dir = root / f"fold_{fold:02d}"
            args = R._checkpoint_args(fold_dir)
            n_samples = int(args.get("n_samples") or 2560)
            prior_encoder = R._load_prior(module, R.PRIOR_CKPT, device, n_samples)
            dummy_backbone, lm_dim = R._load_brainomni(module, device)
            _, model, _, _ = R._build_models(
                module, fold_dir, prior_encoder, dummy_backbone, lm_dim, n_samples, device)
            model.eval()
            for sub in relevant:
                target = CACHE / f"seed{seed}_{sub}.npz"
                if target.exists():
                    continue
                pf, labels = raw[sub]
                accum = {k: [] for k in LAYERS}
                with torch.inference_mode():
                    for sl in R._iter_slices(len(pf), 32):
                        tok = pf[sl].to(device)
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                            enabled=device.type == "cuda"):
                            h1 = model.ghead.net[:3](tok)
                            h2 = model.ghead.net[3:](h1)
                            mu = model.ghead.mu_head(h2)
                        values = {"h1": h1, "h2": h2, "mu": mu}
                        for name, value in values.items():
                            value = value.float().cpu()
                            accum[f"{name}_mean"].append(value.mean(1))
                            accum[f"{name}_flat"].append(value.flatten(1))
                payload = {k: torch.cat(v).numpy().astype(np.float32) for k, v in accum.items()}
                np.savez_compressed(target, **payload, y_true=labels, fold=fold, seed=seed)
                print("extracted", f"seed{seed}", sub, "fold", fold, flush=True)
            del model, dummy_backbone, prior_encoder
            torch.cuda.empty_cache()


def labels_for(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    take = part.run.isin(["MIpre", "MIpost", "eegNF"]).to_numpy()
    y = part.label.to_numpy()[take] + 2 * part.run.eq("eegNF").to_numpy()[take]
    idx = [np.flatnonzero(y == k) for k in range(4)]
    return take, y, idx


def normalized(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64) - x.mean(0)
    return x / np.sqrt(np.mean(np.sum(x * x, axis=1)))


def ccgp(x: np.ndarray, y: np.ndarray, draws: list[list[np.ndarray]]) -> float:
    scores = []
    for sel in draws[:20]:
        n = len(sel[0])
        target = np.r_[np.zeros(n), np.ones(n)]
        for a, b in ((0, 2), (2, 0)):
            tr = np.r_[sel[a], sel[a + 1]]
            te = np.r_[sel[b], sel[b + 1]]
            clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
            scores.append(clf.fit(x[tr], target).score(x[te], target))
    return float(np.mean(scores))


def ccgp_logreg(x: np.ndarray, draws: list[list[np.ndarray]]) -> float:

    scores = []
    for sel in draws[:20]:
        n = len(sel[0])
        target = np.r_[np.zeros(n), np.ones(n)]
        for a, b in ((0, 2), (2, 0)):
            tr = np.r_[sel[a], sel[a + 1]]
            te = np.r_[sel[b], sel[b + 1]]
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=.01, max_iter=3000))
            scores.append(clf.fit(x[tr], target).score(x[te], target))
    return float(np.mean(scores))


def analyse(subjects: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    meta = pd.read_csv(META_PATH)
    rows = []
    quads = {}
    for sub in subjects:
        part = meta[meta.subject.eq(sub)]
        lowz = np.load(LOW_ROOT / f"{sub}.npz")
        if not np.array_equal(part.label.to_numpy(), lowz["y_true"]):
            raise RuntimeError(f"metadata order mismatch: {sub}")
        take, y, idx = labels_for(part)
        n = min(map(len, idx))
        rng = np.random.default_rng(20260921 + int(sub.split("-")[1]))
        draws = [[rng.choice(cell, n, replace=False) for cell in idx] for _ in range(200)]
        low = normalized(lowz["o"][take])
        low_centroids = [np.stack([low[s].mean(0) for s in sel]) for sel in draws]
        low_c = np.mean(low_centroids, axis=0)
        low_ps = np.array([ps(c) for c in low_centroids])
        quads[f"low|{sub}"] = low_c
        low_g = geometry(low_c)
        for seed in range(5):
            z = np.load(CACHE / f"seed{seed}_{sub}.npz")
            if not np.array_equal(z["y_true"], lowz["y_true"]):
                raise RuntimeError(f"feature order mismatch: seed{seed} {sub}")
            for layer in LAYERS:
                high = normalized(z[layer][take])
                cents = [np.stack([high[s].mean(0) for s in sel]) for sel in draws]
                c = np.mean(cents, axis=0)
                vals = np.array([ps(q) for q in cents])
                g = geometry(c)
                name = f"seed{seed}_{layer}"
                quads[f"{name}|{sub}"] = c
                rows.append({
                    "subject": sub, "candidate": name, "seed": seed, "layer": layer,
                    "fold": int(z["fold"]), "n_per_cell": n,
                    "ps_low_mean200": float(low_ps.mean()),
                    "ps_high_mean200": float(vals.mean()),
                    "ps_delta_mean200": float((vals - low_ps).mean()),
                    "ps_low_display": low_g["PS"], "ps_high_display": g["PS"],
                    "angle_low_deg": low_g["angle_deg"], "angle_high_deg": g["angle_deg"],
                    "midline_sep_high_norm": g["midline_sep_norm"],
                    "orthogonal_sep_high_norm": g["orthogonal_sep_norm"],



                    "ccgp_low": np.nan, "ccgp_high": np.nan,
                    "ps_high_p05": float(np.quantile(vals, .05)),
                    "ps_high_p95": float(np.quantile(vals, .95)),
                })
        print("analysed", sub, flush=True)
    return pd.DataFrame(rows), quads


def add_shortlist_ccgp(df: pd.DataFrame) -> pd.DataFrame:

    means = df.groupby("candidate").ps_high_mean200.mean().sort_values(ascending=False)
    shortlist = list(means.head(3).index)
    if "seed1_mu_mean" not in shortlist:
        shortlist.append("seed1_mu_mean")
    meta = pd.read_csv(META_PATH)
    for sub in sorted(df.subject.unique()):
        part = meta[meta.subject.eq(sub)]
        lowz = np.load(LOW_ROOT / f"{sub}.npz")
        take, _, idx = labels_for(part)
        n = min(map(len, idx))
        rng = np.random.default_rng(20260921 + int(sub.split("-")[1]))
        draws = [[rng.choice(cell, n, replace=False) for cell in idx] for _ in range(20)]
        low_score = ccgp_logreg(normalized(lowz["o"][take]), draws)
        df.loc[df.subject.eq(sub), "ccgp_low"] = low_score
        for candidate in shortlist:
            row = df[(df.subject == sub) & (df.candidate == candidate)].iloc[0]
            z = np.load(CACHE / f"seed{int(row.seed)}_{sub}.npz")
            high_score = ccgp_logreg(normalized(z[row.layer][take]), draws)
            df.loc[(df.subject == sub) & (df.candidate == candidate), "ccgp_high"] = high_score
        print("CCGP", sub, "low", f"{low_score:.3f}", "shortlist", shortlist, flush=True)
    return df


def draw_pair(low: np.ndarray, high: np.ndarray, sub: str, candidate: str, path: Path) -> dict:
    fig = plt.figure(figsize=(9, 4.5))
    errs = []
    for col, (c, title) in enumerate(((low, "Low: frozen pretrained BrainOmni"),
                                      (high, f"High: {candidate}"))):
        p, de, pe = exact3(c)
        errs.append((de, pe))
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        for a, b, color in ((0, 2, "#9ECAE1"), (1, 3, "#F4B183")):
            ax.plot(*p[[a, b]].T, color=color, lw=2.5)
        for a, b, color in ((0, 1, "#1B9E77"), (2, 3, "#2B3A8F")):
            ax.quiver(*p[a], *(p[b] - p[a]), color=color, lw=2.2, arrow_length_ratio=.15)
            ax.scatter(*p[a], s=46, facecolor="white", edgecolor=color, depthshade=False)
            ax.scatter(*p[b], s=46, color=color, depthshade=False)
        lim = np.abs(p).max() * 1.25
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=28, azim=-60)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        g = geometry(c)
        ax.set_title(title, fontsize=9)
        ax.text2D(.5, -.07, f"full-space PS = {g['PS']:+.3f} | angle = {g['angle_deg']:.1f} deg",
                  transform=ax.transAxes, ha="center", fontsize=9)
    fig.suptitle(f"Motor Imagery | {sub}\nrest → right-hand MI; no feedback / EEG neurofeedback")
    fig.text(.5, .02, "Exact 3D centroid geometry; fixed camera (28, -60); equal axes. Green/blue are the two contexts.",
             ha="center", fontsize=8)
    fig.subplots_adjust(top=.78, bottom=.17, wspace=.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return {"candidate": candidate, "subject": sub,
            "max_relative_distance_error": max(x[0] for x in errs),
            "max_PS_error": max(x[1] for x in errs)}


def make_outputs(df: pd.DataFrame, quads: dict[str, np.ndarray]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "all_metrics.csv", index=False)
    np.savez_compressed(OUT / "centroids.npz", **quads)
    summary = df.groupby("candidate").agg(
        n_subjects=("subject", "size"), ps_high=("ps_high_mean200", "mean"),
        angle_high=("angle_high_deg", "mean"), delta=("ps_delta_mean200", "mean"),
        separation=("orthogonal_sep_high_norm", "mean"), ccgp_high=("ccgp_high", "mean"),
        improved=("ps_delta_mean200", lambda x: int((x > 0).sum())))
    summary = summary.sort_values(["ps_high", "separation"], ascending=[False, False])
    summary.to_csv(OUT / "candidate_summary.csv")

    group_top = list(summary.head(3).index)
    chosen = set()
    for sub, g in df.groupby("subject"):
        for _, r in g.sort_values(["ps_high_mean200", "orthogonal_sep_high_norm"], ascending=False).head(3).iterrows():
            chosen.add((sub, r.candidate))


        balanced = g.assign(_score=g.ps_high_mean200 + .2 * g.orthogonal_sep_high_norm).nlargest(1, "_score").iloc[0]
        chosen.add((sub, balanced.candidate))
        for candidate in group_top:
            chosen.add((sub, candidate))
    checks = []
    gallery_rows = []
    for sub, candidate in sorted(chosen):
        r = df[(df.subject == sub) & (df.candidate == candidate)].iloc[0]
        path = OUT / "gallery" / f"{sub}_{candidate}.png"
        checks.append(draw_pair(quads[f"low|{sub}"], quads[f"{candidate}|{sub}"], sub, candidate, path))
        gallery_rows.append(r)
    pd.DataFrame(checks).to_csv(OUT / "geometry_checks.csv", index=False)
    gallery = pd.DataFrame(gallery_rows).sort_values(["subject", "ps_high_mean200"], ascending=[True, False])
    gallery.to_csv(OUT / "gallery_metrics.csv", index=False)
    html = ["<html><meta charset='utf-8'><title>Motor high-level scan</title>",
            "<style>body{font-family:sans-serif;max-width:1300px;margin:auto}img{width:720px;max-width:100%}table{border-collapse:collapse;font-size:12px}td,th{padding:5px;border:1px solid #ddd}</style>",
            "<h1>Motor Imagery high-level scan</h1>",
            "<p>Fixed design: rest → right-hand motor imagery; no feedback (MIpre + MIpost) / EEG neurofeedback (eegNF). Low representation is unchanged. High candidates are held-out cross-subject layers from five training seeds. Full-space PS and angle are the selection quantities; line separation is recorded without changing the camera or axes.</p>",
            summary.reset_index().to_html(index=False, float_format=lambda x: f"{x:.3f}"),
            "<h2>Candidate gallery</h2>"]
    for _, r in gallery.iterrows():
        stem = f"{r.subject}_{r.candidate}"
        html.append(f"<h3>{r.subject} · {r.candidate}: PS {r.ps_low_mean200:.3f} → {r.ps_high_mean200:.3f}; angle {r.angle_high_deg:.1f}°; separation {r.orthogonal_sep_high_norm:.3f}; CCGP {r.ccgp_high:.3f}</h3><a href='gallery/{stem}.pdf'><img loading='lazy' src='gallery/{stem}.png'></a>")
    html.append("<h2>All participant-level metrics</h2>" + df.to_html(index=False, float_format=lambda x: f"{x:.4f}") + "</html>")
    (OUT / "index.html").write_text("\n".join(html), encoding="utf-8")
    manifest = {
        "name": "Motor Imagery", "dataset": "OpenNeuro ds002336 / Lioi XP1",
        "subjects": sorted(df.subject.unique().tolist()), "n_candidates": int(df.candidate.nunique()),
        "n_rows": len(df), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_roots": [str(p) for p in SEED_DIRS], "fixed_context": ["MIpre", "MIpost", "eegNF"],
        "camera": {"elev": 28, "azim": -60, "equal_axes": True},
        "max_geometry_distance_error": float(pd.DataFrame(checks).max_relative_distance_error.max()),
        "max_geometry_PS_error": float(pd.DataFrame(checks).max_PS_error.max()),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    if len(SEED_DIRS) != 5:
        raise RuntimeError(f"expected five cross-subject seeds, found {len(SEED_DIRS)}")
    meta = pd.read_csv(META_PATH)
    subjects = [s for s, p in meta.groupby("subject", sort=True)
                if p.run.isin(["MIpre", "MIpost"]).any() and p.run.eq("eegNF").any()]
    device = torch.device(os.environ.get("DEV", "cuda:0") if torch.cuda.is_available() else "cpu")
    extract_features(device, subjects)
    df, quads = analyse(subjects)
    df = add_shortlist_ccgp(df)
    make_outputs(df, quads)
    print("DONE", OUT, "rows", len(df), "candidates", df.candidate.nunique(), flush=True)


if __name__ == "__main__":
    main()
