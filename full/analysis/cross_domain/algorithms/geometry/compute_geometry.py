











import json, sys
from pathlib import Path
import numpy as np, pandas as pd, h5py
from scipy.stats import wilcoxon

ROOT = Path('/media/wsqlab/data/gy/code/Eigen_brain_decoding')
OUT = ROOT / 'claude_add/geometry_reorg_20260918'
CGE = ROOT / 'figures/compositional_geometry_experiment'
GEO = ROOT / 'lizhuo_exp/lizhuo_exp_result/20260910_geometry_obs_vs_mnsr/out'
EXPERT = ROOT / 'guoyi_exp/geometric_add_exp/gaussian_expert_experiment_20260916'
R = 200
rng = np.random.default_rng(0)


CONFIG = {
    'Motor': dict(dataset='BCIC IV-2a (EEG, 4-class MI)', A='hand vs non-hand', B='within-pair identity',
                  labels=['LH', 'RH', 'FT', 'TG'], low='observation expert mean', high='prior expert mean'),
    'Perception': dict(dataset='NSD (fMRI, natural images)', A='person absent/present', B='vehicle absent/present',
                       labels=['P-V-', 'P+V-', 'P-V+', 'P+V+'], low='observation expert mean', high='PoE posterior mean'),
    'Internal mentation': dict(dataset='Lee2021 (fMRI, autobiographical recall)', A='negative/positive valence', B='low/high vividness',
                               labels=['N/L', 'P/L', 'N/H', 'P/H'], low='observation encoder', high='PoE posterior mean'),
}


def inputs():

    for s in [f'sub-{i:03d}' for i in range(1, 10)]:
        Xo, Xp, ys = [], [], []
        for f in range(5):
            d = np.load(GEO / 'within' / f'{s}_fold{f}' / 'predictions.npz', allow_pickle=True)
            Xo.append(d['mu_obs']); Xp.append(d['mu_prior']); ys.append(d['y_true'])
        y = np.concatenate(ys)

        yield 'Motor', s, dict(obs=np.concatenate(Xo), prior=np.concatenate(Xp)), y
    labels = pd.read_csv(CGE / 'three_domains_real/nsd_person_vehicle_labels.csv').set_index('stim_id')
    for p in sorted(EXPERT.glob('NSD_sub-*_experts.npz')):
        z = np.load(p); y = labels.loc[z['stim_ids'], 'cell'].to_numpy(int)
        yield 'Perception', p.name.split('_')[1], dict(obs=z['observation'], prior=z['prior'], posterior=z['posterior']), y
    z = np.load(CGE / 'real_lee2021/inferred_features.npz')
    subs = sorted({k.split('_')[0] for k in z.files})
    for s in subs:
        yield 'Internal mentation', s, dict(obs512=z[f'{s}_Observation'], obs=z[f'{s}_Observation_expert'], posterior=z[f'{s}_PoE_posterior']), z[f'{s}_labels']


def cv_stats(X, y, cells, R=R):

    X = X - X.mean(0)
    X /= np.sqrt(np.mean(np.sum(X * X, 1)))
    idx = [np.where(y == c)[0] for c in cells]
    n = min(len(i) for i in idx); h = n // 2
    acc = dict(num_A=0., n11=0., n22=0., num_B=0., m11=0., m22=0., clo=0., D2=np.zeros((4, 4)), G=np.zeros((4, 4)),
               ax_num=0., ax_a=0., ax_b=0., within=0.)
    Cfull = np.zeros((4, X.shape[1]))
    for r in range(R):
        C1, C2 = [], []
        for k, ii in enumerate(idx):
            jj = rng.permutation(ii)[:n]
            C1.append(X[jj[:h]].mean(0)); C2.append(X[jj[h:2 * h]].mean(0))
            Cfull[k] += X[jj].mean(0) / R
            within_acc = X[jj].var(0).sum() / 4
            acc['within'] += within_acc
        C1, C2 = np.stack(C1), np.stack(C2)

        a1_1, a1_2 = C1[1] - C1[0], C2[1] - C2[0]
        a2_1, a2_2 = C1[3] - C1[2], C2[3] - C2[2]
        acc['num_A'] += 0.5 * (a1_1 @ a2_2 + a1_2 @ a2_1); acc['n11'] += a1_1 @ a1_2; acc['n22'] += a2_1 @ a2_2
        b1_1, b1_2 = C1[2] - C1[0], C2[2] - C2[0]
        b2_1, b2_2 = C1[3] - C1[1], C2[3] - C2[1]
        acc['num_B'] += 0.5 * (b1_1 @ b2_2 + b1_2 @ b2_1); acc['m11'] += b1_1 @ b1_2; acc['m22'] += b2_1 @ b2_2
        v1, v2 = C1[0] + C1[3] - C1[1] - C1[2], C2[0] + C2[3] - C2[1] - C2[2]
        acc['clo'] += v1 @ v2
        d = C1[:, None] - C1[None]; e = C2[:, None] - C2[None]
        acc['D2'] += np.einsum('ijk,ijk->ij', d, e)
        Z1, Z2 = C1 - C1.mean(0), C2 - C2.mean(0)
        acc['G'] += 0.5 * (Z1 @ Z2.T + Z2 @ Z1.T)
        A1, A2 = 0.5 * (a1_1 + a2_1), 0.5 * (a1_2 + a2_2)
        B1, B2 = 0.5 * (b1_1 + b2_1), 0.5 * (b1_2 + b2_2)
        acc['ax_num'] += 0.5 * (A1 @ B2 + A2 @ B1); acc['ax_a'] += A1 @ A2; acc['ax_b'] += B1 @ B2
    for k in acc:
        acc[k] = acc[k] / R
    D2 = acc['D2']; iu = np.triu_indices(4, 1)
    md = D2[iu].mean()
    relA = min(acc['n11'], acc['n22']) / max(md, 1e-12); relB = min(acc['m11'], acc['m22']) / max(md, 1e-12)
    PS_A = acc['num_A'] / np.sqrt(acc['n11'] * acc['n22']) if min(acc['n11'], acc['n22']) > 0.05 * md and md > 0 else np.nan
    PS_B = acc['num_B'] / np.sqrt(acc['m11'] * acc['m22']) if min(acc['m11'], acc['m22']) > 0.05 * md and md > 0 else np.nan
    lam = np.sort(np.linalg.eigvalsh(acc['G']))[::-1][:3]
    lamp = np.clip(lam, 0, None)
    planarity = lamp[2] / max(lamp.sum(), 1e-12)
    axis_cos = acc['ax_num'] / np.sqrt(acc['ax_a'] * acc['ax_b']) if min(acc['ax_a'], acc['ax_b']) > 0 else np.nan
    planarity = planarity if lamp.sum() > 0.05 * abs(lam).sum() and md > 0 else np.nan

    cos = lambda u, v: u @ v / np.linalg.norm(u) / np.linalg.norm(v)
    return dict(n_per_cell=n, PS_A=float(np.clip(PS_A, -1, 1)), PS_B=float(np.clip(PS_B, -1, 1)),
                PS_A_raw=cos(Cfull[1] - Cfull[0], Cfull[3] - Cfull[2]), PS_B_raw=cos(Cfull[2] - Cfull[0], Cfull[3] - Cfull[1]),
                rel_A=float(relA), rel_B=float(relB), **decodability(X, y, cells),
                closure=float(np.sqrt(max(acc['clo'], 0)) / np.sqrt(max(md, 1e-12))),
                planarity=float(planarity), axis_cos=float(np.clip(axis_cos, -1, 1)),
                separation=float(md / max(acc['within'], 1e-12)), eig=lam.tolist(),
                D2=D2, Cfull=Cfull)


def decodability(X, y, cells):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = np.isin(y, cells); Xm = X[m]; ym = np.array([cells.index(v) for v in y[m]])
    out = {}
    for name, lab in (('acc_A', ym % 2), ('acc_B', ym // 2), ('acc_XOR', (ym % 2) ^ (ym // 2))):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        out[name] = float(cross_val_score(clf, Xm, lab, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='balanced_accuracy').mean())
    return out


def bcic_pairing(feats, y):


    best, bp = -2, None
    for pairing in ([0, 1, 2, 3], [0, 1, 3, 2]):

        cells = [pairing[0], pairing[2], pairing[1], pairing[3]]
        s = cv_stats(feats['obs'], y, cells, R=20)['PS_A_raw'] + cv_stats(feats['prior'], y, cells, R=20)['PS_A_raw']
        if s > best:
            best, bp = s, cells
    return bp


def signflip_p(d, n_mc=100000):
    d = np.asarray(d); n = len(d); obs = abs(d.mean())
    if n <= 16:
        signs = np.array(np.meshgrid(*[[1, -1]] * n)).reshape(n, -1).T
        null = np.abs((signs * d).mean(1)); return float((null >= obs - 1e-12).mean()), 'exact sign-flip'
    signs = rng.choice([1, -1], size=(n_mc, n)); null = np.abs((signs * d).mean(1))
    return float(((null >= obs - 1e-12).sum() + 1) / (n_mc + 1)), f'Monte Carlo sign-flip {n_mc}'


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    rows, group, quads = [], {}, {}
    for domain, subject, feats, y in inputs():
        if domain == 'Motor':
            cells = bcic_pairing(feats, y)
        elif domain == 'Perception':
            cells = [0, 1, 2, 3]
        else:
            cells = [0, 1, 2, 3]
        for rep in feats:
            st = cv_stats(np.asarray(feats[rep], float), y, cells)
            D2, C = st.pop('D2'), st.pop('Cfull'); eig = st.pop('eig')
            rows.append(dict(domain=domain, subject=subject, representation=rep, dim=feats[rep].shape[1],
                             cells=str(cells), **st, eig1=eig[0], eig2=eig[1], eig3=eig[2]))
            group.setdefault(f'{domain}|{rep}', []).append(D2)
            quads[f'{domain}|{subject}|{rep}'] = C
        for r in rows[-len(feats):]:
            print(domain, subject, r['representation'], cells, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    df = pd.DataFrame(rows)

    iu = np.triu_indices(4, 1); shape = []
    for (dom, sub), g in df.groupby(['domain', 'subject']):
        for hi_rep in ('prior', 'posterior'):
            if f'{dom}|{hi_rep}' not in group: continue
            subs = list(df[(df.domain == dom) & (df.representation == 'obs')].subject)
            lo = group[f'{dom}|obs'][subs.index(sub)][iu]; hi = group[f'{dom}|{hi_rep}'][subs.index(sub)][iu]
            r = np.corrcoef(lo / lo.mean(), hi / hi.mean())[0, 1] if lo.mean() > 0 and hi.mean() > 0 else np.nan
            shape.append(dict(domain=dom, subject=sub, high=hi_rep, shape_r=r, scale_ratio=hi.mean() / lo.mean() if lo.mean() > 0 else np.nan))
    shape = pd.DataFrame(shape); shape.to_csv(OUT / 'shape_change.csv', index=False)
    df.to_csv(OUT / 'participant_metrics.csv', index=False)
    np.savez_compressed(OUT / 'group_rdm.npz', **{k: np.stack(v) for k, v in group.items()})
    np.savez_compressed(OUT / 'quads.npz', **quads)
    stats = []
    for dom in CONFIG:
      for hi_rep in ('prior', 'posterior'):
        if not ((df.domain == dom) & (df.representation == hi_rep)).any(): continue
        for m in ['PS_A', 'PS_B', 'closure', 'planarity', 'axis_cos', 'separation', 'PS_A_raw', 'PS_B_raw', 'rel_A', 'rel_B', 'acc_A', 'acc_B', 'acc_XOR']:
            lo = df[(df.domain == dom) & (df.representation == 'obs')].set_index('subject')[m]
            hi = df[(df.domain == dom) & (df.representation == hi_rep)].set_index('subject')[m]
            d = (hi - lo.reindex(hi.index)).to_numpy(); d = d[np.isfinite(d)]
            if len(d) < 3: continue
            p, test = signflip_p(d)
            try:
                pw = wilcoxon(d).pvalue
            except ValueError:
                pw = np.nan
            stats.append(dict(domain=dom, high=hi_rep, metric=m, n=len(d), low=lo.mean(), high_mean=hi.mean(), delta=d.mean(),
                              n_up=int((d > 0).sum()), p_signflip=p, test=test, p_wilcoxon=pw))
    st = pd.DataFrame(stats)
    main_mask = st.metric.isin(['PS_A_raw', 'PS_B_raw', 'closure', 'planarity'])
    pv = st.loc[main_mask, 'p_signflip'].to_numpy(); order = np.argsort(pv); m = len(pv)
    q = np.empty(m); q[order] = np.minimum.accumulate((pv[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    st['q_bh'] = np.nan; st.loc[main_mask, 'q_bh'] = np.clip(q, 0, 1)
    st.to_csv(OUT / 'paired_stats.csv', index=False)
    pd.set_option('display.width', 250); print(st.to_string())
    (OUT / 'config.json').write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
