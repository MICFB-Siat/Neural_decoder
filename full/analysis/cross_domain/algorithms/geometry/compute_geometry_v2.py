






import json, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_geometry import cv_stats, bcic_pairing, signflip_p, ROOT, CGE, GEO, EXPERT, rng

OUT = ROOT / 'claude_add/geometry_reorg_20260918/v2'; OUT.mkdir(exist_ok=True, parents=True)
LEE_RAW = Path('/media/wsqlab/nas2/Dataset/ds002835')
CONFIG = {
    'Motor': dict(dataset='Lioi XP1 EEG-fMRI motor imagery (EEG)', A='rest / motor imagery', B='no feedback (MIpre) / EEG neurofeedback',
                  labels=['Rest\nnoFB', 'MI\nnoFB', 'Rest\nFB', 'MI\nFB'], reps=['obs', 'prior']),
    'Perception': dict(dataset='NSD (fMRI)', A='inanimate / animate', B='small / large real-world size',
                       labels=['inan.\nsmall', 'anim.\nsmall', 'inan.\nlarge', 'anim.\nlarge'], reps=['obs', 'prior', 'posterior'],
                       alt=dict(PxV=dict(A='person absent / present', B='vehicle absent / present', labels=['P-V-', 'P+V-', 'P-V+', 'P+V+']))),
    'Internal mentation': dict(dataset='Lee2021 prospection (fMRI)', A='negative / positive valence', B='low / high vividness', C='near / far future',
                               labels=['N/L', 'P/L', 'N/H', 'P/H'], reps=['obs', 'posterior']),
    'Motor (BCIC)': dict(dataset='BCIC IV-2a (EEG)', A='hand / non-hand', B='within-pair identity',
                         labels=['LH', 'FT', 'RH', 'TG'], reps=['obs', 'prior']),
}


def inputs():

    meta = pd.read_csv(CGE / 'three_domains_real/motor_trial_context.csv')
    ck = ROOT / 'data_check_20260507/Exp_tyf_Results/Results/Exp_Fig2/MOTOR/BT-ND+BrainOmni_cross5fold/MOTOR_cross_unifiedv3_seed1_eeg_cross_frozen_cache_20260601_104738'
    folds = {}
    for fold in range(5):
        for s in json.loads((ck / f'fold_{fold:02d}/split.json').read_text())['val_subjects']:
            folds[s] = fold
    for subject, part in meta.groupby('subject', sort=True):
        z = np.load(GEO / 'motor_cross' / f'frame{folds[subject]}_{subject}.npz')
        assert z['unseen'].item() and np.array_equal(part.label.to_numpy(), z['y_true'])
        take = part.run.isin(['MIpre', 'eegNF']).to_numpy()
        if part.run.eq('MIpre').sum() == 0:
            continue
        y = part.label.to_numpy()[take] + 2 * (part.run.to_numpy()[take] == 'eegNF')
        yield 'Motor', subject, dict(obs=z['mu_obs'][take], prior=z['mu_prior'][take]), y, None

    labels = pd.read_csv(CGE / 'three_domains_real/nsd_person_vehicle_labels.csv').set_index('stim_id')
    labels2 = pd.read_csv(OUT / 'nsd_animacy_size_labels.csv').set_index('stim_id')
    labels3 = pd.read_csv(OUT / 'nsd_animacy_spiky_labels.csv').set_index('stim_id')
    for p in sorted(EXPERT.glob('NSD_sub-*_experts.npz')):
        z = np.load(p); y = labels.loc[z['stim_ids'], 'cell'].to_numpy(int); y2 = labels2.loc[z['stim_ids'], 'cell'].to_numpy(int)
        y3 = labels3.reindex(z['stim_ids'])['cell'].fillna(-1).to_numpy(int)
        yield 'Perception', p.name.split('_')[1], dict(obs=z['observation'], prior=z['prior'], posterior=z['posterior']), (y2, y, y3), 'nsd'

    z = np.load(CGE / 'real_lee2021/inferred_features.npz')
    for s in sorted({k.split('_')[0] for k in z.files}):
        ev = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(glob.glob(str(LEE_RAW / s / 'func' / '*events.tsv')))], ignore_index=True)
        assert np.array_equal((ev.positive + 2 * ev.vivid).to_numpy(), z[f'{s}_labels'])
        y8 = (ev.positive + 2 * ev.vivid + 4 * ev.farfuture).to_numpy()
        yield 'Internal mentation', s, dict(obs=z[f'{s}_Observation_expert'], posterior=z[f'{s}_PoE_posterior']), y8, 'cube'

    for s in [f'sub-{i:03d}' for i in range(1, 10)]:
        Xo, Xp, ys = [], [], []
        for f in range(5):
            d = np.load(GEO / 'within' / f'{s}_fold{f}' / 'predictions.npz', allow_pickle=True)
            Xo.append(d['mu_obs']); Xp.append(d['mu_prior']); ys.append(d['y_true'])
        yield 'Motor (BCIC)', s, dict(obs=np.concatenate(Xo), prior=np.concatenate(Xp)), np.concatenate(ys), 'bcic'


def cube_stats(X, y, R=200):


    X = X - X.mean(0); X /= np.sqrt(np.mean(np.sum(X * X, 1)))
    idx = [np.where(y == c)[0] for c in range(8)]; n = min(len(i) for i in idx); h = n // 2
    edges = {'A': [(0, 1), (2, 3), (4, 5), (6, 7)], 'B': [(0, 2), (1, 3), (4, 6), (5, 7)], 'C': [(0, 4), (1, 5), (2, 6), (3, 7)]}
    D = np.array([[1, a, b, c] for c in (0, 1) for b in (0, 1) for a in (0, 1)], float)
    H = D @ np.linalg.pinv(D)
    Cfull = np.zeros((8, X.shape[1])); acc = {f: dict(num=0., den=0.) for f in edges}; r2cv_num = 0.; r2cv_den = 0.
    for r in range(R):
        C1, C2 = [], []
        for k, ii in enumerate(idx):
            jj = rng.permutation(ii)[:n]; C1.append(X[jj[:h]].mean(0)); C2.append(X[jj[h:2 * h]].mean(0)); Cfull[k] += X[jj].mean(0) / R
        C1, C2 = np.stack(C1), np.stack(C2)
        for f, E in edges.items():
            v1 = [C1[b] - C1[a] for a, b in E]; v2 = [C2[b] - C2[a] for a, b in E]
            for i in range(4):
                for j in range(i + 1, 4):
                    acc[f]['num'] += 0.5 * (v1[i] @ v2[j] + v2[i] @ v1[j])
            acc[f]['den'] += np.mean([v1[i] @ v2[i] for i in range(4)]) * 6
        Z1, Z2 = C1 - C1.mean(0), C2 - C2.mean(0)
        r2cv_num += np.sum((H @ Z1) * (H @ Z2)); r2cv_den += np.sum(Z1 * Z2)
    cos = lambda u, v: u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    out = {}
    for f, E in edges.items():
        v = [Cfull[b] - Cfull[a] for a, b in E]
        out[f'PS_{f}_raw'] = float(np.mean([cos(v[i], v[j]) for i in range(4) for j in range(i + 1, 4)]))
        out[f'PS_{f}_cv'] = float(np.clip(acc[f]['num'] / acc[f]['den'], -1, 1)) if acc[f]['den'] > 0 else np.nan
    Zf = Cfull - Cfull.mean(0)
    out['additive_R2_raw'] = float(np.sum((H @ Zf) ** 2) / np.sum(Zf ** 2))
    out['additive_R2_cv'] = float(np.clip(r2cv_num / r2cv_den, 0, 1)) if r2cv_den > 0 else np.nan
    out['n_per_cell'] = n
    return out, Cfull


def main():
    rows, cube_rows, quads, cubes = [], [], {}, {}
    for domain, subject, feats, y, kind in inputs():
        designs = {'main': ([0, 1, 2, 3], y)}
        if kind == 'cube':

            A, B, C = y % 2, (y // 2) % 2, y // 4
            designs = {'AxB': ([0, 1, 2, 3], A + 2 * B), 'AxC': ([0, 1, 2, 3], A + 2 * C), 'BxC': ([0, 1, 2, 3], B + 2 * C)}
            for rep in feats:
                m, C8 = cube_stats(np.asarray(feats[rep], float), y)
                cube_rows.append(dict(domain=domain, subject=subject, representation=rep, **m)); cubes[f'{subject}|{rep}'] = C8
        elif kind == 'bcic':
            designs = {'main': (bcic_pairing(feats, y), y)}
        elif kind == 'nsd':
            designs = {'AxS': ([0, 1, 2, 3], y[0]), 'PxV': ([0, 1, 2, 3], y[1]), 'AxK': ([0, 1, 2, 3], y[2])}
        for dname, (cells, yy) in designs.items():
            for rep in feats:
                st = cv_stats(np.asarray(feats[rep], float), yy, cells)
                st.pop('D2'); C = st.pop('Cfull'); st.pop('eig')
                rows.append(dict(domain=domain, design=dname, subject=subject, representation=rep, **st))
                quads[f'{domain}|{dname}|{subject}|{rep}'] = C
        print(domain, subject, {k: v for k, v in rows[-1].items() if k in ('n_per_cell',)}, flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / 'participant_metrics.csv', index=False)
    cdf = pd.DataFrame(cube_rows); cdf.to_csv(OUT / 'cube_metrics.csv', index=False)
    np.savez_compressed(OUT / 'quads.npz', **quads); np.savez_compressed(OUT / 'cubes.npz', **cubes)
    stats = []
    for (dom, des), g in df.groupby(['domain', 'design']):
        for hi in [r for r in CONFIG[dom]['reps'] if r != 'obs']:
            for m in ['PS_A_raw', 'PS_B_raw', 'closure', 'PS_A', 'PS_B', 'planarity', 'axis_cos', 'acc_A', 'acc_B', 'acc_XOR', 'rel_A', 'rel_B']:
                lo = g[g.representation == 'obs'].set_index('subject')[m]; h = g[g.representation == hi].set_index('subject')[m]
                d = (h - lo.reindex(h.index)).to_numpy(); d = d[np.isfinite(d)]
                if len(d) < 3: continue
                p, test = signflip_p(d)
                stats.append(dict(domain=dom, design=des, high=hi, metric=m, n=len(d), low=lo.mean(), high_mean=h.mean(), delta=d.mean(),
                                  n_up=int((d > 0).sum()), n_down=int((d < 0).sum()), p_signflip=p, test=test))
    for hi in ['posterior']:
        for m in ['PS_A_raw', 'PS_B_raw', 'PS_C_raw', 'PS_A_cv', 'PS_B_cv', 'PS_C_cv', 'additive_R2_raw', 'additive_R2_cv']:
            lo = cdf[cdf.representation == 'obs'].set_index('subject')[m]; h = cdf[cdf.representation == hi].set_index('subject')[m]
            d = (h - lo.reindex(h.index)).to_numpy(); d = d[np.isfinite(d)]
            if len(d) < 3: continue
            p, test = signflip_p(d)
            stats.append(dict(domain='Internal mentation', design='cube', high=hi, metric=m, n=len(d), low=lo.mean(), high_mean=h.mean(), delta=d.mean(),
                              n_up=int((d > 0).sum()), n_down=int((d < 0).sum()), p_signflip=p, test=test))
    st = pd.DataFrame(stats); st.to_csv(OUT / 'paired_stats.csv', index=False)
    pd.set_option('display.width', 250)
    print(st[st.metric.isin(['PS_A_raw', 'PS_B_raw', 'PS_C_raw', 'closure', 'acc_A', 'acc_B', 'additive_R2_raw', 'additive_R2_cv'])].round(3).to_string())
    (OUT / 'config.json').write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
