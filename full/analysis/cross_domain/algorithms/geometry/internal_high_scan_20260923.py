

from pathlib import Path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl_internal_high_scan')
import argparse
import hashlib
import json
import sys
from types import SimpleNamespace
from itertools import product, combinations
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / 'v2/internal_high_scan_20260923'
ROOT = BASE.parents[1]
CODE = ROOT / 'lizhuo_exp/lizhuo_exp_class2/SKIP/code/enc_mod'
EXP = Path('/media/wsqlab/nas2/Dataset/ds005256-download/experiments/alignvideo_selected30_continue20_fourtask_lr3e5_loso80_20260723')
CKPT = EXP / 'result_loso30_direct_ours_80seed/alignvideo/fold-25/seed-00/poe/checkpoint.pt'
COHORT_ROOT = ROOT / 'lizhuo_exp/lizhuo_exp_NSD/test_class2_0715/39_alignvideo_oldmethod_17sub_common35_20260723/source_data'
LAYERS = ('obs_hidden', 'indiv_hidden', 'obs_mu', 'indiv_mu', 'poe_mu')
CAMERAS = {'standard': (28, -60), 'depth': (22, -115), 'elevated': (48, -68)}
BLUE, ORANGE, GRAY = '#3668B4', '#E69235', '#C9CDD3'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fixpath(path):
    return Path(str(path).replace('/home/guoyi/nas2/share', '/media/wsqlab/nas2')
                .replace('/home/guoyi/a800/share/code', '/media/wsqlab/data/gy/code'))


def train():
    import h5py
    import torch
    sys.path.insert(0, str(CODE))
    import decode_cross_subject_residual_5fold as source
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in ('checkpoints', 'features', 'code'):
        (OUT / folder).mkdir(exist_ok=True)
    c = torch.load(CKPT, map_location='cpu', weights_only=False)
    outer_history = next(value for key, value in c.items()
                         if key.startswith('outer_') and key.endswith('_history'))
    cache = np.load(BASE / 'v2/spacetop_trial_features.npz')
    subjects = cache['subjects'].astype(str)
    videos = pd.read_csv(COHORT_ROOT / 'K_PER_VIDEO_17sub_pooled.csv')
    keys = list(zip(videos.session.astype(int), videos.objective_id.astype(int)))
    all_obs, all_indiv, all_ratings, selected_trials = [], [], [], []
    for subject in subjects:
        with h5py.File(EXP / f'features_selected30/alignvideo/{subject}.h5', 'r') as h:
            for attr, name in [('cifti_ckpt_sha256', 'obs'), ('indiv_modes_ckpt_sha256', 'indiv')]:
                assert str(h.attrs[attr]) == c['frozen_feature_sources']['encoder_checkpoint_sha256'][name]
            lookup = {tuple(map(int, pair)): i for i, pair in enumerate(zip(h['session'][:], h['labels_objective'][:]))}
            ids = np.array([lookup[k] for k in keys])
            all_obs.append(h['bci_obs'][:].astype(np.float32).mean(1)[ids])
            all_indiv.append(h['bci_prior_indiv'][:].astype(np.float32).mean(1)[ids])
            all_ratings.append(h['labels_subjective'][:][ids])
            selected_trials.append(ids)
    ratings = np.stack(all_ratings)
    keep = np.all(np.isfinite(ratings[:, :, cache['five']]) & (ratings[:, :, cache['five']] >= 0), axis=2).all(0)
    low, indiv = np.stack(all_obs)[:, keep], np.stack(all_indiv)[:, keep]
    ratings = ratings[:, keep]
    assert np.array_equal(low, cache['low']), 'Frozen low features changed'
    assert np.array_equal(ratings, cache['ratings']), 'Rating/trial selection changed'
    cells = ((ratings[:, :, 6] > np.median(ratings[:, :, 6], axis=1)[:, None]).astype(int)
             + 2 * (ratings[:, :, 2] > np.median(ratings[:, :, 2], axis=1)[:, None]).astype(int))
    valid = np.array([min(np.bincount(y, minlength=4)) >= 3 for y in cells])
    assert valid.sum() == 16
    previous = pd.read_csv(BASE / 'v2/internal_engaged_sad_stats_20260923/participant_metrics.csv')
    assert set(subjects[valid]) == set(previous.subject)
    np.savez_compressed(OUT / 'fixed_analysis_data.npz', low=low, indiv=indiv, ratings=ratings,
                        subjects=subjects, cells=cells, eligible=valid,
                        selected_trials=np.stack(selected_trials)[:, keep], videos=np.array(keys)[keep])
    rows = []
    for s, subject in enumerate(subjects):
        for v, key in enumerate(np.array(keys)[keep]):
            rows.append(dict(subject=subject, eligible=bool(valid[s]), session=int(key[0]), video=int(key[1]),
                             original_trial=int(np.stack(selected_trials)[s, keep][v]), cell=int(cells[s, v]),
                             engaged=float(ratings[s, v, 6]), sad=float(ratings[s, v, 2])))
    pd.DataFrame(rows).to_csv(OUT / 'fixed_trial_labels.csv', index=False)
    dev = torch.device('cuda:0')
    scalers = [source.ArrayScaler.from_state(s) for s in c['feature_scalers']]
    obs_t = torch.tensor(scalers[0].transform(low.reshape(-1, low.shape[-1])), device=dev)
    ind_t = torch.tensor(scalers[1].transform(indiv.reshape(-1, indiv.shape[-1])), device=dev)

    @torch.inference_mode()
    def extract(model, name):
        model.eval()
        ho, hi = model.obs_expert.trunk(obs_t), model.indiv_expert.trunk(ind_t)
        mo, mi = model.obs_expert.mu(ho), model.indiv_expert.mu(hi)
        to, ti = (-model.obs_expert.logvar(ho).clamp(-5, 5)).exp(), (-model.indiv_expert.logvar(hi).clamp(-5, 5)).exp()

        posterior = (to * mo + ti * mi) / (1 + to + ti)
        vals = dict(zip(LAYERS, [ho, hi, mo, mi, posterior]))
        arrays = {k: v.cpu().numpy().reshape(len(subjects), low.shape[1], -1) for k, v in vals.items()}
        np.savez_compressed(OUT / f'features/{name}.npz', **arrays)
        return arrays

    original = source.build_model_from_checkpoint(c, dev)
    original_feats = extract(original, 'original_epoch13')
    original_error = float(np.max(np.abs(original_feats['poe_mu'] - cache['high'])))
    assert original_error < 2e-6, f'Original high failed replay: {original_error}'
    print('Original replay max error', original_error, 'eligible', valid.sum(), flush=True)
    sources = c['frozen_feature_sources']
    training_subjects = source.load_task('alignvideo', fixpath(sources['ours_root']),
        fixpath(sources['neurostorm_root']), fixpath(sources['manifest_root']),
        fixpath(sources['obs_encoder_checkpoint']), fixpath(sources['indiv_encoder_checkpoint']),
        False, methods=['poe'])
    lookup = {s.subject: s for s in training_subjects}
    outer_train = [lookup[s] for s in c['outer_train_subjects']]
    ref = source.ResidualReference.from_state(c['outer_training_reference'])
    args = SimpleNamespace(**c['model_config'], **c['training_config'],
        target_mode=c['target_mode'], min_reference_subjects=c['min_reference_subjects'])
    prep = source.prepare_arrays(outer_train, ref, 'poe', args.min_reference_subjects, target_mode=args.target_mode)
    for computed, saved in zip(prep[3], scalers):
        assert np.array_equal(computed.mean, saved.mean) and np.array_equal(computed.std, saved.std)
    saved_target = source.ArrayScaler.from_state(c['target_scaler'])
    assert np.array_equal(prep[4].mean, saved_target.mean) and np.array_equal(prep[4].std, saved_target.std)
    tensors = source.tensorize(prep, dev)
    source.set_seed(c['model_seed'] + 17)
    model = source.new_model('poe', c['input_dim'], len(c['target_names']), args, dev)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(c['model_seed'] + 17 + 7919)
    history, replay = [], {}
    for epoch in range(27):
        if epoch:
            loss = source.train_epoch(model, tensors, 'poe', args, optim, rng)
            history.append(dict(epoch=epoch, loss=loss,
                original_loss=outer_history[epoch-1]['train_loss'] if epoch <= 13 else None))
        state = source.cpu_state_dict(model)
        payload = {k: c[k] for k in ('method', 'input_dim', 'target_names', 'model_config', 'feature_scalers',
            'target_scaler', 'outer_train_subjects', 'outer_test_subjects', 'training_config', 'model_seed', 'target_mode')}
        payload.update(model_state=state, epoch=epoch, optimizer_state=optim.state_dict(),
                       numpy_rng_state=rng.bit_generator.state, torch_rng_state=torch.get_rng_state(),
                       cuda_rng_state=torch.cuda.get_rng_state_all(), source_checkpoint=str(CKPT))
        torch.save(payload, OUT / f'checkpoints/epoch_{epoch:03d}.pt')
        features = extract(model, f'epoch_{epoch:03d}')
        if epoch == 13:
            replay = {'max_weight_abs_error': max(float((state[k] - c['model_state'][k]).abs().max()) for k in state),
                      'max_poe_feature_abs_error': float(np.max(np.abs(features['poe_mu'] - cache['high']))),
                      'original_checkpoint_inference_error': original_error}
            replay['matched_original'] = replay['max_weight_abs_error'] < 2e-5 and replay['max_poe_feature_abs_error'] < 2e-5
            dump(OUT / 'epoch13_replay.json', replay)
            print('Epoch13 replay', replay, flush=True)
        pd.DataFrame(history).to_csv(OUT / 'training_history.csv', index=False)
        print('Saved high epoch', epoch, flush=True)
    manifest = dict(design='engaged low->high x sad low/high', eligible_subjects=subjects[valid].tolist(),
        excluded_subjects=subjects[~valid].tolist(), low_fixed=True, low_cache_sha256=sha(BASE / 'v2/spacetop_trial_features.npz'),
        source_checkpoint=str(CKPT), source_checkpoint_sha256=sha(CKPT), fold=25, seed=0,
        transductive=True, original_training_epoch=13, trained_epochs=list(range(27)),
        epoch0='randomly initialized HIGH inference head, control only; LOW remains pretrained',
        train_subjects=c['outer_train_subjects'], target_names=c['target_names'],
        model_config=c['model_config'], training_config=c['training_config'], replay=replay,
        layers={'obs_hidden':'observation expert trained hidden activation, 256d',
                'indiv_hidden':'individual expert trained hidden activation, 256d',
                'obs_mu':'observation expert trained Gaussian mean, 128d',
                'indiv_mu':'individual expert trained Gaussian mean, 128d',
                'poe_mu':'original fused Gaussian PoE posterior mean, 128d'},
        cameras=CAMERAS, palette=dict(context_A=BLUE, context_A_prime=ORANGE, links=GRAY),
        centroids='all fixed observations in each median-split cell; original participant centering and isotropic RMS scale',
        statistics='two-sided paired Wilcoxon + exact two-sided sign-flip; BH q across all trained candidates, separate metric/test families',
        display='orthographic projection, per-axis tight limits with equal data-unit scale via matching box aspect; no anisotropic stretching')
    dump(OUT / 'manifest.json', manifest)


def centroid(x, labels):
    x = x.astype(np.float64)
    x = x - x.mean(0)
    x /= np.sqrt(np.mean(np.sum(x*x, axis=1)))
    return np.stack([x[labels == cell].mean(0) for cell in range(4)])


def metrics(c):
    c = np.asarray(c, dtype=np.float64)
    u, v = c[1] - c[0], c[3] - c[2]
    ps = float(u @ v / (np.linalg.norm(u)*np.linalg.norm(v)))
    singular = np.linalg.svd(c - c.mean(0), compute_uv=False)
    return dict(PS=ps, angle=float(np.degrees(np.arccos(np.clip(ps, -1, 1)))),
                depth=float(singular[2]**2 / np.sum(singular**2)))


def ccgp(x, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    rng = np.random.default_rng(0)
    cells = [np.flatnonzero(labels == i) for i in range(4)]
    n = min(map(len, cells))
    ys = np.r_[np.zeros(n), np.ones(n)]
    vals = []
    for _ in range(20):
        selected = [rng.permutation(ids)[:n] for ids in cells]
        for a, b in ((0, 2), (2, 0)):
            itrain, itest = np.r_[selected[a], selected[a+1]], np.r_[selected[b], selected[b+1]]
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=.01, penalty='l2', max_iter=3000, random_state=0))
            vals.append(clf.fit(x[itrain], ys).score(x[itest], ys))
    return float(np.mean(vals))


def paired(low, high):
    from scipy.stats import wilcoxon
    d = np.asarray(high) - np.asarray(low)
    signs = np.array(list(product((-1, 1), repeat=len(d))), dtype=float)
    return dict(n=len(d), low_mean=float(np.mean(low)), high_mean=float(np.mean(high)),
        delta_mean=float(d.mean()), n_up=int((d > 0).sum()),
        wilcoxon_p=float(wilcoxon(d, alternative='two-sided').pvalue) if np.any(d) else 1.,
        signflip_p=float(np.mean(np.abs(signs @ d / len(d)) >= abs(d.mean()) - 1e-12)))


def evaluate_one(job):
    name, layer = job
    fixed = np.load(OUT / 'fixed_analysis_data.npz')
    highs = np.load(OUT / f'features/{name}.npz')[layer]
    lowtab = pd.read_csv(OUT / 'low_metrics.csv').set_index('subject')
    rows, centers = [], {}
    for s in np.flatnonzero(fixed['eligible']):
        subject = str(fixed['subjects'][s])
        c = centroid(highs[s], fixed['cells'][s])
        lo = lowtab.loc[subject]
        hi = metrics(c)
        rows.append(dict(candidate=f'{name}__{layer}', checkpoint=name, layer=layer, subject=subject,
            low_PS=lo.PS, low_angle=lo.angle, low_depth=lo.depth, low_CCGP=lo.CCGP,
            **{f'high_{k}': v for k,v in hi.items()}, high_CCGP=ccgp(highs[s].astype(float), fixed['cells'][s]),
            n_min_cell=int(lo.n_min_cell)))
        centers[subject] = c
    frame = pd.DataFrame(rows)
    candidate = f'{name}__{layer}'
    frame.to_csv(OUT / f'metrics/{candidate}.csv', index=False)
    np.savez_compressed(OUT / f'centroids/{candidate}.npz', **centers)
    summaries = [dict(candidate=candidate, checkpoint=name, layer=layer, metric=m,
                      **paired(frame[f'low_{m}'], frame[f'high_{m}'])) for m in ('PS', 'CCGP')]
    return summaries


def bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum(1, np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    return q


def evaluate():
    for folder in ('metrics', 'centroids'):
        (OUT / folder).mkdir(exist_ok=True)
    fixed = np.load(OUT / 'fixed_analysis_data.npz')
    rows, centers = [], {}
    for s in np.flatnonzero(fixed['eligible']):
        c = centroid(fixed['low'][s], fixed['cells'][s])
        subject = str(fixed['subjects'][s])
        rows.append(dict(subject=subject, **metrics(c), CCGP=ccgp(fixed['low'][s].astype(float), fixed['cells'][s]),
                         n_min_cell=min(np.bincount(fixed['cells'][s], minlength=4))))
        centers[subject] = c
    pd.DataFrame(rows).to_csv(OUT / 'low_metrics.csv', index=False)
    np.savez_compressed(OUT / 'centroids/low.npz', **centers)
    jobs = [(name, layer) for name in ['original_epoch13'] + [f'epoch_{i:03d}' for i in range(27)] for layer in LAYERS]
    allrows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, res in enumerate(pool.map(evaluate_one, jobs)):
            allrows.extend(res)
            pd.DataFrame(allrows).to_csv(OUT / 'group_summary_partial.csv', index=False)
            print('Evaluated', i+1, '/', len(jobs), res[0]['candidate'], flush=True)
    group = pd.DataFrame(allrows)
    group['control_only'] = group.checkpoint.eq('epoch_000')
    for metric in ('PS', 'CCGP'):
        mask = (group.metric == metric) & ~group.control_only
        for test in ('wilcoxon', 'signflip'):
            group.loc[mask, f'{test}_q_BH'] = bh(group.loc[mask, f'{test}_p'])
    group['positive_raw_both_tests'] = (group.delta_mean > 0) & (group.wilcoxon_p < .05) & (group.signflip_p < .05) & ~group.control_only
    group['positive_BH_both_tests'] = (group.delta_mean > 0) & (group.wilcoxon_q_BH < .05) & (group.signflip_q_BH < .05) & ~group.control_only
    group.to_csv(OUT / 'group_summary.csv', index=False)
    participants = pd.concat([pd.read_csv(OUT / f'metrics/{name}__{layer}.csv') for name,layer in jobs], ignore_index=True)
    participants['angle_improvement'] = participants.low_angle - participants.high_angle
    participants.to_csv(OUT / 'participant_metrics.csv', index=False)
    old = pd.read_csv(BASE / 'v2/internal_engaged_sad_stats_20260923/participant_metrics.csv').set_index('subject')
    orig = participants[participants.candidate == 'original_epoch13__poe_mu'].set_index('subject').loc[old.index]
    errors = {k: float(np.max(np.abs(orig[k] - old[v]))) for k,v in
              [('low_PS','PS_low_transductive'), ('high_PS','PS_high_transductive'),
               ('low_CCGP','CCGP_low_transductive_design'), ('high_CCGP','CCGP_high_transductive')]}
    dump(OUT / 'baseline_metric_replay.json', errors)
    assert max(errors.values()) < 1e-6, errors
    print('All candidate statistics complete; baseline replay:', errors, flush=True)


def exact3(c):
    c = np.asarray(c, dtype=np.float64)
    x = c - c.mean(0)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    vectors = [x[1]-x[0], x[3]-x[2], (x[2]+x[3]-x[0]-x[1])/2, *vh]
    axes = []
    for v in vectors:
        z = v.copy()
        for a in axes:
            z -= (z @ a)*a
        if np.linalg.norm(z) > 1e-10:
            axes.append(z / np.linalg.norm(z))
        if len(axes) == 3:
            break
    p = x @ np.stack(axes, axis=1)
    if p.shape[1] < 3:
        p = np.pad(p, ((0,0),(0,3-p.shape[1])))
    if p[2,2] < 0:
        p[:,2] *= -1
    d0 = np.array([np.linalg.norm(x[i]-x[j]) for i,j in combinations(range(4),2)])
    d1 = np.array([np.linalg.norm(p[i]-p[j]) for i,j in combinations(range(4),2)])
    err = float(np.max(np.abs(d0-d1)) / max(d0.max(),1e-12))
    assert err < 1e-10
    scale = float(d1.mean())
    p /= scale
    assert abs(metrics(p)['PS'] - metrics(c)['PS']) < 1e-9
    return p, err, scale


def panel(ax, c, title, view='standard', compact=False):
    p, err, scale = exact3(c)
    m = metrics(c)
    for a,b in ((0,2),(1,3)):
        ax.plot(*zip(p[a],p[b]), color=GRAY, lw=2.8)
    for a,b,col in ((0,1,BLUE),(2,3,ORANGE)):
        ax.quiver(*p[a],*(p[b]-p[a]),color=col,lw=3.1,arrow_length_ratio=.12)
        ax.scatter(*p[a],s=62,facecolor='white',edgecolor=col,lw=2,depthshade=False)
        ax.scatter(*p[b],s=62,color=col,depthshade=False)
    center = (p.min(0)+p.max(0))/2
    span = np.maximum(np.ptp(p,axis=0), np.ptp(p,axis=0).max()*.12)*1.12
    ax.set_xlim(center[0]-span[0]/2,center[0]+span[0]/2)
    ax.set_ylim(center[1]-span[1]/2,center[1]+span[1]/2)
    ax.set_zlim(center[2]-span[2]/2,center[2]+span[2]/2)

    ax.set_box_aspect(span, zoom=1.05)
    assert np.ptp(np.asarray(ax.get_box_aspect()) / span) < 1e-10
    ax.set_proj_type('ortho')
    ax.view_init(*CAMERAS[view])
    ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.set_facecolor((.96,.97,.98,.14))
        axis.pane.set_edgecolor((.6,.65,.7,.3))
        axis.line.set_color('#ADB5BD')
    ax.set_title(title,fontsize=10 if compact else 12,pad=1)
    ax.text2D(.5,.025,f"full-space PS {m['PS']:+.3f} | angle {m['angle']:.1f}°\ndepth {m['depth']:.3f}",
              transform=ax.transAxes,ha='center',fontsize=8 if compact else 10,
              bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1))
    return p, err, scale


def setup_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    return plt


def pair_png(low, high, subject, candidate, dest, view='standard', domain='Internal mentation', pdf=False):
    plt = setup_plot()
    fig = plt.figure(figsize=(5.5,8.4))
    coords = []
    for i, (c,label) in enumerate(((low,'Low: frozen pretrained encoder'),(high,f'High: {candidate}'))):
        ax = fig.add_axes([.04,.515 if i==0 else .035,.92,.425],projection='3d')
        coords.append(panel(ax,c,label,view))
    title = (f'{domain} | {subject}\nengagement low → high; sadness low / high | transductive'
             if domain=='Internal mentation' else f'Motor Imagery | {subject}\nrest → right-hand imagery; no feedback / EEG feedback')
    fig.suptitle(title,fontsize=11,y=.995)
    fig.text(.5,.002,'Blue: context A  •  Orange: context A′  |  open → filled: state low → high',ha='center',fontsize=8)
    dest.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest,dpi=170,facecolor='white')
    if pdf:fig.savefig(dest.with_suffix('.pdf'),facecolor='white')
    plt.close(fig)
    return coords


def contact_one(candidate):
    plt = setup_plot()
    lows = np.load(OUT / 'centroids/low.npz')
    highs = np.load(OUT / f'centroids/{candidate}.npz')
    subjects = sorted(lows.files)
    fig = plt.figure(figsize=(16,29))
    gs = fig.add_gridspec(8,4,left=.025,right=.975,bottom=.015,top=.965,wspace=.02,hspace=.10)
    for s,subject in enumerate(subjects):
        row,col = divmod(s,4)
        for level,source in enumerate((lows,highs)):
            panel(fig.add_subplot(gs[row*2+level,col],projection='3d'),source[subject],
                  f"{subject} | {'low' if level==0 else 'high'}",compact=True)
    fig.suptitle(f'Internal mentation | {candidate}\nEngaged × sad; all 16 participants; fixed standard view; transductive\nBlue = low sadness; orange = high sadness; open → filled = low → high engagement',fontsize=17,y=.994)
    dest = OUT / f'all_subjects/{candidate}.png'
    fig.savefig(dest,dpi=100,facecolor='white')
    plt.close(fig)
    return candidate


def export_points():
    low = np.load(OUT / 'centroids/low.npz')
    plotrows, fullrows, checks = [], [], []
    for file in sorted((OUT/'centroids').glob('*.npz')):
        if file.stem=='low':continue
        high=np.load(file)
        for subject in sorted(high.files):
            for rep,c in [('low',low[subject]),('high',high[subject])]:
                p,err,scale=exact3(c)
                checks.append(dict(candidate=file.stem,subject=subject,representation=rep,distance_error=err,
                                   ps_error=abs(metrics(p)['PS']-metrics(c)['PS'])))
                for cell in range(4):
                    meta=dict(candidate=file.stem,subject=subject,representation=rep,cell=cell,
                              engagement='high' if cell%2 else 'low',sadness='high' if cell//2 else 'low',
                              dimension=c.shape[1],**metrics(c))
                    plotrows.append(dict(**meta,x=p[cell,0],y=p[cell,1],z=p[cell,2],uniform_scale_divisor=scale))
                    fullrows.append(dict(**meta,**{f'f{i:03d}':v for i,v in enumerate(c[cell])}))
    pd.DataFrame(plotrows).to_csv(OUT/'centroids_plot3d.csv',index=False)
    pd.DataFrame(fullrows).to_csv(OUT/'centroids_fullspace.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'geometry_checks.csv',index=False)


def plot():
    for d in ('all_subjects','selected','motor_palette_preview'):(OUT/d).mkdir(exist_ok=True)
    group=pd.read_csv(OUT/'group_summary.csv')
    part=pd.read_csv(OUT/'participant_metrics.csv')
    ps=group[group.metric=='PS'].set_index('candidate')
    cc=group[group.metric=='CCGP'].set_index('candidate')
    part['both_metrics_raw_significant']=part.candidate.map(ps.positive_raw_both_tests & cc.positive_raw_both_tests)
    part['both_metrics_BH_significant']=part.candidate.map(ps.positive_BH_both_tests & cc.positive_BH_both_tests)


    part['low_at_least_60deg']=part.low_angle>=60
    part['high_at_most_30deg']=part.high_angle<=30
    part=part.sort_values(['both_metrics_BH_significant','both_metrics_raw_significant','low_at_least_60deg','high_angle','angle_improvement'],
                         ascending=[False,False,False,True,False])
    part.to_csv(OUT/'ranked_candidates.csv',index=False)
    low=np.load(OUT/'centroids/low.npz')
    diverse=part[part.both_metrics_raw_significant & part.low_at_least_60deg].drop_duplicates(['subject','layer'])
    shortlist=diverse.head(12)

    shortlist=pd.concat([shortlist,part[(part.layer=='poe_mu') & part.both_metrics_raw_significant & part.low_at_least_60deg].drop_duplicates('subject').head(8),
        part[(part.low_angle>=90) & part.both_metrics_BH_significant].drop_duplicates('subject').head(3),
        part[part.both_metrics_raw_significant & part.low_at_least_60deg].sort_values('high_angle').head(1),
        part[(part.candidate=='original_epoch13__poe_mu') & part.subject.isin(['sub-0058','sub-0043','sub-0081'])]]).drop_duplicates(['candidate','subject'])
    shortlist.to_csv(OUT/'shortlist.csv',index=False)
    for r in shortlist.itertuples():
        high=np.load(OUT/f'centroids/{r.candidate}.npz')[r.subject]
        for view in CAMERAS:
            pair_png(low[r.subject],high,r.subject,r.candidate,OUT/f'selected/{r.candidate}/{r.subject}_{view}.png',view,pdf=True)

    motor=BASE/'v2/selected_motor_sub108_mu_mean_3d_20260923/data/selected_centroids.npz'
    if motor.exists():
        data=np.load(motor)
        if 'low' in data and 'high' in data:
            motor_rows=[]
            for representation in ('low','high'):
                p,error,scale=exact3(data[representation])
                for cell in range(4):
                    motor_rows.append(dict(subject='sub-108',representation=representation,cell=cell,
                        state='imagery' if cell%2 else 'rest',context='EEG feedback' if cell//2 else 'no feedback',
                        x=p[cell,0],y=p[cell,1],z=p[cell,2],uniform_scale_divisor=scale,
                        distance_error=error,**metrics(data[representation])))
            pd.DataFrame(motor_rows).to_csv(OUT/'motor_palette_preview/centroids_plot3d.csv',index=False)
            for view in CAMERAS:
                pair_png(data['low'],data['high'],'sub-108','seed1_mu_mean',OUT/f'motor_palette_preview/sub-108_{view}.png',view,domain='Motor',pdf=True)
    export_points()
    candidates=group.candidate.unique().tolist()
    pending=[c for c in candidates if not (OUT/f'all_subjects/{c}.png').exists()]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for i,candidate in enumerate(pool.map(contact_one,pending)):
            print('Gallery',i+1,'/',len(pending),candidate,flush=True)
    html=['<!doctype html><meta charset="utf-8"><title>Internal high-layer and epoch scan</title>',
          '<style>body{max-width:1450px;margin:30px auto;font:16px sans-serif;color:#24324a}img{width:32%;vertical-align:top}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ddd}a{color:#3668B4}</style>',
          '<h1>Internal mentation：高层轮次 × 层候选</h1>',
          '<p>固定 engaged × sad、16 人、低层和样本；蓝色为 sad 低，橙色为 sad 高，空心→实心为 engaged 低→高。高层训练包含评价被试评分（transductive）。均为探索性候选，PS 与 CCGP 群体统计不能替代单人图的稳健性。</p>',
          '<p>上下图使用相同预设视角；坐标边界收紧、保持等比例。每个点是条件中心，不是单个 trial。epoch 0 为随机高层对照，不入选。分支层与最终融合 PoE 层分别标注。筛选后 P 值不可当作独立验证。</p>',
          '<p><a href="group_summary.csv">全部群体统计（原始 P / BH q）</a> · <a href="ranked_candidates.csv">逐人排名</a> · <a href="centroids_plot3d.csv">3D点坐标</a> · <a href="centroids_fullspace.csv">完整空间中心</a></p>',
          '<h2>候选预览</h2>']
    for r in shortlist.itertuples():
        p,c=ps.loc[r.candidate],cc.loc[r.candidate]
        verdict='PS及CCGP均通过两种检验与BH校正' if r.both_metrics_BH_significant else '仅原始P达标；两指标未全部通过BH校正'
        layer_label='原融合后验层' if r.layer=='poe_mu' else '监督专家分支层（非融合后验）'
        html.append(f'<h3>{r.subject} · {r.candidate}</h3><p>{layer_label}；{verdict}。</p><p>PS {r.low_PS:.3f} → {r.high_PS:.3f}; angle {r.low_angle:.1f}° → {r.high_angle:.1f}°; low depth {r.low_depth:.3f}. PS P={p.wilcoxon_p:.5g}, CCGP P={c.wilcoxon_p:.5g}; BH q={p.wilcoxon_q_BH:.5g}/{c.wilcoxon_q_BH:.5g}.</p>')
        for view in CAMERAS:
            rel=f'selected/{r.candidate}/{r.subject}_{view}.png'
            html.append(f'<a href="{rel}"><img loading="lazy" src="{rel}" title="{view}"></a>')
    html.append('<h2>Motor 已选结果：仅更新配色和留白</h2><p>sub-108 / seed1_mu_mean；使用已备份的低高层中心，数据与模型不变。</p>')
    for view in CAMERAS:
        rel=f'motor_palette_preview/sub-108_{view}.png'
        html.append(f'<a href="{rel}"><img loading="lazy" src="{rel}"></a>')
    html.append('<h2>全部轮次与层：16 人低高层对照</h2>')
    for candidate in candidates:
        html.append(f'<p><a href="all_subjects/{candidate}.png">{candidate} — 全部16人</a></p>')
    html.append('<h2>完整群体统计</h2>'+group.to_html(index=False,float_format=lambda v:f'{v:.5g}'))
    (OUT/'index.html').write_text('\n'.join(html),encoding='utf-8')
    manifest=json.loads((OUT/'manifest.json').read_text())
    report=['# Internal 高层轮次与层扫描','',
            '固定 engaged × sad、16 名有效被试、原始30影片集合及每人四格分配；低层逐元素与既有缓存相同。',
            '保持原架构、fold25/seed0、29名训练被试、7项评分训练目标、损失、学习率、batch、scaler和随机种子。训练0–26轮逐轮保存。',
            '低层是冻结预训练MAE；epoch0指随机初始化高层，只作对照，不作为训练后候选。',
            f"第13轮复现检查：{manifest['replay']}",
            '5个节点：obs_hidden、indiv_hidden（256维），obs_mu、indiv_mu、poe_mu（128维）。只有poe_mu是原融合高层；其余是同一监督模型分支表征候选，不能冒称原PoE后验。',
            '所有高层方案保持transductive限制；本扫描没有为每个新节点重训LOSO，因此旧LOSO显著性不能迁移到新方案。',
            'PS是主指标；CCGP使用原C=.01、20次平衡抽样、双向分类与训练集标准化。每种候选两指标均做16人Wilcoxon和精确sign-flip；另报告跨候选BH q。',
            '图中PS由平均条件中心计算，与群体逐人指标一致；3D精确保距，均匀缩放，收紧边界且box aspect与坐标范围同尺度。蓝橙配色，灰连接，空心起点/实心终点。',
            '全体候选中心保存NPZ、CSV及3D坐标CSV；训练checkpoint、优化器和随机状态、训练日志、特征、固定样本标签均保存。原结果不覆盖。','',
            '## 候选预览',shortlist[['candidate','subject','low_PS','high_PS','low_angle','high_angle','low_depth','both_metrics_BH_significant']].to_markdown(index=False,floatfmt='.3f')]
    pass
    import shutil
    shutil.copy2(__file__,OUT/'code'/Path(__file__).name)
    for filename in ('decode_cross_subject_residual_5fold.py','decode_narratives_shortvideo_poe.py'):
        shutil.copy2(CODE/filename,OUT/'code'/filename)
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name not in ('SHA256SUMS','run.log') and '__pycache__' not in str(p))
    (OUT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(OUT)}\n' for p in files))
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['train','evaluate','plot','all'])
    args=parser.parse_args()
    for name,func in [('train',train),('evaluate',evaluate),('plot',plot)]:
        if args.phase in (name,'all'):func()
