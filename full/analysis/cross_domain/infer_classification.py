from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

BASE = Path(__file__).resolve().parent
EXP = BASE / 'experiment'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


from resources import resolve


def restore(model, state, allowed_missing):
    missing, unexpected = model.load_state_dict(state, strict=False)
    invalid = [key for key in missing if not any(key.startswith(p) for p in allowed_missing)]
    if invalid or unexpected:
        raise RuntimeError({'missing': invalid, 'unexpected': unexpected})


def infer_fold(cls, root, fold, output, device, batch_size):
    cfg = read(root / 'args.json')
    if cfg['train_mode'] != 'frozen':
        raise RuntimeError('This replay supports frozen-backbone checkpoints only.')
    split = read(fold / 'split.json')
    cross = cfg['cv_mode'] == 'cross'
    subjects = split['val_subjects'] if cross else [fold.parent.name]
    prior = cls.load_prior_from_ckpt(str(resolve(cfg['encoder_ckpt'])), device, False,
                                    n_samples_override=cfg.get('n_samples'))
    backbone, lm_dim = cls.load_brainomni_tiny(device, False)
    h5_dir = resolve(cfg['h5_dir'])
    if cfg.get('d_path'):
        d = np.load(resolve(cfg['d_path']))
        if cfg.get('lam_cortex_path'):
            lam = np.load(resolve(cfg['lam_cortex_path']))
        else:
            eig_root = BASE / 'templates'
            left = np.load(eig_root / 'fsLR_32k_lh_eval_1024.npy')
            right = np.load(eig_root / 'fsLR_32k_rh_eval_1024.npy')
            lam = np.concatenate([left[:(d.shape[1] + 1)//2], right[:d.shape[1]//2]])
        eigval = torch.from_numpy(cls.compute_eigval_from_D(d, lam_cortex=lam, ratio=cfg['ratio'])[0])
    else:
        eigval = None
    n_tok = prior.expected_n_samples // cls.PATCH_SIZE
    dim = prior.d_model_dim
    obs_enc = cls.ObsEncoder(backbone, lm_dim, n_tok, dim, backbone_trainable=False).to(device)
    models = []
    for encoder, is_obs, filename in [(obs_enc, True, 'obs.pt'), (prior, False, 'prior.pt')]:
        model = cls.ExpertModel(encoder, cls.GaussHead(dim, cfg['lat_dim']).to(device),
                               cls.ClsHead(cfg['lat_dim'], n_classes=cfg['n_classes'],
                                           dropout=cfg['dropout']).to(device),
                               is_obs=is_obs, encoder_trainable=False).to(device)
        state = torch.load(fold / filename, map_location=device, weights_only=False)
        if any(k.startswith('encoder.backbone.') for k in state) or (not is_obs and any(k.startswith('encoder.') for k in state)):
            raise RuntimeError('Checkpoint includes backbone updates; frozen-cache replay is not applicable.')
        restore(model, state, ['encoder.backbone.'] if is_obs else ['encoder.'])
        model.eval()
        models.append(model)
    logits = [[], []]
    labels, identities, indices = [], [], []
    for subject in subjects:
        item = cls.load_subject(str(h5_dir), subject, prior.expected_n_samples,
                                modality=cfg['modality'], label_key=cfg['label_key'],
                                label_offset=cfg.get('label_offset', 0), n_classes=cfg['n_classes'],
                                override_eigval=eigval)
        if item is None:
            raise FileNotFoundError(h5_dir / (subject + '.h5'))
        idx = np.arange(len(item['labels'])) if cross else np.asarray(split['test_idx'])
        ev = item['eigval'].to(device)
        features = cls.precompute_features(item, idx, backbone, prior, n_tok, ev, device, batch_size)
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            for start in range(0, len(idx), batch_size):
                for branch, model in enumerate(models):
                    result, _, _ = model.cached_forward(features[branch][start:start+batch_size].to(device))
                    logits[branch].append(result.float().cpu().numpy())
        labels.append(features[2].numpy())
        identities.extend([subject] * len(idx))
        indices.extend(idx.tolist())
    y = np.concatenate(labels)
    z_obs, z_prior = [np.concatenate(parts) for parts in logits]
    parameters = read(fold / 'metrics.json')
    keys = ['temp_T_obs_A', 'temp_T_pri_A', 'temp_T_obs_B', 'temp_T_pri_B']
    if not all(key in parameters for key in keys):
        raise RuntimeError('Saved temperature parameters are missing; no retraining or score fallback is allowed.')
    first, second = cls._stratified_half_split(y, cfg['n_classes'], 0)
    prediction = np.empty(len(y), dtype=np.int64)
    for ids, suffix in [(second, 'A'), (first, 'B')]:
        prediction[ids] = (z_obs[ids] / parameters['temp_T_obs_' + suffix] +
                           z_prior[ids] / parameters['temp_T_pri_' + suffix]).argmax(1)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'predictions.npz', y=y, prediction=prediction,
                        z_obs=z_obs, z_pri=z_prior, subject=np.array(identities), trial=np.array(indices))
    np.savetxt(output / 'confusion_counts.csv', confusion_matrix(y, prediction, labels=np.arange(cfg['n_classes'])),
               delimiter=',', fmt='%d')
    result = {'accuracy': float(np.mean(prediction == y)), 'n_trials': len(y), 'subjects': subjects,
              'checkpoint': str(fold), 'input_directory': str(h5_dir),
              'calibration': {key: parameters[key] for key in keys},
              'calibration_protocol': 'Retained split-half cross-fit temperatures, partition initialization 0.',
              'source': 'Fresh raw-signal and eigenmode forward passes; no saved logits or accuracy read as results.'}
    (output / 'inference.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    del prior, backbone, models, obs_enc, model, features
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', choices=['C', 'D', 'E'], required=True)
    parser.add_argument('--datasets', nargs='+', default=['BCIC', 'MOTOR', 'FACED', 'SEEDV'])
    parser.add_argument('--cross-protocol',choices=['loso','cross5'],default='loso',help='Explicit protocol; cross5 is not a substitute for LOSO.')
    parser.add_argument('--subjects', nargs='+')
    parser.add_argument('--limit-folds', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or BASE / args.panel / 'outputs'
    output.mkdir(parents=True, exist_ok=True)
    cls = load_module('classification_runtime', EXP / 'Exp_Classification/code/classify_baseline_v2.py')
    rows, coverage = [], []
    for dataset in args.datasets:
        parent = EXP / 'Exp_Classification/result' / dataset
        if args.panel in ('C', 'D'):
            roots = [parent / 'BT-ND+BrainOmni_within5fold']
        else:
            dirname='BT-ND+BrainOmni_LOSO_rerun_20260718' if args.cross_protocol=='loso' else 'BT-ND+BrainOmni_cross5fold'
            roots = sorted(p for p in (parent / dirname).glob('*')
                           if any(p.glob('fold_*/obs.pt')))
        for root in roots:
            pattern = 'sub-*/fold_*/split.json' if args.panel in ('C', 'D') else 'fold_*/split.json'
            folds = [p.parent for p in sorted(root.glob(pattern))]
            if args.subjects:
                folds = [p for p in folds if (p.parent.name in args.subjects if args.panel in ('C', 'D')
                         else any(s in args.subjects for s in read(p/'split.json')['val_subjects']))]
            if args.limit_folds:
                folds = folds[:args.limit_folds]
            for fold in folds:
                relative = fold.relative_to(root)
                if not all((fold / name).is_file() for name in ['obs.pt', 'prior.pt', 'metrics.json']):
                    coverage.append({'checkpoint': str(fold), 'status': 'missing required checkpoint or calibration'})
                    continue
                print('INFER', dataset, relative, flush=True)
                result = infer_fold(cls, root, fold, output / dataset / root.name / relative,
                                    torch.device(args.device), args.batch_size)
                rows.append({'panel': '3'+args.panel, 'dataset': {'MOTOR':'Motor Imagery', 'SEEDV':'SEED-V'}.get(dataset,dataset),
                             'participant': ','.join(result['subjects']), 'accuracy': result['accuracy'],
                             'checkpoint': str(fold), 'n_trials': result['n_trials'], 'protocol':args.cross_protocol if args.panel=='E' else 'within5'})
                print('RESULT', rows[-1], flush=True)
        if not any(row['dataset'] == {'MOTOR':'Motor Imagery', 'SEEDV':'SEED-V'}.get(dataset,dataset) for row in rows):
            coverage.append({'dataset': dataset, 'status': 'no completed inference'})
    (output / 'coverage.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')
    if not rows:
        raise RuntimeError('No inference completed.')
    with (output / 'fresh_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault((row['dataset'], row['participant']), []).append(row['accuracy'])
    plot_rows = [{'panel':'3'+args.panel, 'dataset':name, 'participant':person, 'accuracy':float(np.mean(values))}
                 for (name, person), values in grouped.items()]
    if args.panel == 'D':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for dataset in args.datasets:
            counts = list((output/dataset).rglob('confusion_counts.csv'))
            if not counts: continue
            matrix = sum(np.loadtxt(f,delimiter=',',ndmin=2) for f in counts)
            norm = np.divide(matrix,matrix.sum(1,keepdims=True),out=np.zeros_like(matrix),where=matrix.sum(1,keepdims=True)>0)
            np.savetxt(output/(dataset+'_confusion_counts.csv'),matrix,delimiter=',',fmt='%d')
            fig,ax=plt.subplots();im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues');fig.colorbar(im,ax=ax)
            ax.set(xlabel='Predicted class',ylabel='True class',title=dataset)
            fig.savefig(output/(dataset+'_D.pdf'));fig.savefig(output/(dataset+'_D.png'),dpi=200);plt.close(fig)
    else:
        plotter = load_module('classification_plot', BASE / args.panel / 'source/fig3_classification.py')
        plotter.draw(plot_rows, output, panels=('3'+args.panel,))
    print('Fresh inference output:', output, flush=True)


if __name__ == '__main__':
    main()
