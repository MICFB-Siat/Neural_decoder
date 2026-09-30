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
EXP = BASE / '_source/data_check_20260507/Exp_tyf_Results'


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


def infer_fold(cls, root, fold, output, device, batch_size, config_override=None):
    cfg = read(root / 'args.json')
    if config_override is not None:cfg.update(config_override)
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
            eig_root = BASE / '_source/guoyi_exp/data/eigenmode_test/fs32k'
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



    parameters = cls._temp_scaled_fusion(z_obs, z_prior, y, cfg['n_classes'], seed=0)
    keys = ['temp_T_obs_A', 'temp_T_pri_A', 'temp_T_obs_B', 'temp_T_pri_B']
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
              'calibration_protocol': 'Fresh split-half cross-fit temperatures, partition initialization 0; opposite evaluation-half labels used.',
              'source': 'Fresh raw-signal and eigenmode forward passes; no saved logits or accuracy read as results.'}
    (output / 'inference.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    del prior, backbone, models, obs_enc, model, features
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description='Evaluate retained frozen checkpoints on real input signals, then plot fresh accuracy.')
    p.add_argument('--run',type=Path,required=True,help='Checkpoint run directory, relative to this workflow')
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--limit-folds',type=int);p.add_argument('--output',type=Path,default=BASE/'outputs/saved_inference')
    a=p.parse_args();root=(BASE/a.run).resolve()
    if not root.is_relative_to(BASE):raise ValueError('Checkpoint must be within this workflow')
    cfg=read(root/'args.json');cross=cfg['cv_mode']=='cross'
    folds=sorted(p.parent for p in root.glob('fold_*/split.json' if cross else 'sub-*/fold_*/split.json'))
    if a.limit_folds:folds=folds[:a.limit_folds]
    if not folds:raise FileNotFoundError('No saved fold with split.json: '+str(root))
    cls=load_module('classification_runtime',EXP/'Exp_Classification/code/classify_baseline_v2.py')
    rows=[]
    for fold in folds:
        result=infer_fold(cls,root,fold,a.output/fold.relative_to(root),torch.device(a.device),a.batch_size)
        rows.append(dict(fold=str(fold.relative_to(root)),accuracy=result['accuracy'],n_trials=result['n_trials']))
    with (a.output/'fresh_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(5,4));ax.bar(range(len(rows)),[x['accuracy'] for x in rows]);ax.set(xlabel='Retained fold',ylabel='Accuracy',ylim=(0,1))
    fig.tight_layout();fig.savefig(a.output/'accuracy.pdf');fig.savefig(a.output/'accuracy.png',dpi=200);plt.close(fig)
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
