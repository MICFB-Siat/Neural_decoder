from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
from resources import path as resource_path
INPUT = BASE.parent / 'inputs'
RECORDS = BASE
DAYS = [8, 14, 42, 43, 204, 208]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit-runs', type=int)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', type=Path, default=BASE / 'outputs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    calibrations = json.loads((BASE/'calibration.json').read_text())
    checkpoint_rows = json.loads((BASE/'fixed_checkpoints.json').read_text())['rows']
    checkpoints = {int(r['calibration_key']): r['checkpoint'] for r in checkpoint_rows}
    runs = sorted(checkpoints)
    if args.limit_runs:
        runs = runs[:args.limit_runs]
    module = load_module('ecog_inference_runtime', BASE.parent / 'algorithms/cross_session/dual_expert_poe_exp12_calib20.py')
    with h5py.File(resource_path('ecog_h5'), 'r') as data:
        raw = data['eeg_raw'][:].astype(np.float32)
        labels = data['labels'][:].astype(np.int64)
        sessions = data['session'][:].astype(np.int64)
    signal = module.S.zscore(module.S.resample_to(module.S.TARGET_SR, raw, 256))
    prior = np.load(INPUT / 'prior/p_prior.npy')
    if not np.array_equal(labels, np.load(INPUT / 'prior/labels.npy')):
        raise RuntimeError('Prior prediction labels do not match raw input order.')
    if not np.array_equal(sessions, np.load(INPUT / 'prior/session.npy')):
        raise RuntimeError('Prior prediction sessions do not match raw input order.')
    device = torch.device(args.device)
    model = module.S.SPaRCNet(in_channels=signal.shape[1], num_classes=6, growth_rate=32,
                             block_config=(4,4,4,4,4,4,4), num_init_features=64,
                             drop_rate=.2, drop_fc=.5, batch_norm=True, conv_bias=False).to(device)
    rows, provenance = [], []
    for run in runs:
        print('INFER run', run, flush=True)
        parameters = calibrations[str(run)]
        checkpoint = BASE / checkpoints[run]
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['state_dict'], strict=True)
        days = sorted(np.unique(sessions).tolist())
        training, testing = module.build_split(sessions, labels, days[0], days[1:], parameters['calib_frac'], run)
        if not np.array_equal(training, np.array(state['train_pool'])):
            raise RuntimeError('Reconstructed data split differs from checkpoint training indices.')
        indices = np.concatenate([testing[day] for day in DAYS])
        observed = module.softmax_probs(model, signal[indices], device, args.batch_size)
        prediction = module.fuse_pred(observed, prior[indices], parameters['a_obs'], parameters['a_prior'], parameters['gamma'])
        np.savez_compressed(args.output / f'run_{run}_predictions.npz', trial=indices, y=labels[indices],
                            prediction=prediction, observation_probability=observed, prior_probability=prior[indices])
        for day in DAYS:
            mask = sessions[indices] == day
            rows.append(dict(run=run, day=day, method='Ours', accuracy=float(np.mean(prediction[mask] == labels[indices][mask]))))
            rows.append(dict(run=run, day=day, method='SPaRCNet', accuracy=float(np.mean(observed[mask].argmax(1) == labels[indices][mask]))))
        provenance.append(dict(run=run, checkpoint=str(checkpoint), a_obs=parameters['a_obs'],
                               a_prior=parameters['a_prior'], gamma=parameters['gamma']))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / 'fresh_metrics.csv', index=False)
    summary = frame.groupby(['day','method']).accuracy.agg(['mean','std','count']).reset_index()
    summary.to_csv(args.output / 'fresh_summary.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':9, 'svg.fonttype':'none', 'pdf.fonttype':42})
    figure, axis = plt.subplots(figsize=(7.1,4.2))
    for method, color in [('SPaRCNet','#4c78a8'),('Ours','#e45756')]:
        values = summary[summary.method.eq(method)]
        axis.errorbar(np.arange(len(DAYS)), values['mean']*100, yerr=values['std'].fillna(0)*100,
                      marker='o', capsize=3, lw=1.7, label=method, color=color)
    axis.set(xlabel='Session (days from D0)', ylabel='Decoding accuracy (%)',
             xticks=np.arange(len(DAYS)), xticklabels=[str(day) for day in DAYS], ylim=(20,100))
    axis.spines[['top','right']].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.output / 'Fig4L.png', dpi=300)
    figure.savefig(args.output / 'Fig4L.pdf')
    figure.savefig(args.output / 'Fig4L.svg')
    plt.close(figure)
    report = {'mode':'fresh observation-weight inference with retained trial-level prior probabilities',
              'raw_input':str(resource_path('ecog_h5')), 'runs':provenance,
              'display':'Sessions are equally spaced; tick labels retain actual elapsed days. Error bars are sample SD across runs.',
              'prior_input':str(INPUT/'prior/p_prior.npy'),
              'selection':'Retained 20-run membership; selection score columns are not used as plotted values.'}
    (args.output/'inference.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
