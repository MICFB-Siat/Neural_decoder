from pathlib import Path
import argparse,csv,json,sys,importlib.util
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'C/code'))
def resolve(value):
    p=Path(value);return p if p.is_absolute() else BASE/p
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

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
    from scipy.special import softmax
    prediction = (softmax(z_obs,axis=1) + softmax(z_prior,axis=1)).argmax(1)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'predictions.npz', y=y, prediction=prediction,
                        z_obs=z_obs, z_pri=z_prior, subject=np.array(identities), trial=np.array(indices))
    np.savetxt(output / 'confusion_counts.csv', confusion_matrix(y, prediction, labels=np.arange(cfg['n_classes'])),
               delimiter=',', fmt='%d')
    result = {'accuracy': float(np.mean(prediction == y)), 'n_trials': len(y), 'subjects': subjects,
              'checkpoint': str(fold), 'input_directory': str(h5_dir),
              'source': 'Fresh raw-signal and eigenmode forward passes; no saved logits or accuracy read as results.'}
    (output / 'inference.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    del prior, backbone, models, obs_enc, model, features
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--subjects',nargs='+');p.add_argument('--limit-folds',type=int);p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=32);p.add_argument('--data',type=Path);a=p.parse_args()
    torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=True)
    cls=load_module('meg_runtime',BASE/'C/code/classify_baseline_v2.py');root=BASE/'C/weights';rows=[]
    if a.data:
        original=read
        globals()['read']=lambda path: dict(original(path),h5_dir=str(a.data)) if path.name=='args.json' else original(path)
    for subject in sorted(p for p in root.glob('sub-*') if p.is_dir()):
        if a.subjects and subject.name not in a.subjects:continue
        folds=sorted(subject.glob('fold_*'))
        if a.limit_folds:folds=folds[:a.limit_folds]
        for fold in folds:
            result=infer_fold(cls,root,fold,a.output/subject.name/fold.name,torch.device(a.device),a.batch_size)
            rows.append(dict(subject=subject.name,fold=fold.name,accuracy=result['accuracy'],n_trials=result['n_trials']))
            print(rows[-1],flush=True)
    frame=pd.DataFrame(rows)
    if frame.empty:raise ValueError('No inference completed')
    frame.to_csv(a.output/'fold_metrics.csv',index=False);summary=frame.groupby('subject').accuracy.mean();summary.to_csv(a.output/'subject_metrics.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots();ax.boxplot(summary*100);ax.set(ylabel='Accuracy (%)',xticks=[1],xticklabels=['Ours']);fig.savefig(a.output/'C.pdf');fig.savefig(a.output/'C.png',dpi=200);plt.close(fig)
if __name__=='__main__':main()
