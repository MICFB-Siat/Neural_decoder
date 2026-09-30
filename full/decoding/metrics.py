import csv
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Retrieval(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(1664), nn.Linear(1664, 1664),
                                     nn.GELU(), nn.Dropout(.1), nn.Linear(1664, dim))

    def forward(self, x):
        return self.network(x)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def rsa(data, weights, device):
    state = {k: v.to(device).float() for k, v in load(weights / 'fusion.pt')['model'].items()}
    with np.load(data / 'inputs.npz') as arrays:
        x = {k: torch.as_tensor(arrays[k], device=device).float() for k in ['cifti', 'modes']}
    means, precision = {}, {}
    for k, value in x.items():
        def linear(suffix):
            name = f'fusion.{k}_{suffix}'
            return F.linear(value, state[name + '.weight'], state[name + '.bias'])
        means[k] = linear('mean')
        precision[k] = torch.exp(-linear('logvar').clamp(-4, 4))
    z = sum(means[k] * precision[k] for k in x) / (1 + sum(precision.values()))
    z = F.normalize(z - z.mean(1, keepdim=True), dim=1)
    corr = (z @ z.T).cpu().numpy()
    rdm = 1 - np.clip((corr + corr.T) / 2, -1, 1).astype(np.float64)
    target = np.load(data.parent / 'data/clip_rdm.npy')
    if target.shape != rdm.shape:
        raise ValueError('CLIP and neural RDM shapes differ')
    i = np.triu_indices(len(rdm), 1)
    return {'RSA_Pearson_r': float(np.corrcoef(rdm[i], target[i])[0, 1])}


def retrieve(data, weights, device, examples=False):
    saved = load(weights / 'global_retrieval_head_best.pt')
    model = Retrieval(saved['output_dim']).to(device).eval()
    model.load_state_dict(saved['model'], strict=True)
    x = torch.as_tensor(np.load(data / 'brain_direct_mean_test.npy'), device=device).float()
    y = torch.as_tensor(np.load(data / 'global_target_test.npy'), device=device).float()
    scores = (F.normalize(model(x), dim=-1) @ F.normalize(y, dim=-1).T).cpu().numpy()
    candidates = np.load(data / 'candidate_indices_100way_seeds_42_51.npy')
    if examples:
        return [{'query_index': q, 'top1_index': int(candidates[0, q, scores[q, candidates[0, q]].argmax()])}
                for q in [984, 316, 46, 809, 474]]
    query = np.arange(len(scores))
    selected = scores[query[None, :, None], candidates]
    ranks = 1 + (selected > np.diag(scores)[None, :, None]).sum(-1)
    return {f'Top-{k}_percent': float((ranks <= k).mean() * 100) for k in [1, 5, 10]}


def features(data, weights, out, device, limit):
    from visual import BrainNetwork, AnchoredPoE
    if device.startswith('cuda') and torch.cuda.get_device_capability(device)[0] < 8:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    saved = load(weights / 'brain_final.pt')
    fusion = AnchoredPoE().to(device).eval()
    fusion.load_state_dict(saved['fusion'], strict=True)
    model = BrainNetwork(h=1024, in_dim=1024, seq_len=1, clip_size=1664,
                         out_dim=1664 * 256, n_blocks=4, blurry_recon=True).to(device).eval()
    model.load_state_dict(saved['backbone'], strict=True)
    x = torch.as_tensor(np.load(data / 'cifti_test.npy')[:limit], device=device).float()
    y = torch.as_tensor(np.load(data / 'modes_test.npy')[:limit], device=device).float()
    with torch.autocast('cuda', dtype=torch.float16, enabled=device.startswith('cuda')):
        z, _ = fusion(x, y)
        condition, projected, low = model(z)
    arrays = dict(condition=condition.float().cpu().numpy(), projected=projected.float().cpu().numpy(),
                  lowlevel=low[0].float().cpu().numpy())
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Non-finite features')
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'features.npz', **arrays)
    return {'samples': len(x)}


def evaluate(panel, data, root, out, subjects, device, limit):
    torch.set_num_threads(4)
    subjects = subjects or (['sub-01'] if panel in ['G', 'I'] else [f'sub-{s:02d}' for s in range(1, 9)])
    if panel == 'I' and subjects != ['sub-01']:
        raise ValueError('Panel I examples are from sub-01')
    if panel in ['E', 'F', 'I'] and limit is not None:
        raise ValueError('--limit is only supported for language and G features')
    rows = []
    with torch.inference_mode():
        for subject in subjects:
            weights = root / panel / 'weights' / subject
            if panel == 'E':
                result = [rsa(data / subject, weights, device)]
            elif panel == 'F':
                result = [retrieve(data / subject, weights, device)]
            elif panel == 'I':
                result = retrieve(data, weights, device, examples=True)
            else:
                result = [features(data / subject, weights, out / subject, device, limit or 1)]
            rows.extend({'subject': subject, **r} for r in result)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    render_metrics(panel, rows, data, out)


def render_metrics(panel, rows, data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if panel in ['E','F']:
        keys=['RSA_Pearson_r'] if panel=='E' else [f'Top-{k}_percent' for k in [1,5,10]]
        fig,ax=plt.subplots(figsize=(5,4));ax.boxplot([[r[k] for r in rows] for k in keys],labels=keys)
        ax.set_ylabel('Pearson RSA' if panel=='E' else 'Retrieval accuracy (%)')
    elif panel=='I':
        image_file=data/'images_test.npy'
        if not image_file.is_file():
            raise FileNotFoundError('Fresh retrieval metrics were saved; image panel I additionally requires dataset images_test.npy')
        images=np.load(image_file,mmap_mode='r');fig,axes=plt.subplots(2,len(rows),figsize=(3*len(rows),6),squeeze=False)
        for j,row in enumerate(rows):
            for i,key in enumerate(['query_index','top1_index']):
                image=images[row[key]]
                if image.shape[0]==3:image=image.transpose(1,2,0)
                axes[i,j].imshow(image);axes[i,j].set_title(f'{key}: {row[key]}');axes[i,j].axis('off')
    else:return
    fig.tight_layout();fig.savefig(output/(panel+'.pdf'));fig.savefig(output/(panel+'.png'),dpi=200);plt.close(fig)
