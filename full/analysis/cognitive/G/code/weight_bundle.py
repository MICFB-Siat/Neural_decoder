pass

import hashlib
import io
from pathlib import Path

import torch

DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / 'weights/fig6g_bundle.pt'


def load_bundle(path=DEFAULT_BUNDLE):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing Fig. 6G weight bundle: {path}. Run '
            'python full/download_assets.py --scope '
            'analysis/cognitive/G/weights/fig6g_bundle.pt'
        )
    bundle = torch.load(path, map_location='cpu', weights_only=True)
    if bundle.get('format') != 'fig6g.fixed-replay.v1':
        raise ValueError('Unsupported Fig. 6G weight bundle format')
    referenced = {p for row in bundle['rows'] for p in row['checkpoints']}
    if referenced != set(bundle['checkpoints']) or referenced != set(bundle['sha256']):
        raise ValueError('Fig. 6G bundle membership mismatch')
    for name, data in bundle['checkpoints'].items():
        if hashlib.sha256(data.numpy().tobytes()).hexdigest() != bundle['sha256'][name]:
            raise ValueError(f'Fig. 6G checkpoint checksum mismatch: {name}')
    return bundle


def load_checkpoint(bundle, name):


    return torch.load(io.BytesIO(bundle['checkpoints'][name].numpy().tobytes()),
                      map_location='cpu', weights_only=False)
