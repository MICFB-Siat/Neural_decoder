import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MPLBACKEND','Agg')
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'config.json').read_text())
INPUTS=ROOT/CONFIG['inputs']
WEIGHTS=ROOT/CONFIG['weights']
TEMPLATES=ROOT/CONFIG['templates']

def output(panel):
    path=ROOT/panel/'outputs'
    path.mkdir(parents=True,exist_ok=True)
    return path

def device():
    import torch
    value=os.environ.get('FIG2_DEVICE',CONFIG['device'])
    if value.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Set device to cpu in config.json or choose an available CUDA runtime.')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    return torch.device(value)

def save_figure(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),dpi=300,bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),dpi=300,bbox_inches='tight')

def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
