

import os
from pathlib import Path
import torch
from torchvision.models import resnet50

def load_swav():
    model=resnet50(weights=None)
    path=os.environ.get('SWAV_WEIGHTS')
    if path:
        state=torch.load(Path(path),map_location='cpu',weights_only=True)
    else:
        state=torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar',map_location='cpu')
    state={k.replace('module.',''):v for k,v in state.items()}
    result=model.load_state_dict(state,strict=False)
    missing=[k for k in result.missing_keys if k not in ('fc.weight','fc.bias')]
    if missing:raise RuntimeError('Missing SwAV feature weights: '+repr(missing))
    return model
