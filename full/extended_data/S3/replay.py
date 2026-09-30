
import sys
import os
from pathlib import Path
import json
import subprocess
import tempfile
import numpy as np
import pandas as pd
import h5py
import torch
from torch import nn
from scipy.spatial.distance import pdist,squareform
from sklearn.manifold import MDS
from numerics import transform_pca,geometry,normalize,classical,orient

MDS_PYTHON=sys.executable
SOURCE_ROOT=Path(__file__).resolve().parent
DS000210=Path(os.environ.get('NEURAL_DS000210', '/media/wsqlab/nas2/Dataset/ds000210'))


def motor(root,g,device,register):
    import models as m
    folder = root/'data_check_20260507/Exp_tyf_Results/Exp_Classification/result/BCIC/BT-ND+BrainOmni_within5fold/sub-009/fold_01'
    state = torch.load(register(folder/'prior.pt','model'),map_location='cpu',weights_only=False)
    assert not any(k.startswith('encoder.') for k in state)
    with np.load(register(g/'bcic_extended_models_20260917/sub-009_frozen_prior_tokens.npz','frozen_input')) as z:
        tokens = torch.from_numpy(z['tokens']).to(device)
        labels = z['labels']
    with np.load(register(g/'bcic_extended_models_20260917/44cf5bf9c335_layers.npz','saved_pca_and_selection')) as z:
        params = {k:z[k].copy() for k in ['reference_center','pca_mean','components']}
    split = json.loads(register(folder/'split.json','split').read_text())
    allids = np.array(split['test_idx'],dtype=int)
    ids = allids[np.isin(labels[allids],[0,1])]
    y = labels[ids].astype(int)
    assert len(y)==58 and not set(ids)&set(split['train_idx'])
    gauss = m.GaussHead(tokens.shape[-1],256).to(device).eval()
    head = m.ClsHead(256,n_classes=4,dropout=0).to(device).eval()
    gauss.load_state_dict({k.removeprefix('ghead.'):v for k,v in state.items() if k.startswith('ghead.')},strict=True)
    head.load_state_dict({k.removeprefix('cls_head.'):v for k,v in state.items() if k.startswith('cls_head.')},strict=True)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
        mu,_ = gauss(tokens[allids])
        hidden = head.net[:-1](mu.mean(1)).float().cpu().numpy()
    hidden = hidden[np.isin(labels[allids],[0,1])]
    with np.load(register(root/'jch-0909/eeg_cross_subject_geometry/source_data_features/BCIC_trial_features.npz','frozen_input')) as z:
        mask = z['subjects']=='sub-009'
        raw = z['observation'][mask].astype(float)
        np.testing.assert_array_equal(z['labels'][mask],labels)
    low = transform_pca(raw[ids],params,True)
    inputs = [normalize(low,y),normalize(geometry(hidden),y)]
    with tempfile.TemporaryDirectory(prefix='s3_fresh_mds_') as temp:
        inputfile=Path(temp)/'fresh_inputs.npz'
        resultfile=Path(temp)/'fresh_projection.npz'
        np.savez_compressed(inputfile,low=inputs[0],high=inputs[1],labels=y)
        helper=register(Path(__file__).with_name('legacy_mds.py'),'projection_code')
        subprocess.run([MDS_PYTHON,str(helper),str(inputfile),str(resultfile)],check=True)
        with np.load(resultfile) as result:
            qs=[q.copy() for q in result['coordinates']]
    return dict(domain='Motor',subject='sub-009',names=['Left hand','Right hand'],
                y=y,ids=ids,q=orient(qs,y),low_input=inputs[0],high_input=inputs[1],high_native=hidden)


def perception(root,g,device,register):
    from models import AnchoredPoE
    base=root/'lizhuo_exp/lizhuo_exp_NSD/test_cifti_modes/result/subject1_poe_mindeye2_1024'
    opt=root/'lizhuo_exp/lizhuo_exp_NSD/test_cifti_modes/result/subject1_poe_optimization'
    ck=torch.load(register(opt/'checkpoints/r2_anchored_cont_stage2_best.pt','model'),map_location='cpu',weights_only=False,mmap=True)
    assert ck['metadata']['variant']=='anchored_poe'
    fusion=AnchoredPoE().to(device).eval()
    fusion.load_state_dict(ck['fusion'],strict=True)
    with np.load(register(base/'cache/split_and_normalization.npz','split')) as z:
        allids=z['test_stim_ids'].copy()
    native=pd.read_csv(register(g/'mds_nsd_native80_20260916/NSD_native_labels_and_selection.csv','labels'))
    mapping={int(r.stim_id):json.loads(r.native_category_ids) for r in native.itertuples()}
    rows=np.array([i for i,sid in enumerate(allids) if mapping.get(int(sid)) in [[5],[24]]])
    ids=allids[rows]
    y=np.array([int(mapping[int(s)]==[24]) for s in ids])
    assert np.bincount(y).tolist()==[19,29]
    with np.load(register(g/'early_observation_20260917/replicate_quantification/NSD_sub-01_inputs.npz','saved_pca')) as z:
        params={k:z[k].copy() for k in ['reference_center','pca_mean','components']}
        np.testing.assert_array_equal(z['ids'],ids)
    raw=np.load(register(opt/'cache/cifti_test.npy','preprocessed_input'))
    modes=np.load(register(base/'cache/modes_test.npy','preprocessed_input'))
    chunks=[]

    with torch.inference_mode():
        for start in range(0,len(raw),128):
            a,b=[torch.as_tensor(x[start:start+128],dtype=torch.float32,device=device) for x in [raw,modes]]
            _,stats=fusion(a,b)
            chunks.append(stats['fused_mean'].cpu().numpy())
    high=np.concatenate(chunks)[rows]
    low=transform_pca(raw[rows],params,False)
    inputs=[]
    for x in [low,high]:
        x=geometry(x,False)
        inputs.append(x/np.sqrt(np.mean(np.sum(x*x,axis=1))))
    return dict(domain='Perception',subject='sub-01',names=['Airplane','Zebra'],
                y=y,ids=ids,q=[classical(x) for x in inputs],low_input=inputs[0],high_input=inputs[1],high_native=high)


def internal(root,g,device,register):
    from modeling import GaussianPoE
    artifacts=root/'lizhuo_exp/lizhuo_exp_zizhuan/artifacts/full55_joint_mae_subjectcentered_v1'
    source=register(artifacts/'features/full55_joint_mae_all_re_v1/poe/ds000210/sub-07.h5','frozen_input')
    with h5py.File(source) as f:
        obs=f['bci_obs'][:].astype(np.float32).mean(1)
        prior=f['bci_prior_indiv'][:].astype(np.float32).mean(1)
        labels=f['labels_objective'][:].reshape(-1).astype(int)
        allids=np.array([r.decode()+'|'+str(int(e)) for r,e in zip(f['trial_run'][:],f['trial_source_event_index'][:])])
    ckpath=register(g/'mds_preoutput_layer_20260916/checkpoints/ds000210/ours/sub-07/seed-05-re.pt','model')
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    assert list(ck['condition_names'])==['Past','Future','Other']
    assert 'sub-07' not in ck['provenance']['re_train_subjects']

    model=GaussianPoE(512,lambda d:nn.Linear(d,3),128,256,0).eval()
    model.load_state_dict(ck['model_state'],strict=True)
    def scale(x,k):
        s=ck[k]
        return torch.from_numpy(((x-np.array(s['mean'],np.float32))/np.array(s['std'],np.float32)).astype(np.float32))
    with torch.inference_mode():
        output=model(scale(obs,'x_scaler'),scale(prior,'indiv_scaler'))
    high_all=output['mu_poe'].numpy()
    take=np.flatnonzero(np.isin(labels,[0,2]))
    y=(labels[take]==2).astype(int)
    ids=allids[take]
    assert np.bincount(y).tolist()==[24,24]
    for run in sorted(set(s.split('|')[0] for s in ids)):
        events=register((DS000210/'sub-07/func')/f'sub-07_task-cuedSGT_run-{run}_events.tsv','raw_labels')
        table=pd.read_csv(events,sep='\t').set_index('TrialNumber')
        for sid,label in zip(ids,y):
            r,num=sid.split('|')
            if r==run:
                assert table.loc[int(num),'Condition']==['Past','Other'][label]
    patchck=torch.load(register(artifacts/'checkpoints/mae_joint_full55_re_v1/obs/best.pt','patch_encoder'),map_location='cpu',weights_only=False,mmap=True)
    meta=patchck['source_checkpoint_metadata']
    nl,nr=meta['n_lh'],meta['n_rh']
    with h5py.File(register((DS000210/'preprocess/stage2/sub-07.h5'),'preprocessed_fmri')) as f:
        raw=f['fmri_cift'][:,:nl+nr,:].astype(np.float32)
        np.testing.assert_array_equal(f['labels_objective'][:].reshape(-1),labels)
    raw=(raw-raw.mean(axis=(0,2),keepdims=True))/(raw.std(axis=(0,2),keepdims=True)+1e-8)
    n,_,nt=raw.shape
    groups,size=meta['n_groups_per_hemi'],meta['max_group_size']
    lh=raw[:,:nl].reshape(n,groups,nl//groups,nt)
    rh=raw[:,nl:].reshape(n,groups,nr//groups,nt)
    lh=np.pad(lh,((0,0),(0,0),(0,size-lh.shape[2]),(0,0)))
    rh=np.pad(rh,((0,0),(0,0),(0,size-rh.shape[2]),(0,0)))
    early=np.concatenate([lh,rh],axis=1).astype(float).mean(axis=(1,3))@patchck['encoder']['patch_embed.weight'].numpy().astype(float).T
    with np.load(register(g/'early_observation_20260917/replicate_quantification/Internal_sub-07_inputs.npz','saved_pca')) as z:
        params={k:z[k].copy() for k in ['reference_center','pca_mean','components']}
    low=transform_pca(early,params,True)[take]
    high=geometry(high_all[take])
    return dict(domain='Internal mentation',subject='sub-07',names=['Past','Other (ToM)'],
                y=y,ids=ids,q=[classical(x) for x in [low,high]],low_input=low,high_input=high,
                high_native=high_all[take],original_labels=labels[take])
