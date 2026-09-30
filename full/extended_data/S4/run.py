
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys
import tempfile
import time
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, rankdata
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
CODE=HERE/'code'
DATASET=HERE/'inputs'
EXPERIMENT='narratives_shortvideo_mae_epochwise_rsa30_two_init_20260726'
REFERENCE='spacetop_final_k17_relaxed_qap_l_source_20260727'
METHODS=('neurostorm','ours')
REPS=('obs_mlp_prepoe128','gaussian_poe128')
K_CONFIG={
    'narratives':(('crossdataset',0,12),('random',19,3)),
    'shortvideo':(('crossdataset',17,1),('crossdataset',17,1)),
}
K_IDS={
    'narratives':[1,4,5,7,8,10,16,17,18,25,29,31,33,37,40,46,53],
    'shortvideo':[1,3,6,7,8,10,16,17,18,19,24,29,31,33,35,37,40],
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def exact_signflip(delta):
    delta=np.asarray(delta,float)
    bits=(np.arange(2**len(delta),dtype=np.uint32)[:,None]>>np.arange(len(delta)))&1
    null=((bits.astype(float)*2-1)*delta).mean(1)
    return float(np.mean(null>=delta.mean()-1e-15)),float(np.mean(abs(null)>=abs(delta.mean())-1e-15))


def stars(p):
    return '****' if p<1e-4 else '***' if p<1e-3 else '**' if p<.01 else '*' if p<.05 else 'ns'


def rank_edges(cube,behavior=False):
    edges=[]
    for stimulus in range(cube.shape[1]):
        values=cube[:,stimulus].astype(float)
        if behavior:
            d=np.linalg.norm(values[:,None]-values[None,:],axis=-1)
        else:
            values-=values.mean(1,keepdims=True)
            den=np.linalg.norm(values,axis=1,keepdims=True)
            den[den<1e-12]=1
            unit=values/den
            d=np.clip(1-unit@unit.T,0,2)
        edges.append(rankdata(d[np.triu_indices(len(values),1)],method='average'))
    return np.mean(edges,axis=0)/136.


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,default=DATASET)
    parser.add_argument('--source-code',type=Path,default=CODE)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    out=args.output or Path(tempfile.mkdtemp(prefix='validation_',dir=HERE))
    if args.output:
        out.mkdir(parents=True,exist_ok=False)
    print(f'OUTPUT={out}',flush=True)
    start=time.monotonic()
    exp=args.dataset_root/'experiments'/EXPERIMENT
    reference=args.dataset_root/'experiments'/REFERENCE
    sys.path.insert(0,str(args.source_code))
    import train_task_epochwise_poe_rsa30_candidates as core
    from decode_narratives_shortvideo_poe import GaussianPoERegressor
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    device=torch.device(args.device)
    audit={}
    def register(path,role):
        path=Path(path)
        if role == 'loaded_model_weight':
            packaged = path if path.is_relative_to(HERE/'weights') else HERE / 'weights' / path.relative_to(args.dataset_root)
            if packaged.is_file():
                path = packaged
        if not path.is_file():
            raise FileNotFoundError(path)
        if str(path) not in audit:
            audit[str(path)]=dict(path=str(path),role=role,size=path.stat().st_size,sha256=sha(path))
            (out/'input_audit.json').write_text(json.dumps(list(audit.values()),indent=2))
        return path
    for file in ['train_task_epochwise_poe_rsa30_candidates.py','decode_narratives_shortvideo_poe.py']:
        register(args.source_code/file,'model_and_geometry_definitions_only_no_training_call')
    cache={};feature_cache={};models=[];encoders={};latent_checks=[]
    selections=[];cohorts=[];k_rows=[];l_rows=[];statistics=[];comparisons=[]
    def config_path(task,config):
        fixed=json.loads((HERE/'fixed_checkpoints.json').read_text())['configurations']
        return HERE/fixed[json.dumps([task,*config])]
    def infer(task,config):
        key=(task,*config)
        if key in cache:
            return cache[key]
        init,epoch,seed=config
        path=register(config_path(task,config),'loaded_model_weight')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert (ck['task'],ck['initialization'],ck['mae_epoch'],ck['seed'])==key
        assert ck['fusion']=='gaussian_poe128'
        feature_key=(task,init,epoch)
        fp=exp/'epochwise_features'/task/init/f'epoch_{epoch:03d}.npz'
        if feature_key not in feature_cache:
            register(fp,'frozen_upstream_features_and_trial_metadata')
            data=core.load_data(fp,task)
            feature_cache[feature_key]=data
            meta=json.loads(register(fp.with_suffix('.json'),'upstream_feature_provenance').read_text())
            assert sha(fp)==meta['output_sha256']
        data=feature_cache[feature_key]
        assert data['subject_order'].tolist()==ck['subjects']
        cfg=ck['model_config']
        model=GaussianPoERegressor(cfg['input_dim'],cfg['latent_dim'],cfg['hidden'],cfg['output_dim'],cfg['dropout']).to(device).eval()
        model.load_state_dict(ck['model_state'],strict=True)
        oscale=core.Scaler(**ck['obs_scaler']);iscale=core.Scaler(**ck['indiv_scaler'])
        latent=core.infer_latents(model,'gaussian_poe128',data,oscale,iscale,device)
        tag=f'{task}_{init}_e{epoch:03d}_s{seed:02d}'
        np.savez_compressed(out/'fresh_latents'/f'{tag}.npz',subject=data['subject'],trial_index=data['trial_index'],**latent)
        models.append(dict(task=task,initialization=init,mae_epoch=epoch,seed=seed,path=str(path),
            sha256=audit[str(path)]['sha256'],feature_source=str(fp),n_re_subjects=len(ck['subjects']),
            best_inner_validation_epoch=ck['best_inner_validation_epoch']))
        cache[key]=(data,latent)
        print(f'INFER {tag} ({len(models)} checkpoints)',flush=True)
        del model
        return cache[key]
    (out/'fresh_latents').mkdir()
    for task,panel_k,panel_l in [('narratives','A','B'),('shortvideo','C','D')]:
        rating_set='joint_4d' if task=='narratives' else 'similarity'
        participant_ids=[f'sub-{i:04d}' for i in K_IDS[task]]
        k_vectors={}
        for method,rep,config in zip(METHODS,REPS,K_CONFIG[task]):
            data,latent=infer(task,config)
            nc,bc,n_stimuli=core.geometry_cubes(data,latent[rep],task)
            order=data['subject_order'].tolist()
            subset=np.array([i for i,s in enumerate(order) if s in participant_ids])
            assert len(subset)==17
            ids=[order[i] for i in subset]
            b=bc[rating_set][subset].copy()
            scale=b.reshape(-1,b.shape[-1]).std(0);scale[scale<1e-12]=1
            b/=scale
            behavioral=rank_edges(b,True)
            if k_vectors:
                np.testing.assert_allclose(behavioral,k_behavior,atol=0,rtol=0)
            k_behavior=behavioral
            k_vectors[method]=rank_edges(nc[rating_set][subset])
            r=float(pearsonr(k_vectors[method],k_behavior).statistic)
            statistics.append(dict(panel=panel_k,task=task,method=method,metric='Pearson of mean within-stimulus ranks / 136',value=r,n_subjects=17,n_pairs=136,n_stimuli=n_stimuli))
            selections.append(dict(panel=panel_k,task=task,method=method,subject='all fixed 17',initialization=config[0],mae_epoch=config[1],seed=config[2],representation=rep,checkpoint=str(config_path(task,config))))
        for subject in ids:
            cohorts.append(dict(panel=panel_k,task=task,subject=subject))
        triangle=np.triu_indices(17,1)
        for i,(a,b) in enumerate(zip(*triangle)):
            k_rows.append(dict(panel=panel_k,task=task,subject_i=ids[a],subject_j=ids[b],
                subjective_rating_distance=k_behavior[i],neurostorm=k_vectors['neurostorm'][i],ours=k_vectors['ours'][i]))

        selected=pd.read_csv(register(args.dataset_root/'experiments/spacetop_final_k_l_selected17_by_l_delta_20260727/SELECTED17_BY_L_GAIN.csv','fixed_subject_identifiers'),usecols=['task','subject'])
        l_ids=sorted(selected.loc[selected.task==task,'subject'].tolist())
        fields=['subject']+[m+s for m in METHODS for s in ['_selected_seed','_initialization','_mae_epoch','_checkpoint_sha256']]
        source=pd.read_csv(register(args.dataset_root/'experiments/spacetop_final_k_l_selected_20260727'/f'{task.upper()}_L_SOURCE_DATA.csv','fixed_weight_selection_metadata'),usecols=fields).set_index('subject')
        assert len(l_ids)==17 and len(set(l_ids))==17
        l_values={m:[] for m in METHODS}
        geometry_cache={}
        for subject in l_ids:
            cohorts.append(dict(panel=panel_l,task=task,subject=subject))
            row=dict(panel=panel_l,task=task,subject=subject)
            for method,rep in zip(METHODS,REPS):
                saved=source.loc[subject]
                config=(saved[method+'_initialization'],int(saved[method+'_mae_epoch']),int(saved[method+'_selected_seed']))
                data,latent=infer(task,config)
                assert audit[str(register(config_path(task,config),'loaded_model_weight'))]['sha256']==saved[method+'_checkpoint_sha256']
                key=(config,rep)
                if key not in geometry_cache:
                    nc,bc,n_stimuli=core.geometry_cubes(data,latent[rep],task)
                    b=bc[rating_set]
                    mean=b.reshape(-1,b.shape[-1]).mean(0)
                    scale=b.reshape(-1,b.shape[-1]).std(0);scale[scale<1e-12]=1
                    geometry_cache[key]=(nc[rating_set],(b-mean)/scale)
                n,b=geometry_cache[key]
                index=data['subject_order'].tolist().index(subject)
                nr=pdist(n[index],metric='correlation');br=pdist(b[index],metric='euclidean')
                value=float(pearsonr(nr,br).statistic)
                row[method]=value;l_values[method].append(value)
                selections.append(dict(panel=panel_l,task=task,method=method,subject=subject,initialization=config[0],mae_epoch=config[1],seed=config[2],representation=rep,checkpoint=str(config_path(task,config))))
            l_rows.append(row)
        z=[np.arctanh(np.clip(l_values[m],-.999999,.999999)) for m in METHODS]
        one,two=exact_signflip(z[1]-z[0])
        statistics.append(dict(panel=panel_l,task=task,method='paired',metric='exact sign flip on Fisher-z difference; conditional',
            value=float(np.mean(z[1]-z[0])),p_one_sided=one,p_two_sided=two,stars=stars(two),n_subjects=17,n_stimuli=n_stimuli))
        for method,values in zip(METHODS,z):
            statistics.append(dict(panel=panel_l,task=task,method=method,metric='Fisher-z group mean',value=float(np.tanh(np.mean(values))),n_subjects=17))
        print(f'{task}: metrics recomputed',flush=True)
    kframe=pd.DataFrame(k_rows);lframe=pd.DataFrame(l_rows)
    for task,pk,pl in [('narratives','A','B'),('shortvideo','C','D')]:
        for frame,panel,filename,keys,values in [
            (kframe,pk,f'{task.upper()}_K_SOURCE_DATA.csv',['subject_i','subject_j'],['subjective_rating_distance',*METHODS]),
            (lframe,pl,f'{task.upper()}_L_PLOT_DATA.csv',['subject'],list(METHODS))]:
            fresh=frame[frame.task==task].copy()
            fresh[keys+values].to_csv(out/f'{panel}_plot_data.csv',index=False)
    stats=pd.DataFrame(statistics)
    stats.to_csv(out/'statistics.csv',index=False)
    pd.DataFrame(cohorts).to_csv(out/'participant_ids.csv',index=False)
    pd.DataFrame(selections).to_csv(out/'selection_manifest.csv',index=False)
    pd.DataFrame(models).to_csv(out/'weights_manifest.csv',index=False)
    summary=dict(status='COMPUTED',n_loaded_checkpoints=len(models),n_pairs_per_scatter=136,n_participants_per_panel=17,
        no_training=True,label_correction='Original NeuroSTORM label maps to observation-expert MLP/pre-PoE, not a separate NeuroSTORM model.',
        elapsed_seconds=time.monotonic()-start)
    (out/'validation.json').write_text(json.dumps(summary,indent=2))
    (out/'runtime_versions.json').write_text(json.dumps({p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','torch','matplotlib','h5py']},indent=2))
    (out/'code_sha256.json').write_text(json.dumps({p.name:sha(p) for p in HERE.glob('*.py')},indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    from plot import render
    render(out)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
