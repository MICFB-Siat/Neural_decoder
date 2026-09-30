from __future__ import annotations
from pathlib import Path
import sys, random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parent
MINDEYE_SRC = ROOT / 'vendor/mindeye'
CLIP_DIM, CLIP_TOKENS = 1664, 256
UNCLIP_CHECKPOINT_FAST = None

def resolve_artifact(name, preferred=None):
    if preferred is None or not Path(preferred).is_file():
        raise FileNotFoundError('Set --unclip to the frozen unCLIP checkpoint')
    return Path(preferred)

def build_unclip_engine(device: torch.device, steps: int):
    sys.path.insert(0, str(MINDEYE_SRC))
    sys.path.insert(0, str(MINDEYE_SRC / "generative_models"))
    from generative_models.sgm.models.diffusion import DiffusionEngine
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(
        OmegaConf.load(MINDEYE_SRC / "generative_models/configs/unclip6.yaml"),
        resolve=True,
    )
    params = config["model"]["params"]
    first_stage = params["first_stage_config"]
    first_stage["target"] = "sgm.models.autoencoder.AutoencoderKL"
    first_stage["params"]["ddconfig"]["attn_type"] = "vanilla"
    params["sampler_config"]["params"]["num_steps"] = steps
    params["sampler_config"]["params"]["device"] = str(device)
    engine = DiffusionEngine(
        network_config=params["network_config"],
        denoiser_config=params["denoiser_config"],
        first_stage_config=first_stage,
        conditioner_config=params["conditioner_config"],
        sampler_config=params["sampler_config"],
        scale_factor=params["scale_factor"],
        disable_first_stage_autocast=params["disable_first_stage_autocast"],
    )
    checkpoint_path = resolve_artifact(
        "unclip6_epoch0_step110000.ckpt", preferred=UNCLIP_CHECKPOINT_FAST
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    engine.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    engine = engine.to(device).eval().requires_grad_(False)
    batch = {
        "jpg": torch.randn(1, 3, 1, 1, device=device),
        "original_size_as_tuple": torch.ones(1, 2, device=device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2, device=device),
    }
    vector_suffix = engine.conditioner(batch)["vector"].to(device)
    return engine, vector_suffix, checkpoint_path

def make_diffusion_prior() -> nn.Module:
    sys.path.insert(0, str(MINDEYE_SRC))
    from models import BrainDiffusionPrior, PriorNetwork

    prior_network = PriorNetwork(
        dim=CLIP_DIM,
        depth=6,
        dim_head=52,
        heads=CLIP_DIM // 52,
        causal=False,
        num_tokens=CLIP_TOKENS,
        learned_query_mode="pos_emb",
    )
    return BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=CLIP_DIM,
        condition_on_text_encodings=False,
        timesteps=100,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    )

def unclip_recon(x, diffusion_engine, vector_suffix,
                 num_samples=1, offset_noise_level=0.04):
    assert x.ndim==3
    device = x.device
    if x.shape[0]==1:
        x = x[[0]]
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), diffusion_engine.ema_scope():
        z = torch.randn(num_samples,4,96,96).to(device)



        token_shape = x.shape
        tokens = x
        c = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        tokens = torch.randn_like(x)
        uc = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        for k in c:
            c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))

        noise = torch.randn_like(z)
        sigmas = diffusion_engine.sampler.discretization(diffusion_engine.sampler.num_steps)
        sigma = sigmas[0].to(z.device)

        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(
                torch.randn(z.shape[0], device=z.device), z.ndim
            )
        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(
            1.0 + sigmas[0] ** 2.0
        )

        def denoiser(x, sigma, c):
            return diffusion_engine.denoiser(diffusion_engine.model, x, sigma, c)

        samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        samples_x = diffusion_engine.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x*.8+.2), min=0.0, max=1.0)

        return samples


def append_dims(x, target_dims):
    if x.ndim > target_dims: raise ValueError('Invalid broadcast dimensionality')
    return x[(...,) + (None,) * (target_dims - x.ndim)]


def reconstruct(data, weights, output, device, unclip, limit=None, indices=None, seed=42, prior_steps=20, steps=38):
    import gc
    from PIL import Image
    from visual import BrainNetwork, AnchoredPoE
    global UNCLIP_CHECKPOINT_FAST
    UNCLIP_CHECKPOINT_FAST = Path(unclip)
    device = torch.device(device)
    if device.type != 'cuda': raise ValueError('The historical diffusion pipeline requires CUDA.')

    x = np.load(data/'cifti_test.npy', mmap_mode='r')
    y = np.load(data/'modes_test.npy', mmap_mode='r')
    ids = list(indices) if indices is not None else list(range(len(x)))
    if limit is not None: ids = ids[:limit]
    if not ids or min(ids)<0 or max(ids)>=len(x): raise ValueError('Invalid sample indices')
    output.mkdir(parents=True, exist_ok=True)
    ck = torch.load(weights/'brain_final.pt',map_location='cpu',weights_only=False,mmap=True)
    fusion=AnchoredPoE().to(device).eval()
    fusion.load_state_dict(ck['fusion'],strict=True)
    brain=BrainNetwork(h=1024,in_dim=1024,seq_len=1,clip_size=1664,out_dim=1664*256,n_blocks=4,blurry_recon=True).to(device).eval()
    brain.load_state_dict(ck['backbone'],strict=True)
    del ck
    conditions=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for i in ids:
            z,_=fusion(torch.tensor(np.array(x[i:i+1]),device=device),torch.tensor(np.array(y[i:i+1]),device=device))
            conditions.append(brain(z)[0].float().cpu())
    del fusion,brain;gc.collect();torch.cuda.empty_cache()
    prior=make_diffusion_prior().to(device).eval().requires_grad_(False)
    ck=torch.load(weights/'diffusion_prior.pt',map_location='cpu',weights_only=False,mmap=True)
    prior.load_state_dict(ck['prior'],strict=True);del ck
    tokens=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        for i,condition in zip(ids,conditions):
            torch.manual_seed(seed+i);torch.cuda.manual_seed_all(seed+i)
            c=condition.to(device)
            tokens.append(prior.p_sample_loop(c.shape,text_cond={'text_embed':c},cond_scale=1.,timesteps=prior_steps).float().cpu())
    del prior,conditions;gc.collect();torch.cuda.empty_cache()
    engine,suffix,_=build_unclip_engine(device,steps)
    generated=[]
    with torch.inference_mode():
        for i,token in zip(ids,tokens):
            torch.manual_seed(seed+i);torch.cuda.manual_seed_all(seed+i)
            sample=unclip_recon(token.to(device,dtype=torch.float16),engine,suffix)[0]
            sample=F.interpolate(sample[None].float(),size=(256,256),mode='bilinear',align_corners=False)[0].clamp(0,1).cpu().numpy()
            generated.append(sample)
            Image.fromarray((sample.transpose(1,2,0)*255).round().astype('uint8')).save(output/f'trial_{i:04d}.png')
    np.savez_compressed(output/'reconstructions.npz',indices=np.asarray(ids),images=np.stack(generated))
    del engine;gc.collect();torch.cuda.empty_cache()
    return ids,np.stack(generated)
