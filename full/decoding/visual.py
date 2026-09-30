import math
import torch
from torch import nn
from diffusers.models.autoencoders.vae import Decoder
CIFTI_DIM=4096
MODES_DIM=2000
HIDDEN_DIM=1024
class BrainNetwork(nn.Module):

    def __init__(self, h=4096, in_dim=15724, out_dim=768, seq_len=2, n_blocks=4, drop=0.15, clip_size=768, blurry_recon=True, clip_scale=1):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size
        self.blurry_recon = blurry_recon
        self.clip_scale = clip_scale
        self.mixer_blocks1 = nn.ModuleList([self.mixer_block1(h, drop) for _ in range(n_blocks)])
        self.mixer_blocks2 = nn.ModuleList([self.mixer_block2(seq_len, drop) for _ in range(n_blocks)])
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True)
        self.clip_proj = self.projector(clip_size, clip_size, h=clip_size)
        if self.blurry_recon:
            self.blin1 = nn.Linear(h * seq_len, 4 * 28 * 28, bias=True)
            self.bdropout = nn.Dropout(0.3)
            self.bnorm = nn.GroupNorm(1, 64)
            self.bupsampler = Decoder(in_channels=64, out_channels=4, up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'], block_out_channels=[32, 64, 128], layers_per_block=1)
            self.b_maps_projector = nn.Sequential(nn.Conv2d(64, 512, 1, bias=False), nn.GroupNorm(1, 512), nn.ReLU(True), nn.Conv2d(512, 512, 1, bias=False), nn.GroupNorm(1, 512), nn.ReLU(True), nn.Conv2d(512, 512, 1, bias=True))

    def projector(self, in_dim, out_dim, h=2048):
        return nn.Sequential(nn.LayerNorm(in_dim), nn.GELU(), nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, out_dim))

    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(drop), nn.Linear(out_dim, out_dim))

    def mixer_block1(self, h, drop):
        return nn.Sequential(nn.LayerNorm(h), self.mlp(h, h, drop))

    def mixer_block2(self, seq_len, drop):
        return nn.Sequential(nn.LayerNorm(seq_len), self.mlp(seq_len, seq_len, drop))

    def forward(self, x):
        c, b = (torch.Tensor([0.0]), torch.Tensor([[0.0], [0.0]]))
        residual1 = x
        residual2 = x.permute(0, 2, 1)
        for block1, block2 in zip(self.mixer_blocks1, self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0, 2, 1)
            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0, 2, 1)
        x = x.reshape(x.size(0), -1)
        backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_size)
        if self.clip_scale > 0:
            c = self.clip_proj(backbone)
        if self.blurry_recon:
            b = self.blin1(x)
            b = self.bdropout(b)
            b = b.reshape(b.shape[0], -1, 7, 7).contiguous()
            b = self.bnorm(b)
            b_aux = self.b_maps_projector(b).flatten(2).permute(0, 2, 1)
            b_aux = b_aux.view(len(b_aux), 49, 512)
            b = (self.bupsampler(b), b_aux)
        return (backbone, c, b)

class GlobalPoE(nn.Module):


    def __init__(self):
        super().__init__()
        self.cifti_mean = nn.Linear(CIFTI_DIM, HIDDEN_DIM)
        self.modes_mean = nn.Linear(MODES_DIM, HIDDEN_DIM)
        self.cifti_log_precision = nn.Parameter(torch.full((HIDDEN_DIM,), -math.log(2.0)))
        self.modes_log_precision = nn.Parameter(torch.full((HIDDEN_DIM,), -math.log(2.0)))

    def forward(self, cifti: torch.Tensor, modes: torch.Tensor):
        cifti_mean = self.cifti_mean(cifti)
        modes_mean = self.modes_mean(modes)
        cifti_precision = torch.exp(3.0 * torch.tanh(self.cifti_log_precision / 3.0))
        modes_precision = torch.exp(3.0 * torch.tanh(self.modes_log_precision / 3.0))
        total_precision = cifti_precision + modes_precision
        fused_mean = (cifti_mean * cifti_precision + modes_mean * modes_precision) / total_precision
        fused_variance = total_precision.reciprocal().expand_as(fused_mean)
        return (fused_mean[:, None], {'fused_mean': fused_mean, 'fused_variance': fused_variance, 'cifti_precision': cifti_precision.expand_as(fused_mean), 'modes_precision': modes_precision.expand_as(fused_mean)})

class AnchoredPoE(GlobalPoE):


    def __init__(self):
        super().__init__()
        with torch.no_grad():
            self.cifti_log_precision.zero_()
            self.modes_log_precision.fill_(-2.0)
