import torch
import torch.nn as nn
from einops import rearrange


def get_pcc(rec: torch.Tensor, raw: torch.Tensor):

    B, C, W, D = rec.shape
    x = rearrange(rec, "B C W D->(B C W) 1 D")
    y = rearrange(raw, "B C W D -> (B C W) 1 D")
    c = (
        (x - x.mean(dim=-1, keepdim=True))
        @ ((y - y.mean(dim=-1, keepdim=True)).transpose(1, 2))
        * (1.0 / (D - 1))
    ).squeeze()
    sigma = (torch.std(x, dim=-1) * torch.std(y, dim=-1)).squeeze() + 1e-6
    return (c / sigma).mean()


def compute_l1_loss(rec, raw):




    l1_distance = torch.abs(rec - raw)
    return torch.mean(l1_distance)


def get_time_loss(predicted, target):




    return compute_l1_loss(predicted, target)


def get_frequency_domain_loss(predicted, target):
    window = torch.hamming_window(target.shape[-1], device=predicted.device)
    predicted = window * predicted
    target = window * target

    pred_fft = torch.fft.rfft(predicted, dim=-1, norm="ortho")
    target_fft = torch.fft.rfft(target, dim=-1, norm="ortho")

    pred_magnitude = torch.abs(pred_fft)
    target_magnitude = torch.abs(target_fft)

    pred_phase = torch.angle(pred_fft)
    target_phase = torch.angle(target_fft)

    magnitude_loss = compute_l1_loss(pred_magnitude, target_magnitude)
    phase_loss = compute_l1_loss(pred_phase, target_phase)
    return magnitude_loss, phase_loss
