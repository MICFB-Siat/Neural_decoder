import torch


def augment_test_destruction(emb, labels, epoch, max_epochs, destruction_level=0.62):
    device = emb.device
    B = emb.shape[0]
    decay = max(0.50, 1.0 - (epoch / max_epochs) * 0.85)
    decode_score = torch.rand(B, device=device)


    base_sigma_data = 0.045 + destruction_level * 0.31
    sigma_data = base_sigma_data * (0.7 + 0.65 * decode_score) * decay
    if emb.dim() == 3:
        noise = torch.randn_like(emb) * sigma_data.view(-1, 1, 1)
    else:
        noise = torch.randn_like(emb) * sigma_data.view(-1, 1)
    noisy_emb = emb + noise


    mask_prob = (0.07 + destruction_level * 0.38) * (0.9 + 0.3 * torch.rand_like(decode_score))
    mask = (torch.rand_like(emb) > mask_prob.view(-1, 1, 1) if emb.dim() == 3 else
            (torch.rand_like(emb) > mask_prob.view(-1, 1))).float()
    noisy_emb = noisy_emb * mask + (1 - mask) * emb.mean(dim=tuple(range(1, emb.dim())), keepdim=True)


    scale_factor = 0.09 + destruction_level * 0.16
    ds = decode_score.view(-1, 1, 1) if emb.dim() == 3 else decode_score.view(-1, 1)
    scale = 1.0 + scale_factor * torch.randn_like(emb) * (0.75 + 0.55 * ds)
    noisy_emb = noisy_emb * scale.clamp(0.82, 1.21)


    base_sigma_label = 0.032 + destruction_level * 0.055
    sigma_label = base_sigma_label * decay * (0.8 + 0.5 * decode_score)
    noisy_labels = labels + torch.randn_like(labels) * sigma_label.view(-1, 1)

    actual_data = sigma_data.mean().item()
    actual_label = sigma_label.mean().item()

    return noisy_emb, noisy_labels, actual_data, actual_label
