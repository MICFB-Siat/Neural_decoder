import torch

def augment_test_destruction(emb, labels, epoch, max_epochs, destruction_level=0.5):














    device = emb.device
    B = emb.shape[0]
    progress = epoch / max_epochs


    decode_score = torch.rand(B, device=device)


    base_sigma_data = 0.04 + destruction_level * 0.29
    sigma_data = base_sigma_data * (0.65 + 0.7 * decode_score) * (1 - progress * 0.65)


    if emb.dim() == 3:
        noise = torch.randn_like(emb) * sigma_data.view(-1, 1, 1)
    else:
        noise = torch.randn_like(emb) * sigma_data.view(-1, 1)
    noisy_emb = emb + noise


    mask_prob = (0.06 + destruction_level * 0.37) * (0.9 + 0.25 * torch.rand_like(decode_score))
    mask = (torch.rand_like(emb) > mask_prob.view(-1, 1, 1) if emb.dim() == 3 else
            (torch.rand_like(emb) > mask_prob.view(-1, 1))).float()
    noisy_emb = noisy_emb * mask + (1 - mask) * emb.mean(dim=tuple(range(1, emb.dim())), keepdim=True)


    scale_factor = 0.08 + destruction_level * 0.15
    score_view = decode_score.view(-1, 1, 1) if emb.dim() == 3 else decode_score.view(-1, 1)
    scale = 1.0 + scale_factor * torch.randn_like(emb) * (0.7 + 0.6 * score_view)
    noisy_emb = noisy_emb * scale.clamp(0.83, 1.20)



    base_sigma_label = 0.028 + destruction_level * 0.038
    sigma_label = base_sigma_label * (1 - progress) * (0.75 + 0.5 * decode_score)
    noisy_labels = labels + torch.randn_like(labels) * sigma_label.view(-1, 1)





    actual_data_dest = sigma_data.mean().item()
    actual_label_dest = sigma_label.mean().item()

    return noisy_emb, noisy_labels, actual_data_dest, actual_label_dest
