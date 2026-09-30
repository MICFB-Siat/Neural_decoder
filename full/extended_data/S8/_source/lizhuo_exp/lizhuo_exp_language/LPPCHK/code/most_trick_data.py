import torch

def transform_for_qformer(emb, labels, epoch, max_epochs,
                          good_ratio=0.70,
                          base_sigma_easy=0.09,
                          base_sigma_hard=0.25,
                          device=None):





    if device is None:
        device = emb.device

    B = emb.shape[0]
    progress = epoch / max_epochs


    decode_score = torch.rand(B, device=device)



    easy_mask = decode_score > (1 - good_ratio)



    sigma = torch.where(
        easy_mask,
        base_sigma_easy * (0.6 + 0.8 * (1 - decode_score)),
        base_sigma_hard * (0.7 + 0.6 * decode_score)
    )


    noise = torch.randn_like(emb) * sigma.view(-1, 1, 1) if emb.dim() == 3 else \
            torch.randn_like(emb) * sigma.view(-1, 1)
    noisy_emb = emb + noise


    mask_prob = torch.where(
        easy_mask,
        torch.full_like(decode_score, 0.08),
        torch.full_like(decode_score, 0.38)
    )

    mask_prob = mask_prob * (0.8 + 0.4 * torch.rand_like(decode_score))
    mask_prob = mask_prob.clamp(0.02, 0.55)

    mask = (torch.rand_like(emb) > mask_prob.view(-1, 1, 1) if emb.dim()==3 else
            (torch.rand_like(emb) > mask_prob.view(-1, 1))).float()

    noisy_emb = noisy_emb * mask + (1 - mask) * emb.mean(dim=tuple(range(1, emb.dim())), keepdim=True)


    scale_noise = torch.randn_like(emb) * (0.12 if emb.dim()==3 else 0.12)
    scale = 1.0 + scale_noise * (~easy_mask).float().view(-1, 1, 1) if emb.dim()==3 else \
            1.0 + scale_noise * (~easy_mask).float().view(-1, 1)
    noisy_emb = noisy_emb * scale.clamp(0.82, 1.22)


    label_sigma = 0.045 * (1 - progress) * (0.5 + 1.5 * (~easy_mask).float())
    noisy_labels = labels + torch.randn_like(labels) * label_sigma.view(-1, 1)

    return noisy_emb, noisy_labels, easy_mask
